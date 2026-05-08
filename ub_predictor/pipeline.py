# ------------------------------------------------------------
# pipeline.py
#
# orchestrates the full ub_predictor pipeline.
#
# two modes:
#
#   predict - loads a pre-trained model and scores lysine sites.
#             input: protein_id, lysine_position
#             output: probability scores with shap explanations
#
#   train   - trains a new model on labelled data, evaluates
#             with cross-validation, saves model and reports.
#             input: protein_id, lysine_position, ub (0 or 1)
#             output: model file, training report, plots
#
# intermediate feature files are checkpointed after each step.
# re-running resumes from the last completed step rather than
# starting over - useful for long runs on hpc clusters.
#
# input : sites csv
# output: predictions.csv + training outputs (train mode)
# ------------------------------------------------------------

import pandas as pd
import logging
import sys
from pathlib import Path
from datetime import datetime

from ub_predictor.fetch_structures   import fetch_all
from ub_predictor.idr_filter         import run as filter_idrs
from ub_predictor.features.sequence  import add_sequence_features
from ub_predictor.features.rasa      import add_rasa_features
from ub_predictor.features.spatial   import add_spatial_features
from ub_predictor.features.structure import add_structure_features
from ub_predictor.models.preprocess  import Preprocessor
from ub_predictor.models.train       import run_training
from ub_predictor.models.predict     import run_prediction, load_model
from ub_predictor.models.evaluate    import run_evaluation
from ub_predictor.search import run_search
log = logging.getLogger(__name__)


# ------------------------------------------------------------
# checkpoint helpers
# ------------------------------------------------------------

def load_checkpoint(path):
    """
    Load a checkpoint csv if it exists.

    params:
        path : Path to checkpoint file

    returns dataframe if exists, None otherwise
    """
    path = Path(path)
    if path.exists():
        print(f"  checkpoint found - loading {path.name}")
        return pd.read_csv(path)
    return None


def save_checkpoint(df, path):
    """
    Save a dataframe as a checkpoint csv.

    params:
        df   : dataframe to save
        path : Path to write to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.debug(f"  checkpoint saved - {path.name}")


# ------------------------------------------------------------
# step 1 | validate input
# ------------------------------------------------------------

def validate_input(sites_path, mode):
    """
    Load and validate the user input csv.

    predict mode requires: protein_id, lysine_position
    train mode also requires: ub (0 or 1)

    params:
        sites_path : path to input csv
        mode       : "predict" or "train"

    returns validated dataframe
    """
    sites_path = Path(sites_path)

    if not sites_path.exists():
        print(f"\n  error: input file not found - {sites_path}")
        sys.exit(1)

    df = pd.read_csv(sites_path)

    required = {"protein_id", "lysine_position"}
    if mode == "train":
        required.add("ub")

    missing = required - set(df.columns)
    if missing:
        print(f"\n  error: input file missing columns: {missing}")
        if mode == "train":
            print(
                f"  train mode requires protein_id, lysine_position, "
                f"and ub (0 = not ubiquitinated, 1 = ubiquitinated)"
            )
        sys.exit(1)

    if mode == "train":
        invalid = ~df["ub"].isin([0, 1])
        if invalid.any():
            print(
                f"\n  error: ub column contains values other than "
                f"0 and 1 ({invalid.sum()} invalid rows)"
            )
            sys.exit(1)

        print(f"  {len(df)} sites loaded")
        print(
            f"  {df['ub'].sum()} ubiquitinated, "
            f"{(df['ub'] == 0).sum()} not ubiquitinated"
        )
    else:
        print(f"  {len(df)} sites loaded")

    return df


# ------------------------------------------------------------
# step 2 | fetch structures
# ------------------------------------------------------------

def run_fetch(sites_df, structures_dir):
    """
    Download cif and pae files for all proteins in the input.

    params:
        sites_df       : validated input dataframe
        structures_dir : directory to save structure files
    """
    protein_ids = sites_df["protein_id"].unique().tolist()
    print(f"\n  -- fetch structures --\n")
    fetch_all(protein_ids, structures_dir)


# ------------------------------------------------------------
# step 3 | idr filter
# ------------------------------------------------------------

def run_filter(sites_path, structures_dir, processed_dir):
    """
    Filter lysine sites by disorder prediction.

    checkpointed - skips if sites_structured.csv already exists.

    params:
        sites_path     : path to input csv
        structures_dir : directory containing cif files
        processed_dir  : directory to write output files

    returns structured sites dataframe
    """
    checkpoint = Path(processed_dir) / "sites_structured.csv"
    existing   = load_checkpoint(checkpoint)

    if existing is not None:
        return existing

    print(f"\n  -- idr filter --\n")
    structured, _, _ = filter_idrs(
        sites_path     = sites_path,
        structures_dir = structures_dir,
        out_dir        = processed_dir
    )

    return structured


# ------------------------------------------------------------
# step 4 | feature generation
# ------------------------------------------------------------

def run_features(structured_df, structures_dir, processed_dir):
    """
    Run all feature modules in sequence with checkpointing.

    skips any step whose checkpoint file already exists,
    allowing the pipeline to resume after a crash.

    params:
        structured_df  : structured sites from idr filter
        structures_dir : directory containing cif files
        processed_dir  : directory for checkpoint files

    returns complete feature matrix dataframe
    """
    processed_dir = Path(processed_dir)

    # sequence features
    seq_ckpt = processed_dir / "features_sequence.csv"
    df       = load_checkpoint(seq_ckpt)

    if df is None:
        print(f"\n  -- sequence features --")
        df = add_sequence_features(structured_df)
        save_checkpoint(df, seq_ckpt)

    # rasa features
    rasa_ckpt = processed_dir / "features_rasa.csv"
    df_rasa   = load_checkpoint(rasa_ckpt)

    if df_rasa is None:
        print(f"\n  -- rasa features --")
        df_rasa = add_rasa_features(df, cif_dir=structures_dir)
        save_checkpoint(df_rasa, rasa_ckpt)

    # spatial features
    spatial_ckpt = processed_dir / "features_spatial.csv"
    df_spatial   = load_checkpoint(spatial_ckpt)

    if df_spatial is None:
        print(f"\n  -- spatial features --")
        df_spatial = add_spatial_features(
            df_rasa, cif_dir=structures_dir
        )
        save_checkpoint(df_spatial, spatial_ckpt)

    # secondary structure features
    complete_ckpt = processed_dir / "features_complete.csv"
    df_complete   = load_checkpoint(complete_ckpt)

    if df_complete is None:
        print(f"\n  -- secondary structure features --")
        df_complete = add_structure_features(
            df_spatial, cif_dir=structures_dir
        )
        save_checkpoint(df_complete, complete_ckpt)

    print(f"\n  feature matrix complete")
    print(f"  {df_complete.shape[0]} sites, "
          f"{df_complete.shape[1]} columns")

    return df_complete


# ------------------------------------------------------------
# step 5a | train mode
# ------------------------------------------------------------

def run_train_mode(
    features_df,
    sites_df,
    models_dir,
    output_dir,
    test_sites_path=None,
    structures_dir=None,
    processed_dir=None,
    threshold=0.5
):
    """
    Train a new model on labelled data and evaluate it.

    params:
        features_df     : complete feature matrix
        sites_df        : original input with ub labels
        models_dir      : directory to save model file
        output_dir      : directory for reports and plots
        test_sites_path : optional path to independent test csv
        structures_dir  : needed if test_sites_path provided
        processed_dir   : needed if test_sites_path provided
        threshold       : probability cutoff for binary prediction
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # drop ub if already present in feature matrix
    # can happen if a previous run saved labels into the checkpoint
    features_df = features_df.drop(columns=["ub"], errors="ignore")

    # merge ub labels back onto feature matrix
    label_df    = sites_df[["protein_id", "lysine_position", "ub"]]
    features_df = features_df.merge(
        label_df,
        on  = ["protein_id", "lysine_position"],
        how = "inner"
    )

    n_pos = features_df["ub"].sum()
    n_neg = (features_df["ub"] == 0).sum()
    print(f"\n  {n_pos} ubiquitinated sites")
    print(f"  {n_neg} non-ubiquitinated sites")

    # train
    model_path           = Path(models_dir) / "ub_predictor_model.pkl"
    model, preprocessor, metrics = run_training(
        features_df,
        model_save_path=model_path
    )

    # get cv predictions for evaluation
    from ub_predictor.models.preprocess import Preprocessor
    X  = preprocessor.transform(
        features_df.drop(columns=["ub"], errors="ignore")
    )
    y  = features_df["ub"].astype(int)
    yp = model.predict_proba(X)[:, 1]

    # load and process independent test set if provided
    test_features_df = None

    if test_sites_path is not None:
        print(f"\n  loading independent test set from {test_sites_path}")
        test_df = validate_input(test_sites_path, mode="train")

        # run features on test set using same checkpoints dir
        # but separate checkpoint files
        test_processed = Path(processed_dir) / "test_set"
        test_structured = run_filter(
            test_sites_path, structures_dir, test_processed
        )
        test_features_df = run_features(
            test_structured, structures_dir, test_processed
        )

        # merge test labels
        test_labels      = test_df[
            ["protein_id", "lysine_position", "ub"]
        ]
        test_features_df = test_features_df.merge(
            test_labels,
            on  = ["protein_id", "lysine_position"],
            how = "inner"
        )

    # evaluate
    run_evaluation(
        model           = model,
        preprocessor    = preprocessor,
        cv_scores       = metrics["cv_scores"],
        features_df     = features_df,
        y_true_cv       = y,
        y_pred_proba_cv = yp,
        output_dir      = output_dir,
        test_df         = test_features_df,
        threshold       = threshold
    )

    return model, preprocessor


# ------------------------------------------------------------
# step 5b | predict mode
# ------------------------------------------------------------

def run_predict_mode(
    features_df,
    model_path,
    output_dir,
    threshold=0.5
):
    """
    Load a pre-trained model and predict on new sites.

    params:
        features_df : complete feature matrix
        model_path  : path to saved model bundle (.pkl)
        output_dir  : directory for output files
        threshold   : probability cutoff for binary prediction
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = run_prediction(
        df         = features_df,
        model_path = model_path,
        output_dir = output_dir,
        threshold  = threshold
    )

    return results


# ------------------------------------------------------------
# step 5c | search mode
# ------------------------------------------------------------

def run_search_mode(
    ref_protein_id,
    ref_position,
    targets_path,
    structures_dir,
    output_dir,
    n_results=None,
):
    """
    run structural similarity search against target proteins.

    generates features for a reference site and all lysines
    in target proteins, then ranks by gower distance.

    params:
        ref_protein_id : uniprot id of the reference protein
        ref_position   : lysine position in the reference protein
        targets_path   : path to csv with protein_id column
        structures_dir : directory containing cif files
        output_dir     : directory to write results
        n_results      : max results to return (None = all)
    """
    results = run_search(
        ref_protein_id=ref_protein_id,
        ref_position=ref_position,
        targets_path=targets_path,
        structures_dir=structures_dir,
        output_dir=output_dir,
        n_results=n_results,
    )

    return results

# ------------------------------------------------------------
# main entry point
# ------------------------------------------------------------

def run(
    sites_path,
    mode           = "predict",
    structures_dir = "data/structures",
    processed_dir  = "data/processed",
    models_dir     = "models",
    output_dir     = "outputs",
    model_path     = None,
    test_sites_path= None,
    threshold      = 0.5,
    ref_protein_id = None,
    ref_position   = None,
    targets_path   = None,
    n_results      = None,
):
    """
    Run the full ub_predictor pipeline.
    ...
    """
    # generate a unique run id from timestamp + input filename
    # this keeps all outputs and checkpoints from each run separate
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_stem  = Path(sites_path).stem
    run_id      = f"{timestamp}_{input_stem}"

    # all outputs and checkpoints go into run-specific subfolders
    processed_dir = str(Path(processed_dir) / run_id)
    output_dir    = str(Path(output_dir)    / run_id)

    print(f"\n  ub_predictor - {mode} mode")
    print(f"  input    : {sites_path}")
    print(f"  run id   : {run_id}")
    print(f"  outputs  : {output_dir}")
    print(f"  {'='*48}\n")

# search mode - separate flow, exits early
    if mode == "search":
        targets_df = pd.read_csv(targets_path)
        target_ids = targets_df["protein_id"].unique().tolist()
        if ref_protein_id not in target_ids:
            target_ids.append(ref_protein_id)

        fetch_all(target_ids, out_dir=structures_dir)

        run_search_mode(
            ref_protein_id=ref_protein_id,
            ref_position=ref_position,
            targets_path=targets_path,
            structures_dir=structures_dir,
            output_dir=output_dir,
            n_results=n_results,
        )

        print(f"\n  {'='*48}")
        print(f"  pipeline complete")
        print(f"  outputs saved to {output_dir}/")
        return
    
    # validate input
    sites_df = validate_input(sites_path, mode)

    # fetch structures
    run_fetch(sites_df, structures_dir)

    # idr filter
    structured_df = run_filter(
        sites_path, structures_dir, processed_dir
    )

    if len(structured_df) == 0:
        print(
            f"\n  no structured sites remaining after idr filter.\n"
            f"  check {processed_dir}/sites_disordered.csv and "
            f"sites_no_structure.csv"
        )
        sys.exit(0)

    # feature generation
    features_df = run_features(
        structured_df, structures_dir, processed_dir
    )

    # model step
    if mode == "train":
        model, preprocessor = run_train_mode(
            features_df     = features_df,
            sites_df        = sites_df,
            models_dir      = models_dir,
            output_dir      = output_dir,
            test_sites_path = test_sites_path,
            structures_dir  = structures_dir,
            processed_dir   = processed_dir,
            threshold       = threshold
        )

    else:
        # predict mode - use default model if none specified
        if model_path is None:
            model_path = Path(models_dir) / "ub_predictor_model.pkl"

        if not Path(model_path).exists():
            print(
                f"\n  error: no model found at {model_path}\n"
                f"  run in train mode first, or provide a "
                f"pre-trained model with --model"
            )
            sys.exit(1)

        results = run_predict_mode(
            features_df = features_df,
            model_path  = model_path,
            output_dir  = output_dir,
            threshold   = threshold
        )

    print(f"\n  {'='*48}")
    print(f"  pipeline complete")
    print(f"  outputs saved to {output_dir}/")