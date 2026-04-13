# ------------------------------------------------------------
# models/predict.py
#
# loads a trained model and generates predictions with
# per-site shap explanations for a feature matrix.
#
# shap (shapley additive explanations) assigns each feature
# a contribution score for each individual prediction.
# this tells you not just whether a site is predicted
# ubiquitinated, but which features drove that prediction -
# directly useful for biological interpretation.
#
# the model and preprocessor are always loaded together from
# the same saved file - this guarantees the same encoding
# is applied to new data as was used during training.
#
# input : feature matrix dataframe
#         saved model file (.pkl)
# output: dataframe with probability scores and shap values
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

log = logging.getLogger(__name__)

METADATA_COLS = ["protein_id", "lysine_position", "region_type", "ub"]


# ------------------------------------------------------------
# step 1 | load model bundle
# ------------------------------------------------------------

def load_model(model_path):
    """
    Load model and preprocessor from a saved bundle file.

    params:
        model_path : path to saved .pkl file

    returns (model, preprocessor) tuple
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"model file not found: {model_path}\n"
            f"run in train mode first to create a model"
        )

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model        = bundle["model"]
    preprocessor = bundle["preprocessor"]

    print(f"  model loaded from {model_path.name}")
    print(f"  {len(preprocessor.feature_names)} features expected")

    return model, preprocessor


# ------------------------------------------------------------
# step 2 | generate predictions
# ------------------------------------------------------------

def generate_predictions(model, preprocessor, df, threshold=0.5):
    """
    Generate probability scores and binary predictions.

    params:
        model        : fitted xgboost classifier
        preprocessor : fitted preprocessor from training
        df           : raw feature matrix dataframe
        threshold    : probability cutoff for binary prediction

    returns dataframe with probability and prediction columns added
    """
    X = preprocessor.transform(df)

    probabilities = model.predict_proba(X)[:, 1]
    predictions   = (probabilities >= threshold).astype(int)

    results = df[
        [c for c in METADATA_COLS if c in df.columns]
    ].copy()

    results["ub_probability"] = probabilities.round(4)
    results["predicted_ub"] = predictions
    results["prediction"]   = pd.Series(predictions).map(
        {1: "ubiquitinated", 0: "not ubiquitinated"}
    ).values

    n_predicted_ub = predictions.sum()
    print(f"  {n_predicted_ub} of {len(results)} sites "
          f"predicted ubiquitinated (threshold: {threshold})")

    return results


# ------------------------------------------------------------
# step 3 | generate shap explanations
# ------------------------------------------------------------

def generate_shap_values(model, preprocessor, df, top_n=5):
    """
    Generate per-site shap feature contribution scores.

    shap values explain why each site received its prediction.
    positive shap = feature pushed prediction towards ubiquitinated.
    negative shap = feature pushed prediction away from it.

    only the top_n most influential features per site are included
    to keep results readable.

    params:
        model        : fitted xgboost classifier
        preprocessor : fitted preprocessor from training
        df           : raw feature matrix dataframe
        top_n        : number of top features to report per site

    returns dataframe with top shap features and values per site
    """
    import xgboost as xgb

    X = preprocessor.transform(df)

    booster     = model.get_booster()
    dmatrix     = xgb.DMatrix(X)
    shap_values = booster.predict(dmatrix, pred_contribs=True)

    # last column is the bias term - remove it
    shap_values   = shap_values[:, :-1]
    feature_names = preprocessor.feature_names

    shap_rows = []

    for i in range(len(df)):
        site_shap   = shap_values[i]
        top_indices = np.argsort(np.abs(site_shap))[::-1][:top_n]

        row = {
            "protein_id"      : df["protein_id"].iloc[i],
            "lysine_position" : df["lysine_position"].iloc[i],
        }

        for rank, idx in enumerate(top_indices, 1):
            row[f"shap_feature_{rank}"] = feature_names[idx]
            row[f"shap_value_{rank}"]   = round(float(site_shap[idx]), 4)

        shap_rows.append(row)

    return pd.DataFrame(shap_rows)


# ------------------------------------------------------------
# step 4 | save output
# ------------------------------------------------------------

def save_predictions(predictions_df, shap_df, output_dir):
    """
    Merge predictions and shap values and save to csv.

    sorts by probability - highest confidence predictions first.

    params:
        predictions_df : from generate_predictions
        shap_df        : from generate_shap_values
        output_dir     : directory to write output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = predictions_df.merge(
        shap_df,
        on  = ["protein_id", "lysine_position"],
        how = "left"
    )

    combined = combined.sort_values(
        "ub_probability", ascending=False
    ).reset_index(drop=True)

    output_path = output_dir / "predictions.csv"
    combined.to_csv(output_path, index=False)

    print(f"  predictions saved to {output_path}")
    print(f"  {len(combined)} sites in output")

    return combined


# ------------------------------------------------------------
# main entry point
# ------------------------------------------------------------

def run_prediction(df, model_path, output_dir, threshold=0.5):
    """
    Run the full prediction pipeline.

    params:
        df          : feature matrix dataframe
        model_path  : path to saved model bundle
        output_dir  : directory to write predictions
        threshold   : probability cutoff for binary prediction

    returns combined predictions and shap dataframe
    """
    print(f"\n  -- prediction --\n")

    model, preprocessor = load_model(model_path)

    print(f"\n  generating predictions for {len(df)} sites")
    predictions_df = generate_predictions(
        model, preprocessor, df, threshold
    )

    print(f"\n  generating shap explanations")
    shap_df = generate_shap_values(model, preprocessor, df)

    combined = save_predictions(predictions_df, shap_df, output_dir)

    # print readable summary
    print(f"\n  top predicted ubiquitination sites:")
    print(f"  {'protein':<12} {'position':<10} "
          f"{'probability':<14} {'top feature':<30} {'shap'}")
    print(f"  {'-'*75}")

    for _, row in combined.head(10).iterrows():
        print(
            f"  {row['protein_id']:<12} "
            f"{row['lysine_position']:<10} "
            f"{row['ub_probability']:<14} "
            f"{str(row.get('shap_feature_1', '')):<30} "
            f"{row.get('shap_value_1', '')}"
        )

    return combined