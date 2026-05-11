#!/usr/bin/env python3
# ------------------------------------------------------------
# predict.py
#
# command-line entry point for ub_predictor.
#
# usage:
#
#   predict mode - score lysine sites using a pre-trained model:
#     python predict.py --input sites.csv --mode predict
#
#   predict with a custom model:
#     python predict.py --input sites.csv --mode predict
#                       --model path/to/model.pkl
#
#   train mode - train a new model on your own labelled data:
#     python predict.py --input labelled_sites.csv --mode train
#
#   train with an independent test set:
#     python predict.py --input labelled_sites.csv --mode train
#                       --test held_out_sites.csv
#
# input file format:
#
#   predict mode - csv with columns:
#     protein_id      : uniprot accession e.g. P04637
#     lysine_position : position of lysine in protein sequence
#
#   train mode - csv with additional column:
#     ub              : 1 = ubiquitinated, 0 = not ubiquitinated
#
# outputs are written to the outputs/ directory by default.
# use --output to specify a different location.
# ------------------------------------------------------------

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose=False):
    """
    Configure logging to both terminal and log file.

    terminal shows warnings and above by default.
    log file captures everything including debug messages.
    """
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ub_predictor.log"

    # root logger captures everything
    logging.basicConfig(
        level   = logging.DEBUG,
        format  = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers= [
            logging.FileHandler(log_file),
        ]
    )

    # terminal handler - warnings only unless verbose
    console          = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog        = "predict.py",
        description = "ub_predictor - ubiquitination site prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  predict using pre-trained model:
    python predict.py --input sites.csv --mode predict

  train on your own labelled data:
    python predict.py --input labelled.csv --mode train

  train with independent test set:
    python predict.py --input labelled.csv --mode train --test held_out.csv

  use a custom model for prediction:
    python predict.py --input sites.csv --mode predict --model my_model.pkl

input file columns:
  predict mode : protein_id, lysine_position
  train mode   : protein_id, lysine_position, ub (0 or 1)
        """
    )

    parser.add_argument(
            "--input", "-i",
            required = False,
            default = None,
            help     = "path to input csv file"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["predict", "train", "search"],
        default  = "predict",
        help     = "predict: use pre-trained model  |  "
                   "train: train on your own data  "
                   "(default: predict)"
    )

    parser.add_argument(
        "--model",
        default = None,
        help    = "path to model file (.pkl) - "
                  "predict mode only, overrides default model"
    )

    parser.add_argument(
        "--test",
        default = None,
        help    = "path to independent test csv - "
                  "train mode only, used for external validation"
    )

    parser.add_argument(
        "--output", "-o",
        default = "outputs",
        help    = "directory for output files (default: outputs/)"
    )

    parser.add_argument(
        "--structures",
        default = "data/structures",
        help    = "directory for alphafold structure files "
                  "(default: data/structures/)"
    )

    parser.add_argument(
        "--processed",
        default = "data/processed",
        help    = "directory for intermediate files "
                  "(default: data/processed/)"
    )

    parser.add_argument(
        "--models-dir",
        default = "models",
        help    = "directory for saved model files "
                  "(default: models/)"
    )

    parser.add_argument(
        "--threshold",
        type    = float,
        default = 0.5,
        help    = "probability threshold for ubiquitinated call "
                  "(default: 0.5)"
    )

    parser.add_argument(
            "--verbose", "-v",
            action  = "store_true",
            help    = "print detailed progress to terminal"
        )
    parser.add_argument(
        "--download-only",
        action  = "store_true",
        help    = "only download alphafold structure files and stop. "
                "useful for separating download and compute steps "
                "on hpc clusters where different resources are needed "
                "for each stage."
    )
        
        # search mode arguments
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="reference site for search mode, format: PROTEIN_ID,POSITION (e.g. Q8IXI2,572)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="path to csv of target protein ids to scan (search mode)"
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=None,
        help="max number of search results to return (default: all)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # set up logging
    setup_logging(args.verbose)

    # validate a few things before importing the heavy modules
    if args.mode != "search":
            if args.input is None:
                print("\n  error: --input is required for predict and train modes")
                sys.exit(1)
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"\n  error: input file not found - {args.input}")
                sys.exit(1)

    if args.mode == "predict" and args.test is not None:
        print(
            f"\n  note: --test is only used in train mode. "
            f"it will be ignored for predict mode."
        )

    if args.mode == "train" and args.model is not None:
        print(
            f"\n  note: --model is only used in predict mode. "
            f"in train mode a new model is always created."
        )
        
    if args.mode == "search":
        if args.reference is None:
            print("\n  error: --reference is required for search mode")
            sys.exit(1)
        if args.targets is None:
            print("\n  error: --targets is required for search mode")
            sys.exit(1)

        try:
            ref_parts = args.reference.split(",")
            ref_protein = ref_parts[0].strip()
            ref_position = int(ref_parts[1].strip())
        except (IndexError, ValueError):
            print(
                "\n  error: --reference must be in format "
                "PROTEIN_ID,POSITION (e.g. Q8IXI2,572)"
            )
            sys.exit(1)
        
        

    # import pipeline after arg validation
    # avoids slow imports if user just ran --help
    
    
    
    # handle download-only mode before importing heavy modules
    if args.download_only:
        import pandas as pd
        from ub_predictor.fetch_structures import fetch_all

        input_path  = Path(args.input)
        sites_df    = pd.read_csv(input_path)
        protein_ids = sites_df["protein_id"].unique().tolist()

        print(f"\n  download-only mode")
        print(f"  input    : {args.input}")
        print(f"  proteins : {len(protein_ids)}")
        print(f"  saving to: {args.structures}\n")

        fetch_all(protein_ids, args.structures)

        print(f"\n  downloads complete")
        print(f"  re-run without --download-only to generate "
              f"features and train/predict")
        sys.exit(0)

    from ub_predictor.pipeline import run

    # run the pipeline
    if args.mode == "search":
            from ub_predictor.search import run_search
            run_search(
                ref_protein_id = ref_protein,
                ref_position   = ref_position,
                targets_path   = args.targets,
                structures_dir = args.structures,
                output_dir     = args.output,
                n_results      = args.n_results,
            )
            sys.exit(0)

    else:
        run(
            sites_path      = args.input,
            mode            = args.mode,
            structures_dir  = args.structures,
            processed_dir   = args.processed,
            models_dir      = args.models_dir,
            output_dir      = args.output,
            model_path      = args.model,
            test_sites_path = args.test,
            threshold       = args.threshold,
        )

if __name__ == "__main__":
    main()