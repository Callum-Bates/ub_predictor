# ------------------------------------------------------------
# models/evaluate.py
#
# generates evaluation metrics and plots for a trained model.
#
# used in two contexts:
#
#   during training  - evaluates model performance on the
#                      cross-validation results and optional
#                      held-out test set
#
#   after prediction - if the user provides known labels
#                      alongside their sites, evaluates how
#                      well the pre-trained model performed
#
# outputs:
#   training_report.txt  - metrics in plain readable text
#   roc_curve.png        - roc curve plot
#   feature_importance.png - top features by importance
#
# all plots are saved as files rather than displayed -
# the tool runs on hpc clusters where there is no screen.
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend - no display needed
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve,
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------
# step 1 | calculate metrics
# ------------------------------------------------------------

def calc_metrics(y_true, y_pred_proba, y_pred, label=""):
    """
    Calculate a full set of classification metrics.

    params:
        y_true       : true labels (0/1)
        y_pred_proba : predicted probabilities
        y_pred       : binary predictions
        label        : descriptor for this evaluation set
                       e.g. "cross-validation" or "test set"

    returns dict of metrics
    """
    roc_auc  = roc_auc_score(y_true, y_pred_proba)
    pr_auc   = average_precision_score(y_true, y_pred_proba)
    cm       = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "label"      : label,
        "roc_auc"    : round(roc_auc, 4),
        "pr_auc"     : round(pr_auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "tp"         : int(tp),
        "tn"         : int(tn),
        "fp"         : int(fp),
        "fn"         : int(fn),
        "n_positive" : int(y_true.sum()),
        "n_negative" : int((y_true == 0).sum()),
        "y_true"     : y_true,
        "y_pred_proba": y_pred_proba,
        "y_pred"     : y_pred,
    }


def calc_cv_metrics(cv_scores):
    """
    Summarise cross-validation fold scores.

    params:
        cv_scores : array of per-fold roc-auc scores

    returns dict of cv summary metrics
    """
    if len(cv_scores) == 0:
        return {}

    return {
        "cv_mean_auc" : round(float(np.mean(cv_scores)), 4),
        "cv_std_auc"  : round(float(np.std(cv_scores)), 4),
        "cv_min_auc"  : round(float(np.min(cv_scores)), 4),
        "cv_max_auc"  : round(float(np.max(cv_scores)), 4),
        "cv_n_folds"  : len(cv_scores),
        "cv_scores"   : [round(float(s), 4) for s in cv_scores],
    }


# ------------------------------------------------------------
# step 2 | write training report
# ------------------------------------------------------------

def write_report(cv_metrics, test_metrics=None, output_dir="."):
    """
    Write a plain text training report.

    readable by anyone - no code needed to interpret results.
    saved alongside the model file.

    params:
        cv_metrics   : dict from calc_cv_metrics
        test_metrics : optional dict from calc_metrics for test set
        output_dir   : directory to write report into
    """
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "training_report.txt"

    lines = []
    lines.append("ub_predictor - model training report")
    lines.append("=" * 50)
    lines.append(f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # cross-validation results
    lines.append("cross-validation results")
    lines.append("-" * 30)

    if cv_metrics:
        lines.append(
            f"folds             : {cv_metrics['cv_n_folds']}"
        )
        lines.append(
            f"mean roc-auc      : {cv_metrics['cv_mean_auc']}"
        )
        lines.append(
            f"std roc-auc       : {cv_metrics['cv_std_auc']}"
        )
        lines.append(
            f"min roc-auc       : {cv_metrics['cv_min_auc']}"
        )
        lines.append(
            f"max roc-auc       : {cv_metrics['cv_max_auc']}"
        )
        lines.append(
            f"per-fold scores   : {cv_metrics['cv_scores']}"
        )
    else:
        lines.append("cross-validation was not run")

    lines.append("")
    lines.append(
        "note: cross-validation estimates performance within your"
    )
    lines.append(
        "training dataset. all folds share the same experimental"
    )
    lines.append(
        "conditions - this is an optimistic estimate of real-world"
    )
    lines.append(
        "performance. use --test with a held-out dataset for a"
    )
    lines.append(
        "more conservative independent estimate."
    )

    # independent test results if provided
    if test_metrics:
        lines.append("")
        lines.append("independent test set results")
        lines.append("-" * 30)
        lines.append(
            f"roc-auc           : {test_metrics['roc_auc']}"
        )
        lines.append(
            f"pr-auc            : {test_metrics['pr_auc']}"
        )
        lines.append(
            f"sensitivity       : {test_metrics['sensitivity']}"
        )
        lines.append(
            f"specificity       : {test_metrics['specificity']}"
        )
        lines.append("")
        lines.append("confusion matrix:")
        lines.append(
            f"  true positive   : {test_metrics['tp']}"
        )
        lines.append(
            f"  true negative   : {test_metrics['tn']}"
        )
        lines.append(
            f"  false positive  : {test_metrics['fp']}"
        )
        lines.append(
            f"  false negative  : {test_metrics['fn']}"
        )
        lines.append("")
        lines.append(
            f"test set size     : {test_metrics['n_positive']} "
            f"ubiquitinated, {test_metrics['n_negative']} not ubiquitinated"
        )
        lines.append("")
        lines.append(
            "note: this is a genuine independent evaluation - "
            "these sites were not used during training."
        )

    lines.append("")
    lines.append("=" * 50)

    report_text = "\n".join(lines)

    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"  training report saved to {report_path.name}")
    return report_path


# ------------------------------------------------------------
# step 3 | roc curve plot
# ------------------------------------------------------------

def plot_roc_curve(cv_metrics, test_metrics=None, output_dir="."):
    """
    Plot roc curve(s) and save to file.

    shows cross-validation mean roc curve and optionally
    the independent test set roc curve for comparison.

    params:
        cv_metrics   : dict from calc_cv_metrics - needs y_true
                       and y_pred_proba from final cv model
        test_metrics : optional dict from calc_metrics
        output_dir   : directory to save plot into
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path  = output_dir / "roc_curve.png"

    fig, ax = plt.subplots(figsize=(7, 6))

    # plot cv roc curve if we have predictions
    if "y_true" in cv_metrics and cv_metrics["y_true"] is not None:
        fpr, tpr, _ = roc_curve(
            cv_metrics["y_true"],
            cv_metrics["y_pred_proba"]
        )
        ax.plot(
            fpr, tpr,
            color="steelblue",
            linewidth=2,
            label=f"cross-validation "
                  f"(auc = {cv_metrics['roc_auc']:.3f})"
        )

    # plot independent test roc curve if provided
    if test_metrics and "y_true" in test_metrics:
        fpr_t, tpr_t, _ = roc_curve(
            test_metrics["y_true"],
            test_metrics["y_pred_proba"]
        )
        ax.plot(
            fpr_t, tpr_t,
            color="firebrick",
            linewidth=2,
            linestyle="--",
            label=f"independent test "
                  f"(auc = {test_metrics['roc_auc']:.3f})"
        )

    # random classifier baseline
    ax.plot(
        [0, 1], [0, 1],
        color="grey",
        linewidth=1,
        linestyle=":",
        label="random classifier (auc = 0.500)"
    )

    ax.set_xlabel("false positive rate", fontsize=12)
    ax.set_ylabel("true positive rate", fontsize=12)
    ax.set_title("roc curve - ubiquitination site prediction", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  roc curve saved to {plot_path.name}")
    return plot_path


# ------------------------------------------------------------
# step 4 | feature importance plot
# ------------------------------------------------------------

def plot_feature_importance(model, feature_names, output_dir=".",
                            top_n=20):
    """
    Plot top feature importances from the trained xgboost model.

    uses xgboost's built-in feature importance scores (gain) -
    gain measures how much each feature improves the model when
    it is used in a tree split. higher gain = more useful feature.

    params:
        model         : fitted xgboost classifier
        feature_names : list of feature names from preprocessor
        output_dir    : directory to save plot into
        top_n         : number of top features to show
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path  = output_dir / "feature_importance.png"

    # get feature importance scores from xgboost
    importance = model.feature_importances_

    # build dataframe and sort
    importance_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, top_n * 0.4 + 1))

    ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color="steelblue",
        alpha=0.8
    )

    ax.set_xlabel("feature importance (gain)", fontsize=11)
    ax.set_title(
        f"top {top_n} features - ubiquitination site prediction",
        fontsize=12
    )
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  feature importance plot saved to {plot_path.name}")
    return plot_path


# ------------------------------------------------------------
# main entry point
# ------------------------------------------------------------

def run_evaluation(
    model,
    preprocessor,
    cv_scores,
    features_df,
    y_true_cv,
    y_pred_proba_cv,
    output_dir,
    test_df=None,
    threshold=0.5
):
    """
    Run full evaluation and save all outputs.

    params:
        model            : fitted xgboost classifier
        preprocessor     : fitted preprocessor
        cv_scores        : array of per-fold cv roc-auc scores
        features_df      : feature matrix (for final cv predictions)
        y_true_cv        : true labels used in final cv evaluation
        y_pred_proba_cv  : predicted probabilities from cv
        output_dir       : directory to write all outputs
        test_df          : optional independent test feature matrix
                           must contain "ub" column
        threshold        : probability cutoff for binary prediction
    """
    print(f"\n  -- evaluation --\n")
    output_dir = Path(output_dir)

    # cv metrics
    cv_summary = calc_cv_metrics(cv_scores)
    y_pred_cv  = (y_pred_proba_cv >= threshold).astype(int)

    # add predictions to cv summary for roc plot
    cv_summary["y_true"]       = y_true_cv
    cv_summary["y_pred_proba"] = y_pred_proba_cv
    cv_summary["roc_auc"]      = cv_summary["cv_mean_auc"]

    # independent test metrics if provided
    test_metrics = None

    if test_df is not None:
        print(f"  evaluating on independent test set")

        if "ub" not in test_df.columns:
            log.warning(
                "  test dataframe has no 'ub' column - "
                "skipping independent test evaluation"
            )
        else:
            y_true_test      = test_df["ub"].astype(int)
            X_test           = preprocessor.transform(test_df)
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
            y_pred_test      = (y_pred_proba_test >= threshold).astype(int)

            test_metrics = calc_metrics(
                y_true_test,
                y_pred_proba_test,
                y_pred_test,
                label="independent test set"
            )

            print(
                f"  independent test roc-auc: "
                f"{test_metrics['roc_auc']}"
            )

    # write report
    write_report(cv_summary, test_metrics, output_dir)

    # roc curve
    plot_roc_curve(cv_summary, test_metrics, output_dir)

    # feature importance
    plot_feature_importance(
        model,
        preprocessor.feature_names,
        output_dir
    )

    # print summary to terminal
    print(f"\n  cross-validation auc : {cv_summary['cv_mean_auc']:.3f} "
          f"(+/- {cv_summary['cv_std_auc']:.3f})")

    if test_metrics:
        print(
            f"  independent test auc : {test_metrics['roc_auc']:.3f}"
        )

    return cv_summary, test_metrics