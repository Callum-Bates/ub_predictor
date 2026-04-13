# ------------------------------------------------------------
# models/train.py
#
# trains an xgboost classifier on a labelled feature matrix
# and evaluates its performance.
#
# the training process:
#   1. split data into train and test sets (stratified)
#   2. fit preprocessor on training data only
#   3. train xgboost with cross-validation to assess stability
#   4. evaluate on held-out test set
#   5. save model and preprocessor together as one file
#
# the model and preprocessor are always saved together - you
# need both to make predictions on new data. saving them
# separately risks using a mismatched preprocessor.
#
# input : feature matrix dataframe with "ub" label column
# output: saved model file (.pkl)
#         evaluation metrics dict
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import roc_auc_score, classification_report, confusion_matrix

import xgboost as xgb

from ub_predictor.models.preprocess import Preprocessor

log = logging.getLogger(__name__)

# columns that are not features and should never go into the model
METADATA_COLS = ["protein_id", "lysine_position", "region_type", "ub"]


# ------------------------------------------------------------
# step 1 | prepare data for training
# ------------------------------------------------------------

def prepare_data(df):
    """
    Split feature matrix into features and labels.

    removes metadata columns and separates the ub label.

    params:
        df : feature matrix dataframe with ub column

    returns X (features dataframe), y (labels series)
    """
    if "ub" not in df.columns:
        raise ValueError(
            "dataframe has no 'ub' column - "
            "train mode requires labelled data"
        )

    y = df["ub"].astype(int)
    X = df.drop(columns=[c for c in METADATA_COLS if c in df.columns])

    print(f"  {len(X)} sites, {X.shape[1]} raw features")
    print(f"  {y.sum()} ubiquitinated (ub=1), "
          f"{(y == 0).sum()} not ubiquitinated (ub=0)")
    print(f"  positive rate: {y.mean():.3f}")

    return X, y


# ------------------------------------------------------------
# step 2 | train the model
# ------------------------------------------------------------

def train(X_train, y_train, scale_pos_weight=None):
    """
    Train an xgboost classifier.

    scale_pos_weight handles class imbalance - set to
    n_negative / n_positive so the model pays equal attention
    to both classes. if none, calculated automatically.

    params:
        X_train          : preprocessed training feature array
        y_train          : training labels
        scale_pos_weight : class weight for imbalance correction

    returns fitted xgboost classifier
    """
    if scale_pos_weight is None:
        n_pos            = y_train.sum()
        n_neg            = (y_train == 0).sum()
        scale_pos_weight = n_neg / n_pos
        print(f"  scale_pos_weight set to {scale_pos_weight:.2f} "
              f"({n_neg} negative / {n_pos} positive)")

    model = xgb.XGBClassifier(
        n_estimators     = 500,
        max_depth        = 6,
        learning_rate    = 0.1,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos_weight,
        random_state     = 42,
        eval_metric      = "logloss",
        verbosity        = 0,
    )

    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------
# step 3 | cross-validation
# ------------------------------------------------------------

def cross_validate(X, y, n_folds=5):
    """
    Assess model stability with stratified k-fold cross-validation.

    cross-validation trains and evaluates the model n_folds times
    on different splits of the data. the mean and standard deviation
    of roc-auc scores tells you how stable the model is - a high
    standard deviation suggests the model is sensitive to which
    sites end up in the training vs test set.

    params:
        X       : preprocessed feature array
        y       : labels
        n_folds : number of cross-validation folds

    returns array of roc-auc scores, one per fold
    """
    # with small datasets, cap folds at number of samples
    # this only affects toy datasets - real data will always
    # have enough samples for 5-fold cv
    min_class_count = y.value_counts().min()
    n_folds         = min(n_folds, min_class_count)
    if n_folds < 2:
        print(f"  skipping cross-validation - too few samples ({len(y)})")
        return np.array([])

    print(f"\n  running {n_folds}-fold cross-validation")

    n_pos            = y.sum()
    n_neg            = (y == 0).sum()
    scale_pos_weight = n_neg / n_pos

    cv_model = xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.1,
        scale_pos_weight = scale_pos_weight,
        random_state     = 42,
        eval_metric      = "logloss",
        verbosity        = 0,
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    scores = cross_val_score(
        cv_model, X, y,
        cv      = cv,
        scoring = "roc_auc"
    )

    print(f"  roc-auc per fold: {[round(s, 3) for s in scores]}")
    print(f"  mean: {scores.mean():.3f}  std: {scores.std():.3f}")

    return scores


# ------------------------------------------------------------
# step 4 | evaluate on held-out test set
# ------------------------------------------------------------

def evaluate(model, X_test, y_test):
    """
    Evaluate model performance on held-out test data.

    params:
        model  : fitted xgboost classifier
        X_test : preprocessed test feature array
        y_test : test labels

    returns dict of evaluation metrics
    """
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n  test set evaluation")
    print(f"  roc-auc: {roc_auc:.3f}")
    print(f"\n  classification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["not ub", "ubiquitinated"],
                                zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  confusion matrix:")
    print(f"    true negative : {cm[0][0]}  false positive: {cm[0][1]}")
    print(f"    false negative: {cm[1][0]}  true positive : {cm[1][1]}")

    return {
        "roc_auc"       : roc_auc,
        "y_pred"        : y_pred,
        "y_pred_proba"  : y_pred_proba,
        "confusion_matrix": cm,
    }


# ------------------------------------------------------------
# step 5 | save model and preprocessor together
# ------------------------------------------------------------

def save_model(model, preprocessor, path):
    """
    Save model and preprocessor together as one file.

    they are always saved together because you need both to
    make predictions - the preprocessor encodes new data the
    same way training data was encoded.

    params:
        model        : fitted xgboost classifier
        preprocessor : fitted Preprocessor instance
        path         : file path to save to (.pkl)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model"        : model,
        "preprocessor" : preprocessor,
    }

    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n  model saved to {path}")


# ------------------------------------------------------------
# main entry point
# ------------------------------------------------------------

def run_training(df, model_save_path, test_size=0.2):
    """
    Run the full training pipeline.

    params:
        df               : feature matrix with ub label column
        model_save_path  : where to save the trained model
        test_size        : fraction of data held out for evaluation

    returns fitted model, fitted preprocessor, metrics dict
    """
    print(f"\n  -- model training --\n")

    # prepare features and labels
    X_df, y = prepare_data(df)

    # stratified split - preserves class ratio in train and test
    # stratified means both sets have the same proportion of
    # ubiquitinated sites as the full dataset
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y,
        test_size    = test_size,
        random_state = 42,
        stratify     = y
    )

    print(f"\n  train set: {len(X_train_df)} sites")
    print(f"  test set : {len(X_test_df)} sites")

    # fit preprocessor on training data only
    # never fit on test data - that would leak information
    print(f"\n  fitting preprocessor on training data")
    preprocessor = Preprocessor()
    X_train      = preprocessor.fit_transform(X_train_df)
    X_test       = preprocessor.transform(X_test_df)

    # cross-validation on training data
    cv_scores = cross_validate(X_train, y_train)

    # train final model on full training set
    print(f"\n  training final model on full training set")
    model = train(X_train, y_train)

    # evaluate on held-out test set
    metrics = evaluate(model, X_test, y_test)
    metrics["cv_scores"] = cv_scores

    # save model and preprocessor together
    save_model(model, preprocessor, model_save_path)

    return model, preprocessor, metrics