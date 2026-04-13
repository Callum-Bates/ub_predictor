# ------------------------------------------------------------
# models/preprocess.py
#
# prepares the raw feature matrix for model training and
# prediction.
#
# three jobs:
#   1. identify which columns are features vs metadata (identifiers)
#   2. one-hot encode categorical columns (amino acid identity)
#   3. handle missing values (nulls from failed calculations)
#
# the fitted preprocessor is saved alongside the model so
# prediction always uses identical encoding to training.
# this is critical - if E at position k-1 maps to column 4
# during training, it must map to column 4 during prediction.
#
# input : raw feature matrix dataframe from feature generation
# output: preprocessed numpy array ready for xgboost
#         fitted preprocessor object (save this with the model)

# ------------------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

log = logging.getLogger(__name__)

# columns that identify a site but are not features
# these are never passed to the model
METADATA_COLS = [
    "protein_id",
    "lysine_position",
    "region_type",
    "ub",            # label column - handled separately
]

# columns that contain amino acid identity as single letters
# these need one-hot encoding
# includes sequence window positions and spatial neighbour identities
AA_CATEGORICAL_PREFIXES = [
    "aa_k",    # sequence window positions e.g. aa_k-1, aa_k+1
    "nb",      # spatial neighbour amino acids e.g. nb1_aa, nb2_aa
    "ss_k",    # per-position secondary structure e.g. ss_k-1
    "ss_lysine",  # secondary structure of the lysine itself
]


# ------------------------------------------------------------
# preprocessor class
# ------------------------------------------------------------

class Preprocessor:
    """
    Fits and applies feature preprocessing for ub_predictor.

    handles identification of categorical vs numeric columns,
    one-hot encoding of amino acid identity features, and
    imputation of missing values.

    fit() on training data, then transform() on any new data.
    save and load with pickle to ensure consistency.
    """

    def __init__(self):
        self.categorical_cols = []
        self.numeric_cols     = []
        self.encoder          = None
        self.imputer          = None
        self.feature_names    = []
        self.is_fitted        = False

    def _identify_columns(self, df):
    #
        """
        Split columns into categorical and numeric feature sets.

        params:
            df : raw feature matrix dataframe
        """
        feature_cols = [
            col for col in df.columns
            if col not in METADATA_COLS
        ]

        categorical_cols = []
        numeric_cols     = []

        for col in feature_cols:
            # check if this column matches any categorical prefix
            is_categorical = any(
                col.startswith(prefix)
                for prefix in AA_CATEGORICAL_PREFIXES
            )

            if is_categorical:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)

        return categorical_cols, numeric_cols

    def fit(self, df):
        """
        Fit the preprocessor on training data.

        learns the encoding from training data - call this once
        on training data then use transform() for all subsequent
        data including prediction inputs.

        params:
            df : raw feature matrix dataframe (training data)
        """
        self.categorical_cols, self.numeric_cols = (
            self._identify_columns(df)
        )

        print(f"  {len(self.categorical_cols)} categorical columns")
        print(f"  {len(self.numeric_cols)} numeric columns")

        # fit one-hot encoder on categorical columns
        # handle_unknown="ignore" means unseen categories during
        # prediction (e.g. rare amino acids) become all-zeros
        # rather than causing an error
        if self.categorical_cols:
            cat_data     = df[self.categorical_cols].astype(str).fillna("X")
            self.encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
            self.encoder.fit(cat_data)

        # fit imputer on numeric columns
        # uses median imputation - robust to outliers
        # fills nulls from failed rasa/spatial calculations
        if self.numeric_cols:
            num_data     = df[self.numeric_cols]
            self.imputer = SimpleImputer(strategy="median")
            self.imputer.fit(num_data)

        # record final feature names for interpretability
        self._build_feature_names()
        self.is_fitted = True

        print(f"  {len(self.feature_names)} total features after encoding")

        return self

    def transform(self, df):
        """
        Apply preprocessing to a feature matrix.

        params:
            df : raw feature matrix dataframe

        returns numpy array ready for xgboost
        """
        if not self.is_fitted:
            raise RuntimeError(
                "preprocessor has not been fitted - call fit() first"
            )

        parts = []

        # one-hot encode categorical columns
        if self.categorical_cols and self.encoder is not None:
            cat_data    = df[self.categorical_cols].astype(str).fillna("X")
            cat_encoded = self.encoder.transform(cat_data)
            parts.append(cat_encoded)

        # impute numeric columns
        if self.numeric_cols and self.imputer is not None:
            num_data    = df[self.numeric_cols].copy()
            num_imputed = self.imputer.transform(num_data)
            parts.append(num_imputed)

        if not parts:
            raise ValueError("no features found to preprocess")

        return np.hstack(parts)

    def fit_transform(self, df):
        """
        Fit on and transform the same dataframe.
        convenience method for training.

        params:
            df : raw feature matrix dataframe
        """
        return self.fit(df).transform(df)

    def _build_feature_names(self):
        """
        Build human-readable feature names after fitting.
        used for shap value interpretation.
        """
        names = []

        # one-hot encoded feature names
        if self.encoder is not None:
            for col, categories in zip(
                self.categorical_cols,
                self.encoder.categories_
            ):
                for cat in categories:
                    names.append(f"{col}_{cat}")

        # numeric feature names - unchanged
        names.extend(self.numeric_cols)

        self.feature_names = names

    def save(self, path):
        """
        Save fitted preprocessor to disk.

        params:
            path : file path to save to (use .pkl extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  preprocessor saved to {path.name}")

    @classmethod
    def load(cls, path):
        """
        Load a fitted preprocessor from disk.

        params:
            path : path to saved preprocessor pkl file
        """
        with open(path, "rb") as f:
            preprocessor = pickle.load(f)
        print(f"  preprocessor loaded from {Path(path).name}")
        return preprocessor