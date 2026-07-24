"""
Data loading, cleaning, and the sklearn preprocessing pipeline.

The key fix from Week 1: instead of using pd.get_dummies() on the whole
dataframe (which produces different columns depending on what data is
passed in), we use a ColumnTransformer + OneHotEncoder wrapped inside a
single sklearn Pipeline together with the model. That pipeline is what
gets saved to disk, so preprocessing at inference time is guaranteed to
match training exactly, even for a single-row API request.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def load_and_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and apply the minimal cleaning every model needs."""
    df = pd.read_csv(path)

    # customerID carries no predictive signal
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # TotalCharges is stored as a string with some blank entries for new
    # customers (tenure == 0). Coercing turns those blanks into NaN, which
    # the numeric imputer in the pipeline handles downstream.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode target as 0/1
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build the reusable preprocessing step.
    - Numeric: median impute (handles the TotalCharges NaNs) + scale
    - Categorical: most-frequent impute + one-hot encode
      (handle_unknown="ignore" prevents crashes if the API ever receives
      a category value that wasn't seen during training)
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def get_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y
