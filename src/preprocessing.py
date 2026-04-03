"""
Preprocessing utilities: encoding, scaling, train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
import sys
sys.path.append("..")
from config import (
    TARGET, TARGET_CLASSES, CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES, RANDOM_STATE, TEST_SIZE,
)


def encode_target(df):
    """Encode the target column as integers following the ordinal class order."""
    mapping = {label: i for i, label in enumerate(TARGET_CLASSES)}
    df = df.copy()
    df[TARGET] = df[TARGET].map(mapping)
    return df


def encode_categoricals(df, method="onehot"):
    """
    Encode categorical features.

    Parameters
    ----------
    df : DataFrame
    method : 'onehot' or 'ordinal'

    Returns
    -------
    DataFrame with encoded features.
    """
    df = df.copy()
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    if method == "onehot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    elif method == "ordinal":
        enc = OrdinalEncoder()
        df[cat_cols] = enc.fit_transform(df[cat_cols])
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    return df


def scale_features(X_train, X_test, cols=None):
    """
    Fit a StandardScaler on training data and transform both sets.

    Parameters
    ----------
    X_train, X_test : DataFrames
    cols : list of column names to scale (default: CONTINUOUS_FEATURES)

    Returns
    -------
    X_train, X_test (scaled), scaler
    """
    cols = cols or [c for c in CONTINUOUS_FEATURES if c in X_train.columns]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_test, scaler


def split_data(df, target_col=TARGET):
    """Stratified train/test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
