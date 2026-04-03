"""
Load and validate the Obesity dataset.
"""

import pandas as pd
import sys
sys.path.append("..")
from config import RAW_FILE, TARGET, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES


def load_raw(path=None):
    """Load the raw Excel dataset and return a DataFrame."""
    path = path or RAW_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download it from UCI and place it in data/raw/."
        )
    df = pd.read_excel(path)
    return df


def validate(df):
    """Run basic sanity checks and print a summary."""
    print(f"Shape: {df.shape}")
    print(f"Target classes: {df[TARGET].nunique()} — {sorted(df[TARGET].unique())}")
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values:\n", missing[missing > 0])
    else:
        print("No missing values.")
    print(f"\nCategorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f"Continuous features  ({len(CONTINUOUS_FEATURES)}): {CONTINUOUS_FEATURES}")
    return df


if __name__ == "__main__":
    df = load_raw()
    validate(df)
