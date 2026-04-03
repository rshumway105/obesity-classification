"""
Central configuration for the Obesity Classification project.
All paths, constants, and shared settings live here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
FIGURES_DIR = PROJECT_ROOT / "figures"

RAW_FILE = DATA_RAW / "ObesityDataSet_raw_and_data_sinthetic.xlsx"

# ── Dataset ────────────────────────────────────────────────────────────
TARGET = "NObeyesdad"

# Ordered from lightest to heaviest for ordinal encoding of the target
TARGET_CLASSES = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]

CONTINUOUS_FEATURES = [
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
]

# Features that may dominate — useful for ablation experiments
BODY_FEATURES = ["Height", "Weight"]

# ── Modeling ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
