"""
Central configuration for the Obesity Classification project.
"""

# ── Dataset ────────────────────────────────────────────────────────────
TARGET = "NObeyesdad"

# Ordered from lightest to heaviest
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
