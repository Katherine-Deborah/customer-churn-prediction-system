"""
config.py — Central configuration for ChurnSight.

All paths, column definitions, model hyperparameters, and constants
live here so every other module imports from a single source of truth.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "churn.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")

# ─── Reproducibility ─────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.20

# ─── Target column ───────────────────────────────────────────────────
TARGET_COL = "churn_risk_score"

# ─── Columns to drop (identifiers / not useful for modeling) ─────────
DROP_COLS = ["security_no", "referral_id", "joining_date", "last_visit_time"]

# ─── Column types (auto-detected during preprocessing, but we define
#     known categoricals explicitly for clarity) ───────────────────────
CATEGORICAL_COLS = [
    "gender",
    "region_category",
    "membership_category",
    "joined_through_referral",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "used_special_discount",
    "offer_application_preference",
    "past_complaint",
    "complaint_status",
    "feedback",
]

NUMERICAL_COLS = [
    "age",
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "points_in_wallet",
]

# ─── Model hyperparameter grids (used with GridSearchCV) ─────────────
PARAM_GRIDS = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "max_iter": [1000],
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
    "MLP": {
        "hidden_layer_sizes": [(64, 32), (128, 64)],
        "alpha": [0.0001, 0.001],
        "max_iter": [500],
    },
}

# ─── Cross-validation folds ──────────────────────────────────────────
CV_FOLDS = 5
