"""
preprocess.py — Load raw data, clean it, encode, scale, and split.

Usage:
    python src/preprocess.py

What happens:
    1. Reads  data/raw/churn.csv
    2. Drops identifier / datetime columns
    3. Replaces '?' with NaN and fills missing values
    4. One-hot encodes categorical columns
    5. Standard-scales numerical columns
    6. Splits into 80 % train / 20 % test
    7. Saves X_train, X_test, y_train, y_test  →  data/processed/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import config
from utils import get_logger, ensure_dirs

log = get_logger("preprocess")


def load_raw_data() -> pd.DataFrame:
    """Read the raw CSV and do basic type fixes."""
    df = pd.read_csv(config.RAW_DATA_PATH)
    log.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # The first unnamed column is just a row index — drop it
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, drop useless columns, fix types."""

    # Drop identifier / date columns that can't help the model
    cols_to_drop = [c for c in config.DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.info(f"Dropped columns: {cols_to_drop}")

    # Replace '?' (present in some fields) with NaN
    df.replace("?", np.nan, inplace=True)

    # ── Fill missing values ──────────────────────────────────────────
    for col in df.columns:
        if df[col].dtype == "object":
            # Categorical: fill with the most frequent value (mode)
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            # Numerical: fill with the median (robust to outliers)
            df[col].fillna(df[col].median(), inplace=True)

    # Ensure numerical columns that were read as strings are converted
    for col in config.NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    log.info(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def encode_and_scale(df: pd.DataFrame):
    """One-hot encode categoricals, standard-scale numericals.

    Returns:
        X (DataFrame) — feature matrix ready for modeling
        y (Series)     — target vector
        scaler         — fitted StandardScaler (saved for later use)
    """
    y = df[config.TARGET_COL]
    df = df.drop(columns=[config.TARGET_COL])

    # One-hot encode categorical columns that are still in the dataframe
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    log.info(f"After one-hot encoding: {df.shape[1]} features")

    # Standard-scale numerical columns
    num_cols = [c for c in config.NUMERICAL_COLS if c in df.columns]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, y, scaler


def main():
    ensure_dirs(config.PROCESSED_DIR)

    # 1. Load
    df = load_raw_data()

    # 2. Clean
    df = clean(df)

    # 3. Encode + Scale
    X, y, scaler = encode_and_scale(df)

    # 4. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,  # keep class balance in both sets
    )
    log.info(
        f"Split → train {X_train.shape[0]} rows | test {X_test.shape[0]} rows"
    )
    log.info(
        f"Class balance in train — churn: {y_train.mean():.2%}"
    )

    # 5. Save processed artifacts
    X_train.to_csv(f"{config.PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{config.PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{config.PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{config.PROCESSED_DIR}/y_test.csv", index=False)
    joblib.dump(scaler, f"{config.PROCESSED_DIR}/scaler.joblib")

    log.info(f"Saved processed data to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
