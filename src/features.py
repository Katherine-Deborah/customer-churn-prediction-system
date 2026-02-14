"""
features.py — Engineer new features from the processed data.

Usage:
    python src/features.py        (run AFTER preprocess.py)

New features created:
    - tenure_bucket        : binned age groups (proxy for tenure)
    - avg_charge_per_login : avg_transaction_value / avg_frequency_login_days
    - usage_intensity      : avg_time_spent * avg_frequency_login_days
    - payment_reliability  : points_in_wallet / (days_since_last_login + 1)

The enriched datasets overwrite the processed CSVs so that
train.py always picks up the latest feature set.
"""

import pandas as pd
import numpy as np

import config
from utils import get_logger, ensure_dirs

log = get_logger("features")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features.  Works on the already-scaled data,
    but the ratios / products of scaled features are still meaningful."""

    # ── Tenure bucket (binned age as a proxy for customer tenure) ─────
    # age was scaled, so we bin on the scaled values into 4 quantile buckets
    if "age" in df.columns:
        df["tenure_bucket"] = pd.qcut(
            df["age"], q=4, labels=False, duplicates="drop"
        )

    # ── Average charge per login day ─────────────────────────────────
    if "avg_transaction_value" in df.columns and "avg_frequency_login_days" in df.columns:
        denom = df["avg_frequency_login_days"].replace(0, np.nan)
        df["avg_charge_per_login"] = df["avg_transaction_value"] / denom
        df["avg_charge_per_login"].fillna(0, inplace=True)

    # ── Usage intensity (time × frequency) ───────────────────────────
    if "avg_time_spent" in df.columns and "avg_frequency_login_days" in df.columns:
        df["usage_intensity"] = (
            df["avg_time_spent"] * df["avg_frequency_login_days"]
        )

    # ── Payment reliability (wallet points relative to inactivity) ───
    if "points_in_wallet" in df.columns and "days_since_last_login" in df.columns:
        df["payment_reliability"] = df["points_in_wallet"] / (
            df["days_since_last_login"] + 1
        )

    log.info(f"Feature engineering complete — {df.shape[1]} total features")
    return df


def main():
    ensure_dirs(config.PROCESSED_DIR)

    for split in ("train", "test"):
        path = f"{config.PROCESSED_DIR}/X_{split}.csv"
        df = pd.read_csv(path)
        df = add_features(df)
        df.to_csv(path, index=False)
        log.info(f"Saved enriched X_{split} ({df.shape})")


if __name__ == "__main__":
    main()
