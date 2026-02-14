"""
evaluate.py — Evaluate all trained models on the held-out test set.

Usage:
    python src/evaluate.py       (run AFTER train.py)

For each model in  models/*.joblib  it computes:
    Accuracy, Precision, Recall, F1-score, ROC-AUC

Results are:
    - Printed as a table in the terminal
    - Saved to  reports/metrics.json
    - Compared against the Logistic Regression baseline
"""

import json
import os

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import config
from utils import get_logger, ensure_dirs

log = get_logger("evaluate")


def load_test_data():
    """Load the processed test split."""
    X = pd.read_csv(f"{config.PROCESSED_DIR}/X_test.csv")
    y = pd.read_csv(f"{config.PROCESSED_DIR}/y_test.csv").squeeze()
    return X, y


def score_model(model, X_test, y_test) -> dict:
    """Compute all five metrics for a single model."""
    y_pred = model.predict(X_test)
    # For AUC we need probability estimates
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }


def main():
    ensure_dirs(config.REPORTS_DIR)
    X_test, y_test = load_test_data()

    # Discover all saved models
    model_files = sorted(
        f for f in os.listdir(config.MODELS_DIR) if f.endswith(".joblib")
    )
    if not model_files:
        log.error("No models found in models/. Run train.py first.")
        return

    all_metrics = {}

    for mf in model_files:
        name = mf.replace(".joblib", "")
        model = joblib.load(os.path.join(config.MODELS_DIR, mf))
        metrics = score_model(model, X_test, y_test)
        all_metrics[name] = metrics

    # ── Print comparison table ────────────────────────────────────────
    header = f"{'Model':25s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s}"
    log.info(f"\n{'='*62}")
    log.info(header)
    log.info("-" * 62)
    for name, m in all_metrics.items():
        log.info(
            f"{name:25s} {m['accuracy']:7.4f} {m['precision']:7.4f} "
            f"{m['recall']:7.4f} {m['f1']:7.4f} {m['roc_auc']:7.4f}"
        )

    # ── Compare against baseline ─────────────────────────────────────
    baseline_name = "LogisticRegression"
    if baseline_name in all_metrics:
        baseline_f1 = all_metrics[baseline_name]["f1"]
        best_name = max(all_metrics, key=lambda n: all_metrics[n]["f1"])
        best_f1 = all_metrics[best_name]["f1"]
        if baseline_f1 > 0:
            improvement = (best_f1 - baseline_f1) / baseline_f1 * 100
        else:
            improvement = float("inf")
        log.info(f"\nBaseline F1 ({baseline_name}): {baseline_f1:.4f}")
        log.info(f"Best F1     ({best_name}): {best_f1:.4f}")
        log.info(f"Improvement: {improvement:+.1f}%")

    # ── Save to JSON ─────────────────────────────────────────────────
    with open(config.METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"\nMetrics saved to {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
