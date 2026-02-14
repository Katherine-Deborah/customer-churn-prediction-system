"""
explain.py — Generate SHAP explanations for the best model.

Usage:
    python src/explain.py        (run AFTER evaluate.py)

What it produces (saved to reports/figures/):
    1. shap_summary_bar.png   — Global feature importance (bar chart)
    2. shap_summary_dot.png   — Global SHAP beeswarm plot
    3. shap_local_0.png …     — Local waterfall plots for 3 sample customers

It auto-selects the best model by reading reports/metrics.json.
"""

import json
import os

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

import config
from utils import get_logger, ensure_dirs

log = get_logger("explain")


def pick_best_model() -> str:
    """Read metrics.json and return the name of the model with highest F1."""
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    best = max(metrics, key=lambda n: metrics[n]["f1"])
    log.info(f"Best model by F1: {best} (F1={metrics[best]['f1']})")
    return best


def load_model_and_data(model_name: str):
    """Load the best model and the test set."""
    model = joblib.load(os.path.join(config.MODELS_DIR, f"{model_name}.joblib"))
    X_test = pd.read_csv(f"{config.PROCESSED_DIR}/X_test.csv")
    return model, X_test


def compute_shap_values(model, X_test, model_name: str):
    """Compute SHAP values using the appropriate explainer.

    - Tree models (RF, XGB) → TreeExplainer  (fast, exact)
    - Other models (LR, MLP) → KernelExplainer on a sample (slower)
    """
    if model_name in ("RandomForest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # For binary classification, TreeExplainer may return a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = churn
    else:
        # KernelExplainer needs a background dataset (use 100-sample summary)
        background = shap.sample(X_test, min(100, len(X_test)))
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            predict_fn = model.decision_function
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(
            X_test.iloc[:200], nsamples=100  # limit for speed
        )
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        X_test = X_test.iloc[:200]

    return explainer, shap_values, X_test


def plot_global(shap_values, X_test):
    """Save global feature importance plots."""

    # Bar chart of mean |SHAP|
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "shap_summary_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved {path}")

    # Beeswarm / dot plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "shap_summary_dot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved {path}")


def plot_local(explainer, shap_values, X_test, n_samples: int = 3):
    """Save waterfall plots for individual customer explanations."""
    for i in range(min(n_samples, len(X_test))):
        plt.figure(figsize=(10, 6))
        # Build an Explanation object for the waterfall plot
        explanation = shap.Explanation(
            values=shap_values[i],
            base_values=(
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                   and len(explainer.expected_value) > 1
                else explainer.expected_value
            ),
            data=X_test.iloc[i].values,
            feature_names=X_test.columns.tolist(),
        )
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        path = os.path.join(config.FIGURES_DIR, f"shap_local_{i}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        log.info(f"Saved {path}")


def main():
    ensure_dirs(config.FIGURES_DIR)

    model_name = pick_best_model()
    model, X_test = load_model_and_data(model_name)

    log.info("Computing SHAP values (this may take a minute) ...")
    explainer, shap_values, X_test_used = compute_shap_values(
        model, X_test, model_name
    )

    plot_global(shap_values, X_test_used)
    plot_local(explainer, shap_values, X_test_used)

    log.info("Explainability report complete.")


if __name__ == "__main__":
    main()
