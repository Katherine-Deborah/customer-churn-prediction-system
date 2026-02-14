"""
train.py — Train all models with cross-validated hyperparameter tuning.

Usage:
    python src/train.py        (run AFTER features.py)

Pipeline:
    1. Load processed train data
    2. For each model (LR, RF, XGB, MLP):
       - Run GridSearchCV with the param grid from config.py
       - Pick best estimator by F1-score
       - Save the best model to  models/<name>.joblib
    3. Print a summary table of best CV scores
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import joblib

import config
from utils import get_logger, ensure_dirs

log = get_logger("train")


def load_train_data():
    """Load the processed training split."""
    X = pd.read_csv(f"{config.PROCESSED_DIR}/X_train.csv")
    y = pd.read_csv(f"{config.PROCESSED_DIR}/y_train.csv").squeeze()
    log.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def build_models(y_train: pd.Series) -> dict:
    """Instantiate the four model objects.

    class_weight / scale_pos_weight handles class imbalance so the
    models don't simply predict the majority class.
    """
    # Compute imbalance ratio for XGBoost
    neg, pos = np.bincount(y_train.astype(int))
    scale = neg / pos

    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale,
            random_state=config.RANDOM_SEED,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            random_state=config.RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }
    return models


def train_all(X_train: pd.DataFrame, y_train: pd.Series):
    """Run GridSearchCV for every model and save the best estimator."""
    ensure_dirs(config.MODELS_DIR)
    models = build_models(y_train)
    results = {}

    for name, estimator in models.items():
        log.info(f"{'─'*50}")
        log.info(f"Training {name} ...")
        param_grid = config.PARAM_GRIDS[name]

        search = GridSearchCV(
            estimator,
            param_grid,
            scoring="f1",             # primary metric per PRD
            cv=config.CV_FOLDS,
            n_jobs=-1,
            verbose=0,
            refit=True,               # refit best params on full train set
        )
        search.fit(X_train, y_train)

        best = search.best_estimator_
        best_score = search.best_score_

        # Save the best model
        model_path = f"{config.MODELS_DIR}/{name}.joblib"
        joblib.dump(best, model_path)

        results[name] = {
            "best_params": search.best_params_,
            "cv_f1": round(best_score, 4),
        }
        log.info(f"  Best CV F1 = {best_score:.4f}")
        log.info(f"  Params: {search.best_params_}")
        log.info(f"  Saved → {model_path}")

    # Summary
    log.info(f"\n{'='*50}")
    log.info("Training complete — CV F1 summary:")
    for name, r in results.items():
        log.info(f"  {name:25s}  F1 = {r['cv_f1']}")

    return results


def main():
    X_train, y_train = load_train_data()
    train_all(X_train, y_train)


if __name__ == "__main__":
    main()
