# ChurnSight — Interpretable Customer Churn Prediction

An end-to-end ML project that predicts customer churn, compares multiple
models, and uses SHAP to explain *why* customers are likely to leave.

---

## Architecture

```
churnsight/
│
├── data/
│   ├── raw/churn.csv              ← original dataset
│   └── processed/                 ← cleaned, encoded, scaled splits
│
├── src/
│   ├── config.py                  ← all paths, params, column lists
│   ├── utils.py                   ← logger + dir helpers
│   ├── preprocess.py              ← clean → encode → scale → split
│   ├── features.py                ← engineer new features
│   ├── train.py                   ← GridSearchCV for 4 models
│   ├── evaluate.py                ← test-set metrics + comparison
│   └── explain.py                 ← SHAP global + local plots
│
├── models/                        ← saved .joblib model files
├── reports/
│   ├── figures/                   ← SHAP plots (PNG)
│   └── metrics.json               ← evaluation results
│
├── requirements.txt
└── README.md
```

---

## Pipeline Flow

```
raw CSV ──► preprocess.py ──► features.py ──► train.py ──► evaluate.py ──► explain.py
  │              │                │               │              │              │
  │         clean/encode     add derived      GridSearchCV   test metrics   SHAP plots
  │         scale/split      features         save models    metrics.json   PNG files
```

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

Run each step sequentially from the `churnsight/` directory:

```bash
cd churnsight

# Step 1: Clean, encode, scale, and split the data
python src/preprocess.py

# Step 2: Engineer new features
python src/features.py

# Step 3: Train all 4 models (LR, RF, XGBoost, MLP)
python src/train.py

# Step 4: Evaluate on the test set and save metrics
python src/evaluate.py

# Step 5: Generate SHAP explainability plots
python src/explain.py
```

---

## Models Trained

| Model               | Imbalance Handling     | Tuning               |
|---------------------|------------------------|----------------------|
| Logistic Regression | `class_weight=balanced`| GridSearchCV (C)     |
| Random Forest       | `class_weight=balanced`| GridSearchCV (depth, trees) |
| XGBoost             | `scale_pos_weight`     | GridSearchCV (lr, depth)    |
| MLP Neural Network  | early stopping         | GridSearchCV (layers, alpha)|

All models are tuned via 5-fold cross-validation optimizing **F1-score**.

---

## Metrics

After running `evaluate.py`, check `reports/metrics.json` for results like:

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| LogisticRegression  | —        | —         | —      | —      | —       |
| RandomForest        | —        | —         | —      | —      | —       |
| XGBoost             | —        | —         | —      | —      | —       |
| MLP                 | —        | —         | —      | —      | —       |

*(Values populated after running the pipeline)*

---

## SHAP Explainability

After `explain.py` runs, find these in `reports/figures/`:

- **shap_summary_bar.png** — Top features by mean |SHAP value|
- **shap_summary_dot.png** — Beeswarm plot showing feature value effects
- **shap_local_0.png** — Waterfall for individual customer prediction

---

## Reproducibility

- Random seed fixed at `42` across all scripts
- All hyperparameters defined in `src/config.py`
- Train/test split stratified on the target variable
