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
├── app.py                         ← Streamlit web UI
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


---

## Metrics


| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| LogisticRegression  | 0.8774   | 0.925     | 0.8416 | 0.8814 | 0.9544  |
| RandomForest        | 0.9309   | 0.9197    | 0.9558 | 0.9374 | 0.9753  |
| XGBoost             | 0.9343   | 0.9303    | 0.9498 | 0.9399 | 0.9761  |
| MLP                 | 0.9157   | 0.9278    | 0.9153 | 0.9215 | 0.973   |


---

## SHAP Explainability

After `explain.py` runs, find these in `reports/figures/`:

- **shap_summary_bar.png** — Top features by mean |SHAP value|
- **shap_summary_dot.png** — Beeswarm plot showing feature value effects
- **shap_local_0.png** — Waterfall for individual customer prediction

---

## Web UI (Streamlit)

ChurnSight includes an interactive web app for business users.

```bash
streamlit run app.py
```

**Features:**
- Upload any customer CSV and get instant churn predictions
- Choose between all 4 trained models from the sidebar
- Adjust the churn probability threshold with a slider
- View color-coded results table with risk levels (Low / Medium / High)
- Download predictions as CSV
- Explore SHAP global feature importance
- Select any individual customer to see a waterfall explanation of their prediction
- Pre-computed SHAP plots displayed on the landing page

---

## Reproducibility

- Random seed fixed at `42` across all scripts
- All hyperparameters defined in `src/config.py`
- Train/test split stratified on the target variable
