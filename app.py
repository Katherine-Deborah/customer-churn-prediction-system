"""
app.py — ChurnSight Streamlit Web Application.

Launch with:
    streamlit run app.py

Flow:
    1. User uploads a CSV (same schema as the raw training data)
    2. App preprocesses it (clean → encode → scale → engineer features)
    3. Loads the best trained model
    4. Generates churn predictions + probabilities
    5. Displays results table, charts, and SHAP explanations
"""

import os
import json

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Project paths (relative to this file) ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")

# Column definitions (mirrors src/config.py)
TARGET_COL = "churn_risk_score"
DROP_COLS = ["security_no", "referral_id", "joining_date", "last_visit_time"]
CATEGORICAL_COLS = [
    "gender", "region_category", "membership_category",
    "joined_through_referral", "preferred_offer_types",
    "medium_of_operation", "internet_option", "used_special_discount",
    "offer_application_preference", "past_complaint",
    "complaint_status", "feedback",
]
NUMERICAL_COLS = [
    "age", "days_since_last_login", "avg_time_spent",
    "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet",
]

# ─── Cached loaders ──────────────────────────────────────────────────

@st.cache_resource
def load_model(name: str):
    """Load a saved model from the models/ directory."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    return joblib.load(path)


@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler from preprocessing."""
    return joblib.load(os.path.join(PROCESSED_DIR, "scaler.joblib"))


@st.cache_data
def load_metrics() -> dict:
    """Load evaluation metrics for display."""
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def get_training_columns() -> list:
    """Return the exact column order the model was trained on."""
    return pd.read_csv(
        os.path.join(PROCESSED_DIR, "X_train.csv"), nrows=0
    ).columns.tolist()


# ─── Preprocessing (same logic as src/preprocess.py + features.py) ───

def preprocess_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, encode, scale, and engineer features for an uploaded CSV.
    Returns a DataFrame aligned to the training feature columns."""

    # Drop row-index column if present
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    # Drop the target column if the user included it
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # Drop identifier / datetime columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Convert numerical columns
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    # One-hot encode
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scale numerical columns with the saved scaler
    scaler = load_scaler()
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    df[num_cols] = scaler.transform(df[num_cols])

    # Engineer features (same as src/features.py)
    if "age" in df.columns:
        df["tenure_bucket"] = pd.qcut(
            df["age"], q=4, labels=False, duplicates="drop"
        )
    if "avg_transaction_value" in df.columns and "avg_frequency_login_days" in df.columns:
        denom = df["avg_frequency_login_days"].replace(0, np.nan)
        df["avg_charge_per_login"] = df["avg_transaction_value"] / denom
        df["avg_charge_per_login"].fillna(0, inplace=True)
    if "avg_time_spent" in df.columns and "avg_frequency_login_days" in df.columns:
        df["usage_intensity"] = df["avg_time_spent"] * df["avg_frequency_login_days"]
    if "points_in_wallet" in df.columns and "days_since_last_login" in df.columns:
        df["payment_reliability"] = df["points_in_wallet"] / (df["days_since_last_login"] + 1)

    # Align columns to exactly match training set
    train_cols = get_training_columns()
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0  # missing one-hot columns default to 0
    df = df[train_cols]

    return df


def validate_columns(df: pd.DataFrame) -> list:
    """Check that required raw columns are present. Returns list of missing."""
    required = NUMERICAL_COLS + CATEGORICAL_COLS
    present = df.columns.tolist()
    return [c for c in required if c not in present]


# ─── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChurnSight",
    page_icon="📊",
    layout="wide",
)

# ─── Sidebar: model selector ─────────────────────────────────────────

st.sidebar.title("Settings")

available_models = sorted(
    f.replace(".joblib", "")
    for f in os.listdir(MODELS_DIR)
    if f.endswith(".joblib")
)

selected_model_name = st.sidebar.selectbox(
    "Select model",
    available_models,
    index=available_models.index("XGBoost") if "XGBoost" in available_models else 0,
)

# Threshold slider
threshold = st.sidebar.slider(
    "Churn probability threshold",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Customers above this probability are flagged as churners.",
)

# Show model metrics in sidebar
metrics = load_metrics()
if selected_model_name in metrics:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Performance")
    m = metrics[selected_model_name]
    for k, v in m.items():
        st.sidebar.metric(k.upper(), f"{v:.4f}")

# ─── Main page ────────────────────────────────────────────────────────

st.title("📊 ChurnSight – Customer Churn Prediction")
st.markdown(
    "Upload a customer CSV file to predict which customers are likely to churn, "
    "and explore the reasons behind each prediction using SHAP explainability."
)

# ─── File upload ──────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload customer data (.csv)",
    type=["csv"],
    help="Use the same column format as the training dataset.",
)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)
    st.caption(f"{raw_df.shape[0]} rows × {raw_df.shape[1]} columns")

    # Validate columns
    missing = validate_columns(raw_df)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    # ─── Run Prediction ──────────────────────────────────────────────

    if st.button("🔍 Run Prediction", type="primary"):
        with st.spinner("Preprocessing and predicting..."):
            # Preprocess
            X = preprocess_upload(raw_df.copy())

            # Load model and predict
            model = load_model(selected_model_name)
            probabilities = model.predict_proba(X)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            # Build results dataframe
            results = raw_df.copy()
            results["churn_probability"] = np.round(probabilities, 4)
            results["churn_prediction"] = predictions
            results["risk_level"] = pd.cut(
                probabilities,
                bins=[0, 0.3, 0.6, 1.0],
                labels=["Low", "Medium", "High"],
            )

        # Store in session state for SHAP section
        st.session_state["results"] = results
        st.session_state["X_processed"] = X
        st.session_state["model_name"] = selected_model_name

        # ── Results section ──────────────────────────────────────────
        st.subheader("Prediction Results")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        n_churn = int(predictions.sum())
        col1.metric("Total Customers", len(predictions))
        col2.metric("Predicted Churners", n_churn)
        col3.metric("Churn Rate", f"{n_churn / len(predictions):.1%}")

        # Color-coded results table
        st.dataframe(
            results.style.apply(
                lambda row: [
                    "background-color: #ffcccc" if row["churn_prediction"] == 1
                    else "background-color: #ccffcc"
                ] * len(row),
                axis=1,
            ),
            use_container_width=True,
            height=400,
        )

        # Download button
        csv_out = results.to_csv(index=False)
        st.download_button(
            "⬇️ Download Predictions CSV",
            csv_out,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

        # ── Charts ───────────────────────────────────────────────────
        st.subheader("Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(probabilities, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Number of Customers")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        # Top high-risk customers
        st.subheader("Top High-Risk Customers")
        high_risk = results[results["churn_prediction"] == 1].sort_values(
            "churn_probability", ascending=False
        )
        st.dataframe(high_risk.head(20), use_container_width=True)

    # ─── SHAP Explainability ─────────────────────────────────────────

    if "results" in st.session_state:
        st.markdown("---")
        st.subheader("SHAP Explainability")

        results = st.session_state["results"]
        X = st.session_state["X_processed"]
        model_name = st.session_state["model_name"]
        model = load_model(model_name)

        # Compute SHAP values
        with st.spinner("Computing SHAP values..."):
            if model_name in ("RandomForest", "XGBoost"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                background = shap.sample(X, min(100, len(X)))
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X.iloc[:200], nsamples=100)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                X = X.iloc[:200]

        # Global importance
        st.markdown("#### Global Feature Importance")
        fig_global, ax_global = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig_global)
        plt.close(fig_global)

        # Local explanation for a selected customer
        st.markdown("#### Individual Customer Explanation")
        max_idx = min(len(X), len(results)) - 1
        selected_idx = st.selectbox(
            "Select a customer row to explain",
            range(max_idx + 1),
            format_func=lambda i: f"Row {i} — Prob: {results.iloc[i]['churn_probability']:.2%}"
            if "churn_probability" in results.columns else f"Row {i}",
        )

        fig_local, ax_local = plt.subplots(figsize=(10, 5))
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)) and len(expected) > 1:
            expected = expected[1]
        explanation = shap.Explanation(
            values=shap_values[selected_idx],
            base_values=expected,
            data=X.iloc[selected_idx].values,
            feature_names=X.columns.tolist(),
        )
        shap.waterfall_plot(explanation, show=False)
        st.pyplot(fig_local)
        plt.close(fig_local)

else:
    # Landing state — show project info
    st.info("Upload a CSV file to get started.")

    st.markdown("### Expected CSV Columns")
    st.code(
        ", ".join(NUMERICAL_COLS + CATEGORICAL_COLS),
        language=None,
    )

    # Show existing SHAP plots if available
    if os.path.exists(os.path.join(FIGURES_DIR, "shap_summary_bar.png")):
        st.markdown("### Pre-computed SHAP Analysis (Training Data)")
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                os.path.join(FIGURES_DIR, "shap_summary_bar.png"),
                caption="Global Feature Importance",
            )
        with col2:
            st.image(
                os.path.join(FIGURES_DIR, "shap_summary_dot.png"),
                caption="SHAP Beeswarm Plot",
            )
