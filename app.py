
"""Streamlit dashboard for UCI Heart Disease classification."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease ML Evaluation",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
TARGET = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

# --------------------------------------------------
# Compact page styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1450px;
        }
        div[data-testid="stMetric"] {
            padding: 8px;
            border: 1px solid #cccccc;
            border-radius: 8px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
        h1 {
            font-size: 1.8rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Page heading
# --------------------------------------------------

st.title("Heart Disease Classification Dashboard")

st.caption(
    "BITS Pilani | Machine Learning Assignment 2 | "
    "Combined UCI Heart Disease Dataset"
)

st.info(
    "Dataset: 920 records | 13 features | "
    "Training: 736 (80%) | Testing: 184 (20%)"
)

# --------------------------------------------------
# Load available trained models
# --------------------------------------------------

available_models = {
    name: MODEL_DIR / filename
    for name, filename in MODEL_FILES.items()
    if (MODEL_DIR / filename).is_file()
}

if not available_models:
    st.error(
        "Saved models not found. "
        "Please execute python train_models.py first."
    )
    st.stop()

# --------------------------------------------------
# Model selection and test-data upload
# --------------------------------------------------

selection_col, upload_col = st.columns([1, 2])

with selection_col:
    selected_name = st.selectbox(
        "Select Classification Model",
        list(available_models.keys()),
    )

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"],
    )

# --------------------------------------------------
# Load the selected saved model
# --------------------------------------------------

try:
    bundle = joblib.load(available_models[selected_name])
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

if uploaded_file is None:
    st.info(
        "Please upload test_data.csv to display "
        "evaluation metrics and classification results."
    )
    st.stop()

# --------------------------------------------------
# Evaluate uploaded test data
# --------------------------------------------------

try:
    test_df = pd.read_csv(uploaded_file)

    required_columns = feature_columns + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in test_df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing required columns: {missing_columns}"
        )
        st.stop()

    X_test = test_df[feature_columns].copy()

    y_test = pd.to_numeric(
        test_df[TARGET],
        errors="coerce",
    )

    if y_test.isna().any():
        st.error(
            "The target column contains missing or invalid values."
        )
        st.stop()

    if not set(y_test.unique()).issubset({0, 1}):
        st.error(
            "The target column must contain only 0 and 1."
        )
        st.stop()

    y_test = y_test.astype(int)

    if y_test.nunique() < 2:
        st.error(
            "Both target classes are required to calculate AUC."
        )
        st.stop()

    # Predictions use the previously trained model.
    # No retraining is performed here.
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions),
    }

    st.subheader("Model Evaluation Metrics")

    metric_columns = st.columns(6)

    for column, (name, value) in zip(
        metric_columns, metrics.items()
    ):
        column.metric(name, f"{value:.4f}")

    # --------------------------------------------------
    # Two-column evaluation dashboard
    # --------------------------------------------------

    chart_col, report_col = st.columns(
        [1, 2],
        gap="medium",
    )

    # --------------------------------------------------
    # Small confusion matrix
    # --------------------------------------------------

    with chart_col:
        st.markdown("**Confusion Matrix**")

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        fig, ax = plt.subplots(
            figsize=(2.8, 2.3),
            dpi=90,
        )

        ax.imshow(matrix, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["No disease", "Disease"],
            fontsize=7,
        )

        ax.set_yticklabels(
            ["No disease", "Disease"],
            fontsize=7,
        )

        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

        for row in range(2):
            for col in range(2):
                value = matrix[row, col]

                ax.text(
                    col,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=(
                        "white"
                        if value > matrix.max() / 2
                        else "black"
                    ),
                )

        fig.tight_layout(pad=0.5)

        st.pyplot(
            fig,
            use_container_width=False,
        )

        plt.close(fig)

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    with report_col:
        st.markdown("**Classification Report**")

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "No disease (0)",
                "Disease (1)",
            ],
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True,
            height=240,
        )

    # --------------------------------------------------
    # Completion message
    # --------------------------------------------------

    st.success(
        f"Evaluation completed for {len(test_df)} "
        f"test records using {selected_name}."
    )

except Exception as error:
    st.error("An error occurred during evaluation.")
    st.exception(error)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption(
    "Source: UCI Heart Disease Dataset "
    "(Cleveland, Hungary, Switzerland and VA Long Beach). "
    "Educational use only; not a medical diagnostic tool."
)
