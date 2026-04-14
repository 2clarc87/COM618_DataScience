import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app_core.data_pipeline import (
    classification_metrics,
    regression_metrics,
    run_kmeans,
    train_supervised_model,
)

st.set_page_config(page_title="Model Training & Prediction", layout="wide")
st.title("Model Training & Prediction Results")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page first.")
    st.stop()

source_option = st.radio(
    "Choose data source",
    options=["Cleaned data (recommended)", "Uploaded raw data"],
    horizontal=True,
)

if source_option == "Cleaned data (recommended)" and "cleaned_data" in st.session_state:
    df = st.session_state["cleaned_data"].copy()
else:
    df = st.session_state["uploaded_data"].copy()

if df.empty or df.shape[1] < 2:
    st.error("Dataset must have at least 2 columns and 1 row.")
    st.stop()

st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

model_name = st.selectbox(
    "Select model",
    ["K-Means", "Linear Regression", "Random Forest", "XGBoost", "Logistic Regression"],
)

test_size_pct = st.slider(
    "How much data to use for testing vs training",
    min_value=10,
    max_value=50,
    value=20,
    step=5,
)


def render_predictions_table(
    y_true: pd.Series,
    y_pred: np.ndarray,
    label_true: str = "Actual",
    label_pred: str = "Predicted",
):
    result_df = pd.DataFrame(
        {label_true: y_true.reset_index(drop=True), label_pred: pd.Series(y_pred).reset_index(drop=True)}
    )
    st.subheader("Prediction Results")
    st.dataframe(result_df.head(100), use_container_width=True)


if model_name == "K-Means":
    st.info("K-Means is unsupervised. It does not need a target column.")
    feature_cols = st.multiselect(
        "Select feature columns for clustering",
        options=df.columns.tolist(),
        default=df.select_dtypes(include=[np.number]).columns.tolist()[:6] or df.columns.tolist()[:3],
    )
    n_clusters = st.slider("Number of clusters (K)", 2, 10, 3)

    if st.button("Train & Predict"):
        if len(feature_cols) < 1:
            st.warning("Select at least one feature column.")
            st.stop()

        output = run_kmeans(df, feature_cols, n_clusters)

        st.success("K-Means training complete.")
        st.subheader("Cluster Distribution")
        cluster_counts = output["cluster"].value_counts().sort_index().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]

        fig = px.bar(cluster_counts, x="Cluster", y="Count", title="Cluster Sizes")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prediction Results")
        st.dataframe(output[[*feature_cols, "cluster"]].head(200), use_container_width=True)

else:
    target_col = st.selectbox(
        "Select target column",
        options=df.columns.tolist(),
        index=df.columns.get_loc("stroke") if "stroke" in df.columns else 0,
    )

    if st.button("Train & Predict"):
        try:
            result = train_supervised_model(df, target_col, model_name, test_size_pct)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        preds = result["preds"]
        y_test = result["y_test"]
        model_type = result["model_type"]

        st.success(f"{model_name} training complete.")

        if model_type == "classification":
            metrics = classification_metrics(y_test, preds, result["is_binary_target"])

            a, b, c, d = st.columns(4)
            a.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            b.metric("Precision", f"{metrics['precision']:.4f}")
            c.metric("Recall", f"{metrics['recall']:.4f}")
            d.metric("F1", f"{metrics['f1']:.4f}")

            cm_df = pd.DataFrame(
                metrics["cm"],
                index=[f"Actual {x}" for x in metrics["labels"]],
                columns=[f"Pred {x}" for x in metrics["labels"]],
            )

            st.subheader("Confusion Matrix")
            st.dataframe(cm_df, use_container_width=True)

            render_predictions_table(pd.Series(metrics["y_test_eval"]), np.array(metrics["pred_eval"]))

        else:
            metrics = regression_metrics(y_test, preds)

            a, b = st.columns(2)
            a.metric("MSE", f"{metrics['mse']:.4f}")
            b.metric("R²", f"{metrics['r2']:.4f}")

            render_predictions_table(y_test, preds)
