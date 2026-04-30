import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Patterns")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page.")
    st.stop()

raw_df = st.session_state["uploaded_data"].copy()
cleaned_df = st.session_state.get("cleaned_data")
if cleaned_df is None:
    st.info("Clean data firest on cleaning page")
    st.stop()

if "stroke" not in raw_df.columns:
    st.error("This page expects a 'stroke' target column to identify significant patterns.")
    st.stop()

# -----------------------------------------------------------------------------
# Age

n_estimators=350
age_df = raw_df[["age", "stroke"]].dropna().copy()
if not age_df.empty:

    age_bins = [0, 20, 30, 40, 50, 60, 70, 80, np.inf]
    age_labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "80+"]
    age_df["age_group"] = pd.cut(age_df["age"], bins=age_bins, labels=age_labels, right=True)
    age_rate = (
        age_df.groupby("age_group", observed=False)["stroke"]
        .mean()
        .mul(100)
        .reset_index(name="Stroke Rate (%)")
        .dropna()
    )

    fig = px.line(
        age_rate,
        x="age_group",
        y="Stroke Rate (%)",
        markers=True,
        title="Stroke Rate (%) by Age Group",
        labels={"age_group": "Age Group"},
    )
    fig.update_traces(line_color="#F28E2B")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Hypertension + Heart Disease

n_estimators=350
risk_cols = [col for col in ["hypertension", "heart_disease"] if col in raw_df.columns]
if risk_cols:
    risk_rates = []
    for col in risk_cols:
        grouped = raw_df.groupby(col, dropna=True)["stroke"].mean().mul(100)
        for value, rate in grouped.items():
            risk_rates.append({"Risk Factor": col, "State": str(value), "Stroke Rate (%)": rate})

    risk_plot_df = pd.DataFrame(risk_rates)
    if not risk_plot_df.empty:
        fig = px.bar(
            risk_plot_df,
            x="Risk Factor",
            y="Stroke Rate (%)",
            color="State",
            barmode="group",
            title="Stroke Rate by Risk Factor State",
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Work

n_estimators=350
if "work_type" in raw_df.columns:
    work_rate = (
        raw_df[["work_type", "stroke"]]
        .dropna()
        .groupby("work_type", observed=False)["stroke"]
        .mean()
        .sort_values(ascending=False)
        .mul(100)
        .reset_index(name="Stroke Rate (%)")
    )
    fig = px.bar(
        work_rate,
        x="work_type",
        y="Stroke Rate (%)",
        title="Stroke Rate (%) by Work Type",
        color_discrete_sequence=["#59A14F"],
    )
    fig.update_xaxes(tickangle=25)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Smoking

n_estimators=350
if "smoking_status" in raw_df.columns:
    smoking_rate = (
        raw_df[["smoking_status", "stroke"]]
        .dropna()
        .groupby("smoking_status", observed=False)["stroke"]
        .mean()
        .sort_values(ascending=False)
        .mul(100)
        .reset_index(name="Stroke Rate (%)")
    )
    fig = px.bar(
        smoking_rate,
        x="smoking_status",
        y="Stroke Rate (%)",
        title="Stroke Rate (%) by Smoking Status",
        color_discrete_sequence=["#76B7B2"],
    )
    fig.update_xaxes(tickangle=25)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Glucose + BMI

n_estimators=350
numeric_compare_cols = [col for col in ["avg_glucose_level", "bmi"] if col in raw_df.columns]
if numeric_compare_cols:
    melted = raw_df[["stroke", *numeric_compare_cols]].copy()
    melted["Outcome"] = melted["stroke"].map({0: "No Stroke", 1: "Stroke"})
    long_df = melted.melt(id_vars="Outcome", value_vars=numeric_compare_cols, var_name="Feature", value_name="Value").dropna()

    fig = px.box(
        long_df,
        x="Outcome",
        y="Value",
        color="Outcome",
        facet_col="Feature",
        title="Numeric Features by Stroke Outcome",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Correlation Heat Map

n_estimators=350
numeric_df = cleaned_df.select_dtypes(include=["number"])
if numeric_df.shape[1] > 1:
    corr = numeric_df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Coefficient

n_estimators=350
if "stroke" in numeric_df.columns and numeric_df.shape[1] > 2:
    stroke_corr = numeric_df.corr(numeric_only=True)["stroke"].drop("stroke").sort_values()

    corr_df = stroke_corr.reset_index()
    corr_df.columns = ["Feature", "Correlation Coefficient"]
    fig = px.bar(
        corr_df,
        x="Correlation Coefficient",
        y="Feature",
        orientation="h",
        title="Feature Correlation Coefficient",
        color_discrete_sequence=["#EDC948"],
    )
    st.plotly_chart(fig, use_container_width=True)
