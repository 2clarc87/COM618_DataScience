import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app_core.data_quality import apply_stroke_preprocessing

st.title("Data Mining, Patterns, and Trends")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page.")
    st.stop()

raw_df = st.session_state["uploaded_data"].copy()
cleaned_df = st.session_state.get("cleaned_data")
if cleaned_df is None:
    cleaned_df = apply_stroke_preprocessing(raw_df)
    st.session_state["cleaned_data"] = cleaned_df

if "stroke" not in raw_df.columns:
    st.error("This page expects a 'stroke' target column to identify significant patterns.")
    st.stop()

st.markdown(
    "This page applies **data-mining techniques** (grouped trend analysis, correlation-based analysis, "
    "and pattern segmentation) to identify significant patterns in the uploaded dataset."
)
st.caption("Tip: Use the two-column dashboard below to compare related patterns side by side.")

left_col, right_col = st.columns(2)

# -----------------------------------------------------------------------------
with left_col:
    st.subheader("1) Class Distribution (Outcome Balance)")
    outcome_counts = raw_df["stroke"].value_counts(dropna=False).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    outcome_counts.plot(kind="bar", ax=ax, color=["#4E79A7", "#E15759"])
    ax.set_title("Stroke Outcome Distribution")
    ax.set_xlabel("Stroke (0 = No, 1 = Yes)")
    ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0])
    st.pyplot(fig)

    stroke_rate = outcome_counts.get(1, 0) / max(outcome_counts.sum(), 1) * 100
    st.metric("Overall stroke rate", f"{stroke_rate:.2f}%")

# -----------------------------------------------------------------------------
with right_col:
    st.subheader("2) Trend: Stroke Rate by Age Group")
    age_df = raw_df[["age", "stroke"]].dropna()
    if not age_df.empty:
        age_bins = [0, 20, 30, 40, 50, 60, 70, 80, np.inf]
        age_labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "80+"]
        age_df["age_group"] = pd.cut(age_df["age"], bins=age_bins, labels=age_labels, right=True)
        age_rate = age_df.groupby("age_group", observed=False)["stroke"].mean().mul(100)

        fig, ax = plt.subplots(figsize=(8, 4))
        age_rate.plot(kind="line", marker="o", ax=ax, color="#F28E2B")
        ax.set_title("Stroke Rate (%) by Age Group")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Stroke Rate (%)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# -----------------------------------------------------------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("3) Trend: Stroke Rate by Key Risk Conditions")
    risk_cols = [col for col in ["hypertension", "heart_disease"] if col in raw_df.columns]
    if risk_cols:
        risk_rates = []
        for col in risk_cols:
            grouped = raw_df.groupby(col, dropna=True)["stroke"].mean().mul(100)
            for value, rate in grouped.items():
                risk_rates.append({"Risk Factor": col, "Value": value, "Stroke Rate (%)": rate})

        risk_plot_df = pd.DataFrame(risk_rates)
        if not risk_plot_df.empty:
            pivot_df = risk_plot_df.pivot(index="Risk Factor", columns="Value", values="Stroke Rate (%)")
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot_df.plot(kind="bar", ax=ax)
            ax.set_title("Stroke Rate by Risk Factor State")
            ax.set_xlabel("Risk Factor")
            ax.set_ylabel("Stroke Rate (%)")
            ax.legend(title="State", loc="best")
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)

# -----------------------------------------------------------------------------
with right_col:
    st.subheader("4) Pattern: Stroke Rate Across Work Type")
    if "work_type" in raw_df.columns:
        work_rate = (
            raw_df[["work_type", "stroke"]]
            .dropna()
            .groupby("work_type", observed=False)["stroke"]
            .mean()
            .sort_values(ascending=False)
            .mul(100)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        work_rate.plot(kind="bar", ax=ax, color="#59A14F")
        ax.set_title("Stroke Rate (%) by Work Type")
        ax.set_xlabel("Work Type")
        ax.set_ylabel("Stroke Rate (%)")
        plt.xticks(rotation=25, ha="right")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("5) Pattern: Stroke Rate Across Smoking Status")
    if "smoking_status" in raw_df.columns:
        smoking_rate = (
            raw_df[["smoking_status", "stroke"]]
            .dropna()
            .groupby("smoking_status", observed=False)["stroke"]
            .mean()
            .sort_values(ascending=False)
            .mul(100)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        smoking_rate.plot(kind="bar", ax=ax, color="#76B7B2")
        ax.set_title("Stroke Rate (%) by Smoking Status")
        ax.set_xlabel("Smoking Status")
        ax.set_ylabel("Stroke Rate (%)")
        plt.xticks(rotation=25, ha="right")
        st.pyplot(fig)

with right_col:
    st.subheader("6) Numeric Comparison: Glucose and BMI by Outcome")
    numeric_compare_cols = [col for col in ["avg_glucose_level", "bmi"] if col in raw_df.columns]
    if numeric_compare_cols:
        fig, axes = plt.subplots(1, len(numeric_compare_cols), figsize=(4.5 * len(numeric_compare_cols), 4))
        if len(numeric_compare_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, numeric_compare_cols):
            groups = [
                raw_df.loc[raw_df["stroke"] == 0, col].dropna(),
                raw_df.loc[raw_df["stroke"] == 1, col].dropna(),
            ]
            ax.boxplot(groups, labels=["No Stroke", "Stroke"], patch_artist=True)
            ax.set_title(f"{col} by Stroke Outcome")
            ax.set_ylabel(col)
            ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig)

# -----------------------------------------------------------------------------
left_col, right_col = st.columns(2)

numeric_df = cleaned_df.select_dtypes(include=["number"])
with left_col:
    st.subheader("7) Correlation Structure (Encoded Data)")
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Correlation Heatmap")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

with right_col:
    st.subheader("8) Feature Importance Proxy: Correlation with Stroke")
    if "stroke" in numeric_df.columns and numeric_df.shape[1] > 2:
        stroke_corr = numeric_df.corr(numeric_only=True)["stroke"].drop("stroke").sort_values()

        fig, ax = plt.subplots(figsize=(8, 5))
        stroke_corr.plot(kind="barh", ax=ax, color="#EDC948")
        ax.set_title("Feature Correlation with Stroke (Proxy Importance)")
        ax.set_xlabel("Correlation Coefficient")
        ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)

        top_patterns = stroke_corr.abs().sort_values(ascending=False).head(5)
        pattern_df = pd.DataFrame(
            {
                "Feature": top_patterns.index,
                "|Correlation with Stroke|": top_patterns.values.round(4),
            }
        )

        st.markdown("**Top correlation-based patterns detected:**")
        st.dataframe(pattern_df, use_container_width=True)
