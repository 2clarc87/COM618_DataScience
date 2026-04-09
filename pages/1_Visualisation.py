import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from app_core.data_quality import duplicate_summary, missing_summary


def render_missing_values(df):
    st.subheader("Missing Values")
    summary = missing_summary(df)

    fig, ax = plt.subplots(figsize=(10, 4))
    summary["Missing Count"].plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_ylabel("Missing Count")
    ax.set_xlabel("Columns")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    st.dataframe(summary)


def render_duplicates(df):
    st.subheader("Duplication")
    duplicate_count, duplicate_rows = duplicate_summary(df)
    st.metric("Duplicate Rows", duplicate_count)
    st.dataframe(duplicate_rows)


def render_column_insights(df):
    st.subheader("Column Insights")

    unique_counts = df.nunique(dropna=True).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    unique_counts.plot(kind="bar", ax=ax)
    ax.set_title("Number of Unique Values by Column")
    ax.set_ylabel("Unique Count")
    ax.set_xlabel("Columns")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    summary_rows = []
    for col in df.columns:
        non_null = df[col].dropna()
        mode_series = non_null.mode()
        most_common_value = mode_series.iloc[0] if not mode_series.empty else None
        most_common_count = int((non_null == most_common_value).sum()) if most_common_value is not None else 0
        mean_value = pd.to_numeric(df[col], errors="coerce").mean()

        summary_rows.append(
            {
                "Column": col,
                "Unique Values": int(df[col].nunique(dropna=True)),
                "Most Common Value": most_common_value,
                "Most Common Count": most_common_count,
                "Mean (numeric columns)": round(float(mean_value), 3) if pd.notna(mean_value) else None,
            }
        )

    st.dataframe(pd.DataFrame(summary_rows))


def render_graphs(df, cols):
    st.subheader("Graphs")

    col1, col2 = st.columns(2)

    if "Label" in df.columns:
        col1.text("Label Counts")
        col1.bar_chart(df["Label"].value_counts())
    else:
        col1.info("`Label` column not found, so label counts are unavailable.")

    numeric_df = df[cols].select_dtypes(include="number")
    col2.text("Correlation Matrix")
    if numeric_df.empty:
        col2.info("No numeric columns available for correlation matrix.")
    else:
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Blues")
        col2.plotly_chart(fig_corr, use_container_width=True)

    for i, col in enumerate(cols):
        fig = px.histogram(df, x=col, nbins=40, title=f"Distribution: {col}")
        if i % 2 == 0:
            col1.plotly_chart(fig, use_container_width=True)
        else:
            col2.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------

st.set_page_config(page_title="Visualisation", layout="wide")
st.title("Visualisation")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page to view visualisations.")
    st.stop()

data = st.session_state["uploaded_data"]
st.session_state.setdefault("df", data)
st.session_state.setdefault("cols", list(data.columns))

render_missing_values(data)
st.divider()
render_duplicates(data)
st.divider()
render_column_insights(data)
st.divider()

session_df = st.session_state.get("df")
session_cols = st.session_state.get("cols")
seed = st.session_state.get("seed")

if seed is not None:
    st.caption(f"Current seed: {seed}")

if session_df is None or session_cols is None:
    st.warning("Load data first.")
    st.stop()

render_graphs(session_df, session_cols)
