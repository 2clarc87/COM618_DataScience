import pandas as pd
import plotly.express as px
import streamlit as st

from app_core.data_functions import duplicate_summary, missing_summary, random_colour


def render_missing_values(df):
    summary = missing_summary(df)
    fig = px.bar(summary, y="Missing Count", title="Missing Values by Column", labels={"index": "Columns"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary)


def render_duplicates(df):
    duplicate_count, duplicate_rows = duplicate_summary(df)
    st.metric("Duplicate Rows", duplicate_count)


def render_column_insights(df):
    unique_counts = df.nunique(dropna=True).sort_values(ascending=False).reset_index()
    unique_counts.columns = ["Column", "Unique Count"]

    fig = px.bar(unique_counts, x="Column", y="Unique Count", title="Number of Unique Values by Column")
    st.plotly_chart(fig, use_container_width=True)

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
    for i, col in enumerate(cols):
        fig = px.histogram(
            df,
            x=col,
            nbins=40,
            title=f"Distribution: {col}",
            color_discrete_sequence=[random_colour()]
        )
        st.plotly_chart(fig, use_container_width=True)

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
render_duplicates(data)
render_column_insights(data)

session_df = st.session_state.get("df")
session_cols = st.session_state.get("cols")
seed = st.session_state.get("seed")

if seed is not None:
    st.caption(f"Current seed: {seed}")

if session_df is None or session_cols is None:
    st.warning("Load data first.")
    st.stop()

render_graphs(session_df, session_cols)
