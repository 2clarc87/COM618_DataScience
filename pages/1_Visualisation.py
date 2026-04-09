import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app_core.data_quality import (
    duplicate_summary,
    missing_summary
)


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


def render_column_insights(df: pd.DataFrame) -> None:
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


st.title("Visualisation")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page to view visualisations.")
    st.stop()

data = st.session_state["uploaded_data"]

render_missing_values(data)
st.divider()
render_duplicates(data)
st.divider()
render_column_insights(data)
