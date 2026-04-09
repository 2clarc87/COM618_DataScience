import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app_core.data_quality import (
    duplicate_summary,
    generic_error_checks,
    missing_summary,
    stroke_dataset_error_checks,
)


def render_missing_values(df: pd.DataFrame) -> None:
    st.subheader("Missing Values")
    summary = missing_summary(df)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Total Missing Cells", int(df.isna().sum().sum()))
        st.metric("Columns with Missing Data", int((summary["Missing Count"] > 0).sum()))

    with col2:
        fig, ax = plt.subplots(figsize=(10, 4))
        summary["Missing Count"].plot(kind="bar", ax=ax)
        ax.set_title("Missing Values by Column")
        ax.set_ylabel("Missing Count")
        ax.set_xlabel("Columns")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    st.caption("Table")
    st.dataframe(summary)


def render_duplicates(df: pd.DataFrame) -> None:
    st.subheader("Duplication")
    duplicate_count, duplicate_rows = duplicate_summary(df)

    st.metric("Duplicate Rows", duplicate_count)
    st.caption("Values")
    st.write(f"Duplicate row percentage: {(duplicate_count / len(df) * 100):.2f}%")

    st.caption("Table")
    if duplicate_rows.empty:
        st.success("No duplicate rows detected.")
    else:
        st.dataframe(duplicate_rows)


def render_column_insights(df: pd.DataFrame) -> None:
    st.subheader("Column Insights")

    unique_counts = df.nunique(dropna=True).sort_values(ascending=False)

    st.caption("Graph")
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

    st.caption("Table")
    st.dataframe(pd.DataFrame(summary_rows))


def render_error_data(df: pd.DataFrame) -> None:
    st.subheader("Error Data")

    if {"age", "avg_glucose_level", "bmi", "gender"}.issubset(df.columns):
        checks = stroke_dataset_error_checks(df)
        st.info("Using validation rules tailored for the stroke dataset (data types + validity ranges).")
    else:
        checks = generic_error_checks(df)
        st.info("Using generic validation rules (data type consistency + numeric format + outlier checks).")

    error_counts = {
        check_name: len(error_rows)
        for check_name, error_rows in checks.items()
        if len(error_rows) > 0
    }

    st.caption("Values")
    st.write(f"Total error categories triggered: {len(error_counts)}")

    if not error_counts:
        st.success("No error data found by current checks.")
        return

    st.caption("Graph")
    fig, ax = plt.subplots(figsize=(10, 4))
    pd.Series(error_counts).sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_title("Error Records by Check")
    ax.set_ylabel("Row Count")
    ax.set_xlabel("Validation Checks")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    st.caption("Tables")
    for check_name, error_rows in checks.items():
        if len(error_rows) == 0:
            continue
        with st.expander(f"{check_name} ({len(error_rows)} rows)"):
            st.dataframe(error_rows)


st.title("Visualisation")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page to view visualisations.")
    st.stop()

data = st.session_state["uploaded_data"]
st.caption("Data source: Uploaded CSV")

render_missing_values(data)
st.divider()
render_duplicates(data)
st.divider()
render_column_insights(data)
st.divider()
render_error_data(data)
