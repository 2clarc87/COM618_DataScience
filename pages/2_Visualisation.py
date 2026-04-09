import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app_core.data_quality import (
    DEFAULT_DATASET,
    duplicate_summary,
    generic_error_checks,
    load_data,
    missing_summary,
    stroke_dataset_error_checks,
)


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return load_data(default_path=DEFAULT_DATASET)


def resolve_data() -> pd.DataFrame:
    if "uploaded_data" in st.session_state:
        return st.session_state["uploaded_data"]
    return load_default_data()


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


def render_error_data(df: pd.DataFrame) -> None:
    st.subheader("Error Data")

    if {"age", "avg_glucose_level", "bmi", "gender"}.issubset(df.columns):
        checks = stroke_dataset_error_checks(df)
        st.info("Using validation rules tailored for the stroke dataset.")
    else:
        checks = generic_error_checks(df)
        st.info("Using generic validation rules (numeric format + outlier checks).")

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

try:
    data = resolve_data()
except FileNotFoundError:
    st.error(f"Default dataset '{DEFAULT_DATASET}' not found. Please upload from Home page.")
    st.stop()

source = "Uploaded CSV" if "uploaded_data" in st.session_state else f"Default: {DEFAULT_DATASET}"
st.caption(f"Data source: {source}")

section = st.selectbox("Choose section", ["Missing Values", "Duplication", "Error Data"])
if section == "Missing Values":
    render_missing_values(data)
elif section == "Duplication":
    render_duplicates(data)
else:
    render_error_data(data)
