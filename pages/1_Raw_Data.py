import streamlit as st
import pandas as pd

from app_core.data_quality import DEFAULT_DATASET, load_data


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return load_data(default_path=DEFAULT_DATASET)


def resolve_data() -> pd.DataFrame:
    if "uploaded_data" in st.session_state:
        return st.session_state["uploaded_data"]
    return load_default_data()


st.title("Raw Data")

try:
    data = resolve_data()
except FileNotFoundError:
    st.error(f"Default dataset '{DEFAULT_DATASET}' not found. Please upload from Home page.")
    st.stop()

source = "Uploaded CSV" if "uploaded_data" in st.session_state else f"Default: {DEFAULT_DATASET}"
st.caption(f"Data source: {source}")
st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
st.dataframe(data)

st.download_button(
    label="Download current raw data",
    data=data.to_csv(index=False),
    file_name="raw_data_export.csv",
    mime="text/csv",
)
