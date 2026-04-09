import pandas as pd
import streamlit as st

from app_core.data_quality import DEFAULT_DATASET, load_data
from app_pages.raw_data_page import render_raw_data_page
from app_pages.visualisation_page import render_visualisation_page


st.set_page_config(page_title="Data Quality Dashboard", layout="wide")


@st.cache_data
def load_data_cached(file_buffer=None, default_path: str = DEFAULT_DATASET) -> pd.DataFrame:
    return load_data(file_buffer=file_buffer, default_path=default_path)


st.title("Data Science Project: Raw Data & Visualisation")
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

try:
    data = load_data_cached(uploaded)
except FileNotFoundError:
    st.error(
        f"Default dataset '{DEFAULT_DATASET}' not found. Please upload a CSV file to continue."
    )
    st.stop()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Raw Data", "Visualisation"])

if page == "Raw Data":
    render_raw_data_page(data)
else:
    render_visualisation_page(data)
