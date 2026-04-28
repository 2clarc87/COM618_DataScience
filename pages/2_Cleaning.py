import pandas as pd
import streamlit as st

from app_core.data_functions import apply_stroke_preprocessing

st.title("Data Preparation")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page.")
    st.stop()

raw_df = st.session_state["uploaded_data"]

cleaned_df = apply_stroke_preprocessing(raw_df)
st.session_state["cleaned_data"] = cleaned_df

st.subheader("Cleaned Data")
st.dataframe(cleaned_df)
