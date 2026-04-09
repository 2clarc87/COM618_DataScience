import pandas as pd
import streamlit as st

from app_core.data_quality import clean_data_context_aware, missing_summary

st.title("Data Preparation")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page.")
    st.stop()

raw_df = st.session_state["uploaded_data"]

cleaned_df, fill_map = clean_data_context_aware(raw_df)
st.session_state["cleaned_data"] = cleaned_df

fill_table = pd.DataFrame(
    [
        {
            "Column": col,
            "Strategy": details["strategy"],
            "Value Used": details["value"],
        }
        for col, details in fill_map.items()
    ]
)

st.subheader("Column Averages")
st.dataframe(fill_table)

st.write("Fill in missing value with averages")

st.subheader("Cleaned Data")
st.dataframe(cleaned_df)
