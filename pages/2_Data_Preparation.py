import pandas as pd
import streamlit as st

from app_core.data_quality import clean_data_context_aware, duplicate_summary, missing_summary

st.title("Data Preparation")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page.")
    st.stop()

raw_df = st.session_state["uploaded_data"]

st.subheader("Before Cleaning")
col1, col2, col3 = st.columns(3)
missing_before = int(raw_df.isna().sum().sum())
duplicates_before, _ = duplicate_summary(raw_df)

col1.metric("Rows", raw_df.shape[0])
col2.metric("Missing Cells", missing_before)
col3.metric("Duplicate Rows", duplicates_before)

st.dataframe(missing_summary(raw_df))

cleaned_df, fill_map, duplicates_removed = clean_data_context_aware(raw_df)

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

st.subheader("Cleaning Applied")
st.dataframe(fill_table)

st.subheader("After Cleaning")
col4, col5, col6 = st.columns(3)
missing_after = int(cleaned_df.isna().sum().sum())
duplicates_after, _ = duplicate_summary(cleaned_df)

col4.metric("Rows", cleaned_df.shape[0])
col5.metric("Missing Cells", missing_after)
col6.metric("Duplicate Rows", duplicates_after)

st.metric("Duplicates Removed", duplicates_removed)
st.dataframe(missing_summary(cleaned_df))

st.subheader("Cleaned Data Preview")
st.dataframe(cleaned_df.head(30))

if st.button("Use cleaned dataset"):
    st.session_state["uploaded_data"] = cleaned_df
    st.success("Cleaned dataset saved to session.")
