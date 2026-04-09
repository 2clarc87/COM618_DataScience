import pandas as pd
import streamlit as st

from app_core.data_quality import apply_stroke_preprocessing

st.title("Data Preparation")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page.")
    st.stop()

raw_df = st.session_state["uploaded_data"]

cleaned_df = apply_stroke_preprocessing(raw_df)
st.session_state["cleaned_data"] = cleaned_df

preprocess_steps = pd.DataFrame(
    {
        "Step": [
            "Drop id",
            "Drop missing bmi rows",
            "Binary encode gender",
            "Binary encode ever_married",
            "Binary encode Residence_type",
            "One-hot encode work_type (drop_first=True)",
            "One-hot encode smoking_status (drop_first=True)",
        ]
    }
)

st.subheader("Applied Preprocessing Steps")
st.dataframe(preprocess_steps, use_container_width=True)

st.subheader("Cleaned Data")
st.dataframe(cleaned_df)
