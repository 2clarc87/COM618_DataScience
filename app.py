import streamlit as st

from app_core.data_quality import load_data

st.set_page_config(page_title="Health Care Data Science", layout="wide")

st.title("Health Care Data Science")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        data = load_data(file_buffer=uploaded)
        st.session_state["uploaded_data"] = data
        st.success("Success")
        st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
        st.dataframe(data)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
else:
    if "uploaded_data" in st.session_state:
        if st.button("Clear uploaded data"):
            del st.session_state["uploaded_data"]
            st.success("Cleared uploaded data.")
        else:
            data = st.session_state["uploaded_data"]
            st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
            st.dataframe(data)
    else:
        st.info("No upload yet. Please upload a CSV file to begin.")
