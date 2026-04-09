import streamlit as st

from app_core.data_quality import DEFAULT_DATASET, load_data

st.set_page_config(page_title="Health Care Data Science", layout="wide")

st.title("Health Care Data Science")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        data = load_data(file_buffer=uploaded)
        st.session_state["uploaded_data"] = data
        st.success("Success")
        st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
else:
    if "uploaded_data" in st.session_state:
        if st.button("Clear uploaded data and use default dataset"):
            del st.session_state["uploaded_data"]
            st.success("Cleared uploaded data. Pages will use the default dataset.")
    else:
        st.info(
            f"No upload yet. Pages will use the default dataset: `{DEFAULT_DATASET}`."
        )
        try:
            default_df = load_data(default_path=DEFAULT_DATASET)
            st.write(f"Default dataset preview: {default_df.shape[0]} rows x {default_df.shape[1]} columns")
            st.dataframe(default_df.head(10))
        except FileNotFoundError:
            st.warning(
                f"Default dataset '{DEFAULT_DATASET}' not found. Upload a CSV to continue."
            )
