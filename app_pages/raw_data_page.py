import streamlit as st
import pandas as pd


def render_raw_data_page(data: pd.DataFrame) -> None:
    st.header("Raw Data")
    st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
    st.dataframe(data)

    st.download_button(
        label="Download current raw data",
        data=data.to_csv(index=False),
        file_name="raw_data_export.csv",
        mime="text/csv",
    )
