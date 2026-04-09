import io
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_DATASET = "healthcare-dataset-stroke-data.csv"
MISSING_MARKERS = ["", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "?", "Unknown"]


def load_data(file_buffer=None, default_path: str = DEFAULT_DATASET) -> pd.DataFrame:
    if file_buffer is not None:
        content = file_buffer.getvalue()
        return pd.read_csv(io.BytesIO(content), na_values=MISSING_MARKERS)
    return pd.read_csv(default_path, na_values=MISSING_MARKERS)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    return pd.DataFrame(
        {
            "Missing Count": missing_count,
            "Missing %": missing_percent,
        }
    ).sort_values("Missing Count", ascending=False)


def duplicate_summary(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    duplicate_rows = df[df.duplicated(keep=False)]
    return int(df.duplicated().sum()), duplicate_rows