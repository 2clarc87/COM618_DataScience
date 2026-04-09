import io

import pandas as pd

MISSING_MARKERS = ["", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "?", "Unknown"]


def load_data(file_buffer=None, default_path=None):
    if file_buffer is not None or default_path is None :
        content = file_buffer.getvalue()
        return pd.read_csv(io.BytesIO(content), na_values=MISSING_MARKERS)
    return pd.read_csv(default_path, na_values=MISSING_MARKERS)


def missing_summary(df):
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    return pd.DataFrame(
        {
            "Missing Count": missing_count,
            "Missing %": missing_percent,
        }
    ).sort_values("Missing Count", ascending=False)


def duplicate_summary(df):
    duplicate_rows = df[df.duplicated(keep=False)]
    return int(df.duplicated().sum()), duplicate_rows


def fill_numeric_missing_with_mean(df):
    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    mean_map = {}

    for col in numeric_columns:
        mean_value = cleaned[col].mean(skipna=True)
        mean_map[col] = float(mean_value) if pd.notna(mean_value) else None
        if pd.notna(mean_value):
            cleaned[col] = cleaned[col].fillna(mean_value)

    return cleaned, mean_map


def clean_data_context_aware(df):
    cleaned = df.copy()
    fill_map = {}

    for col in cleaned.columns:
        series = cleaned[col]
        if pd.api.types.is_numeric_dtype(series):
            fill_value = series.mean(skipna=True)
            if pd.notna(fill_value):
                cleaned[col] = series.fillna(fill_value)
                fill_map[col] = {"strategy": "mean", "value": float(fill_value)}
            else:
                fill_map[col] = {"strategy": "mean", "value": None}
        else:
            mode = series.mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[col] = series.fillna(fill_value)
            fill_map[col] = {"strategy": "mode", "value": fill_value}

    return cleaned, fill_map
