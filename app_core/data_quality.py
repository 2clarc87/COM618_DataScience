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


def stroke_dataset_error_checks(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    checks = {}

    expected_numeric_columns = [
        "id",
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "stroke",
    ]
    for col in expected_numeric_columns:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            invalid_numeric_mask = df[col].notna() & converted.isna()
            checks[f"Non-numeric values in expected numeric column: {col}"] = df[invalid_numeric_mask]

    if "age" in df.columns:
        age_numeric = pd.to_numeric(df["age"], errors="coerce")
        checks["Age outside 0-120"] = df[(age_numeric.notna()) & ((age_numeric < 0) | (age_numeric > 120))]

    if "bmi" in df.columns:
        bmi_numeric = pd.to_numeric(df["bmi"], errors="coerce")
        checks["BMI outside 10-80"] = df[(bmi_numeric.notna()) & ((bmi_numeric < 10) | (bmi_numeric > 80))]

    if "avg_glucose_level" in df.columns:
        glucose_numeric = pd.to_numeric(df["avg_glucose_level"], errors="coerce")
        checks["Glucose outside 40-400"] = df[
            (glucose_numeric.notna())
            & ((glucose_numeric < 40) | (glucose_numeric > 400))
        ]

    binary_cols = ["hypertension", "heart_disease", "stroke"]
    for col in binary_cols:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            checks[f"Invalid values in {col} (expected 0/1)"] = df[
                (converted.notna()) & (~converted.isin([0, 1]))
            ]

    allowed_categories = {
        "gender": {"Male", "Female", "Other"},
        "ever_married": {"Yes", "No"},
        "work_type": {"Private", "Self-employed", "Govt_job", "children", "Never_worked"},
        "Residence_type": {"Urban", "Rural"},
        "smoking_status": {"formerly smoked", "never smoked", "smokes", "Unknown"},
    }

    for col, allowed in allowed_categories.items():
        if col in df.columns:
            checks[f"Unexpected categories in {col}"] = df[
                (df[col].notna()) & (~df[col].isin(allowed))
            ]

    return checks


def generic_error_checks(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    checks = {}
    numeric_like_columns: List[str] = []

    for col in df.columns:
        non_null_series = df[col].dropna()
        observed_types = non_null_series.map(type).unique()
        if len(observed_types) > 1:
            checks[f"Mixed data types in column: {col}"] = df[df[col].notna()]

        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= (0.8 * len(df)):
            numeric_like_columns.append(col)
            invalid_numeric_mask = df[col].notna() & converted.isna()
            checks[f"Non-numeric values in numeric-like column: {col}"] = df[invalid_numeric_mask]

    for col in numeric_like_columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = converted.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = converted.notna() & ((converted < lower) | (converted > upper))
        checks[f"Potential outliers in {col} (IQR rule)"] = df[outlier_mask]

    return checks
