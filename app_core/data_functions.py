import io
import random
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


MISSING_MARKERS = ["", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "?", "Unknown"]


def load_data(file_buffer=None, default_path=None):
    if file_buffer is not None or default_path is None:
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


def apply_stroke_preprocessing(df):
    preprocessed = df.copy()

    if "id" in preprocessed.columns:
        preprocessed = preprocessed.drop(columns=["id"])

    if "bmi" in preprocessed.columns:
        preprocessed = preprocessed.dropna(subset=["bmi"])

    # -- Binary --
    binary_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "ever_married": {"No": 0, "Yes": 1},
        "Residence_type": {"Rural": 0, "Urban": 1},
    }
    for column, mapping in binary_mappings.items():
        if column in preprocessed.columns:
            preprocessed[column] = preprocessed[column].map(mapping)

    # -- One Hot Encoding --
    one_hot_columns = [
        column
        for column in ["work_type", "smoking_status"]
        if column in preprocessed.columns
    ]
    if one_hot_columns:
        preprocessed = pd.get_dummies(
            preprocessed,
            columns=one_hot_columns,
            drop_first=True,
            dtype=int,
        )

    # -- Bool --
    bool_columns = preprocessed.select_dtypes(include="bool").columns
    if len(bool_columns) > 0:
        preprocessed[bool_columns] = preprocessed[bool_columns].astype(int)

    return preprocessed


def classification_metrics(y_test, preds, is_binary_target):
    if not is_binary_target and y_test.dtype == "object":
        y_test_eval = y_test.astype(str)
        pred_eval = pd.Series(preds).astype(str)
        average_mode = "weighted"
    else:
        y_test_eval = y_test
        pred_eval = preds
        average_mode = "binary" if is_binary_target else "weighted"

    labels = sorted(pd.Series(y_test_eval).astype(str).unique().tolist())
    cm = confusion_matrix(pd.Series(y_test_eval).astype(str), pd.Series(pred_eval).astype(str), labels=labels)

    return {
        "accuracy": accuracy_score(y_test_eval, pred_eval),
        "precision": precision_score(y_test_eval, pred_eval, average=average_mode, zero_division=0),
        "recall": recall_score(y_test_eval, pred_eval, average=average_mode, zero_division=0),
        "f1": f1_score(y_test_eval, pred_eval, average=average_mode, zero_division=0),
        "cm": cm,
        "labels": labels,
        "y_test_eval": y_test_eval,
        "pred_eval": pred_eval,
    }

def random_colour():
    r = random.randint(60, 200)
    g = random.randint(60, 200)
    b = random.randint(60, 200)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)