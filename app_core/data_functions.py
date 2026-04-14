import io

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def apply_stroke_preprocessing(df):
    preprocessed = df.copy()

    if "id" in preprocessed.columns:
        preprocessed = preprocessed.drop(columns=["id"])

    if "bmi" in preprocessed.columns:
        preprocessed = preprocessed.dropna(subset=["bmi"])

    binary_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "ever_married": {"No": 0, "Yes": 1},
        "Residence_type": {"Rural": 0, "Urban": 1},
    }

    for column, mapping in binary_mappings.items():
        if column in preprocessed.columns:
            preprocessed[column] = preprocessed[column].map(mapping)

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

    bool_columns = preprocessed.select_dtypes(include="bool").columns
    if len(bool_columns) > 0:
        preprocessed[bool_columns] = preprocessed[bool_columns].astype(int)

    return preprocessed


def build_feature_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def run_kmeans(df: pd.DataFrame, feature_cols: list[str], n_clusters: int):
    X = df[feature_cols].copy()
    preprocessor = build_feature_preprocessor(X)
    X_encoded = preprocessor.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(X_encoded)

    output = df.copy()
    output["cluster"] = cluster_labels
    return output


def prepare_supervised_data(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    valid_idx = y.notna()
    return X.loc[valid_idx], y.loc[valid_idx]


def choose_model(model_name: str, y: pd.Series):
    is_binary_target = y.nunique(dropna=True) <= 2
    is_numeric_target = pd.api.types.is_numeric_dtype(y)

    if model_name == "Linear Regression":
        return LinearRegression(), "regression", is_binary_target

    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000), "classification", is_binary_target

    if model_name == "Random Forest":
        if is_binary_target or (not is_numeric_target):
            return RandomForestClassifier(n_estimators=250, random_state=42), "classification", is_binary_target
        return RandomForestRegressor(n_estimators=250, random_state=42), "regression", is_binary_target

    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier, XGBRegressor

            if is_binary_target or (not is_numeric_target):
                return (
                    XGBClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=5,
                        random_state=42,
                        eval_metric="logloss",
                    ),
                    "classification",
                    is_binary_target,
                )
            return (
                XGBRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                ),
                "regression",
                is_binary_target,
            )
        except Exception:
            if is_binary_target or (not is_numeric_target):
                return RandomForestClassifier(n_estimators=250, random_state=42), "classification", is_binary_target
            return RandomForestRegressor(n_estimators=250, random_state=42), "regression", is_binary_target

    raise ValueError(f"Unsupported model: {model_name}")


def train_supervised_model(df: pd.DataFrame, target_col: str, model_name: str, test_size_pct: int):
    X, y = prepare_supervised_data(df, target_col)
    if X.empty:
        raise ValueError("No valid rows left after dropping missing target values.")

    stratify = y if y.nunique(dropna=True) <= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_pct / 100,
        random_state=42,
        stratify=stratify,
    )

    estimator, model_type, is_binary_target = choose_model(model_name, y)
    pipe = Pipeline(
        steps=[
            ("preprocess", build_feature_preprocessor(X)),
            ("model", estimator),
        ]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    return {
        "preds": preds,
        "y_test": y_test,
        "model_type": model_type,
        "is_binary_target": is_binary_target,
    }


def classification_metrics(y_test, preds, is_binary_target: bool):
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


def regression_metrics(y_test, preds):
    return {
        "mse": mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }
