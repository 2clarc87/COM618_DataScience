import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from app_core.data_functions import classification_metrics

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Models")

if "uploaded_data" not in st.session_state:
    st.info("Upload a CSV file on the Home page first.")
    st.stop()

df = st.session_state["cleaned_data"].copy()

if df.empty:
    st.error("Dataset is empty.")
    st.stop()

target_col = "stroke"
feature_cols = [c for c in df.columns if c != target_col]

if len(feature_cols) == 0:
    st.error("No feature columns available after selecting target.")
    st.stop()

selected_models = st.multiselect(
    "Select models",
    [
        "Logistic Regression",
        "Random Forest",
        "KNN",
        "Naive Bayes",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ],
    default=["Logistic Regression", "Random Forest", "KNN"],
)

test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
seed = st.session_state.get("seed", 42)

if len(selected_models) == 0:
    st.info("Choose at least one model.")
    st.stop()

x = df[feature_cols].copy()
y = df[target_col].copy()

if y.nunique(dropna=True) < 2:
    st.error("Target must contain at least two classes for classification.")
    st.stop()

x = pd.get_dummies(x, drop_first=True)
x = x.fillna(0)
valid_idx = y.notna()
x = x.loc[valid_idx]
y = y.loc[valid_idx]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=test_size,
    random_state=seed,
    stratify=y,
)

sm = SMOTE(random_state=seed, sampling_strategy=0.3)
x_train, y_train = sm.fit_resample(x_train, y_train)

def fit_xgboost(x_tr: pd.DataFrame, y_tr: pd.Series):
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            random_state=seed,
            eval_metric="logloss",
        )
    except Exception:
        model = RandomForestClassifier(n_estimators=250, random_state=seed)
    return model.fit(x_tr, y_tr)


def fit_lightgbm(x_tr: pd.DataFrame, y_tr: pd.Series):
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=250, learning_rate=0.05, random_state=seed)
    except Exception:
        model = RandomForestClassifier(n_estimators=250, random_state=seed)
    return model.fit(x_tr, y_tr)


def fit_catboost(x_tr: pd.DataFrame, y_tr: pd.Series):
    try:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(iterations=250, learning_rate=0.05, random_seed=seed, verbose=False)
    except Exception:
        model = RandomForestClassifier(n_estimators=250, random_state=seed)
    return model.fit(x_tr, y_tr)


model_dict = {
    "Logistic Regression": lambda x_tr, y_tr: Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))]
    ).fit(x_tr, y_tr),
    "KNN": lambda x_tr, y_tr: Pipeline(
        [("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]
    ).fit(x_tr, y_tr),
    "Naive Bayes": lambda x_tr, y_tr: GaussianNB().fit(x_tr, y_tr),
    "Random Forest": lambda x_tr, y_tr: RandomForestClassifier(class_weight="balanced", n_estimators=250, random_state=seed).fit(
        x_tr, y_tr
    ),
    "XGBoost": fit_xgboost,
    "LightGBM": fit_lightgbm,
    "CatBoost": fit_catboost,
}

results = []
conf_mats = {}

for name in selected_models:
    model = model_dict[name](x_train, y_train)
    predictions = model.predict(x_test)
    metrics = classification_metrics(y_test, predictions, is_binary_target=(y.nunique(dropna=True) <= 2))

    labels = metrics["labels"]
    conf_mats[name] = {
        "cm": confusion_matrix(
            pd.Series(y_test).astype(str),
            pd.Series(predictions).astype(str),
            labels=labels,
        ),
        "labels": labels,
    }

    results.append(
        {
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"],
        }
    )

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)

st.subheader("Model Performance Comparison")
st.dataframe(results_df.style.format({c: "{:.4f}" for c in results_df.columns if c != "Model"}), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Accuracy Comparison")
    fig_acc = px.bar(results_df, x="Model", y="Accuracy", text="Accuracy")
    fig_acc.update_traces(texttemplate="%{text:.4f}")
    st.plotly_chart(fig_acc, use_container_width=True)

with col2:
    st.subheader("F1 Score Comparison")
    fig_f1 = px.bar(results_df, x="Model", y="F1 Score", text="F1 Score")
    fig_f1.update_traces(texttemplate="%{text:.4f}")
    st.plotly_chart(fig_f1, use_container_width=True)

st.subheader("Confusion Matrices")
for name in selected_models:
    st.markdown(f"### {name}")
    cm = conf_mats[name]["cm"]
    labels = conf_mats[name]["labels"]

    heatmap = ff.create_annotated_heatmap(
        z=cm,
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        showscale=True,
    )
    heatmap.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(heatmap, use_container_width=True, key=f"{name}")
