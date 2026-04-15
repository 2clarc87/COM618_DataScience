import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import Birch, KMeans, OPTICS
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Distribution", layout="wide")
st.title("Distribution")

if "uploaded_data" not in st.session_state:
    st.warning("Load data first from the Home page.")
    st.stop()

source_option = st.radio(
    "Choose data source",
    options=["Cleaned data (recommended)", "Uploaded raw data"],
    horizontal=True,
)

if source_option == "Cleaned data (recommended)" and "cleaned_data" in st.session_state:
    df = st.session_state["cleaned_data"].copy()
else:
    df = st.session_state["uploaded_data"].copy()

if df.empty:
    st.error("Dataset is empty.")
    st.stop()

label_candidates = [c for c in ["stroke", "Label", "label"] if c in df.columns]
label_col = st.selectbox(
    "Label column (for symbol coloring)",
    options=df.columns.tolist(),
    index=df.columns.get_loc(label_candidates[0]) if label_candidates else 0,
)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
feature_default = [c for c in numeric_cols if c != label_col]

feature_cols = st.multiselect(
    "Feature columns for clustering",
    options=[c for c in df.columns if c != label_col],
    default=feature_default[:8] if feature_default else [c for c in df.columns if c != label_col][:6],
)

if len(feature_cols) < 2:
    st.info("Please select at least two feature columns.")
    st.stop()

st.header("Clustering")
cluster_algo = st.selectbox("Clustering algorithm", ["KMeans", "Birch", "OPTICS"])
dr_algo = st.selectbox("Dimensionality reduction", ["PCA", "UMAP", "TruncatedSVD"])
n_clusters = st.slider("Number of clusters", 2, 12, 6)
seed = int(st.session_state.get("seed", 42))

x = pd.get_dummies(df[feature_cols], drop_first=True).fillna(0)
x_scaled = StandardScaler().fit_transform(x)

cluster_models = {
    "KMeans": KMeans(n_clusters=n_clusters, random_state=seed, n_init=10),
    "Birch": Birch(n_clusters=n_clusters),
    "OPTICS": OPTICS(min_samples=5),
}

cluster_labels = cluster_models[cluster_algo].fit_predict(x_scaled)

if dr_algo == "PCA":
    reducer = PCA(n_components=2, random_state=seed)
elif dr_algo == "TruncatedSVD":
    reducer = TruncatedSVD(n_components=2, random_state=seed)
else:
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=seed)
    except Exception:
        st.warning("UMAP is not installed; using PCA fallback.")
        reducer = PCA(n_components=2, random_state=seed)

x2 = reducer.fit_transform(x_scaled)

plot_df = pd.DataFrame(x2, columns=["dim1", "dim2"])
plot_df["cluster"] = cluster_labels.astype(str)
plot_df["label"] = df[label_col].astype(str).values

fig = px.scatter(
    plot_df,
    x="dim1",
    y="dim2",
    color="cluster",
    symbol="label",
    hover_data=["label"],
    title=f"{cluster_algo} clusters ({dr_algo})",
)

st.plotly_chart(fig, use_container_width=True)

cluster_counts = plot_df["cluster"].value_counts().sort_index().reset_index()
cluster_counts.columns = ["Cluster", "Count"]

st.subheader("Cluster Distribution")
st.dataframe(cluster_counts, use_container_width=True)
