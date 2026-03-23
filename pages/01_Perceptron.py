import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data import load_circles, load_moons
from utils.export import download_code_snippet, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme, theme_toggle
from utils.viz import plot_decision_boundary


st.set_page_config(page_title="Perceptron", layout="wide")
apply_theme()
render_sidebar("Perceptron")
theme_toggle()

st.title("Perceptron – Single Neuron Classifier")

with st.expander("Theory: why perceptrons work", expanded=True):
    st.markdown(
        """
A perceptron computes a linear score and applies a step function.
It learns by nudging weights toward misclassified samples.
"""
    )
    st.latex(r"\hat{y} = \mathbb{1}(w^T x + b \ge 0)")
    st.latex(r"w \leftarrow w + \eta (y - \hat{y}) x")

with st.expander("Why this happens?"):
    st.markdown("Misclassified points push the boundary in the direction that reduces future mistakes.")

st.markdown("---")

col1, col2 = st.columns([1, 2])
with col1:
    dataset = st.selectbox("Dataset", ["make_moons", "make_circles"]) 
    n_samples = st.slider("Samples", 100, 600, 200, 50)
    noise = st.slider("Noise", 0.0, 0.5, 0.2, 0.05)
    factor = st.slider("Circle factor", 0.1, 0.9, 0.5, 0.05)
    w1 = st.slider("Weight w1", -3.0, 3.0, 0.3, 0.1)
    w2 = st.slider("Weight w2", -3.0, 3.0, -0.2, 0.1)
    bias = st.slider("Bias", -2.0, 2.0, 0.0, 0.1)
    lr = st.slider("Learning rate", 0.01, 1.0, 0.1, 0.01, help="Step size for weight updates")
    epochs = st.slider("Epochs", 1, 50, 10, 1, help="Number of passes over the dataset")

    upload = st.file_uploader("Or upload 2D CSV", type=["csv"])

with col2:
    if dataset == "make_moons":
        X, y = load_moons(n_samples, noise)
    else:
        X, y = load_circles(n_samples, noise, factor)

    if upload:
        df = pd.read_csv(upload)
        X = df.iloc[:, :2].values
        y = df.iloc[:, 2].values.astype(int)

    w = np.array([w1, w2], dtype=float)
    fig = plot_decision_boundary(X, y, w, bias)
    st.pyplot(fig)
    scatter = px.scatter(x=X[:, 0], y=X[:, 1], color=y.astype(str),
                         labels={"x": "x1", "y": "x2"},
                         title="Interactive point cloud")
    st.plotly_chart(scatter, use_container_width=True)

st.subheader("Train from Scratch")
train_btn = st.button("Train Perceptron")

history = []
if train_btn:
    w = np.array([w1, w2], dtype=float)
    b = bias
    bar = st.progress(0)
    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            y_hat = 1 if np.dot(w, X[i]) + b >= 0 else 0
            update = lr * (y[i] - y_hat)
            w += update * X[i]
            b += update
            errors += int(update != 0)
            history.append({
                "epoch": epoch + 1,
                "x1": X[i, 0],
                "x2": X[i, 1],
                "y": int(y[i]),
                "y_hat": y_hat,
                "w1": w[0],
                "w2": w[1],
                "b": b,
            })
        bar.progress(int((epoch + 1) / epochs * 100))
        time.sleep(0.05)

    st.success("Training complete")
    fig = plot_decision_boundary(X, y, w, b)
    st.pyplot(fig)

    st.markdown("#### Step-by-step weight updates")
    st.dataframe(pd.DataFrame(history).head(30))

    download_pickle("Download trained perceptron", {"w": w, "b": b}, "perceptron.pkl")

code = """
import numpy as np

w = np.array([0.3, -0.2])
b = 0.0
lr = 0.1

for epoch in range(10):
    for x, y in data:
        y_hat = 1 if np.dot(w, x) + b >= 0 else 0
        update = lr * (y - y_hat)
        w += update * x
        b += update
"""

download_code_snippet("Export Python Code", code.strip(), "perceptron.py")
