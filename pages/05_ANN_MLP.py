import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from utils.data import csv_to_dataframe, load_iris, standardize
from utils.export import download_code_snippet, download_pickle, download_torch_state
from utils.nav import render_sidebar
from utils.theme import apply_theme, theme_toggle
from utils.viz import plot_loss_curve


st.set_page_config(page_title="ANN (MLP)", layout="wide")
apply_theme()
render_sidebar("ANN (MLP)")
theme_toggle()

st.title("ANN (MLP) – Build a Multi-Layer Perceptron")

with st.expander("Theory: multilayer perceptrons", expanded=True):
    st.markdown("MLPs stack linear layers and nonlinear activations.")
    st.latex(r"a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})")

with st.expander("Why this happens?"):
    st.markdown("Depth plus nonlinearities lets MLPs approximate complex decision boundaries.")

col1, col2 = st.columns([1, 2])
with col1:
    source = st.selectbox("Dataset", ["Iris", "Upload CSV"])
    backend = st.selectbox("Training backend", ["NumPy", "PyTorch"])
    epochs = st.slider("Epochs", 5, 200, 40, 5)
    lr = st.slider("Learning rate", 0.001, 0.5, 0.05, 0.001)
    hidden_layers = st.text_input("Hidden layer sizes (comma-separated)", "16,8", help="Example: 32,16")
    test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)

    uploaded = st.file_uploader("Upload CSV (features + label column)", type=["csv"])

with col2:
    if source == "Iris":
        X_df, y_series = load_iris()
        X = X_df.values
        y = y_series.values
    else:
        if uploaded:
            df = csv_to_dataframe(uploaded.getvalue())
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        else:
            st.warning("Upload a CSV to continue")
            st.stop()

    X = standardize(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

st.subheader("Train from Scratch")

if st.button("Train MLP"):
    sizes: List[int] = [int(v) for v in hidden_layers.split(",") if v.strip().isdigit()]
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y))

    losses = []
    if backend == "NumPy":
        weights = []
        biases = []
        layer_dims = [input_dim] + sizes + [output_dim]
        for i in range(len(layer_dims) - 1):
            weights.append(np.random.randn(layer_dims[i], layer_dims[i + 1]) * 0.1)
            biases.append(np.zeros((1, layer_dims[i + 1])))

        y_onehot = np.eye(output_dim)[y_train]
        bar = st.progress(0)
        for epoch in range(epochs):
            # Forward
            a = X_train
            activations = [a]
            for i in range(len(weights)):
                z = a @ weights[i] + biases[i]
                a = np.maximum(0, z) if i < len(weights) - 1 else z
                activations.append(a)

            logits = activations[-1]
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
            losses.append(loss)

            # Backward
            grad = (probs - y_onehot) / len(X_train)
            for i in reversed(range(len(weights))):
                dW = activations[i].T @ grad
                db = grad.sum(axis=0, keepdims=True)
                weights[i] -= lr * dW
                biases[i] -= lr * db
                if i > 0:
                    grad = grad @ weights[i].T
                    grad[activations[i] <= 0] = 0

            bar.progress(int((epoch + 1) / epochs * 100))
            time.sleep(0.01)

        logits_test = X_test
        for i in range(len(weights)):
            logits_test = logits_test @ weights[i] + biases[i]
            if i < len(weights) - 1:
                logits_test = np.maximum(0, logits_test)
        preds = np.argmax(logits_test, axis=1)

        download_pickle("Download NumPy model", {"weights": weights, "biases": biases}, "mlp_numpy.pkl")

    else:
        torch.manual_seed(42)
        layers = []
        dims = [input_dim] + sizes + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        model = nn.Sequential(*layers)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        bar = st.progress(0)
        for epoch in range(epochs):
            opt.zero_grad()
            logits = model(X_train_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            bar.progress(int((epoch + 1) / epochs * 100))
            time.sleep(0.01)

        preds = model(X_test_t).argmax(dim=1).numpy()
        download_torch_state("Download PyTorch model", model.state_dict(), "mlp_torch.pt")

    st.plotly_chart(plot_loss_curve(losses), use_container_width=True)

    cm = confusion_matrix(y_test, preds)
    st.markdown("#### Confusion Matrix")
    st.dataframe(pd.DataFrame(cm))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (Matplotlib)")
    st.pyplot(fig)

code = """
import numpy as np

# Simplified NumPy MLP training loop
for epoch in range(40):
    logits = X @ W1 @ W2
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
"""

download_code_snippet("Export Python Code", code.strip(), "mlp_train.py")
