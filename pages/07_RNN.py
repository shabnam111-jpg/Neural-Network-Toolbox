import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn

from utils.export import download_code_snippet, download_torch_state
from utils.nav import render_sidebar
from utils.theme import apply_theme


st.set_page_config(page_title="RNN", layout="wide")
apply_theme()
render_sidebar("RNN")

st.title("RNN – Sequence Modeling Lab")

with st.expander("Theory: recurrent nets", expanded=True):
    st.markdown("RNNs carry information across time steps with a hidden state.")
    st.latex(r"h_t = \tanh(W_x x_t + W_h h_{t-1})")

with st.expander("Why this happens?"):
    st.markdown("Hidden states capture context, enabling predictions from past steps.")

col1, col2 = st.columns([1, 2])
with col1:
    task = st.selectbox("Task", ["Sine wave prediction", "Synthetic time series"])
    seq_len = st.slider("Sequence length", 10, 60, 30, 5)
    hidden = st.slider("Hidden size", 8, 64, 24, 4, help="More units capture longer patterns")
    epochs = st.slider("Epochs", 10, 200, 60, 10)
    lr = st.slider("Learning rate", 0.0005, 0.05, 0.01, 0.0005)

with col2:
    st.markdown("#### Hidden state evolution")
    st.info("Train to see predictions roll out over time.")

if st.button("Train RNN"):
    torch.manual_seed(42)
    t = np.linspace(0, 8 * np.pi, 500)
    series = np.sin(t) if task == "Sine wave prediction" else np.sin(t) + 0.2 * np.cos(3 * t)

    X = []
    y = []
    for i in range(len(series) - seq_len - 1):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    model = nn.RNN(input_size=1, hidden_size=hidden, batch_first=True)
    head = nn.Linear(hidden, 1)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.MSELoss()

    bar = st.progress(0)
    for epoch in range(epochs):
        opt.zero_grad()
        out, h = model(X)
        pred = head(out[:, -1, :])
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        bar.progress(int((epoch + 1) / epochs * 100))
        if epoch % 20 == 0:
            time.sleep(0.01)

    st.success("Training complete")
    st.line_chart({"target": y.squeeze().numpy(), "prediction": pred.detach().squeeze().numpy()})

    df_plot = {
        "t": list(range(len(y))),
        "target": y.squeeze().numpy(),
        "prediction": pred.detach().squeeze().numpy(),
    }
    plot = px.line(df_plot, x="t", y=["target", "prediction"], title="Sequence forecast")
    st.plotly_chart(plot, use_container_width=True)

    fig, ax = plt.subplots(figsize=(4, 3))
    hidden_map = out.detach().cpu().numpy()[0]
    hidden_map = hidden_map[:min(30, hidden_map.shape[0]), :min(30, hidden_map.shape[1])]
    ax.imshow(hidden_map, aspect="auto", cmap="magma")
    ax.set_title("Hidden state slice (Matplotlib)")
    st.pyplot(fig)

    download_torch_state("Download RNN model", {
        "rnn": model.state_dict(),
        "head": head.state_dict(),
    }, "rnn_model.pt")

code = """
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=1, hidden_size=24, batch_first=True)
head = nn.Linear(24, 1)
"""

download_code_snippet("Export Python Code", code.strip(), "rnn_model.py")
