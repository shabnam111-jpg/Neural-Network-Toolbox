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
    pred_first = float(pred[0].item())
    y_first = float(y[0].item())
    loss_first = 0.5 * (pred_first - y_first) ** 2
    st.markdown("#### Step-by-step calculations")
    st.write({
        "input_shape": list(X.shape),
        "output_shape": list(out.shape),
        "y0": y_first,
        "pred0": pred_first,
        "mse_sample": float(loss_first),
    })
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

st.divider()
st.title("LSTM – Sentiment Analysis (Chat + Live Video)")

with st.expander("Theory: LSTM gates", expanded=True):
    st.markdown(
        "LSTMs add gates that control what to remember, what to forget, and what to output. "
        "This helps with longer-term dependencies than a vanilla RNN."
    )
    st.latex(
        r"f_t = \sigma(W_f x_t + U_f h_{t-1}) \\"
        r"i_t = \sigma(W_i x_t + U_i h_{t-1}) \\"
        r"o_t = \sigma(W_o x_t + U_o h_{t-1}) \\"
        r"c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1})"
    )

st.caption("Sentiment is computed from chat text. Live video is a preview only.")

SENTIMENT_SAMPLES = [
    ("i love this", 1),
    ("this is amazing", 1),
    ("so happy and excited", 1),
    ("great job", 1),
    ("this is good", 1),
    ("i hate this", 0),
    ("this is terrible", 0),
    ("so sad and angry", 0),
    ("bad experience", 0),
    ("this is awful", 0),
]


def _tokenize(text: str) -> list:
    return [tok.strip() for tok in text.lower().split() if tok.strip()]


def _build_vocab(samples: list) -> dict:
    vocab = {"<pad>": 0, "<unk>": 1}
    for text, _ in samples:
        for tok in _tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def _encode(text: str, vocab: dict, max_len: int) -> list:
    tokens = _tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids[:max_len]


def _train_lstm_sentiment() -> None:
    torch.manual_seed(42)
    vocab = _build_vocab(SENTIMENT_SAMPLES)
    max_len = max(len(_tokenize(text)) for text, _ in SENTIMENT_SAMPLES)
    max_len = max(3, min(max_len, 12))

    X = torch.tensor([
        _encode(text, vocab, max_len) for text, _ in SENTIMENT_SAMPLES
    ], dtype=torch.long)
    y = torch.tensor([label for _, label in SENTIMENT_SAMPLES], dtype=torch.float32).unsqueeze(-1)

    embed_dim = 16
    hidden_dim = 24
    model = nn.Module()
    model.embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=0)
    model.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    model.head = nn.Linear(hidden_dim, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCEWithLogitsLoss()

    bar = st.progress(0)
    epochs = 80
    for epoch in range(epochs):
        optimizer.zero_grad()
        emb = model.embedding(X)
        out, _ = model.lstm(emb)
        logits = model.head(out[:, -1, :])
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bar.progress(int((epoch + 1) / epochs * 100))

    st.session_state["lstm_vocab"] = vocab
    st.session_state["lstm_max_len"] = max_len
    st.session_state["lstm_model"] = model
    st.success("LSTM sentiment model trained.")


def _predict_sentiment(text: str) -> float:
    model = st.session_state.get("lstm_model")
    vocab = st.session_state.get("lstm_vocab")
    max_len = st.session_state.get("lstm_max_len")
    if model is None or vocab is None or max_len is None:
        return 0.5
    ids = torch.tensor([_encode(text, vocab, max_len)], dtype=torch.long)
    with torch.no_grad():
        emb = model.embedding(ids)
        out, _ = model.lstm(emb)
        logits = model.head(out[:, -1, :])
        prob = torch.sigmoid(logits).item()
    return float(prob)


col_lstm1, col_lstm2 = st.columns([1, 1])
with col_lstm1:
    st.markdown("#### Train the LSTM")
    st.caption("Tiny demo dataset for quick training.")
    if st.button("Train LSTM sentiment model"):
        _train_lstm_sentiment()

    st.markdown("#### Live video")
    cam_frame = st.camera_input("Capture a frame")
    if cam_frame:
        st.image(cam_frame, caption="Live frame", use_column_width=True)

with col_lstm2:
    st.markdown("#### Chat sentiment")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_text = st.text_input("Type a message")
    if st.button("Analyze sentiment") and user_text.strip():
        score = _predict_sentiment(user_text)
        label = "Positive" if score >= 0.5 else "Negative"
        percent = int(round(score * 100))
        st.session_state["chat_history"].append((user_text, label, percent))

    for text, label, percent in reversed(st.session_state["chat_history"]):
        st.markdown(f"**{label}** ({percent}%) — {text}")
