import os
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

try:
    import cv2
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
    import av
    import tensorflow as tf
    _video_deps_error = None
except Exception as exc:
    cv2 = None
    VideoTransformerBase = None
    webrtc_streamer = None
    av = None
    tf = None
    _video_deps_error = str(exc)


st.markdown(
    """
<style>
.lstm-hero {
    background: radial-gradient(circle at top left, rgba(0, 255, 208, 0.12), transparent 45%),
                radial-gradient(circle at bottom right, rgba(0, 136, 255, 0.18), transparent 45%),
                rgba(15, 18, 26, 0.75);
    border: 1px solid rgba(0, 255, 208, 0.35);
    box-shadow: 0 0 20px rgba(0, 255, 208, 0.18);
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.lstm-title {
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: #e7f9ff;
}
.lstm-subtitle {
    color: rgba(231, 249, 255, 0.7);
    font-size: 14px;
}
.neon-card {
    background: rgba(18, 22, 32, 0.72);
    border: 1px solid rgba(0, 255, 208, 0.25);
    border-radius: 16px;
    padding: 16px;
    box-shadow: inset 0 0 18px rgba(0, 255, 208, 0.08);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="lstm-hero">
    <div class="lstm-title">RNN – Live Emotion Detection</div>
    <div class="lstm-subtitle">Real-time facial emotion detection with webcam streaming.</div>
</div>
""",
    unsafe_allow_html=True,
)

EMOJI_MAP = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Neutral": "😐",
    "Surprise": "😲",
    "Fear": "😨",
}

MODEL_PATH = os.getenv("EMOTION_MODEL_PATH", "models/emotion_model.h5")
LABELS = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


@st.cache_resource
def _load_emotion_model():
    if tf is None:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def _get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def _predict_emotion(face_gray: np.ndarray, model) -> tuple:
    if model is None:
        return "Neutral", 0.5
    face = cv2.resize(face_gray, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))
    preds = model.predict(face, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    label = LABELS[best_idx] if best_idx < len(LABELS) else "Neutral"
    return label, float(preds[best_idx])


class EmotionVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = _load_emotion_model()
        self.face_cascade = _get_face_cascade()
        self.last_emotion = ("Neutral", 0.5)
        self.fps = 0.0
        self._last_tick = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            emotion_label, confidence = _predict_emotion(face_roi, self.model)
            self.last_emotion = (emotion_label, confidence)

            emoji = EMOJI_MAP.get(emotion_label, "")
            label = f"{emotion_label} {emoji} | Confidence {int(confidence * 100)}%"
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 200), 2)
            cv2.putText(img, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        now = time.time()
        dt = now - self._last_tick
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self._last_tick = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")


if _video_deps_error:
    st.warning("Live webcam dependencies are missing in this environment.")
    st.code(_video_deps_error)
else:
    model_loaded = _load_emotion_model() is not None
    if not model_loaded:
        st.warning("No emotion model found. Using Neutral fallback. Add models/emotion_model.h5 to enable predictions.")

    controls = st.columns([1, 1, 1])
    with controls[0]:
        start_cam = st.button("Start Camera")
    with controls[1]:
        stop_cam = st.button("Stop Camera")
    with controls[2]:
        st.markdown("<div class='neon-card'>Live Emotion Feed</div>", unsafe_allow_html=True)

    if "camera_running" not in st.session_state:
        st.session_state["camera_running"] = False
    if start_cam:
        st.session_state["camera_running"] = True
    if stop_cam:
        st.session_state["camera_running"] = False

    status_col, meter_col = st.columns([1, 1])
    if st.session_state["camera_running"]:
        webrtc_ctx = webrtc_streamer(
            key="emotion-cam",
            video_transformer_factory=EmotionVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

        processor = webrtc_ctx.video_transformer if webrtc_ctx else None
        label = "Neutral"
        conf = 0.0
        fps_val = 0.0
        if processor:
            label, conf = processor.last_emotion
            fps_val = processor.fps

        with status_col:
            emoji = EMOJI_MAP.get(label, "")
            st.markdown("#### Current Emotion")
            st.markdown(f"<div class='neon-card'>{label} {emoji}</div>", unsafe_allow_html=True)
            st.markdown(f"**FPS:** {fps_val:.1f}")

        with meter_col:
            st.markdown("#### Confidence")
            st.progress(int(conf * 100))
    else:
        st.info("Camera is stopped. Click Start Camera.")
