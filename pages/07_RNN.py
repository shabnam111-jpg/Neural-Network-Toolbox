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
    from fer import FER
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    _lstm_deps_error = None
except Exception as exc:
    VideoTransformerBase = None
    webrtc_streamer = None
    av = None
    FER = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None
    _lstm_deps_error = str(exc)


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
.typing {
    display: inline-block;
    padding-left: 6px;
    color: rgba(0, 255, 208, 0.8);
}
.typing span {
    animation: blink 1.4s infinite both;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0% { opacity: 0.2; }
    20% { opacity: 1; }
    100% { opacity: 0.2; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="lstm-hero">
    <div class="lstm-title">LSTM Sentiment + Emotion Studio</div>
    <div class="lstm-subtitle">Real-time text and camera analysis with multi-class sentiment, emotion mapping, and live metrics.</div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("Text models use pretrained classifiers for stronger baseline accuracy. You can replace them with a BiLSTM if you provide a dataset.")

with st.expander("Theory: LSTM and emotion modeling", expanded=True):
    st.markdown(
        "BiLSTMs capture context in both directions, while emotion classifiers map text or facial signals into affective states. "
        "This module combines a pretrained sentiment model, an emotion classifier, and a live facial emotion detector."
    )
    st.latex(
        r"\overrightarrow{h_t} = \text{LSTM}(x_t, \overrightarrow{h_{t-1}}), "
        r"\overleftarrow{h_t} = \text{LSTM}(x_t, \overleftarrow{h_{t+1}})"
    )

if _lstm_deps_error:
    st.error("Missing dependencies for LSTM sentiment/emotion module.")
    st.code(_lstm_deps_error)
    st.stop()

TEXT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
TEXT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

EMOJI_MAP = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Fear": "😨",
    "Surprise": "😮",
    "Love": "❤️",
    "Excited": "🤩",
    "Calm": "😌",
}


@st.cache_resource
def _load_text_pipelines():
    tokenizer_sent = AutoTokenizer.from_pretrained(TEXT_SENTIMENT_MODEL)
    model_sent = AutoModelForSequenceClassification.from_pretrained(TEXT_SENTIMENT_MODEL)
    sent_pipe = pipeline("text-classification", model=model_sent, tokenizer=tokenizer_sent, return_all_scores=True)

    tokenizer_emo = AutoTokenizer.from_pretrained(TEXT_EMOTION_MODEL)
    model_emo = AutoModelForSequenceClassification.from_pretrained(TEXT_EMOTION_MODEL)
    emo_pipe = pipeline("text-classification", model=model_emo, tokenizer=tokenizer_emo, return_all_scores=True)
    return sent_pipe, emo_pipe


def _normalize_scores(scores: list) -> dict:
    total = sum(item["score"] for item in scores) or 1.0
    return {item["label"].lower(): float(item["score"] / total) for item in scores}


def _map_sentiment(scores: dict) -> tuple:
    mapping = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
    best_label = max(scores, key=scores.get)
    return mapping.get(best_label, "Neutral"), scores[best_label]


def _map_emotion(scores: dict, sentiment_label: str) -> tuple:
    base_map = {
        "joy": "Happy",
        "love": "Love",
        "surprise": "Surprise",
        "anger": "Angry",
        "fear": "Fear",
        "sadness": "Sad",
    }
    best_label = max(scores, key=scores.get)
    mapped = base_map.get(best_label, "Calm")
    mapped_score = scores[best_label]

    if sentiment_label == "Neutral" and mapped in {"Happy", "Sad", "Angry", "Fear"}:
        mapped = "Calm"
        mapped_score = max(mapped_score, 0.3)

    if scores.get("joy", 0) > 0.45 and scores.get("surprise", 0) > 0.18:
        mapped = "Excited"
        mapped_score = max(scores.get("joy", 0), scores.get("surprise", 0))

    return mapped, mapped_score


def _emotion_distribution(scores: dict, sentiment_label: str) -> dict:
    dist = {label: 0.0 for label in EMOJI_MAP.keys()}
    base_map = {
        "joy": "Happy",
        "love": "Love",
        "surprise": "Surprise",
        "anger": "Angry",
        "fear": "Fear",
        "sadness": "Sad",
    }
    for label, score in scores.items():
        mapped = base_map.get(label)
        if mapped:
            dist[mapped] = max(dist[mapped], float(score))
    dist["Excited"] = min(1.0, (scores.get("joy", 0.0) + scores.get("surprise", 0.0)) / 2)
    if sentiment_label == "Neutral":
        dist["Calm"] = max(dist["Calm"], 0.45)
    else:
        dist["Calm"] = max(dist["Calm"], 0.15)
    return dist


def _predict_text(text: str) -> dict:
    sent_pipe, emo_pipe = _load_text_pipelines()
    sent_scores = _normalize_scores(sent_pipe(text)[0])
    emo_scores = _normalize_scores(emo_pipe(text)[0])
    sentiment_label, sentiment_score = _map_sentiment(sent_scores)
    emotion_label, emotion_score = _map_emotion(emo_scores, sentiment_label)
    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "emotion_label": emotion_label,
        "emotion_score": emotion_score,
        "sentiment_scores": sent_scores,
        "emotion_scores": emo_scores,
    }


def _confidence_percent(score: float) -> int:
    return int(round(score * 100))


class EmotionVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = FER(mtcnn=False)
        self.last_result = None
        self.last_frame = None
        self.fps = 0.0
        self._last_tick = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img.copy()

        results = self.detector.detect_emotions(img)
        if results:
            face = max(results, key=lambda r: (r["box"][2] * r["box"][3]))
            emotions = face["emotions"]
            self.last_result = emotions
            x, y, w, h = face["box"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 200), 2)

            emotion_label, emotion_score = _map_face_emotion(emotions)
            sentiment_label, sentiment_score = _map_face_sentiment(emotion_label, emotion_score)
            emoji = EMOJI_MAP.get(emotion_label, "")
            label = f"{emoji} {emotion_label} | {sentiment_label} | {_confidence_percent(sentiment_score)}%"
            cv2.putText(img, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        now = time.time()
        dt = now - self._last_tick
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self._last_tick = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def _map_face_emotion(emotions: dict) -> tuple:
    if not emotions:
        return "Calm", 0.0
    best_label = max(emotions, key=emotions.get)
    base_map = {
        "happy": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "fear": "Fear",
        "surprise": "Surprise",
        "neutral": "Calm",
        "disgust": "Angry",
    }
    mapped = base_map.get(best_label, "Calm")
    score = float(emotions.get(best_label, 0.0))
    if emotions.get("happy", 0) > 0.6 and emotions.get("surprise", 0) > 0.2:
        mapped = "Excited"
        score = max(float(emotions.get("happy", 0.0)), float(emotions.get("surprise", 0.0)))
    return mapped, score


def _map_face_sentiment(emotion_label: str, emotion_score: float) -> tuple:
    positive = {"Happy", "Excited", "Love"}
    negative = {"Angry", "Sad", "Fear"}
    if emotion_label in positive:
        return "Positive", emotion_score
    if emotion_label in negative:
        return "Negative", emotion_score
    return "Neutral", max(0.35, emotion_score)


if "text_history" not in st.session_state:
    st.session_state["text_history"] = []

if "camera_running" not in st.session_state:
    st.session_state["camera_running"] = False
if "camera_facing" not in st.session_state:
    st.session_state["camera_facing"] = "user"

tab_text, tab_cam = st.tabs(["Text / Chat Mode", "Live Webcam Mode"])

with tab_text:
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Text Analysis Panel")
        st.markdown(
            "<div class='neon-card'>Type a message and analyze sentiment + emotion.</div>",
            unsafe_allow_html=True,
        )
        chat_text = st.text_area("Chat input", height=120, placeholder="Type here...")
        st.markdown("Typing" + "<span class='typing'><span>.</span><span>.</span><span>.</span></span>", unsafe_allow_html=True)
        analyze = st.button("Analyze Text", type="primary")

        if analyze and chat_text.strip():
            result = _predict_text(chat_text)
            st.session_state["text_history"].append({
                "text": chat_text,
                "sentiment_label": result["sentiment_label"],
                "sentiment_score": result["sentiment_score"],
                "emotion_label": result["emotion_label"],
                "emotion_score": result["emotion_score"],
            })

    with right:
        st.markdown("#### Confidence Charts")
        if analyze and chat_text.strip():
            sentiment_scores = result["sentiment_scores"]
            emotion_scores = result["emotion_scores"]
            sent_df = {
                "label": ["Positive", "Neutral", "Negative"],
                "score": [
                    sentiment_scores.get("positive", 0.0),
                    sentiment_scores.get("neutral", 0.0),
                    sentiment_scores.get("negative", 0.0),
                ],
            }
            emotion_dist = _emotion_distribution(emotion_scores, result["sentiment_label"])
            emo_df = {
                "label": list(emotion_dist.keys()),
                "score": list(emotion_dist.values()),
            }
            sent_plot = px.bar(sent_df, x="label", y="score", range_y=[0, 1])
            emo_plot = px.bar(emo_df, x="label", y="score", range_y=[0, 1])
            st.plotly_chart(sent_plot, use_container_width=True)
            st.plotly_chart(emo_plot, use_container_width=True)
        else:
            st.info("Run a text analysis to render confidence charts.")

    st.markdown("#### Prediction History")
    if st.session_state["text_history"]:
        for item in reversed(st.session_state["text_history"][-8:]):
            sent_pct = _confidence_percent(item["sentiment_score"])
            emo_pct = _confidence_percent(item["emotion_score"])
            emoji = EMOJI_MAP.get(item["emotion_label"], "")
            st.markdown(
                f"**{item['sentiment_label']}** ({sent_pct}%) | "
                f"**{item['emotion_label']}** {emoji} ({emo_pct}%) — {item['text']}"
            )
    else:
        st.info("No predictions yet.")

with tab_cam:
    st.markdown("#### Live Camera Panel")
    cam_controls = st.columns([1, 1, 1, 1])
    with cam_controls[0]:
        if st.button("Start Camera"):
            st.session_state["camera_running"] = True
    with cam_controls[1]:
        if st.button("Stop Camera"):
            st.session_state["camera_running"] = False
    with cam_controls[2]:
        if st.button("Switch Camera"):
            st.session_state["camera_facing"] = (
                "environment" if st.session_state["camera_facing"] == "user" else "user"
            )
        st.caption(f"Camera: {st.session_state['camera_facing']}")
    with cam_controls[3]:
        snapshot = st.button("Snapshot")

    if st.session_state["camera_running"]:
        webrtc_ctx = webrtc_streamer(
            key="emotion-cam",
            video_transformer_factory=EmotionVideoProcessor,
            media_stream_constraints={"video": {"facingMode": st.session_state["camera_facing"]}, "audio": False},
        )
        processor = webrtc_ctx.video_transformer if webrtc_ctx else None
        fps_val = getattr(processor, "fps", 0.0) if processor else 0.0
        st.markdown(f"**FPS:** {fps_val:.1f}")

        if snapshot and processor and processor.last_frame is not None:
            snapshot_rgb = cv2.cvtColor(processor.last_frame, cv2.COLOR_BGR2RGB)
            st.image(snapshot_rgb, caption="Snapshot", use_column_width=True)
    else:
        st.info("Camera is stopped.")

st.markdown("#### Accuracy Metrics")
if st.session_state["text_history"]:
    sent_avg = float(np.mean([item["sentiment_score"] for item in st.session_state["text_history"]]))
    emo_avg = float(np.mean([item["emotion_score"] for item in st.session_state["text_history"]]))
else:
    sent_avg = 0.0
    emo_avg = 0.0

st.write(
    {
        "text_model": TEXT_SENTIMENT_MODEL,
        "emotion_model": TEXT_EMOTION_MODEL,
        "session_predictions": len(st.session_state["text_history"]),
        "avg_sentiment_confidence": _confidence_percent(sent_avg),
        "avg_emotion_confidence": _confidence_percent(emo_avg),
        "note": "Metrics shown here are session-level confidence summaries, not ground-truth accuracy.",
    }
)

if st.session_state["text_history"]:
    export_rows = []
    for item in st.session_state["text_history"]:
        export_rows.append(
            f"{item['text']},{item['sentiment_label']},{_confidence_percent(item['sentiment_score'])},"
            f"{item['emotion_label']},{_confidence_percent(item['emotion_score'])}"
        )
    export_csv = "text,sentiment,sentiment_confidence,emotion,emotion_confidence\n" + "\n".join(export_rows)
    st.download_button("Export Results", export_csv, file_name="lstm_sentiment_history.csv")
