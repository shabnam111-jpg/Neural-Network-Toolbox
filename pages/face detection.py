import datetime
import time
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from utils.theme import apply_theme


st.set_page_config(page_title="Face Detection", layout="wide")
apply_theme()

try:
    import cv2

    _cv2_error = None
except Exception as exc:
    cv2 = None
    _cv2_error = str(exc)


st.title("Face Detection – Match and Timestamp")

if _cv2_error:
    st.error("OpenCV failed to import in this environment.")
    st.code(_cv2_error)
    st.stop()


def _load_image(upload) -> np.ndarray:
    image = Image.open(upload).convert("RGB")
    return np.array(image)


def _get_cascade() -> "cv2.CascadeClassifier":
    if "face_cascade" not in st.session_state:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        st.session_state["face_cascade"] = cv2.CascadeClassifier(cascade_path)
    return st.session_state["face_cascade"]


def _detect_faces(gray: np.ndarray, scale: float, neighbors: int, min_size: int) -> List[Tuple[int, int, int, int]]:
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=neighbors,
        minSize=(min_size, min_size),
    )
    return list(faces) if len(faces) else []


def _largest_face(faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not faces:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def _face_embedding(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA)
    vec = gray.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _annotate_frame(
    frame_bgr: np.ndarray,
    scale: float,
    neighbors: int,
    min_size: int,
    ref_embeddings: List[np.ndarray],
    ref_labels: List[str],
    threshold: float,
) -> Tuple[np.ndarray, bool, List[str], int]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces(gray, scale, neighbors, min_size)

    display = frame_bgr.copy()
    matched_any = False
    match_lines = []

    for (x, y, w, h) in faces:
        face_crop = frame_bgr[y : y + h, x : x + w]
        emb = _face_embedding(face_crop)
        best_label = "Unknown"
        best_score = 0.0

        if ref_embeddings:
            scores = [_cosine_similarity(emb, ref) for ref in ref_embeddings]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score >= threshold:
                best_label = ref_labels[best_idx]

        match_percent = int(round(best_score * 100))
        color = (35, 179, 88) if best_label != "Unknown" else (255, 99, 71)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            display,
            f"{best_label} ({match_percent}%)",
            (x, max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        if best_label != "Unknown":
            matched_any = True
            match_lines.append(f"Match: {best_label} {match_percent}%")

    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB), matched_any, match_lines, len(faces)


st.markdown(
    "Upload reference images (known faces), then capture a frame. "
    "If a face matches, you will see \"Face detected\" with the time."
)

col1, col2 = st.columns([1, 2])

with col1:
    ref_uploads = st.file_uploader(
        "Reference face images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    scale = st.slider("Detection scale", 1.05, 1.5, 1.1, 0.05)
    neighbors = st.slider("Min neighbors", 3, 10, 5, 1)
    min_size = st.slider("Min face size", 30, 180, 60, 10)
    threshold = st.slider("Match threshold", 0.4, 0.95, 0.72, 0.01)

with col2:
    if not ref_uploads:
        st.info("Add reference images to enable matching.")

    ref_embeddings = []
    ref_labels = []

    if ref_uploads:
        st.markdown("#### Reference faces")
        ref_cols = st.columns(min(4, len(ref_uploads)))
        for idx, upload in enumerate(ref_uploads):
            img_rgb = _load_image(upload)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = _detect_faces(gray, scale, neighbors, min_size)
            face_box = _largest_face(faces)
            if face_box is None:
                continue
            x, y, w, h = face_box
            face_crop = img_bgr[y : y + h, x : x + w]
            emb = _face_embedding(face_crop)
            ref_embeddings.append(emb)
            ref_labels.append(upload.name)
            with ref_cols[idx % len(ref_cols)]:
                st.image(img_rgb, caption=upload.name, use_column_width=True)

    st.markdown("#### Live webcam")
    st.caption("Capture a frame to run matching and timestamp.")
    cam_upload = st.camera_input("Capture frame")
    if cam_upload:
        frame_rgb = _load_image(cam_upload)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        display, matched_any, match_lines, face_count = _annotate_frame(
            frame_bgr,
            scale,
            neighbors,
            min_size,
            ref_embeddings,
            ref_labels,
            threshold,
        )
        st.image(display, caption="Live detection", use_column_width=True)
        if face_count == 0:
            st.warning("No faces detected in the captured frame.")
        elif matched_any:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"Face detected at {now}.")
            st.write("\n".join(match_lines))
        else:
            st.info("Faces detected, but no matches above the threshold.")
    else:
        st.info("Capture a frame to run face detection and matching.")

