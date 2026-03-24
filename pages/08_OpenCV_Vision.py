import io

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image

from utils.export import download_code_snippet, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme


st.set_page_config(page_title="OpenCV + Vision", layout="wide")
apply_theme()
render_sidebar("OpenCV + Vision")

try:
    import cv2
    _cv2_error = None
except Exception as exc:
    cv2 = None
    _cv2_error = str(exc)

st.title("OpenCV + Vision – Image Processing Playground")

if _cv2_error:
    st.error("OpenCV failed to import in this environment.")
    st.code(_cv2_error)
    st.stop()

with st.expander("Theory: preprocessing", expanded=True):
    st.markdown("Preprocessing boosts CNN performance and reduces noise.")
    st.latex(r"I_{gray} = 0.299R + 0.587G + 0.114B")

with st.expander("Why this happens?"):
    st.markdown("Filtering and thresholding simplify images so CNNs learn clearer patterns.")

col1, col2 = st.columns([1, 2])
with col1:
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    tool = st.selectbox(
        "Tool",
        ["Grayscale", "Canny Edge", "Gaussian Blur", "Threshold", "Contours"],
    )
    blur = st.slider("Blur kernel", 3, 15, 5, 2)
    thresh = st.slider("Threshold", 0, 255, 120, 5, help="Higher values keep fewer pixels")

with col2:
    if upload:
        image = Image.open(upload).convert("RGB")
        img = np.array(image)
        display = img.copy()

        if tool == "Grayscale":
            display = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif tool == "Canny Edge":
            display = cv2.Canny(img, 80, 160)
        elif tool == "Gaussian Blur":
            display = cv2.GaussianBlur(img, (blur, blur), 0)
        elif tool == "Threshold":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, display = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            display = img.copy()
            cv2.drawContours(display, contours, -1, (255, 0, 0), 2)

        st.image(display, caption="Processed output", use_column_width=True)

        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        center_rgb = img[cy, cx].tolist()
        info = {
            "tool": tool,
            "center_pixel_rgb": center_rgb,
        }
        if tool == "Grayscale":
            r, g, b = center_rgb[0], center_rgb[1], center_rgb[2]
            gray_val = 0.299 * r + 0.587 * g + 0.114 * b
            info["gray_value"] = float(gray_val)
        elif tool == "Threshold":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_val = int(gray[cy, cx])
            info["gray_value"] = gray_val
            info["threshold"] = int(thresh)
            info["thresholded_pixel"] = int(display[cy, cx])
        else:
            if display.ndim == 2:
                info["processed_center_pixel"] = int(display[cy, cx])
            else:
                info["processed_center_pixel"] = display[cy, cx].tolist()

        st.markdown("#### Step-by-step calculation")
        st.write(info)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(display if display.ndim == 2 else cv2.cvtColor(display, cv2.COLOR_RGB2GRAY), cmap="gray")
        ax.set_title("Matplotlib preview")
        st.pyplot(fig)

        hist = px.histogram(display.reshape(-1), nbins=40, title="Pixel intensity histogram")
        st.plotly_chart(hist, use_container_width=True)

        if st.button("Feed to CNN"):
            st.session_state["vision_image"] = display
            st.success("Image queued for CNN tab")

        download_pickle(
            "Download preprocessing config",
            {"tool": tool, "blur": blur, "threshold": thresh},
            "opencv_config.pkl",
        )
    else:
        st.info("Upload an image to start")

code = """
import cv2

img = cv2.imread("image.jpg")
edge = cv2.Canny(img, 80, 160)
"""

download_code_snippet("Export Python Code", code.strip(), "opencv_tools.py")
