import streamlit as st

from utils.nav import render_sidebar
from utils.theme import apply_theme


st.set_page_config(page_title="OpenCV + Vision", layout="wide")
apply_theme()
render_sidebar("OpenCV + Vision")

st.title("OpenCV + Vision – Theory")

st.markdown(
    "OpenCV (Open Source Computer Vision Library) provides building blocks for image and video processing. "
    "Core ideas include image preprocessing (grayscale, blur, thresholding), feature extraction (edges, contours), "
    "and traditional detection pipelines (such as face detection with Haar cascades)."
)

with st.expander("Theory: preprocessing", expanded=True):
    st.markdown("Preprocessing boosts CNN performance and reduces noise before learning.")
    st.latex(r"I_{gray} = 0.299R + 0.587G + 0.114B")

with st.expander("Theory: edges and contours"):
    st.markdown(
        "Edge detection highlights intensity changes, while contours connect edge pixels into shapes. "
        "These steps help isolate objects and simplify downstream analysis."
    )

with st.expander("Theory: face detection in OpenCV"):
    st.markdown(
        "Yes, face detection is available in OpenCV. A classic approach uses Haar cascades to scan an image at "
        "multiple scales and detect face-like patterns. Modern pipelines often use deep learning models, but the "
        "Haar cascade method remains fast and easy to deploy."
    )
