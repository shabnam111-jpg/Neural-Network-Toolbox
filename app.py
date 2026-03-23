import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st

from utils.export import download_code_snippet, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme


st.set_page_config(page_title="Neural Lab", layout="wide")
apply_theme()
render_sidebar("Home")

st.title("Neural Lab – Interactive Neural Network Toolbox")

st.markdown(
    """
<div class="neural-gradient">
  <span class="neural-pill">Play and Learn</span>
  <h3>Welcome to a modern neural network playground.</h3>
  <p>Build, tweak, and visualize neural networks from scratch with step-by-step math,
  animated decision boundaries, and real datasets. Every page mixes theory, intuition,
  and hands-on experiments.</p>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="neural-card">
    <h4>What you will learn</h4>
    <ul>
      <li>Neurons, activations, and decision boundaries</li>
      <li>Forward and backward passes with math</li>
      <li>Optimization and training dynamics</li>
    </ul>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="neural-card">
    <h4>How to use this lab</h4>
    <ul>
      <li>Pick a section in the sidebar</li>
      <li>Slide hyperparameters</li>
      <li>Train and watch the visuals update</li>
    </ul>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="neural-card">
    <h4>Quick start</h4>
    <ul>
      <li>Start with Perceptron</li>
      <li>Move to ANN and CNN</li>
      <li>Try OpenCV + Vision</li>
    </ul>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("Animated intro")
progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i + 1)

st.markdown(
    """
### What's inside
- Multi-page guided labs with theory, sliders, and experiments
- Real-time plots with Matplotlib and Plotly
- CSV and image upload with instant previews
- Export-ready code snippets and model downloads
"""
)


st.subheader("Quick demo")
if st.button("Train a mini neuron"):
  w = np.array([0.2, -0.4])
  b = 0.1
  bar = st.progress(0)
  for i in range(20):
    bar.progress((i + 1) * 5)
    time.sleep(0.02)
  st.success("Mini demo complete")
  download_pickle("Download demo state", {"w": w, "b": b}, "demo_neuron.pkl")

demo_code = """
import numpy as np

w = np.array([0.2, -0.4])
b = 0.1
"""
download_code_snippet("Export Python Code", demo_code.strip(), "demo_neuron.py")

fig, ax = plt.subplots(figsize=(4, 2))
ax.plot([0, 1, 2], [0.2, 0.1, 0.05], color="#0b6b77")
ax.set_title("Mini loss curve (Matplotlib)")
st.pyplot(fig)

scatter = px.scatter(x=[0, 1, 2], y=[0.2, 0.1, 0.05], title="Mini loss curve (Plotly)")
st.plotly_chart(scatter, use_container_width=True)

np.random.seed(42)
