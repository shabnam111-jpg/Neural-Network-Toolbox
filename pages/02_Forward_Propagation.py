import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.export import download_code_snippet, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme
from utils.viz import plot_activation_curve


st.set_page_config(page_title="Forward Propagation", layout="wide")
apply_theme()
render_sidebar("Forward Propagation")

st.title("Forward Propagation – Step-by-step Calculator")

with st.expander("Theory: forward pass", expanded=True):
    st.markdown("A forward pass is a sequence of linear maps and activations.")
    st.latex(r"z = W x + b")
    st.latex(r"a = \sigma(z)")

with st.expander("Why this happens?"):
    st.markdown("Activations squash linear outputs, letting networks model nonlinear patterns.")

col1, col2 = st.columns([1, 2])
with col1:
    activation = st.selectbox("Activation", ["Sigmoid", "ReLU", "Tanh", "Softmax"], help="Controls how z is squashed")
    x1 = st.number_input("Input x1", value=1.0)
    x2 = st.number_input("Input x2", value=-0.5)
    w11 = st.number_input("W11", value=0.5)
    w12 = st.number_input("W12", value=-0.4)
    w21 = st.number_input("W21", value=0.3)
    w22 = st.number_input("W22", value=0.2)
    b1 = st.number_input("b1", value=0.1)
    b2 = st.number_input("b2", value=-0.1)

with col2:
    x = np.array([x1, x2])
    W = np.array([[w11, w12], [w21, w22]])
    b = np.array([b1, b2])
    z = W @ x + b

    if activation == "Sigmoid":
        a = 1 / (1 + np.exp(-z))
    elif activation == "ReLU":
        a = np.maximum(0, z)
    elif activation == "Tanh":
        a = np.tanh(z)
    else:
        exp = np.exp(z - np.max(z))
        a = exp / exp.sum()

    st.markdown("#### Linear step")
    st.code(f"z = W x + b = {z}")
    st.markdown("#### Activation output")
    st.code(f"a = {a}")

    st.plotly_chart(plot_activation_curve(activation), use_container_width=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, a[0]], marker="o")
    ax.set_title("Activation snapshot (Matplotlib)")
    st.pyplot(fig)

st.markdown("---")

st.subheader("Neuron firing animation")
if st.button("Simulate forward pass"):
    st.progress(25)
    st.progress(55)
    st.progress(90)
    st.success("Activation computed")

code = """
import numpy as np

x = np.array([1.0, -0.5])
W = np.array([[0.5, -0.4], [0.3, 0.2]])
b = np.array([0.1, -0.1])

z = W @ x + b

a = 1 / (1 + np.exp(-z))
print(a)
"""

download_code_snippet("Export Python Code", code.strip(), "forward_pass.py")

download_pickle("Download forward-pass state", {"x": x, "W": W, "b": b, "z": z, "a": a}, "forward_state.pkl")
