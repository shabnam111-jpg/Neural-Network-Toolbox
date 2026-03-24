import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st

from utils.export import download_code_snippet, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme


st.set_page_config(page_title="Backpropagation", layout="wide")
apply_theme()
render_sidebar("Backpropagation")

st.title("Backpropagation – Gradient Visualizer")

with st.expander("Theory: chain rule", expanded=True):
    st.markdown("Backprop uses the chain rule to compute gradients efficiently.")
    st.latex(r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}")

with st.expander("Why this happens?"):
    st.markdown("Backprop reuses intermediate derivatives, avoiding repeated computations.")

st.markdown("---")

col1, col2 = st.columns([1, 2])
with col1:
    x = st.number_input("Input x", value=0.8)
    w = st.number_input("Weight w", value=0.5)
    b = st.number_input("Bias b", value=0.1)
    y = st.number_input("Target y", value=1.0)
    activation = st.selectbox("Activation", ["Sigmoid", "Tanh", "ReLU"], help="Changes derivative behavior")

with col2:
    z = w * x + b
    if activation == "Sigmoid":
        a = 1 / (1 + np.exp(-z))
        da_dz = a * (1 - a)
    elif activation == "Tanh":
        a = np.tanh(z)
        da_dz = 1 - a ** 2
    else:
        a = max(0, z)
        da_dz = 1.0 if z > 0 else 0.0

    loss = 0.5 * (a - y) ** 2
    dL_da = a - y
    dL_dz = dL_da * da_dz
    dL_dw = dL_dz * x
    dL_db = dL_dz

    st.markdown("#### Forward pass")
    st.write({"z": float(z), "a": float(a), "loss": float(loss)})
    st.markdown("#### Step-by-step gradients")
    st.write({
        "dL/da": float(dL_da),
        "da/dz": float(da_dz),
        "dL/dz": float(dL_dz),
        "dL/dw": float(dL_dw),
        "dL/db": float(dL_db),
    })
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["loss"], [loss], color="#0b6b77")
    ax.set_title("Loss snapshot (Matplotlib)")
    st.pyplot(fig)

    if st.button("Compute Gradients"):
        st.markdown("#### Chain-rule breakdown")
        st.write({
            "dL/da": float(dL_da),
            "da/dz": float(da_dz),
            "dL/dz": float(dL_dz),
            "dL/dw": float(dL_dw),
            "dL/db": float(dL_db),
        })
        grad_fig = px.bar(x=["dL_dw", "dL_db"], y=[dL_dw, dL_db], title="Gradient magnitudes")
        st.plotly_chart(grad_fig, use_container_width=True)

st.markdown("---")

code = """
import numpy as np

x, w, b, y = 0.8, 0.5, 0.1, 1.0
z = w * x + b

a = 1 / (1 + np.exp(-z))
loss = 0.5 * (a - y) ** 2

dL_da = a - y
da_dz = a * (1 - a)

dL_dw = dL_da * da_dz * x
"""

download_code_snippet("Export Python Code", code.strip(), "backprop.py")

download_pickle(
    "Download gradient state",
    {
        "x": x,
        "w": w,
        "b": b,
        "y": y,
        "z": z,
        "a": a,
        "dL_dw": dL_dw,
        "dL_db": dL_db,
    },
    "backprop_state.pkl",
)
