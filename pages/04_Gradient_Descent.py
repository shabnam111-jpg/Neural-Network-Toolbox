import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.export import download_code_snippet, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme, theme_toggle
from utils.viz import plot_contour_path, plot_3d_surface


st.set_page_config(page_title="Gradient Descent", layout="wide")
apply_theme()
render_sidebar("Gradient Descent")
theme_toggle()

st.title("Gradient Descent – Optimizer Playground")

with st.expander("Theory: optimization", expanded=True):
    st.markdown("We minimize a loss surface by moving opposite to the gradient.")
    st.latex(r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)")

with st.expander("Why this happens?"):
    st.markdown("Momentum and Adam smooth noisy gradients, helping faster convergence.")

col1, col2 = st.columns([1, 2])
with col1:
    optimizer = st.selectbox("Optimizer", ["GD", "SGD", "Momentum", "Adam"])
    lr = st.slider("Learning rate", 0.001, 0.5, 0.05, 0.001, help="Too high can diverge")
    steps = st.slider("Steps", 10, 120, 40, 5)
    init_x = st.slider("Init x", -3.0, 3.0, 2.0, 0.1)
    init_y = st.slider("Init y", -3.0, 3.0, -2.0, 0.1)

with col2:
    st.plotly_chart(plot_3d_surface(), use_container_width=True)

if st.button("Animate optimization"):
    x, y = init_x, init_y
    vx, vy = 0.0, 0.0
    xs, ys = [x], [y]
    beta = 0.9
    m_x, m_y, v_x, v_y = 0.0, 0.0, 0.0, 0.0

    for t in range(1, steps + 1):
        grad_x, grad_y = 2 * x, 2 * y
        if optimizer == "GD":
            x -= lr * grad_x
            y -= lr * grad_y
        elif optimizer == "SGD":
            noise = np.random.normal(scale=0.1, size=2)
            x -= lr * (grad_x + noise[0])
            y -= lr * (grad_y + noise[1])
        elif optimizer == "Momentum":
            vx = beta * vx + (1 - beta) * grad_x
            vy = beta * vy + (1 - beta) * grad_y
            x -= lr * vx
            y -= lr * vy
        else:
            m_x = 0.9 * m_x + 0.1 * grad_x
            m_y = 0.9 * m_y + 0.1 * grad_y
            v_x = 0.999 * v_x + 0.001 * (grad_x ** 2)
            v_y = 0.999 * v_y + 0.001 * (grad_y ** 2)
            m_x_hat = m_x / (1 - 0.9 ** t)
            m_y_hat = m_y / (1 - 0.9 ** t)
            v_x_hat = v_x / (1 - 0.999 ** t)
            v_y_hat = v_y / (1 - 0.999 ** t)
            x -= lr * m_x_hat / (np.sqrt(v_x_hat) + 1e-8)
            y -= lr * m_y_hat / (np.sqrt(v_y_hat) + 1e-8)

        xs.append(x)
        ys.append(y)
        time.sleep(0.02)

    st.plotly_chart(plot_contour_path(xs, ys), use_container_width=True)
    losses = [x ** 2 + y ** 2 for x, y in zip(xs, ys)]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(losses, color="#0b6b77")
    ax.set_title("Loss vs step (Matplotlib)")
    st.pyplot(fig)
    download_pickle("Download optimizer path", {"x": xs, "y": ys}, "optimizer_path.pkl")

code = """
import numpy as np

x, y = 2.0, -2.0
lr = 0.05

for _ in range(40):
    grad_x, grad_y = 2 * x, 2 * y
    x -= lr * grad_x
    y -= lr * grad_y
"""

download_code_snippet("Export Python Code", code.strip(), "gradient_descent.py")
