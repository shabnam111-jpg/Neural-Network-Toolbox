from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_decision_boundary(X, y, w, b):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.8)
    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_vals = np.linspace(xmin, xmax, 100)
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
        ax.plot(x_vals, y_vals, color="black")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Decision Boundary")
    return fig


def plot_activation_curve(name: str):
    x = np.linspace(-6, 6, 200)
    if name == "Sigmoid":
        y = 1 / (1 + np.exp(-x))
    elif name == "ReLU":
        y = np.maximum(0, x)
    elif name == "Tanh":
        y = np.tanh(x)
    else:
        exp = np.exp(x - np.max(x))
        y = exp / exp.sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def plot_loss_curve(losses: List[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, mode="lines+markers"))
    fig.update_layout(title="Loss Curve", height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_contour_path(xs: List[float], ys: List[float]) -> go.Figure:
    X, Y = np.meshgrid(np.linspace(-4, 4, 120), np.linspace(-4, 4, 120))
    Z = X ** 2 + Y ** 2
    fig = go.Figure(data=go.Contour(z=Z, x=np.linspace(-4, 4, 120), y=np.linspace(-4, 4, 120)))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#f97316")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_3d_surface() -> go.Figure:
    X, Y = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
    Z = X ** 2 + Y ** 2
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    return fig
