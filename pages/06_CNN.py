import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from utils.export import download_code_snippet, download_torch_state
from utils.nav import render_sidebar
from utils.theme import apply_theme


st.set_page_config(page_title="CNN", layout="wide")
apply_theme()
render_sidebar("CNN")

st.title("CNN – Convolutional Neural Network Explorer")

with st.expander("Theory: convolutions", expanded=True):
    st.markdown("Convolutions learn local patterns with shared filters.")
    st.latex(r"(f * x)(i, j) = \sum_{m,n} f_{m,n} x_{i-m, j-n}")

with st.expander("Why this happens?"):
    st.markdown("Shared filters let CNNs detect edges and textures across the image.")

col1, col2 = st.columns([1, 2])
with col1:
    dataset_name = st.selectbox("Dataset", ["MNIST", "Fashion-MNIST"])
    epochs = st.slider("Epochs", 1, 5, 2, 1)
    lr = st.slider("Learning rate", 0.0005, 0.01, 0.001, 0.0005)
    filters = st.slider("Filters", 4, 32, 8, 2, help="Number of convolution kernels")
    upload = st.file_uploader("Upload image (28x28 grayscale)", type=["png", "jpg", "jpeg"])

with col2:
    st.markdown("#### Feature map preview")
    st.info("Train to visualize feature maps after first conv layer.")
    if "vision_image" in st.session_state:
        st.image(st.session_state["vision_image"], caption="From OpenCV tab", width=160)

if st.button("Train CNN"):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "MNIST":
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    else:
        train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)

    loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    model = nn.Sequential(
        nn.Conv2d(1, filters, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(filters * 14 * 14, 10),
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    bar = st.progress(0)
    for epoch in range(epochs):
        for i, (x, y) in enumerate(loader):
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            if i % 50 == 0:
                bar.progress(min(100, int((epoch + 1) / epochs * 100)))
        time.sleep(0.05)

    st.success("Training complete")
    download_torch_state("Download CNN model", model.state_dict(), "cnn_model.pt")

    sample, _ = train_ds[0]
    with torch.no_grad():
        feats = model[0](sample.unsqueeze(0))

    st.image(sample.squeeze(0).numpy(), caption="Input sample", width=120)
    fmaps = [f.numpy() for f in feats[0, :min(filters, 8)]]
    fmaps_normalized = []
    for fmap in fmaps:
        fmap = fmap - np.min(fmap)
        fmap = fmap / np.max(fmap)
        fmaps_normalized.append(fmap)
    st.image(
        fmaps_normalized,
        caption=[f"Map {i+1}" for i in range(len(fmaps_normalized))],
    )

    weights = model[0].weight[0, 0].detach().numpy()
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(weights, cmap="viridis")
    ax.set_title("First filter (Matplotlib)")
    st.pyplot(fig)

    batch, _ = next(iter(loader))
    with torch.no_grad():
        logits = model(batch)
        preds = logits.argmax(dim=1).numpy()
    hist = px.histogram(preds, nbins=10, title="Prediction distribution")
    st.plotly_chart(hist, use_container_width=True)

code = """
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(8 * 14 * 14, 10),
)
"""

download_code_snippet("Export Python Code", code.strip(), "cnn_model.py")
