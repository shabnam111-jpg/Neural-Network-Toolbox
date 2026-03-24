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
    filters = st.slider("Filters", 4, 32, 8, 2)
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

    # 🔁 Training loop
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

    # 🖼️ Feature map visualization
    sample, _ = train_ds[0]

    with torch.no_grad():
        feats = model[0](sample.unsqueeze(0))  # first conv layer output

    st.image(sample.squeeze(0).numpy(), caption="Input sample", width=120)

    # ✅ SAFE conversion + normalization
    fmaps = [f.detach().cpu().numpy() for f in feats[0, :min(filters, 8)]]

    fmaps_normalized = []
    for fmap in fmaps:
        fmap = fmap - np.min(fmap)
        max_val = np.max(fmap)

        if max_val != 0:
            fmap = fmap / max_val
        else:
            fmap = np.zeros_like(fmap)

        fmaps_normalized.append(fmap)

    # ✅ No more crash here
    st.image(
        fmaps_normalized,
        caption=[f"Map {i+1}" for i in range(len(fmaps_normalized))],
    )

    if fmaps_normalized:
        f0 = fmaps[0]
        f0_norm = fmaps_normalized[0]
        i0, j0 = f0.shape[0] // 2, f0.shape[1] // 2
        st.markdown("#### Step-by-step normalization (Map 1)")
        st.write({
            "min": float(np.min(f0)),
            "max": float(np.max(f0)),
            "sample_value": float(f0[i0, j0]),
            "normalized_sample": float(f0_norm[i0, j0]),
        })

    # 🎨 Filter visualization
    weights = model[0].weight[0, 0].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(weights, cmap="viridis")
    ax.set_title("First filter (Matplotlib)")
    ax.axis("off")
    st.pyplot(fig)

    # 📊 Prediction distribution
    batch, _ = next(iter(loader))
    with torch.no_grad():
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu().numpy()

    hist = px.histogram(preds, nbins=10, title="Prediction distribution")
    st.plotly_chart(hist, use_container_width=True)


# 📦 Export code
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