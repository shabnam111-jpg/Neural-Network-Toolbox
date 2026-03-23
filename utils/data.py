import io
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner=False)
def load_iris() -> Tuple[pd.DataFrame, pd.Series]:
    iris = datasets.load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y


@st.cache_data(show_spinner=False)
def load_moons(n_samples: int, noise: float):
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y


@st.cache_data(show_spinner=False)
def load_circles(n_samples: int, noise: float, factor: float):
    X, y = datasets.make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=42
    )
    return X, y


@st.cache_data(show_spinner=False)
def standardize(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


@st.cache_data(show_spinner=False)
def csv_to_dataframe(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))
