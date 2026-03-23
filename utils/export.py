import io
import pickle
from typing import Dict

import streamlit as st
import torch


def download_code_snippet(label: str, code: str, filename: str) -> None:
    st.download_button(
        label,
        data=code.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
    )


def download_pickle(label: str, obj: object, filename: str) -> None:
    payload = pickle.dumps(obj)
    st.download_button(
        label,
        data=payload,
        file_name=filename,
        mime="application/octet-stream",
    )


def download_torch_state(label: str, state: Dict, filename: str) -> None:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    st.download_button(
        label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/octet-stream",
    )
