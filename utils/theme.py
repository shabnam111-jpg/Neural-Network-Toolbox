import streamlit as st

LIGHT_CSS = """
:root {
  --bg: #f8f7f4;
  --panel: #ffffff;
  --text: #1b1b1f;
  --muted: #5b5b6b;
  --accent: #0b6b77;
  --accent-2: #f97316;
  --card: #ffffff;
}
"""

DARK_CSS = """
:root {
  --bg: #0e1116;
  --panel: #131820;
  --text: #f4f4f8;
  --muted: #b8c1d1;
  --accent: #24c1d9;
  --accent-2: #f59e0b;
  --card: #151b24;
}
"""

BASE_CSS = """
body {
  background: var(--bg);
  color: var(--text);
}
section[data-testid="stSidebar"] {
  background: var(--panel);
}
.neural-card {
  background: var(--card);
  border: 1px solid rgba(120, 130, 140, 0.2);
  padding: 1.2rem;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}
.neural-pill {
  display: inline-block;
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  background: rgba(36, 193, 217, 0.15);
  color: var(--accent);
  font-weight: 600;
  font-size: 0.8rem;
}
.neural-gradient {
  background: radial-gradient(circle at top left, rgba(36,193,217,0.2), transparent 55%),
              radial-gradient(circle at top right, rgba(249,115,22,0.25), transparent 45%),
              var(--panel);
  border-radius: 20px;
  padding: 1.4rem;
}
"""


def apply_theme() -> None:
    theme = st.session_state.get("theme", "light")
    css = LIGHT_CSS if theme == "light" else DARK_CSS
    st.markdown(f"<style>{css}{BASE_CSS}</style>", unsafe_allow_html=True)


def theme_toggle() -> None:
    with st.sidebar:
        st.markdown("---")
        choice = st.radio("Theme", ["light", "dark"],
                          index=0 if st.session_state.get("theme", "light") == "light" else 1,
                          horizontal=True)
    st.session_state["theme"] = choice
