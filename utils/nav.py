import streamlit as st
from streamlit_option_menu import option_menu

PAGES = [
    ("Home", "house"),
    ("Perceptron", "activity"),
    ("Forward Propagation", "diagram-3"),
    ("Backpropagation", "arrow-repeat"),
    ("Gradient Descent", "graph-up"),
    ("ANN (MLP)", "layers"),
    ("CNN", "grid-3x3"),
    ("RNN", "arrow-left-right"),
    ("OpenCV + Vision", "camera"),
]

PAGE_PATHS = {
    "Home": "app.py",
    "Perceptron": "pages/01_Perceptron.py",
    "Forward Propagation": "pages/02_Forward_Propagation.py",
    "Backpropagation": "pages/03_Backpropagation.py",
    "Gradient Descent": "pages/04_Gradient_Descent.py",
    "ANN (MLP)": "pages/05_ANN_MLP.py",
    "CNN": "pages/06_CNN.py",
    "RNN": "pages/07_RNN.py",
    "OpenCV + Vision": "pages/08_OpenCV_Vision.py",
}


def render_sidebar(current_page: str) -> None:
    with st.sidebar:
        st.markdown("## NeuralViz Lab")
        selection = option_menu(
            "Navigation",
            [name for name, _ in PAGES],
            icons=[icon for _, icon in PAGES],
            menu_icon="map",
            default_index=[name for name, _ in PAGES].index(current_page),
            styles={
                "container": {"padding": "0.4rem"},
                "icon": {"font-size": "16px"},
                "nav-link": {"font-size": "15px", "margin": "2px"},
                "nav-link-selected": {"font-weight": "700"},
            },
        )

    if selection != current_page:
        st.switch_page(PAGE_PATHS[selection])
