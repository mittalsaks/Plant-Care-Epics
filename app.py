import streamlit as st

from pages import dashboard, severity
from utils.theme import inject_global_theme
from utils.translator import t

st.set_page_config(
    page_title="PlantCare AI – Crop Disease Checker",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_global_theme()

lang = "en"

page = st.radio(
    "",
    ["dashboard", "severity"],
    format_func=lambda k: {
        "dashboard": t("nav_dashboard", lang),
        "severity": t("nav_severity", lang),
    }[k],
    horizontal=True,
    label_visibility="collapsed",
    key="navbar_radio",
)

if page == "dashboard":
    dashboard.show(lang)

elif page == "severity":
    severity.show(lang)
