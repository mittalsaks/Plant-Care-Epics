import streamlit as st
from pages import dashboard, severity, weather, voice

# 🔹 Language selection
language = st.selectbox(
    "Choose Language",
    ["English", "Hindi"],
    key="language_select"   # 👈 unique key
)
st.set_page_config(
        page_title="PlantCare AI – Crop Disease Checker",
        page_icon="🌾",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
lang = "hi" if language == "हिंदी" else "en"

# 🔹 Navbar (TOP)
page = st.radio(
    "",
    ["🏠 Dashboard", "📊 Severity Analysis", "🌦 Weather Insights", "🎙 Voice Assistant"],
    horizontal=True
)

# 🔹 Routing
if page == "🏠 Dashboard":
    dashboard.show(lang)

elif page == "📊 Severity Analysis":
    severity.show(lang)

elif page == "🌦 Weather Insights":
    weather.show(lang)

elif page == "🎙 Voice Assistant":
    voice.show(lang)