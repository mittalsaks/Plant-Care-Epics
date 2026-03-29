import streamlit as st
from utils.translator import t

def show(lang):
    st.title(t("weather", lang))

    st.write("🌦 Weather prediction yahan hoga")