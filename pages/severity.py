import streamlit as st
from utils.translator import t

def show(lang):
    st.title(t("severity", lang))

    st.write("📊 Yahan graphs aayenge")