import streamlit as st
from utils.translator import t

def show(lang):
    st.title(t("voice", lang))

    st.write("🎙 Voice assistant yahan hoga")