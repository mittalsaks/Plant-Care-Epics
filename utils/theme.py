"""Shared PlantCare AI visual theme: contrast-safe colors and Streamlit widget overrides."""

import streamlit as st

GLOBAL_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

:root {
  --pc-bg: #faf6f0;
  --pc-surface: #ffffff;
  --pc-card: #ffffff;
  --pc-cream: #faf6f0;
  --pc-text: #1a211d;
  --pc-soil: #2b2b2b;
  --pc-text-muted: #4d5752;
  --pc-border: #c8d4cf;
  --pc-green: #1b4332;
  --pc-green-mid: #2d6a4f;
  --pc-green-light: #52b788;
  --pc-warn: #e9c46a;
  --pc-serious: #f4a261;
  --pc-critical: #c1121f;
  --pc-radius: 18px;
  --pc-shadow: rgba(27, 67, 50, 0.12);
}

/* App shell — light background everywhere */
.stApp,
[data-testid="stAppViewContainer"] > .main {
  background-color: var(--pc-bg) !important;
  background-image: 
    radial-gradient(circle at 25% 25%, rgba(82, 183, 136, 0.03) 0%, transparent 50%),
    radial-gradient(circle at 75% 75%, rgba(233, 196, 106, 0.03) 0%, transparent 50%);
  background-size: 400px 400px;
  color: var(--pc-text) !important;
}

.block-container {
  padding-top: 1.1rem !important;
  max-width: 920px !important;
}

/* Default typography (Streamlit often inherits wrong colors) */
[data-testid="stAppViewContainer"] {
  font-family: 'Nunito', sans-serif !important;
  color: var(--pc-text) !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th {
  color: var(--pc-text) !important;
}

[data-testid="stMarkdownContainer"] strong {
  color: var(--pc-text) !important;
}

h1, h2, h3 {
  color: var(--pc-green) !important;
  font-weight: 800 !important;
}

h4, h5, h6 {
  color: var(--pc-green-mid) !important;
}

[data-testid="stCaptionContainer"] {
  color: var(--pc-text-muted) !important;
}

/* Widget labels */
label[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] p,
.stSelectbox label p,
.stMultiSelect label p,
.stFileUploader label p,
.stTextInput label p,
.stTextArea label p {
  color: var(--pc-text) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background-color: var(--pc-surface) !important;
  color: var(--pc-text) !important;
  border-color: var(--pc-border) !important;
}

/* Radio — options readable */
[data-testid="stRadio"] {
  margin-top: 1.1rem !important;
  padding: 0.35rem 0.55rem !important;
  border: 1px solid rgba(27, 67, 50, 0.2) !important;
  border-radius: 14px !important;
  background: rgba(255, 255, 255, 0.88) !important;
  box-shadow: 0px 6px 22px rgba(15, 46, 34, 0.12) !important;
  position: relative !important;
  top: 0.2rem !important;
}

[data-testid="stRadio"] label,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] div[role="radiogroup"] label {
  color: var(--pc-text) !important;
}

[data-testid="stRadio"] div[role="radiogroup"] {
  display: flex !important;
  gap: 0.5rem;
  background-color: transparent !important;
  border: none !important;
  padding: 0 !important;
}

[data-testid="stRadio"] div[role="radiogroup"] label {
  background-color: rgba(35, 91, 72, 0.08) !important;
  border: 1px solid rgba(27, 67, 50, 0.25) !important;
  border-radius: 10px !important;
  padding: 0.45rem 1rem !important;
  min-width: 160px;
  text-align: center;
  transition: all 0.2s ease;
  color: var(--pc-green) !important;
  font-weight: 700;
}

[data-testid="stRadio"] div[role="radiogroup"] label:hover {
  background-color: rgba(77, 87, 82, 0.12) !important;
  color: var(--pc-green-mid) !important;
}

[data-testid="stRadio"] div[role="radiogroup"] label[data-state="true"],
[data-testid="stRadio"] div[role="radiogroup"] label[data-focused="true"] {
  background-color: var(--pc-green-mid) !important;
  border-color: var(--pc-green-mid) !important;
  color: white !important;
}


/* Buttons */
.stButton > button {
  border-radius: 12px !important;
  font-weight: 700 !important;
  border: 1px solid var(--pc-border) !important;
  background-color: var(--pc-surface) !important;
  color: var(--pc-text) !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
  background-color: var(--pc-green-mid) !important;
  color: #ffffff !important;
  border-color: var(--pc-green-mid) !important;
}

.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
  background-color: #eef5f2 !important;
  color: var(--pc-text) !important;
}

/* File uploader */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] small {
  color: var(--pc-text-muted) !important;
}

[data-testid="stFileUploader"] {
  background: var(--pc-surface) !important;
  border: 1px dashed var(--pc-border) !important;
  border-radius: 14px !important;
}

/* Camera input */
[data-testid="stCameraInput"] label,
[data-testid="stCameraInput"] p {
  color: var(--pc-text) !important;
}

/* Expanders */
.streamlit-expanderHeader {
  color: var(--pc-green) !important;
  font-weight: 700 !important;
  background-color: #f0f6f3 !important;
}

[data-testid="stExpander"] details {
  background: var(--pc-surface) !important;
  border: 1px solid var(--pc-border) !important;
  border-radius: 14px !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
  color: var(--pc-text) !important;
}

/* Alerts — force readable body text (fixes low-contrast theme bugs) */
[data-testid="stAlert"] {
  border-radius: 12px !important;
}
[data-testid="stAlert"],
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
  color: var(--pc-soil) !important;
  opacity: 1 !important;
}

/* Divider */
hr {
  border-color: var(--pc-border) !important;
}

/* Hide menu/footer clutter */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Shared subpage hero (Severity, Voice, etc.) — always light text on gradient */
.pc-page-hero {
  background: linear-gradient(145deg, #143d32 0%, #1b4332 42%, #40916c 100%);
  border-radius: var(--pc-radius);
  padding: 1.85rem 1.35rem 1.55rem;
  text-align: center;
  margin-bottom: 1.35rem;
  box-shadow: 0 10px 32px var(--pc-shadow);
}
.pc-page-hero h1 {
  font-size: clamp(1.35rem, 4vw, 1.9rem);
  font-weight: 800;
  margin: 0 0 0.4rem;
  line-height: 1.2;
  color: #ffffff !important;
}
.pc-page-hero p {
  margin: 0;
  font-size: clamp(0.92rem, 2.6vw, 1.05rem);
  line-height: 1.55;
  opacity: 0.95;
  color: #ffffff !important;
}

/* Beat Streamlit’s markdown container (otherwise dark text on dark gradient) */
[data-testid="stMarkdownContainer"] .pc-page-hero h1,
[data-testid="stMarkdownContainer"] .pc-page-hero p {
  color: #ffffff !important;
}

/* Light content panel for secondary pages */
.pc-content-card {
  background: var(--pc-surface);
  border-radius: var(--pc-radius);
  padding: 1.25rem 1.35rem;
  box-shadow: 0 4px 18px var(--pc-shadow);
  border: 1px solid var(--pc-border);
  color: var(--pc-text) !important;
  margin-bottom: 1rem;
}
.pc-content-card p,
.pc-content-card li {
  color: var(--pc-text) !important;
}

[data-testid="stMarkdownContainer"] .pc-content-card p,
[data-testid="stMarkdownContainer"] .pc-content-card li,
[data-testid="stMarkdownContainer"] .pc-content-card div {
  color: var(--pc-text) !important;
}
</style>
"""


def inject_global_theme():
    st.markdown(GLOBAL_THEME_CSS, unsafe_allow_html=True)


def render_subpage_hero(lang: str, title_key: str, subtitle_key: str) -> None:
    """Same gradient header as Dashboard hero, for Severity / Voice / etc."""
    import html

    from utils.translator import t

    title = html.escape(t(title_key, lang))
    sub = html.escape(t(subtitle_key, lang))
    st.markdown(
        f'<div class="pc-page-hero"><h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True,
    )
