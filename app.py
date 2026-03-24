import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import time
import json
import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model.keras")


# ─── Session State Initialization (FIXED) ─────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

if "model" not in st.session_state:
    st.session_state["model"] = None

if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Class Labels (38 PlantVillage classes) ────────────────────────────────────
CLASS_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ─── Disease Info Database ─────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
        "treatment": ["Apply fungicides (captan, myclobutanil) early season", "Remove and destroy infected leaves", "Prune for better air circulation"],
        "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Apply preventive fungicide sprays"],
        "icon": "🍎"
    },
    "Apple___Black_rot": {
        "severity": "High",
        "description": "Fungal disease causing fruit rot and frogeye leaf spots.",
        "treatment": ["Remove mummified fruits", "Apply copper-based fungicides", "Prune infected wood during dormancy"],
        "prevention": ["Maintain orchard sanitation", "Avoid wounding fruits", "Monitor regularly during wet seasons"],
        "icon": "🍎"
    },
    "Apple___Cedar_apple_rust": {
        "severity": "Moderate",
        "description": "Fungal disease requiring two hosts: cedar/juniper and apple trees.",
        "treatment": ["Apply fungicides during spring", "Remove nearby cedar trees if feasible", "Use myclobutanil or mancozeb"],
        "prevention": ["Plant resistant apple varieties", "Create distance from cedar trees", "Apply preventive sprays at pink bud stage"],
        "icon": "🍎"
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "Moderate",
        "description": "Fungal rust disease producing orange-brown pustules on leaves.",
        "treatment": ["Apply fungicides at early infection", "Use triazole or strobilurin fungicides", "Remove severely infected plants"],
        "prevention": ["Plant resistant hybrids", "Early planting to avoid peak rust season", "Monitor fields regularly"],
        "icon": "🌽"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "High",
        "description": "Fungal disease causing large tan lesions on leaves, reducing yield.",
        "treatment": ["Apply fungicides at first sign", "Use azoxystrobin or propiconazole", "Remove crop debris after harvest"],
        "prevention": ["Plant resistant hybrids", "Crop rotation with non-host crops", "Avoid dense planting"],
        "icon": "🌽"
    },
    "Tomato___Late_blight": {
        "severity": "Critical",
        "description": "Devastating oomycete disease causing rapid plant collapse in wet conditions.",
        "treatment": ["Apply copper-based fungicides immediately", "Remove and destroy infected plants", "Avoid composting infected material"],
        "prevention": ["Use certified disease-free seeds", "Avoid overhead watering", "Apply preventive fungicides"],
        "icon": "🍅"
    },
    "Tomato___Early_blight": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark concentric ring lesions on older leaves.",
        "treatment": ["Apply chlorothalonil or copper fungicides", "Remove lower infected leaves", "Improve air circulation"],
        "prevention": ["Mulch around plants", "Avoid wetting foliage", "Rotate crops yearly"],
        "icon": "🍅"
    },
    "Potato___Late_blight": {
        "severity": "Critical",
        "description": "Same pathogen as tomato late blight; caused the Irish Potato Famine.",
        "treatment": ["Apply systemic fungicides immediately", "Destroy infected tubers and plants", "Harvest early if outbreak is severe"],
        "prevention": ["Use certified seed potatoes", "Hill potatoes to protect tubers", "Apply preventive sprays in wet weather"],
        "icon": "🥔"
    },
    "Grape___Black_rot": {
        "severity": "High",
        "description": "Fungal disease causing circular brown lesions on leaves and mummified berries.",
        "treatment": ["Apply mancozeb or myclobutanil", "Remove mummified berries", "Prune for air circulation"],
        "prevention": ["Apply fungicides from bud break", "Maintain good sanitation", "Avoid overhead irrigation"],
        "icon": "🍇"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "Critical",
        "description": "Viral disease spread by whiteflies causing severe yellowing and curl of leaves.",
        "treatment": ["No cure; remove and destroy infected plants", "Control whitefly vectors with insecticides", "Use reflective mulches to deter whiteflies"],
        "prevention": ["Use virus-resistant varieties", "Install insect-proof netting", "Monitor and control whitefly populations"],
        "icon": "🍅"
    },
}

DEFAULT_DISEASE_INFO = {
    "severity": "Unknown",
    "description": "Consult an agricultural expert for detailed disease assessment.",
    "treatment": ["Consult local agricultural extension office", "Send samples to plant disease lab", "Monitor plant regularly"],
    "prevention": ["Maintain good agricultural practices", "Regular monitoring", "Proper crop rotation"],
    "icon": "🌱"
}

SEVERITY_COLORS = {
    "Critical": "#ff4444",
    "High": "#ff8800",
    "Moderate": "#ffcc00",
    "Low": "#88cc00",
    "Unknown": "#888888",
}

HEALTHY_PLANTS = [c for c in CLASS_LABELS if "healthy" in c]

# ─── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-deep: #1a3a2a;
    --green-mid: #2d6a4f;
    --green-light: #52b788;
    --green-pale: #b7e4c7;
    --cream: #f8f4ed;
    --earth: #8b5e3c;
    --red-disease: #c1121f;
    --gold: #e9c46a;
    --shadow: 0 4px 24px rgba(26,58,42,0.13);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--green-deep);
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green-mid) 60%, var(--green-light) 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.app-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.app-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -20px;
    width: 250px; height: 250px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: #fff;
    margin: 0;
    letter-spacing: -1px;
    line-height: 1.1;
}
.header-subtitle {
    color: var(--green-pale);
    font-size: 1.1rem;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}
.header-badge {
    display: inline-block;
    background: var(--gold);
    color: var(--green-deep);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-top: 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Upload zone */
.upload-section {
    background: #fff;
    border: 2.5px dashed var(--green-pale);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s;
    margin-bottom: 1.5rem;
}
.upload-section:hover {
    border-color: var(--green-light);
    background: #f0faf4;
}

/* Result cards */
.result-card {
    background: #fff;
    border-radius: 16px;
    padding: 1.8rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.2rem;
    border-left: 5px solid var(--green-light);
}
.result-card.disease {
    border-left-color: var(--red-disease);
}
.result-card.healthy {
    border-left-color: #52b788;
}

.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin-bottom: 0.2rem;
}

/* Confidence bar */
.conf-bar-bg {
    background: #eee;
    border-radius: 50px;
    height: 12px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 1s ease;
}

/* Severity badge */
.severity-badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: #fff;
    margin-bottom: 0.8rem;
}

/* Info sections */
.info-box {
    background: #f7fbf8;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.info-box h4 {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: var(--green-mid);
    margin-bottom: 0.6rem;
}
.info-box ul {
    margin: 0;
    padding-left: 1.2rem;
}
.info-box li {
    font-size: 0.9rem;
    margin-bottom: 0.3rem;
    color: #444;
}

/* Top predictions table */
.pred-row {
    display: flex;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f0f0f0;
    gap: 1rem;
}
.pred-row:last-child { border-bottom: none; }
.pred-name { flex: 1; font-size: 0.88rem; }
.pred-pct { font-weight: 600; font-size: 0.9rem; color: var(--green-mid); min-width: 50px; text-align: right; }

/* Stats section */
.stat-card {
    background: linear-gradient(135deg, var(--green-deep), var(--green-mid));
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    color: #fff;
    box-shadow: var(--shadow);
}
.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
}
.stat-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.8;
    margin-top: 0.2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--green-deep) !important;
}
section[data-testid="stSidebar"] * {
    color: #e0f0e8 !important;
}
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--green-pale) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: var(--green-pale) !important;
}

/* Streamlit button overrides */
.stButton>button {
    background: var(--green-mid) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton>button:hover {
    background: var(--green-light) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(82,183,136,0.35) !important;
}

/* File uploader */
.stFileUploader {
    background: transparent !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    background: #fff !important;
    border-radius: 10px !important;
    border: 1.5px solid var(--green-pale) !important;
    color: var(--green-deep) !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--green-mid) !important;
    color: #fff !important;
    border-color: var(--green-mid) !important;
}

.stProgress > div > div {
    background-color: var(--green-light) !important;
}

/* Divider */
hr { border-color: var(--green-pale); opacity: 0.5; }

/* Healthy vs Disease */
.healthy-banner {
    background: linear-gradient(120deg, #d8f3dc, #b7e4c7);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    border: 2px solid #52b788;
}
.disease-banner {
    background: linear-gradient(120deg, #fff0f0, #ffd6d6);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    border: 2px solid #c1121f;
}

/* History item */
.history-item {
    background: #fff;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    try:
        model = tf.keras.models.load_model("model_tf")
        infer = model.signatures["serving_default"]
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, img_array):
    import tensorflow as tf

    infer = model.signatures["serving_default"]

    img_tensor = tf.convert_to_tensor(img_array)

    preds = infer(img_tensor)
    preds = list(preds.values())[0].numpy()[0]

    return preds

def format_class_name(raw: str) -> tuple[str, str]:
    """Returns (plant, condition)."""
    parts = raw.split("___")
    plant = parts[0].replace("_", " ").strip()
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return plant, condition

def get_conf_color(conf: float) -> str:
    if conf >= 85:
        return "#2d6a4f"
    elif conf >= 65:
        return "#52b788"
    elif conf >= 45:
        return "#e9c46a"
    else:
        return "#e76f51"

def make_gradcam(model, img_array):
    """Generate Grad-CAM heatmap."""
    try:
        import tensorflow as tf
        import cv2

        last_conv_layer = None
        for layer in reversed(model.layers):
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            last_conv_layer = "out_relu"

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Get original image
        original = np.uint8(img_array[0] * 255)
        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

        return Image.fromarray(superimposed_rgb)
    except Exception:
        return None


# ─── Session State ─────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "model_path" not in st.session_state:
    st.session_state.model_path = DEFAULT_MODEL_PATH

import tensorflow as tf

if not st.session_state.model_loaded:
    if os.path.exists(DEFAULT_MODEL_PATH):
        with st.spinner("🔄 Loading default model..."):
            model = load_model(DEFAULT_MODEL_PATH)
            if model:
                st.session_state.model = model
                st.session_state.model_loaded = True
    else:
        st.warning("⚠️ Default model not found in app directory.")


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 PlantCare AI")
    st.markdown("---")

    st.markdown("### ⚙️ Model Setup")
    model_path_input = st.text_input(
    "Model File Path (.keras)",
    value=DEFAULT_MODEL_PATH,   # 👈 auto-filled
    )

    if st.button("🔄 Load Model"):
        if model_path_input and os.path.exists(model_path_input):
            with st.spinner("Loading model..."):
                st.session_state.model_path = model_path_input
                st.session_state.model_path_input = model_path_input
                import tensorflow as tf
                try:
                    model = load_model(model_path_input)
                    if model:
                        st.session_state.model = model
                        st.session_state.model_loaded = True
                        st.success("✅ Model loaded!")
                    st.success("✅ Model loaded!")
                except Exception as e:
                    st.error(f"❌ {e}")
        elif model_path_input:
            st.error("❌ File not found.")
        else:
            st.warning("Please enter a model path.")

    # Model status
    if st.session_state.model_loaded:
        st.markdown("""
        <div style='background:rgba(82,183,136,0.2);border-radius:8px;padding:0.7rem 1rem;margin-top:0.5rem;'>
            <span style='color:#52b788;font-weight:600;'>● Model Active</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(255,100,100,0.15);border-radius:8px;padding:0.7rem 1rem;margin-top:0.5rem;'>
            <span style='color:#ff7070;font-weight:600;'>○ No Model Loaded</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ Settings")
    conf_threshold = st.slider("Confidence Threshold (%)", 0, 100, 50,
                                help="Only show results above this confidence")
    show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True,
                                help="Visualize what the model focuses on")
    top_k = st.selectbox("Top Predictions to Show", [3, 5, 10], index=1)

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    total = len(st.session_state.history)
    diseases = sum(1 for h in st.session_state.history if not h["healthy"])
    healthy = total - diseases

    col1s, col2s = st.columns(2)
    with col1s:
        st.metric("Total", total)
    with col2s:
        st.metric("Diseases", diseases)

    if total > 0:
        st.progress(healthy / total if total else 0)
        st.caption(f"{healthy} healthy · {diseases} diseased")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style='font-size:0.82rem;opacity:0.85;line-height:1.6;'>
    MobileNetV2 fine-tuned on PlantVillage dataset.<br>
    <b>38 classes</b> · 14 plant species<br>
    Architecture: MobileNetV2 + custom head<br>
    Training: 3-phase fine-tuning
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ──────────────────────────────────────────────────────────────
st.sidebar.write("Model path:", model_path_input)
st.sidebar.write("Exists:", os.path.exists(model_path_input))
# Header
st.markdown("""
<div class="app-header">
    <div style="position:relative;z-index:1;">
        <p class="header-title">🌿 PlantCare AI</p>
        <p class="header-subtitle">AI-powered plant disease detection for farmers, agronomists & researchers</p>
        <span class="header-badge">MobileNetV2 · 38 Disease Classes · PlantVillage Dataset</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Diagnose", "📋 Disease Library", "📈 History", "💡 How to Use"])

# ─────────────────────────── TAB 1: DIAGNOSE ───────────────────────────────────
with tab1:
    col_upload, col_result = st.columns([1, 1.3], gap="large")

    with col_upload:
        st.markdown("#### 📸 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear photo of the plant leaf. Best results with close-up shots."
        )

        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)

            # Image info
            st.markdown(f"""
            <div style='font-size:0.8rem;color:#888;margin-top:0.3rem;'>
            📐 {img.width}×{img.height}px · {uploaded_file.type} · {uploaded_file.size/1024:.1f} KB
            </div>""", unsafe_allow_html=True)

        else:
            # Placeholder
            st.markdown("""
            <div class="upload-section">
                <div style='font-size:3rem;'>🍃</div>
                <div style='font-size:1rem;color:#666;margin-top:0.5rem;'>Upload a plant leaf image</div>
                <div style='font-size:0.8rem;color:#aaa;margin-top:0.3rem;'>JPG, PNG, WEBP supported</div>
            </div>
            """, unsafe_allow_html=True)

        # Tips
        with st.expander("📸 Photography Tips for Best Results"):
            st.markdown("""
            - **Good lighting**: Natural daylight or bright indoor light
            - **Close-up shot**: Fill the frame with the leaf
            - **Sharp focus**: Avoid blurry images
            - **Leaf only**: Minimize background distractions
            - **Show symptoms**: Capture the affected areas clearly
            """)

        if uploaded_file:
            analyze_btn = st.button("🔬 Analyze Leaf", use_container_width=True)
        else:
            analyze_btn = False

    with col_result:
        st.markdown("#### 🧪 Diagnosis Results")

        if not uploaded_file:
            st.info("Upload a leaf image to get started.")

        elif not st.session_state.model_loaded:
            st.warning("⚠️ No model loaded. Please provide your `.h5` model path in the sidebar and click **Load Model**.")
            st.markdown("""
            <div style='background:#fff8e1;border-radius:12px;padding:1.2rem;border:1.5px solid #e9c46a;'>
                <b>Quick Setup:</b><br>
                1. Open the sidebar (left panel)<br>
                2. Enter the path to your <code>model.keras</code> file<br>
                3. Click <b>Load Model</b><br>
                4. Come back and click <b>Analyze Leaf</b>
            </div>
            """, unsafe_allow_html=True)

        elif analyze_btn:
            img = Image.open(uploaded_file)
            img_array = preprocess_image(img)

            with st.spinner("🔍 Analyzing leaf..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.008)
                    progress.progress(i + 1)

                results, all_preds = predict(st.session_state.model, img_array)
                top_label, top_conf = results[0]
                plant, condition = format_class_name(top_label)
                is_healthy = "healthy" in top_label.lower()
                info = DISEASE_INFO.get(top_label, DEFAULT_DISEASE_INFO)
                severity = info["severity"]
                sev_color = SEVERITY_COLORS.get(severity, "#888")

            # Save to history
            st.session_state.history.append({
                "filename": uploaded_file.name,
                "label": top_label,
                "plant": plant,
                "condition": condition,
                "confidence": top_conf,
                "healthy": is_healthy,
                "time": time.strftime("%H:%M:%S"),
            })

            # ── Result banner ──
            if is_healthy:
                st.markdown(f"""
                <div class="healthy-banner">
                    <div style='font-size:2.5rem;'>✅</div>
                    <div style='font-family:Playfair Display,serif;font-size:1.5rem;font-weight:700;color:#1a3a2a;'>{plant}</div>
                    <div style='color:#2d6a4f;font-weight:500;margin-top:0.3rem;'>Healthy Plant — No Disease Detected</div>
                    <div style='font-size:0.85rem;color:#555;margin-top:0.5rem;'>Confidence: <b>{top_conf:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="disease-banner">
                    <div style='font-size:2.5rem;'>{info['icon']}</div>
                    <div style='font-family:Playfair Display,serif;font-size:1.5rem;font-weight:700;color:#7d0000;'>{plant}</div>
                    <div style='color:#c1121f;font-weight:600;margin-top:0.3rem;'>{condition}</div>
                    <div style='font-size:0.85rem;color:#555;margin-top:0.5rem;'>Confidence: <b>{top_conf:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Confidence bar ──
            conf_color = get_conf_color(top_conf)
            st.markdown(f"""
            <div class="result-label">Confidence Level</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{top_conf:.1f}%;background:{conf_color};"></div>
            </div>
            <div style='text-align:right;font-size:0.85rem;color:{conf_color};font-weight:600;margin-top:0.2rem;'>{top_conf:.1f}%</div>
            """, unsafe_allow_html=True)

            # ── Disease Info ──
            if not is_healthy:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <span class="severity-badge" style="background:{sev_color};">
                    ⚠ {severity} Severity
                </span>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-box">
                    <h4>📖 About this Disease</h4>
                    <p style='font-size:0.9rem;color:#444;margin:0;'>{info['description']}</p>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    treatment_html = "".join(f"<li>{t}</li>" for t in info["treatment"])
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>💊 Treatment</h4>
                        <ul>{treatment_html}</ul>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    prevention_html = "".join(f"<li>{p}</li>" for p in info["prevention"])
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>🛡️ Prevention</h4>
                        <ul>{prevention_html}</ul>
                    </div>""", unsafe_allow_html=True)

            # ── Top K Predictions ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Top {top_k} Predictions**")
            for label, conf in results[:top_k]:
                if conf < conf_threshold:
                    continue
                p, c = format_class_name(label)
                bar_w = int(conf)
                bar_color = "#52b788" if "healthy" in label else "#e76f51"
                healthy_tag = "🟢" if "healthy" in label else "🔴"
                st.markdown(f"""
                <div class="pred-row">
                    <div style='width:22px;text-align:center;'>{healthy_tag}</div>
                    <div class="pred-name"><b>{p}</b><br><span style='color:#888;font-size:0.78rem;'>{c}</span></div>
                    <div style='flex:1;'>
                        <div class="conf-bar-bg" style='height:8px;'>
                            <div class="conf-bar-fill" style="width:{bar_w}%;background:{bar_color};"></div>
                        </div>
                    </div>
                    <div class="pred-pct">{conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Grad-CAM ──
            if show_gradcam:
                st.markdown("<br>**🔥 Grad-CAM Visualization**")
                st.caption("Highlights regions the model focused on for its prediction.")
                img = Image.open(uploaded_file)
                img_arr = preprocess_image(img)
                gradcam_img = make_gradcam(st.session_state.model, img_arr)
                if gradcam_img:
                    c1g, c2g = st.columns(2)
                    with c1g:
                        st.image(img.resize((224, 224)), caption="Original", use_container_width=True)
                    with c2g:
                        st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)
                else:
                    st.info("Grad-CAM not available for this model configuration.")

            # ── Disclaimer ──
            st.markdown("""
            <div style='background:#fffbea;border-radius:10px;padding:0.9rem 1.2rem;margin-top:1rem;border:1.5px solid #e9c46a;font-size:0.82rem;color:#665500;'>
            ⚠️ <b>Disclaimer:</b> This AI tool is for informational purposes only. For critical crop decisions, 
            always consult a certified agricultural expert or plant pathologist.
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────── TAB 2: DISEASE LIBRARY ───────────────────────────
with tab2:
    st.markdown("### 📋 Plant Disease Reference Library")
    st.markdown("Browse all 38 plant conditions that this model can detect.")

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="stat-card"><div class="stat-number">38</div><div class="stat-label">Total Classes</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="stat-card"><div class="stat-number">14</div><div class="stat-label">Plant Species</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="stat-card"><div class="stat-number">26</div><div class="stat-label">Disease Types</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="stat-card"><div class="stat-number">54K+</div><div class="stat-label">Training Images</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        plants = sorted(set(c.split("___")[0].replace("_", " ") for c in CLASS_LABELS))
        plant_filter = st.selectbox("Filter by Plant", ["All"] + plants)
    with col_filter2:
        status_filter = st.selectbox("Filter by Status", ["All", "Diseased", "Healthy"])

    # Display classes
    filtered = CLASS_LABELS.copy()
    if plant_filter != "All":
        filtered = [c for c in filtered if c.split("___")[0].replace("_", " ") == plant_filter]
    if status_filter == "Healthy":
        filtered = [c for c in filtered if "healthy" in c]
    elif status_filter == "Diseased":
        filtered = [c for c in filtered if "healthy" not in c]

    for label in filtered:
        plant, condition = format_class_name(label)
        is_h = "healthy" in label
        info = DISEASE_INFO.get(label, DEFAULT_DISEASE_INFO)
        border = "#52b788" if is_h else SEVERITY_COLORS.get(info["severity"], "#888")
        icon = "✅" if is_h else info["icon"]

        with st.expander(f"{icon} {plant} — {condition}"):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"**Status:** {'🟢 Healthy' if is_h else '🔴 Diseased'}")
                if not is_h:
                    sev = info["severity"]
                    st.markdown(f"**Severity:** <span style='color:{SEVERITY_COLORS.get(sev,'#888')};font-weight:600'>{sev}</span>", unsafe_allow_html=True)
                st.markdown(f"**Description:** {info['description']}")
            with c2:
                if not is_h:
                    st.markdown("**Treatment:**")
                    for t in info["treatment"]:
                        st.markdown(f"• {t}")
                    st.markdown("**Prevention:**")
                    for p in info["prevention"]:
                        st.markdown(f"• {p}")
                else:
                    st.markdown("This plant appears healthy. Maintain current care practices.")

    st.markdown(f"<div style='text-align:center;color:#aaa;font-size:0.85rem;margin-top:1rem;'>Showing {len(filtered)} of {len(CLASS_LABELS)} classes</div>", unsafe_allow_html=True)


# ─────────────────────────── TAB 3: HISTORY ───────────────────────────────────
with tab3:
    st.markdown("### 📈 Analysis History")

    if not st.session_state.history:
        st.info("No analyses yet. Upload a leaf image in the **Diagnose** tab to get started.")
    else:
        # Summary
        total = len(st.session_state.history)
        diseased = sum(1 for h in st.session_state.history if not h["healthy"])
        healthy = total - diseased
        avg_conf = np.mean([h["confidence"] for h in st.session_state.history])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="stat-card"><div class="stat-number">{total}</div><div class="stat-label">Total Analyzed</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="stat-card"><div class="stat-number">{healthy}</div><div class="stat-label">Healthy</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="stat-card"><div class="stat-number">{diseased}</div><div class="stat-label">Diseased</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="stat-card"><div class="stat-number">{avg_conf:.0f}%</div><div class="stat-label">Avg Confidence</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # History list
        for i, h in enumerate(reversed(st.session_state.history)):
            icon = "✅" if h["healthy"] else "🔴"
            color = "#52b788" if h["healthy"] else "#c1121f"
            st.markdown(f"""
            <div class="history-item">
                <div style='font-size:1.4rem;'>{icon}</div>
                <div style='flex:1;'>
                    <div style='font-weight:600;'>{h["plant"]} — {h["condition"]}</div>
                    <div style='font-size:0.8rem;color:#888;'>{h["filename"]} · {h["time"]}</div>
                </div>
                <div style='font-weight:700;color:{color};font-size:1rem;'>{h["confidence"]:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Export
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📥 Export History as CSV"):
            import csv
            import io as sio
            output = sio.StringIO()
            writer = csv.DictWriter(output, fieldnames=["filename", "plant", "condition", "confidence", "healthy", "time"])
            writer.writeheader()
            writer.writerows(st.session_state.history)
            st.download_button(
                "Download CSV",
                data=output.getvalue(),
                file_name="plantcare_history.csv",
                mime="text/csv"
            )


# ─────────────────────────── TAB 4: HOW TO USE ────────────────────────────────
with tab4:
    st.markdown("### 💡 How to Use PlantGuard AI")

    st.markdown("""
    <div class="result-card">
    <div class="result-title">🚀 Quick Start Guide</div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Load Your Model", "In the sidebar, enter the full path to your trained `.h5` model file (e.g., `/path/to/model.keras`) and click **Load Model**."),
        ("2️⃣", "Upload a Leaf Image", "Go to the **Diagnose** tab, upload a clear photo of the plant leaf you want to analyze. Supported formats: JPG, PNG, WEBP."),
        ("3️⃣", "Analyze", "Click **Analyze Leaf** and wait for the AI to process the image. Results appear within seconds."),
        ("4️⃣", "Read the Diagnosis", "The app shows the detected plant, disease (or healthy status), confidence level, and detailed treatment/prevention recommendations."),
        ("5️⃣", "View Grad-CAM", "Enable **Show Grad-CAM Heatmap** in settings to see which parts of the leaf the model focused on."),
        ("6️⃣", "Track History", "All your analyses are saved in the **History** tab. You can export them as CSV for record-keeping."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex;gap:1rem;margin-bottom:1.2rem;align-items:flex-start;'>
            <div style='font-size:1.8rem;min-width:40px;'>{num}</div>
            <div>
                <div style='font-family:Playfair Display,serif;font-size:1.1rem;font-weight:700;color:#1a3a2a;'>{title}</div>
                <div style='font-size:0.9rem;color:#555;margin-top:0.2rem;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-box">
            <h4>✅ Best Practices</h4>
            <ul>
                <li>Use close-up, well-lit photos</li>
                <li>Focus on the most affected area</li>
                <li>Avoid images with heavy shadows</li>
                <li>Capture both sides of the leaf</li>
                <li>Use multiple images for cross-validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ Limitations</h4>
            <ul>
                <li>Trained on 38 specific disease classes</li>
                <li>May not generalize to rare diseases</li>
                <li>Image quality significantly affects accuracy</li>
                <li>Results are probabilistic, not definitive</li>
                <li>Always consult an expert for critical decisions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(120deg,#d8f3dc,#b7e4c7);border-radius:14px;padding:1.5rem 2rem;margin-top:1rem;'>
        <div style='font-family:Playfair Display,serif;font-size:1.1rem;font-weight:700;color:#1a3a2a;'>🌾 Supported Crops & Diseases</div>
        <div style='font-size:0.88rem;color:#2d6a4f;margin-top:0.5rem;'>
        Apple (4) · Blueberry · Cherry (2) · Corn/Maize (4) · Grape (4) · Orange · 
        Peach (2) · Bell Pepper (2) · Potato (3) · Raspberry · Soybean · Squash · 
        Strawberry (2) · Tomato (10)
        </div>
    </div>
    """, unsafe_allow_html=True)