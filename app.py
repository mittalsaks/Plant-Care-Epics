import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json

# Load translations
with open("translations.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

# Language selector
lang = st.selectbox(
    "🌍 Select Language / भाषा चुनें",
    options=["en", "hi"],
    format_func=lambda x: "English" if x == "en" else "हिंदी"
)
# ──────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantCare AI – Crop Disease Checker",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

:root {
    --soil:   #2b2b2b;
    --bark:   #5c3d2e;
    --leaf:   #2d6a4f;
    --lime:   #52b788;
    --sun:    #f4a261;
    --cream:  #faf3e8;
    --red:    #c1121f;
    --white:  #ffffff;
    --shadow: rgba(59,42,26,0.12);
}

html, body  {
    font-family: 'Nunito', sans-serif;
    background-color: var(--cream);
    color: var(--soil);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 800px; }

.hero {
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 55%, #52b788 100%);
    border-radius: 22px;
    padding: 2.2rem 2rem 1.8rem;
    color: white;
    text-align: center;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 36px var(--shadow);
}
.hero .emoji { font-size: 3rem; display: block; margin-bottom: 0.5rem; }
.hero h1     { font-size: 2.1rem; font-weight: 900; margin: 0 0 0.3rem; letter-spacing: -0.5px; }
.hero p      { font-size: 1rem; opacity: 0.88; margin: 0; }

.result-card {
    background: var(--white);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 6px 28px var(--shadow);
    margin-bottom: 1.2rem;
    border-left: 7px solid var(--leaf);
}
.result-card.danger  { border-left-color: var(--red); }
.result-card.healthy { border-left-color: var(--lime); }
.result-card.critical{ border-left-color: #8b0000; }

.badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.badge-danger  { background: #fde8ea; color: var(--red); }
.badge-healthy { background: #d8f3dc; color: var(--leaf); }
.badge-critical{ background: #fde8ea; color: #8b0000; }

.crop-name    { font-size: 1.65rem; font-weight: 900; margin: 0; }
.disease-name { font-size: 1.1rem;  font-weight: 700; color: var(--bark); margin: 5px 0 0; }
.confidence   { font-size: 0.88rem; color: #999; margin-top: 6px; }

.conf-bar-bg {
    background: #eee;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 6px;
}
.conf-bar-fill { height: 100%; border-radius: 999px; }

.tip-section {
    background: var(--white);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 3px 14px var(--shadow);
}
.tip-section h3 {
    font-size: 1rem;
    font-weight: 800;
    margin: 0 0 0.9rem;
}
.tip-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 0.65rem;
    font-size: 0.94rem;
    line-height: 1.6;
}
.tip-dot {
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem;
    flex-shrink: 0;
    margin-top: 1px;
    font-weight: 900;
}
.dot-red    { background:#fde8ea; color: var(--red); }
.dot-green  { background:#d8f3dc; color: var(--leaf); }
.dot-yellow { background:#fef9e7; color: #b7790a; }
.dot-blue   { background:#e8f4f8; color: #1a6985; }

.alert-box {
    background: #fff3cd;
    border: 1.5px solid #ffc107;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
    font-size: 0.93rem;
    line-height: 1.6;
}

.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #bbb;
    margin-top: 2rem;
    padding-bottom: 1.5rem;
}

            .result-card {
    color: #2b2b2b !important;
}

.tip-section {
    color: #333 !important;
}

.tip-section h3 {
    color: #1b4332 !important;
}

.tip-item {
    color: #444 !important;
}

.alert-box {
    color: #5c3d2e !important;
}
            
p, span, label, div {
    color: inherit;
}
            
.stFileUploader label {
    color: #3b2a1a !important;
    font-weight: 600;
}
            

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Disease knowledge base  (farmer-friendly language)
# ──────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "crop": {
            "en": "Apple",
            "hi": "सेब"
        },
        "disease": {
            "en": "Apple Scab",
            "hi": "एप्पल स्कैब"
        },
        "what": {
            "en": "A fungal infection that creates dark, rough spots on leaves and fruits, making them unsellable.",
            "hi": "यह एक फंगल रोग है जो पत्तियों और फलों पर काले धब्बे बनाता है और फल बेचने लायक नहीं रहते।"
        },
        "signs": {
            "en": [
                "Dark olive-green or brown spots on leaves",
                "Rough, scabby patches on the fruit skin",
                "Leaves may turn yellow and fall early"
            ],
            "hi": [
                "पत्तियों पर गहरे हरे या भूरे धब्बे",
                "फलों की त्वचा पर खुरदरे धब्बे",
                "पत्तियां पीली होकर जल्दी गिर सकती हैं"
            ]
        },
        "causes": {
            "en": [
                "Wet and rainy weather in spring",
                "Water staying on leaves for too long",
                "Infected fallen leaves from last season"
            ],
            "hi": [
                "वसंत में अधिक बारिश और नमी",
                "पत्तियों पर लंबे समय तक पानी रहना",
                "पिछले साल की संक्रमित गिरी पत्तियां"
            ]
        },
        "cure": {
            "en": [
                "Spray copper-based fungicide every 7–10 days",
                "Use mancozeb or captan spray",
                "Remove and burn fallen leaves"
            ],
            "hi": [
                "हर 7–10 दिन में कॉपर आधारित दवा का छिड़काव करें",
                "मैनकोजेब या कैप्टान का उपयोग करें",
                "गिरी पत्तियों को हटाकर जला दें"
            ]
        },
        "prevention": {
            "en": [
                "Use resistant varieties",
                "Ensure good airflow by pruning",
                "Avoid overhead watering"
            ],
            "hi": [
                "रोग-रोधी किस्में लगाएं",
                "पेड़ की छंटाई करें",
                "ऊपर से पानी न डालें"
            ]
        },
        "severity": "medium"
    },

    "Apple___Black_rot": {
        "crop": {
            "en": "Apple",
            "hi": "सेब"
        },
        "disease": {
            "en": "Black Rot",
            "hi": "ब्लैक रॉट"
        },
        "what": {
            "en": "A fungal disease that rots fruit and creates purple spots on leaves.",
            "hi": "यह एक फंगल रोग है जो फलों को सड़ा देता है और पत्तियों पर बैंगनी धब्बे बनाता है।"
        },
        "signs": {
            "en": [
                "Purple spots on leaves",
                "Brown rings on fruits",
                "Shrivelled black fruits"
            ],
            "hi": [
                "पत्तियों पर बैंगनी धब्बे",
                "फलों पर भूरे छल्ले",
                "सूखे काले फल"
            ]
        },
        "causes": {
            "en": [
                "Dead branches",
                "Warm humid weather",
                "Insect wounds"
            ],
            "hi": [
                "सूखी शाखाएं",
                "गर्म और नम मौसम",
                "कीड़ों के कारण घाव"
            ]
        },
        "cure": {
            "en": [
                "Remove dead branches",
                "Spray fungicide",
                "Remove infected fruits"
            ],
            "hi": [
                "सूखी शाखाएं काटें",
                "फफूंदनाशक का छिड़काव करें",
                "संक्रमित फल हटाएं"
            ]
        },
        "prevention": {
            "en": [
                "Prune regularly",
                "Keep orchard clean",
                "Protect from insects"
            ],
            "hi": [
                "नियमित छंटाई करें",
                "बगीचे को साफ रखें",
                "कीड़ों से बचाव करें"
            ]
        },
        "severity": "high"
    }
}

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
MODEL_PATH = "plantvillage_phase3_epoch25_FINAL.h5"
IMAGE_SIZE = (224, 224)
LAST_CONV  = "out_relu"

CLASS_NAMES = list(DISEASE_INFO.keys())   # order must match training

SEVERITY_META = {
    "healthy":  {"label": "✅ HEALTHY CROP",       "badge": "badge-healthy", "card": "healthy",  "bar": "#52b788"},
    "medium":   {"label": "⚠️ DISEASE DETECTED",   "badge": "badge-danger",  "card": "danger",   "bar": "#e9c46a"},
    "high":     {"label": "🚨 SERIOUS DISEASE",    "badge": "badge-danger",  "card": "danger",   "bar": "#f4a261"},
    "critical": {"label": "🆘 CRITICAL — ACT NOW", "badge": "badge-critical","card": "critical", "bar": "#c1121f"},
}

# ──────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m

#helper function for multilingual support
def t(key):
    return translations.get(lang, {}).get(key, key)

def get_text(field):
    if isinstance(field, dict):
        return field.get(lang, field.get("en", ""))
    return field

def preprocess(pil_img):
    img = pil_img.resize(IMAGE_SIZE).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

import tensorflow as tf
import numpy as np

def make_gradcam(img_array, model, last_conv_layer_name=None):
    try:
        # Ensure batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)

        # Auto-detect last conv layer
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if "conv" in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output[0]
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

            # ✅ FIX 1: Ensure predictions is tensor (not list)
            if isinstance(predictions, list):
                predictions = predictions[0]

            # ✅ FIX 2: Get class index safely
            class_idx = tf.argmax(predictions[0])

            # ✅ FIX 3: Correct indexing
            class_channel = predictions[:, class_idx]

        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)

        # Mean intensity of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        # Weighted combination
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize safely
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)

        if max_val == 0:
            return None

        heatmap /= max_val

        return heatmap.numpy()

    except Exception as e:
        st.error(f"❌ GradCAM Error: {e}")
        return None

def overlay_gradcam(pil_img, heatmap, alpha=0.4):
    """Overlay GradCAM heatmap on original image with robust error handling."""

    # ✅ CHECK 1: Heatmap None
    if heatmap is None:
        st.warning("⚠️ GradCAM failed. Showing original image.")
        return np.array(pil_img.resize(IMAGE_SIZE))

    try:
        # ✅ CHECK 2: Convert Tensor → NumPy
        if isinstance(heatmap, (tf.Tensor, tf.Variable)):
            heatmap = heatmap.numpy()

        # ✅ CHECK 3: Handle list/tuple (Streamlit serialization issue)
        if isinstance(heatmap, (list, tuple)):
            heatmap = np.array(heatmap)

        # ✅ Convert to numpy float32
        heatmap = np.asarray(heatmap, dtype=np.float32)

        # ✅ CHECK 4: Ensure 2D
        if heatmap.ndim > 2:
            heatmap = heatmap.squeeze()

        if heatmap.ndim != 2:
            st.error(f"❌ Invalid heatmap shape: {heatmap.shape}")
            return np.array(pil_img.resize(IMAGE_SIZE))

        # ✅ CHECK 5: Normalize safely (0–1)
        min_val, max_val = np.min(heatmap), np.max(heatmap)
        if max_val > min_val:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(heatmap)

        # ✅ CHECK 6: Ensure contiguous memory (VERY IMPORTANT for OpenCV)
        heatmap = np.ascontiguousarray(heatmap)

        # ✅ CHECK 7: Resize (IMPORTANT FIX: use width, height correctly)
        heatmap = cv2.resize(
            heatmap,
            (IMAGE_SIZE[1], IMAGE_SIZE[0]),  # (width, height)
            interpolation=cv2.INTER_LINEAR
        )

        # ✅ Convert to heatmap colors
        heatmap = np.uint8(255 * np.clip(heatmap, 0, 1))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # ✅ Prepare original image
        img = np.array(pil_img.resize(IMAGE_SIZE))
        img = np.asarray(img, dtype=np.uint8)

        # ✅ Final overlay
        return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    except Exception as e:
        st.error(f"❌ Overlay failed: {e}")
        return np.array(pil_img.resize(IMAGE_SIZE))

# ──────────────────────────────────────────────────────────────
# HTML helpers
# ──────────────────────────────────────────────────────────────
def tips_html(items, dot_class, symbol):
    rows = "".join(
        f'<div class="tip-item">'
        f'<div class="tip-dot {dot_class}">{symbol}</div>'
        f'<div>{t}</div></div>'
        for t in items
    )
    return rows

# ══════════════════════════════════════════════════════════════
# UI START
# ══════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
  <span class="emoji">🌾</span>
  <h1>PlantCare AI</h1>
  <p>Take a photo of your crop leaf — we will tell you what is wrong and how to fix it</p>
</div>
""", unsafe_allow_html=True)

# Upload
uploaded = st.file_uploader(
    t("upload_label"),
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding:2rem 0; color:#aaa; font-size:0.95rem;">
        👆 Upload a clear photo of a single leaf to get your result.<br>
        <small>Tip: Good lighting and a steady hand gives the best results.</small>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="footer">PlantCare AI• AI Crop Disease Checker • Always confirm with your local agriculture officer before buying medicines.</div>', unsafe_allow_html=True)
    st.stop()

# ── Load & predict ─────────────────────────────────────────
pil_image = Image.open(uploaded).convert("RGB")

with st.spinner(t("analyzing")):
    try:
        model = load_model()
    except Exception as e:
        st.error(
            f"❌ Model file **{MODEL_PATH}** not found in this folder.\n\n"
            f"Make sure the model file is in the same folder as `app.py`.\n\n`{e}`"
        )
        st.stop()

    img_arr    = preprocess(pil_image)
    preds      = model.predict(img_arr, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    confidence = float(preds[pred_idx]) * 100

    pred_class = CLASS_NAMES[pred_idx]
    info       = DISEASE_INFO.get(pred_class)

    if info is None:
        st.error("Detected class not found in knowledge base. Please check CLASS_NAMES order.")
        st.stop()

    severity = info["severity"]
    meta     = SEVERITY_META[severity]

    heatmap = make_gradcam(img_arr, model)

    if heatmap is None:
        st.warning("⚠️ Could not generate GradCAM. Showing original image.")
        overlay_img = pil_image
    else:
        overlay_img = overlay_gradcam(pil_image, heatmap)

# ── Images side by side ────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**📷 {t('your_photo')}**")
    st.image(pil_image, use_container_width=True)
with col2:
    st.markdown(f"**🔬 {t('disease_area')}**")
    st.image(overlay_img, use_container_width=True)
    st.caption("🔴 Red/orange areas = where the disease is concentrated")

st.markdown("<br>", unsafe_allow_html=True)

# ── Result card ────────────────────────────────────────────
bar_color = meta["bar"]
bar_w     = round(confidence)

st.markdown(f"""
<div class="result-card {meta['card']}">
    <span class="badge {meta['badge']}">{meta['label']}</span>
    <div class="crop-name">🌿 {get_text(info['crop'])}</div>
    <div class="disease-name">{get_text(info['disease'])}</div>
    <div class="confidence">The AI is <b>{confidence:.1f}%</b> confident in this result</div>
    <div class="conf-bar-bg" style="margin-top:8px;">
        <div class="conf-bar-fill" style="width:{bar_w}%; background:{bar_color};"></div>
    </div>
    <p style="margin-top:1rem; font-size:0.97rem; line-height:1.7; color:#444;">{get_text(info['what'])}</p>
</div>
""", unsafe_allow_html=True)

# ── Healthy branch ─────────────────────────────────────────
if severity == "healthy":
    if info["prevention"]:
        st.markdown(f"""
        <div class="tip-section">
            <h3>💡 Tips to Keep Your Crop Healthy</h3>
            {tips_html(get_text(info['prevention']), 'dot-green', '✓')}
        </div>
        """, unsafe_allow_html=True)

# ── Disease branch ─────────────────────────────────────────
else:
    if info["signs"]:
        st.markdown(f"""
        <div class="tip-section">
            <h3>🔎 Signs You Will See on Your Crop</h3>
            {tips_html(get_text(info['signs']), 'dot-red', '!')}
        </div>
        """, unsafe_allow_html=True)

    if info["causes"]:
        st.markdown(f"""
        <div class="tip-section">
            <h3>🌧️ Why This is Happening</h3>
            {tips_html(get_text(info['causes']), 'dot-yellow', '?')}
        </div>
        """, unsafe_allow_html=True)

    if info["cure"]:
        st.markdown(f"""
        <div class="tip-section" style="border-left:4px solid #1a6985;">
            <h3>💊 What To Do Right Now — Step by Step</h3>
            {tips_html(get_text(info['cure']), 'dot-blue', '→')}
        </div>
        """, unsafe_allow_html=True)

    if info["prevention"]:
        st.markdown(f"""
        <div class="tip-section">
            <h3>🛡️ How to Prevent This Next Season</h3>
            {tips_html(get_text(info['prevention']), 'dot-green', '✓')}
        </div>
        """, unsafe_allow_html=True)

    # Advisory note for serious cases
    if severity in ("high", "critical"):
        st.markdown("""
        <div class="alert-box">
            <b>⚠️ Important Advisory:</b> This is a serious crop disease.
            Please contact your local Agriculture Officer, Krishi Vigyan Kendra (KVK),
            or call the Kisan Call Centre at <b>1800-180-1551 (free call)</b>
            for guidance on the right medicines available in your area.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PlantCare AI 🌾 • Powered by AI (MobileNetV2) trained on PlantVillage Dataset •
    38 crop diseases supported • Always confirm with your local agriculture expert before purchasing medicines.
</div>
""", unsafe_allow_html=True)