import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pages.dashboard import (
    CLASS_NAMES,
    DISEASE_INFO,
    SEVERITY_META,
    load_model,
    make_gradcam,
    overlay_gradcam,
    preprocess,
)
from utils.theme import render_subpage_hero
from utils.translator import t


def _severity_label(score: int) -> str:
    if score < 30:
        return "healthy"
    if score < 60:
        return "medium"
    if score < 85:
        return "high"
    return "critical"


def _severity_internal_advice(signal: str) -> str:
    text_map = {
        "healthy": "Your crop is healthy. Continue routine monitoring and maintain good farming practice.",
        "medium": "Early symptoms detected. Apply cultural controls and inspect nearby plants today.",
        "high": "Serious risk. Apply treatment recommended below and consult local agriculture expert.",
        "critical": "Critical level. Stop harvesting from affected plants and call local extension service immediately.",
    }
    return text_map.get(signal, "Please monitor conditions closely and repeat analysis needed.")


def show(lang: str) -> None:
    render_subpage_hero(lang, "severity", "severity_subtitle")

    # image input + severity detection
    st.subheader("📷 Severity prediction from leaf image")
    upload = st.file_uploader(
        t("upload_image", lang),
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="visible",
    )

    if upload is not None:
        from PIL import Image

        try:
            model = load_model()
            img = Image.open(upload).convert("RGB")
            st.image(img, caption=t('preview_label', lang), width=300)

            arr = preprocess(img)
            preds = model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx] * 100.0)
            predicted_class = CLASS_NAMES[idx]
            info = DISEASE_INFO.get(predicted_class, {})

            severity_tag = info.get('severity', 'medium')
            meta = SEVERITY_META.get(severity_tag, SEVERITY_META['medium'])

            st.markdown(f"### {t('disease_detected', lang)}: {html.escape(info.get('disease', predicted_class))}")
            st.info(f"{t('severity_label_' + severity_tag, lang)} ({severity_tag.upper()})")
            st.metric("Confidence", f"{confidence:.1f}%")

            st.markdown("### 🔍 Severity summary")
            st.markdown(f"- Crop: **{html.escape(info.get('crop', 'Unknown'))}**")
            st.markdown(f"- Severity tag: **{severity_tag}**")
            st.markdown(f"- Model advice: **{html.escape(info.get('what', 'No details'))}**")

            advice_text = _severity_internal_advice(severity_tag)
            st.warning(f"{advice_text}")

            heatmap = make_gradcam(arr, model, lang)
            if heatmap is not None:
                overlay = overlay_gradcam(img, heatmap, lang)
                st.image(overlay, caption=t('caption_heatmap', lang), width=300)

        except Exception as e:
            st.error(f"Could not analyze image: {e}")
    else:
        st.info(t('upload_prompt', lang))

    # quick manual severity explorer
    st.markdown("---")
    st.subheader("🧾 Manual severity summary by disease")

    all_crops = sorted({info['crop'] for info in DISEASE_INFO.values()})
    crop_choice = st.selectbox(t('section_leaf_photo', lang), all_crops)

    crop_diseases = sorted(
        [name for name, info in DISEASE_INFO.items() if info['crop'] == crop_choice]
    )
    disease_choice = st.selectbox("Choose disease sample", crop_diseases)

    severity_info = DISEASE_INFO.get(disease_choice, {})
    if severity_info:
        st.markdown("### Selected disease details")
        st.write("**Disease:**", severity_info['disease'])
        st.write("**Crop:**", severity_info['crop'])
        st.write("**Severity:**", severity_info['severity'])
        st.write("**What it means:**", severity_info['what'])

    rows = [{'severity': info['severity']} for info in DISEASE_INFO.values()]
    df = pd.DataFrame(rows)
    summary = df['severity'].value_counts().reindex(['healthy', 'medium', 'high', 'critical']).fillna(0)

    st.markdown("### Model dataset severity distribution")
    st.bar_chart(summary)

    # Pie chart for severity distribution
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(summary, labels=summary.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Severity Distribution (%)')
    st.pyplot(fig)

    # Bar chart for diseases per crop
    crop_counts = pd.Series([info['crop'] for info in DISEASE_INFO.values()]).value_counts()
    st.markdown("### Diseases per crop")
    st.bar_chart(crop_counts)

    # Severity levels by crop (stacked bar)
    crop_severity = {}
    for info in DISEASE_INFO.values():
        crop = info['crop']
        sev = info['severity']
        if crop not in crop_severity:
            crop_severity[crop] = {'healthy': 0, 'medium': 0, 'high': 0, 'critical': 0}
        crop_severity[crop][sev] += 1

    severity_df = pd.DataFrame.from_dict(crop_severity, orient='index').fillna(0)
    st.markdown("### Severity levels by crop")
    st.bar_chart(severity_df)

    note = "Use this section for rough risk guidance; always verify with local expert input."
    st.info(note)


