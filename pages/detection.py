import streamlit as st
from PIL import Image

def run():
    st.header("🌿 Disease Detection")

    uploaded = st.file_uploader("Upload leaf image", type=["jpg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image)

        from core.model import load_model
        from core.preprocess import preprocess
        from core.gradcam import make_gradcam, overlay_gradcam
        import numpy as np

        # Load model
        model = load_model()

        # Preprocess image
        img_arr = preprocess(image)

        # Prediction
        preds = model.predict(img_arr, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx]) * 100

        # Class mapping
        CLASS_NAMES = list(DISEASE_INFO.keys())
        pred_class = CLASS_NAMES[pred_idx]
        info = DISEASE_INFO.get(pred_class)

        # GradCAM
        heatmap = make_gradcam(img_arr, model)

        if heatmap is None:
            overlay_img = image
        else:
            overlay_img = overlay_gradcam(image, heatmap)

        # Show results
        st.image(overlay_img, caption="Disease focus area")

        st.success(f"Prediction: {info['disease']}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.write("### Description")
        st.write(info["what"])