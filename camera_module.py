# camera_module.py

import streamlit as st
import cv2
from PIL import Image
import numpy as np


# ──────────────────────────────────────────────────────────────
# 📸 SIMPLE CAMERA (Recommended)
# ──────────────────────────────────────────────────────────────
def camera_input_image():
    """
    Streamlit built-in camera (Best option)
    Works on mobile + cloud
    """

    st.markdown("### 📸 Capture Leaf Photo")

    img_file_buffer = st.camera_input("Take a picture of the leaf")

    if img_file_buffer is not None:
        try:
            image = Image.open(img_file_buffer).convert("RGB")
            return image
        except Exception as e:
            st.error(f"❌ Error reading image: {e}")
            return None

    return None


# ──────────────────────────────────────────────────────────────
# 🎥 LIVE WEBCAM (Optional - Advanced)
# ──────────────────────────────────────────────────────────────
def live_webcam_stream():
    """
    OpenCV webcam stream (only works locally)
    """

    st.markdown("### 🎥 Live Camera Stream")

    start = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    captured_image = None

    if start:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
            return None

        while start:
            ret, frame = cap.read()

            if not ret:
                st.error("❌ Failed to read from camera")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

            if st.button("📸 Capture Frame"):
                captured_image = Image.fromarray(frame)
                break

        cap.release()

    return captured_image


# ──────────────────────────────────────────────────────────────
# 🔄 MAIN WRAPPER FUNCTION
# ──────────────────────────────────────────────────────────────
def get_camera_image(mode="simple"):
    """
    mode = "simple" → recommended
    mode = "live"   → advanced webcam
    """

    if mode == "live":
        return live_webcam_stream()
    else:
        return camera_input_image()