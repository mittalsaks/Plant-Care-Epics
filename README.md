# 🌿 PlantCare AI — Streamlit App

A polished and user-friendly Streamlit app for plant disease detection and severity guidance. Trained on the PlantVillage dataset model `plantvillage_phase3_epoch25_FINAL.h5`.

---

## ✨ Features

- **Leaf Disease Prediction**
  - Upload or capture a leaf image
  - Support jpg, jpeg, png, webp, bmp
  - Model outputs disease class and confidence
- **Severity Tagging**
  - `healthy`, `medium`, `high`, `critical`
  - Advisor text for each severity level
- **Visual Explanation**
  - Grad-CAM heatmap overlay on leaf image
  - Side-by-side input image + attention map
- **Rich Dashboard UI**
  - Animated hero and gradient theme
  - Clean cards, badges, and hover animation
  - Smaller preview images for better UX
- **Severity Analytics Page**
  - Disease/crop selector and details
  - Severity distribution bar chart
  - Severity pie chart (with labels)
  - Diseases per crop bar chart
  - Severity-by-crop stacked bar chart
- **Improved UX**
  - No voice tab in main nav
  - Clear nav text and visible tab style
  - Full screen, responsive layout

---

## 🗂️ Repository Structure

```
PlantCare App/
├── app.py                # Main app entrypoint (nav)
├── pages/
│   ├── dashboard.py      # Main detection page
│   ├── severity.py       # Severity analytics page
│   ├── voice.py          # Optional voice page (not in main nav)
├── utils/
│   ├── theme.py          # CSS + theming
│   ├── translator.py     # Text dictionary + UI labels
├── plantvillage_phase3_epoch25_FINAL.h5 # Model file (required)
├── requirements.txt
└── README.md
```

---

## 📦 Install & Run

1. Clone repository:

   ```bash
git clone https://github.com/saumyadwiv/PlantCare-EPICS.git
cd "PlantCare App"
```

2. Create virtual environment and install:

   ```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Launch:

   ```bash
streamlit run app.py
```

---

## 🌾 Supported Crops & Diseases

- Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- Blueberry: Healthy
- Cherry: Powdery Mildew, Healthy
- Corn (Maize): Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- Grape: Black Rot, Esca, Leaf Blight, Healthy
- Orange: Citrus Greening (Huanglongbing)
- Peach: Bacterial Spot, Healthy
- Pepper (Bell): Bacterial Spot, Healthy
- Potato: Early Blight, Late Blight, Healthy
- Raspberry: Healthy
- Soybean: Healthy
- Squash: Powdery Mildew
- Strawberry: Leaf Scorch, Healthy
- Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

---

## 🛠 Model Notes

- Model architecture is based on a MobileNetV2-style head
- Input: RGB images resized to 224x224
- Prediction: softmax over 38 classes
- Class-to-severity maps inside `dashboard.py` constants

---

## 🧾 GitHub & Deploy

- Remote: `https://github.com/saumyadwiv/PlantCare-EPICS.git`
- Deployment URL: https://plantcare-epics-4rjmxmfzmhxikdytqd5f7q.streamlit.app/
- Recommended deploy: Streamlit Cloud 

---

## ⚠️ Disclaimer

For training or critical farm decisions, consult a local expert. This tool is for guidance and education only.
