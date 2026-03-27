# 🌿 PlantGuard AI — Streamlit App

A full-featured, production-grade Streamlit application for plant disease detection using your MobileNetV2 model trained on the PlantVillage dataset.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 AI Diagnosis | Upload any leaf image → get instant disease prediction |
| 📊 Confidence Scores | Visual confidence bars for top-K predictions |
| 🔥 Grad-CAM | Visualize what the model focuses on |
| 💊 Treatment Info | Disease-specific treatment & prevention advice |
| 📋 Disease Library | Browse all 38 detectable classes with filtering |
| 📈 Session History | Track all analyses, export as CSV |
| ⚙️ Settings Panel | Threshold control, top-K selector, Grad-CAM toggle |

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Load your model
- In the sidebar, enter the path to your `.h5` model file
- Example: `/path/to/plantvillage_phase3_epoch25_FINAL.h5`
- Click **Load Model**

---

## 🗂️ File Structure

```
plant_disease_app/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🌾 Supported Plants & Diseases (38 Classes)

| Plant | Conditions |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn/Maize | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Orange | Huanglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Bell Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## ⚙️ Model Architecture

- **Base**: MobileNetV2 (ImageNet pretrained)
- **Head**: GlobalAveragePooling2D → Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.5) → Dense(128) → Dense(38, softmax)
- **Input**: 224×224 RGB images, normalized to [0, 1]
- **Training**: 3-phase fine-tuning on PlantVillage dataset

---

## 🖥️ Deployment Options

### Local
```bash
streamlit run app.py
```

### Deployed Link
https://plantcare-epics-4rjmxmfzmhxikdytqd5f7q.streamlit.app/

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

---

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. For critical agricultural decisions, always consult a certified agronomist or plant pathologist.
