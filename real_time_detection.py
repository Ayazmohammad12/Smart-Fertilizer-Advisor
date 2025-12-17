import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image

# ================= CONFIGURATION =================
MODEL_PATH = "best.pt" 
CONF_THRESHOLD = 0.85 

DISEASE_DB = {
    "Tomato_Early_blight": {
        "title": "EARLY BLIGHT (Detected)",
        "cause": "Nutrient Imbalance",
        "fert": ["Balanced NPK (10-10-10)", "Epsom Salt (Magnesium)", "Vermicompost"],
        "tip": "Balanced nutrition reduces leaf stress and susceptibility."
    },
    "Tomato_Late_blight": {
        "title": "LATE BLIGHT (Fungal)",
        "cause": "Fungal Spread",
        "fert": ["Potassium-rich (5-10-10)", "Calcium Nitrate", "Potassium Sulphate"],
        "tip": "Potassium strengthens plant resistance and reduces fungal spread.",
        "avoid": "Excess Nitrogen (Urea)"
    },
    "Healthy": {
        "title": "VITAL SIGNS: HEALTHY",
        "cause": "Optimal Condition",
        "fert": ["Standard N-P-K Maintenance"],
        "tip": "Maintenance of soil health prevents future outbreaks."
    }
}

# ================= SYSTEM INITIALIZATION =================
@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except:
        return YOLO('yolov8n-cls.pt')

model = load_model()

st.set_page_config(page_title="AgriGuard AI", layout="wide")
st.title("🌱 Smart Fertilizer Advisor")
st.write("Take a photo of a tomato leaf to analyze its health.")

# ================= CAMERA INTERFACE =================
img_file = st.camera_input("Scan Leaf")

if img_file is not None:
    # Process Image
    image = Image.open(img_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Run AI
    results = model(frame, verbose=False)
    name = model.names[results[0].probs.top1]
    conf = float(results[0].probs.top1conf)

    # Display Results
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if conf > CONF_THRESHOLD and name in DISEASE_DB:
            info = DISEASE_DB[name]
            st.success(f"### {info['title']}")
            st.write(f"**Confidence:** {conf:.2%}")
            st.write(f"**Cause:** {info['cause']}")
            st.write("**Recommended Fertilizers:**")
            for f in info['fert']:
                st.write(f"- {f}")
            st.info(f"**Field Tip:** {info['tip']}")
        else:
            st.warning("Could not identify disease clearly. Please try a closer shot.")
