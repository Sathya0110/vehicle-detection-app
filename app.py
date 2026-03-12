import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Vehicle Detection", page_icon="🚗", layout="wide")

st.title("🚗 Smart Vehicle Detection System")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model(np.array(image), conf=0.3)

    annotated = results[0].plot()

    st.image(annotated, caption="Detected Vehicles", use_container_width=True)

    vehicles = ["car","bus","truck","motorcycle"]

    count = 0

    for box in results[0].boxes:
        cls = int(box.cls)
        label = model.names[cls]

        if label in vehicles:
            count += 1

    st.success(f"🚘 Total Vehicles Detected: {count}")
