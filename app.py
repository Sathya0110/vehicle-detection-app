import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Smart Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

# Title
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚗 Smart Vehicle Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a traffic image to detect vehicles using YOLOv8</p>", unsafe_allow_html=True)

# Load model
model = YOLO("yolov8n.pt")

# Layout
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Upload Traffic Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(np.array(image))
    annotated = results[0].plot()

    with col2:
        st.image(annotated, caption="Detected Vehicles", use_column_width=True)

    vehicles = ["car","bus","truck","motorcycle"]

    count = 0
    detected_types = []

    for box in results[0].boxes:
        cls = int(box.cls)
        label = model.names[cls]

        if label in vehicles:
            count += 1
            detected_types.append(label)

    st.divider()

    m1, m2 = st.columns(2)

    with m1:
        st.metric("🚘 Total Vehicles Detected", count)

    with m2:
        st.write("### Detected Vehicle Types")
        st.write(list(set(detected_types)))
