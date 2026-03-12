import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("yolov8n.pt")

st.title("Vehicle Detection App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)

    results = model(img)

    annotated = results[0].plot()

    st.image(annotated, caption="Detected Vehicles")
