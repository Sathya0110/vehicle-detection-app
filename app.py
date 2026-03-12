import gradio as gr
from ultralytics import YOLO
import numpy as np

# Load model
model = YOLO("yolov8n.pt")

def detect_vehicle(image):

    results = model(image)

    annotated_image = results[0].plot()

    # Count vehicles
    vehicles = ["car", "truck", "bus", "motorcycle"]
    count = 0
    detected = []

    for box in results[0].boxes:
        cls = int(box.cls)
        label = model.names[cls]
        conf = float(box.conf)

        if label in vehicles:
            count += 1
            detected.append(f"{label} ({conf:.2f})")

    return annotated_image, count, ", ".join(detected)


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🚗 Smart Vehicle Detection System")
    gr.Markdown("Upload a traffic image to detect and count vehicles using YOLOv8")

    with gr.Row():

        image_input = gr.Image(type="numpy", label="Upload Image")

        image_output = gr.Image(label="Detection Result")

    detect_btn = gr.Button("Detect Vehicles")

    with gr.Row():

        vehicle_count = gr.Number(label="Total Vehicles Detected")

        vehicle_list = gr.Textbox(label="Detected Vehicle Types")

    detect_btn.click(
        fn=detect_vehicle,
        inputs=image_input,
        outputs=[image_output, vehicle_count, vehicle_list]
    )

demo.launch()
