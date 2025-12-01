import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# -------------------------
# MODELS
# -------------------------

# Image classification model (simple)
import tensorflow as tf
classification_model = tf.keras.applications.MobileNetV2(weights="imagenet")

# YOLO model for object detection
yolo = YOLO("yolov8n.pt")   # auto-downloads and loads model


# -------------------------
# FUNCTIONS
# -------------------------

# 1. Image Classification
def classify_image(img):
    if img is None:
        return "No image provided"

    img_resized = cv2.resize(img, (224, 224))
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)

    preds = classification_model.predict(img_resized)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)
    label = decoded[0][0][1]
    confidence = float(decoded[0][0][2])

    return f"{label} ({confidence*100:.2f}%)"


# 2. Grayscale Image
def grayscale(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


# 3. Image Filter (Blur)
def blur_image(img):
    if img is None:
        return None
    blur = cv2.GaussianBlur(img, (25, 25), 0)
    return blur


# 4. Object Detection — FIXED VERSION
def detect_objects(img):
    if img is None:
        return None

    results = yolo(img)
    result_img = results[0].plot()   # draw bounding boxes

    return result_img


# -------------------------
# UI SETUP
# -------------------------

with gr.Blocks(title="Image Projects Collection") as demo:
    gr.Markdown("## 📸 Image Projects Collection")

    with gr.Tabs():

        # ---------------- IMAGE CLASSIFICATION TAB ----------------
        with gr.Tab("Image Classification"):
            gr.Markdown("### Upload an image to classify")

            input_img = gr.Image(type="numpy")
            output_label = gr.Textbox(label="Prediction")

            classify_btn = gr.Button("Submit")
            classify_btn.click(classify_image, input_img, output_label)

        # ---------------- GRAYSCALE TAB ----------------
        with gr.Tab("Grayscale Editing"):
            gr.Markdown("### Convert Image to Grayscale")

            gray_in = gr.Image(type="numpy")
            gray_out = gr.Image(type="numpy")

            gray_btn = gr.Button("Convert")
            gray_btn.click(grayscale, gray_in, gray_out)

        # ---------------- IMAGE FILTERS TAB ----------------
        with gr.Tab("Image Filters"):
            gr.Markdown("### Apply Blur Filter")

            blur_in = gr.Image(type="numpy")
            blur_out = gr.Image(type="numpy")

            blur_btn = gr.Button("Blur")
            blur_btn.click(blur_image, blur_in, blur_out)

        # ---------------- OBJECT DETECTION TAB ----------------
        with gr.Tab("Object Detection"):
            gr.Markdown("### Detect Objects in Image (YOLOv8)")

            od_in = gr.Image(type="numpy", sources=["upload", "webcam"])
            od_out = gr.Image(type="numpy")

            od_btn = gr.Button("Submit")
            od_btn.click(detect_objects, od_in, od_out)

demo.launch()
