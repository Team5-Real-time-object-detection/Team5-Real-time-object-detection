import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.title("Settings")

model_choice = st.sidebar.selectbox(
    "Select YOLO Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.25, 0.05
)

upload_option = st.sidebar.radio(
    "Choose Input Type",
    ["Upload Image/Video", "Use Webcam"]
)

st.sidebar.markdown("— Developed by George Badea")

# -----------------------------
# Main App UI
# -----------------------------
st.title("YOLO Object Detection Streamlit App")
st.write(
    "Upload an image or video, or use your webcam to perform real-time object detection "
    "using Ultralytics YOLO."
)

# Load YOLO model
@st.cache_resource
def load_yolo(model_path):
    return YOLO(model_path)

model = load_yolo(model_choice)

# -----------------------------
# Helper: Run YOLO on Frames
# -----------------------------
def process_frame(frame, conf):
    results = model(frame, conf=conf)
    annotated = results[0].plot()
    return annotated

# -----------------------------
# Image / Video Upload Option
# -----------------------------
if upload_option == "Upload Image/Video":
    uploaded_file = st.file_uploader(
        "Upload an image or video file",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type

        # -----------------------------
        # Image Processing
        # -----------------------------
        if file_type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            processed = process_frame(img_rgb, confidence_threshold)

            st.subheader("Processed Image")
            st.image(processed, channels="RGB")

        # -----------------------------
        # Video Processing
        # -----------------------------
        elif file_type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            vid_path = tfile.name

            st.video(vid_path)
            run = st.button("Run Detection")

            if run:
                st.subheader("Processing Video...")

                cap = cv2.VideoCapture(vid_path)
                frame_container = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed = process_frame(frame, confidence_threshold)

                    frame_container.image(processed, channels="RGB")

                cap.release()
                st.success("Video processing completed.")

# -----------------------------
# Webcam Option
# -----------------------------
elif upload_option == "Use Webcam":
    st.subheader("Webcam Detection")
    run_webcam = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    if run_webcam:
        cap = cv2.VideoCapture(0)

        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not access webcam.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = process_frame(frame, confidence_threshold)
            FRAME_WINDOW.image(processed, channels="RGB")

        cap.release()
