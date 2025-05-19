import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
import threading

import tempfile
tfile = tempfile.NamedTemporaryFile(delete=False)


# Load YOLOv8s model (once)
model = YOLO("yolov8s.pt")

PERSON_CLASS_ID = 0
ALERT_THRESHOLD = 40

st.title("Crowd Counting & Density Estimation")

# Density estimation functions
def estimate_density_map(frame_shape, boxes, grid_size=8):
    height, width = frame_shape[:2]
    grid = np.zeros((grid_size, grid_size))
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        row = min(cy * grid_size // height, grid_size - 1)
        col = min(cx * grid_size // width, grid_size - 1)
        grid[row, col] += 1
    return grid

def draw_density_map(frame, density_grid, alpha=0.5):
    if np.max(density_grid) == 0:
        return frame
    heatmap = cv2.applyColorMap(
        (density_grid * (255 / np.max(density_grid))).astype(np.uint8),
        cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

# Popup alert handler (runs in thread to avoid blocking UI)
alert_lock = threading.Lock()
alert_triggered = False

def alert_popup():
    global alert_triggered
    with alert_lock:
        if not alert_triggered:
            st.warning(f"🚨 Alert! Crowd count exceeded {ALERT_THRESHOLD}!")
            alert_triggered = True

def reset_alert():
    global alert_triggered
    with alert_lock:
        alert_triggered = False

# Process frame for detection + drawing
def process_frame(frame):
    results = model(frame)
    person_boxes = [box for box in results[0].boxes if int(box.cls[0]) == PERSON_CLASS_ID]
    crowd_count = len(person_boxes)

    # Draw bounding boxes
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw crowd count text
    cv2.putText(frame, f'Crowd Count: {crowd_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Density heatmap
    density_grid = estimate_density_map(frame.shape, person_boxes)
    frame = draw_density_map(frame, density_grid)

    # Trigger alert if crowd is high
    if crowd_count > ALERT_THRESHOLD:
        threading.Thread(target=alert_popup).start()
    else:
        reset_alert()

    return frame, crowd_count

# UI options
option = st.radio("Select Input Type:", ("Upload Image", "Upload Video", "Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        output_img, count = process_frame(img)
        st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), caption=f"People Count: {count}")

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_window = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output_frame, count = process_frame(frame)
            frame_window.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
        cap.release()

elif option == "Webcam":
    def video_frame_callback(frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        output_img, count = process_frame(img)
        return av.VideoFrame.from_ndarray(output_img, format="bgr24")

    webrtc_streamer(key="webcam", video_frame_callback=video_frame_callback)

