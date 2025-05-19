from ultralytics import YOLO
import cv2
import numpy as np

# Load the pretrained YOLOv8s model
model = YOLO("yolov8s.pt")

# COCO class index for 'person'
PERSON_CLASS_ID = 0

# Load video
video_path = 'C:/Users/Reshan H/Downloads/6387-191695740_medium.mp4'
cap = cv2.VideoCapture(video_path)

# Get original video FPS
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)  # milliseconds between frames

# Optional: Save output
save_output = True
output_path = 'output_crowd_count.mp4'
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Get boxes where class is 'person'
    person_boxes = [box for box in results[0].boxes if int(box.cls[0]) == PERSON_CLASS_ID]
    crowd_count = len(person_boxes)

    # Draw detections
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show crowd count
    cv2.putText(frame, f'Crowd Count: {crowd_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Crowd Counting", frame)

    # Save frame
    if save_output:
        out.write(frame)

    # Press 'q' to quit or delay to match original FPS
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
