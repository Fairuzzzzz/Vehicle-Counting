from ultralytics import YOLO
import cv2
import supervision as sv
import torch
import numpy as np
from collections import defaultdict

# Initialize model
model = YOLO('bestv8.pt')

# Initialize video capture
video_path = "british_highway_traffic.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
output_path = "output_british_with_counter_yolov8.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize tracker
tracker = sv.ByteTrack()

# Initialize counters
vehicle_counts = defaultdict(int)
active_tracks = defaultdict(set)  # To keep track of active vehicles by class

def custom_label(label: str, confidence: float | None, track_id: int | None) -> str:
    if confidence is not None and track_id is not None:
        return f"{label} #{track_id} ({confidence:.2f})"
    return label

# Initialize annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Create function to draw counter
def draw_counter(frame, counts):
    y_position = 30
    for vehicle_type, count in counts.items():
        text = f"{vehicle_type}: {count}"
        cv2.putText(
            frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        y_position += 40

# Function to convert YOLOv8 results to supervision Detections
def yolo_to_supervision(results):
    result = results[0]  # Get the first result since we process one frame at a time

    # Extract boxes, confidence scores, and class ids
    xyxy = result.boxes.xyxy.cpu().numpy()
    confidence = result.boxes.conf.cpu().numpy()
    class_id = result.boxes.cls.cpu().numpy().astype(int)  # Convert to integer

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id
    )

frame_count = 0
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
no_detection_frames = 0  # Counter for frames without detections
RESET_THRESHOLD = 5  # Number of frames without detections before resetting counter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        results = model(frame)

    # Convert detections to supervision format using our custom function
    detections = yolo_to_supervision(results)

    # Track detections
    detections = tracker.update_with_detections(detections)

    # Check if there are any detections
    if len(detections) == 0:
        no_detection_frames += 1
        if no_detection_frames >= RESET_THRESHOLD:
            # Reset all counters and active tracks
            vehicle_counts.clear()
            active_tracks.clear()
            no_detection_frames = 0
    else:
        no_detection_frames = 0  # Reset the no detection counter
        # Update vehicle counts
        current_tracks = defaultdict(set)

        for class_id, track_id in zip(detections.class_id, detections.tracker_id):
            vehicle_type = model.names[int(class_id)]
            if track_id is not None:  # Only count if tracking ID exists
                current_tracks[vehicle_type].add(track_id)

        # Update counts based on current tracks
        for vehicle_type in current_tracks:
            # Add new tracks
            new_tracks = current_tracks[vehicle_type] - active_tracks[vehicle_type]
            vehicle_counts[vehicle_type] += len(new_tracks)

            # Remove disappeared tracks
            disappeared_tracks = active_tracks[vehicle_type] - current_tracks[vehicle_type]
            vehicle_counts[vehicle_type] -= len(disappeared_tracks)

            # Update active tracks
            active_tracks[vehicle_type] = current_tracks[vehicle_type]

    # Create labels with tracking IDs
    labels = [
        custom_label(
            model.names[int(class_id)],
            float(confidence),
            int(track_id) if track_id is not None else None
        )
        for class_id, confidence, track_id in zip(
            detections.class_id,
            detections.confidence,
            detections.tracker_id
        )
    ]

    # Annotate frame
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # Draw counter
    draw_counter(annotated_frame, vehicle_counts)

    # Write frame
    out.write(annotated_frame)

    # Update progress
    frame_count += 1
    progress = (frame_count / total_frame) * 100
    print(f"\rProcessing: {progress:.2f}% complete", end="")

cap.release()
out.release()
print(f"\nProcessed video saved as {output_path}")
