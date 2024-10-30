import json
import yolov5
import cv2
import supervision as sv
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from collections import defaultdict

@dataclass(frozen=True)
class LineCounterConfig:
    start_point: tuple
    end_point: tuple
    line_number: int
    line_color: tuple = (255, 0, 0)
    line_thickness: int = 2

class LineCounter:
    def __init__(self, config: LineCounterConfig):
        self.config = config
        self.vehicle_count = 0
        self.previous_centers = {}
        self.counted_objects = set()  # Track objects already counted for this line

    def create_line(self, frame):
        # Draw the line
        cv2.line(
            frame,
            self.config.start_point,
            self.config.end_point,
            self.config.line_color,
            self.config.line_thickness
        )

        # Add line number label
        label_x = (self.config.start_point[0] + self.config.end_point[0]) // 2
        label_y = (self.config.start_point[1] + self.config.end_point[1]) // 2
        cv2.putText(
            frame,
            f"Line {self.config.line_number}",
            (label_x, label_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.config.line_color,
            2
        )

    def get_line_points(self):
        return np.array(self.config.start_point), np.array(self.config.end_point)

    def calculate_distance_to_line(self, point):
        """Calculate the distance from a point to the line"""
        line_start, line_end = self.get_line_points()

        # Line vector
        line_vec = line_end - line_start

        # Vector from line start to point
        point_vec = np.array(point) - line_start

        # Calculate distance
        line_length = np.linalg.norm(line_vec)
        if line_length == 0:
            return np.linalg.norm(point_vec)

        # Project point_vec onto line_vec
        unit_line_vec = line_vec / line_length
        projection_length = np.dot(point_vec, unit_line_vec)

        # Calculate perpendicular distance
        projection = line_start + unit_line_vec * projection_length
        distance = np.linalg.norm(np.array(point) - projection)

        return distance

    def is_near_line(self, point, threshold=30):
        """Check if a point is near the line"""
        return self.calculate_distance_to_line(point) < threshold

    def check_line_crossing(self, detection_id: int, bbox: tuple, class_id: int, confidence: float) -> bool:
        """
        Check if an object crosses the line, ensuring each object is counted only once
        Args:
            detection_id: unique identifier for the detection
            bbox: bounding box coordinates (x1, y1, x2, y2)
            class_id: class identifier of the detected object
            confidence: detection confidence score
        """
        # Calculate center point of bounding box
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        center = (center_x, center_y)

        # Create unique object identifier combining multiple attributes
        object_id = f"{detection_id}_{class_id}_{int(confidence * 100)}"
        track_id = f"{self.config.line_number}_{object_id}"

        # If object is already counted for this line, skip it
        if object_id in self.counted_objects:
            return False

        # Check if object is near the line
        if not self.is_near_line(center):
            return False

        if track_id not in self.previous_centers:
            self.previous_centers[track_id] = center
            return False

        previous_center = self.previous_centers[track_id]
        current_center = np.array(center)
        line_start, line_end = self.get_line_points()

        v1 = line_end - line_start
        v2_previous = np.array(previous_center) - line_start
        v2_current = current_center - line_start

        previous_side = np.cross(v1, v2_previous)
        current_side = np.cross(v1, v2_current)

        # If line is crossed
        if previous_side * current_side < 0:
            self.counted_objects.add(object_id)  # Mark object as counted
            self.previous_centers[track_id] = center
            return True

        self.previous_centers[track_id] = center
        return False

class MultiLineCounter:
    def __init__(self, line_configs: List[LineCounterConfig]):
        self.line_counters = {
            config.line_number: LineCounter(config)
            for config in line_configs
        }

    def create_lines(self, frame):
        for counter in self.line_counters.values():
            counter.create_line(frame)

    def check_crossings(self, detection_id: int, bbox: tuple, class_id: int, confidence: float) -> List[int]:
        crossed_lines = []
        for line_number, counter in self.line_counters.items():
            if counter.check_line_crossing(detection_id, bbox, class_id, confidence):
                crossed_lines.append(line_number)
        return crossed_lines

    def get_counts(self) -> Dict[int, int]:
        return {
            line_number: counter.vehicle_count
            for line_number, counter in self.line_counters.items()
        }

def load_line_coordinates(json_path: str, video_width: int, video_height: int) -> List[LineCounterConfig]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get original image dimensions from JSON
    orig_width = data['image_width']
    orig_height = data['image_height']

    # Calculate scaling factors
    scale_x = video_width / orig_width
    scale_y = video_height / orig_height

    configs = []
    for line in data['lines']:
        # Scale coordinates to video dimensions
        start_x = int(line['start_point'][0] * scale_x)
        start_y = int(line['start_point'][1] * scale_y)
        end_x = int(line['end_point'][0] * scale_x)
        end_y = int(line['end_point'][1] * scale_y)

        config = LineCounterConfig(
            start_point=(start_x, start_y),
            end_point=(end_x, end_y),
            line_number=line['line_number']
        )
        configs.append(config)
    return configs

def draw_counts(frame: np.ndarray, counts: Dict[int, int]):
    y_offset = 40
    for line_number, count in sorted(counts.items()):
        cv2.putText(
            frame,
            f"Line {line_number} Count: {count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        y_offset += 40

def main():
    json_path = "british_line.json" # Update data json sesuai dengan data mu
    video_path = "british_highway_traffic.mp4"  # Update video path sesuai dengan video mu
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Load and scale line coordinates to video dimensions
    line_configs = load_line_coordinates(json_path, width, height)

    # Initialize multi-line counter
    multi_counter = MultiLineCounter(line_configs)

    # Load model
    model = yolov5.load('best2.pt')

    # Output video setup
    output_path = "output_with_counter.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw counting lines
        multi_counter.create_lines(frame)

        # Detect objects
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            results = model(frame)

        detections = sv.Detections.from_yolov5(results)

        # Process each detection
        for i, (box, class_id, confidence) in enumerate(zip(detections.xyxy, detections.class_id, detections.confidence)):
            # Check all line crossings
            crossed_lines = multi_counter.check_crossings(i, box, int(class_id), confidence)
            for line_number in crossed_lines:
                multi_counter.line_counters[line_number].vehicle_count += 1

        # Create labels with confidence
        labels = [
            f"{model.names[int(class_id)]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Draw counts for all lines
        draw_counts(frame, multi_counter.get_counts())

        # Annotate frame
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        out.write(annotated_frame)

        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\rProcessing: {progress:.2f}% complete", end="")

    cap.release()
    out.release()

    print("\nFinal counts for each line:")
    for line_number, count in multi_counter.get_counts().items():
        print(f"Line {line_number}: {count} vehicles")

    print(f"\nProcessed video saved as {output_path}")

if __name__ == "__main__":
    main()
