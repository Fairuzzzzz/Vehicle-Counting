import json
import cv2
import numpy as np
from ultralytics import YOLO
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

class BoundingBoxTracker:
    def __init__(self):
        self.next_id = 0
        self.tracked_objects = {}
        self.max_disappeared = 30

    def update(self, results, frame_count):
        current_objects = {}

        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls)
                confidence = float(box.conf)
                if confidence < 0.4:
                    continue

                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                matched = False

                for track_id, (old_bbox, old_class, last_seen) in self.tracked_objects.items():
                    old_center = ((old_bbox[0] + old_bbox[2]) / 2, (old_bbox[1] + old_bbox[3]) / 2)
                    distance = np.sqrt((center[0] - old_center[0])**2 + (center[1] - old_center[1])**2)

                    if distance < 50 and old_class == class_id:
                        current_objects[track_id] = (bbox, class_id, frame_count)
                        matched = True
                        break

                if not matched:
                    current_objects[self.next_id] = (bbox, class_id, frame_count)
                    self.next_id += 1

        self.tracked_objects = {
            track_id: obj_data
            for track_id, obj_data in current_objects.items()
            if frame_count - obj_data[2] <= self.max_disappeared
        }

        return self.tracked_objects

class LineCounter:
    def __init__(self, config: LineCounterConfig):
        self.config = config
        self.vehicle_count = 0
        self.previous_centers = {}
        self.counted_objects = set()

    def create_line(self, frame):
        cv2.line(
            frame,
            self.config.start_point,
            self.config.end_point,
            self.config.line_color,
            self.config.line_thickness
        )

    def get_line_points(self):
        return np.array(self.config.start_point), np.array(self.config.end_point)

    def calculate_distance_to_line(self, point):
        line_start, line_end = self.get_line_points()
        line_vec = line_end - line_start
        point_vec = np.array(point) - line_start

        line_length = np.linalg.norm(line_vec)
        if line_length == 0:
            return np.linalg.norm(point_vec)

        unit_line_vec = line_vec / line_length
        projection_length = np.dot(point_vec, unit_line_vec)
        projection = line_start + unit_line_vec * projection_length
        distance = np.linalg.norm(np.array(point) - projection)

        return distance

    def is_near_line(self, point, threshold=30):
        return self.calculate_distance_to_line(point) < threshold

    def check_line_crossing(self, track_id: int, bbox: tuple, class_name: str) -> bool:
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        center = (center_x, center_y)

        if track_id in self.counted_objects:
            return False

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

        if previous_side * current_side < 0:
            self.counted_objects.add(track_id)
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
        self.counts = defaultdict(int)

    def create_lines(self, frame):
        for counter in self.line_counters.values():
            counter.create_line(frame)

    def check_crossings(self, track_id: int, bbox: tuple, class_name: str) -> List[int]:
        crossed_lines = []
        for line_number, counter in self.line_counters.items():
            if counter.check_line_crossing(track_id, bbox, class_name):
                crossed_lines.append(line_number)
                self.counts[class_name] += 1
        return crossed_lines

    def get_counts(self) -> Dict[str, int]:
        return dict(self.counts)

def load_line_coordinates(json_path: str, video_width: int, video_height: int) -> List[LineCounterConfig]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    orig_width = data['image_width']
    orig_height = data['image_height']

    scale_x = video_width / orig_width
    scale_y = video_height / orig_height

    configs = []
    for line in data['lines']:
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

def draw_counts(frame: np.ndarray, counts: Dict[str, int]):
    total_height = frame.shape[0]
    x_offset = 20
    y_position = total_height - 30

    count_text = " | ".join([f"{class_name}: {count:02d}" for class_name, count in counts.items()])
    cv2.putText(
        frame,
        count_text,
        (x_offset, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

def main():
    json_path = "british.json"
    video_path = "british_highway_traffic.mp4"
    model_path = "bestv11.pt"

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    line_configs = load_line_coordinates(json_path, width, height)
    multi_counter = MultiLineCounter(line_configs)
    tracker = BoundingBoxTracker()

    model = YOLO(model_path)
    class_names = model.names

    output_path = "output_british_with_counter_11.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    class_colors = {
        'car': (0, 0, 255),
        'motorcycle': (0, 255, 0),
        'truck': (255, 0, 0),
        'bus': (255, 255, 0)
    }

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        multi_counter.create_lines(frame)
        results = model(frame, conf=0.5)
        tracked_objects = tracker.update(results, frame_count)

        for track_id, (bbox, class_id, _) in tracked_objects.items():
            class_name = class_names[int(class_id)]
            color = class_colors.get(class_name, (255, 255, 255))

            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2
            )

            label = f"{class_name}-{track_id}"
            cv2.putText(
                frame,
                label,
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            multi_counter.check_crossings(track_id, bbox, class_name)

        draw_counts(frame, multi_counter.get_counts())
        out.write(frame)

        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\rProcessing: {progress:.2f}% complete", end="")

    cap.release()
    out.release()

    print("\nFinal counts:")
    for class_name, count in multi_counter.get_counts().items():
        print(f"{class_name}: {count} vehicles")

    print(f"\nProcessed video saved as {output_path}")

if __name__ == "__main__":
    main()
