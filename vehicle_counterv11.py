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
        self.last_count_time = {}
        self.count_cooldown = 30
        self.center_history = defaultdict(list)
        self.history_size = 3

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

    def is_near_line(self, point, threshold=15):
        return self.calculate_distance_to_line(point) < threshold

    def get_smooth_center(self, track_id, current_center):
        self.center_history[track_id].append(current_center)
        if len(self.center_history[track_id]) > self.history_size:
            self.center_history[track_id].pop(0)
        return np.mean(self.center_history[track_id], axis=0)

    def check_line_crossing(self, track_id: int, bbox: tuple, class_name: str, frame_count: int) -> bool:
        if track_id in self.last_count_time:
            if frame_count - self.last_count_time[track_id] < self.count_cooldown:
                return False

        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        current_center = np.array([center_x, center_y])

        smooth_center = self.get_smooth_center(track_id, current_center)
        center = (int(smooth_center[0]), int(smooth_center[1]))

        if track_id in self.counted_objects:
            return False

        if not self.is_near_line(center):
            return False

        if track_id not in self.previous_centers:
            self.previous_centers[track_id] = center
            return False

        previous_center = np.array(self.previous_centers[track_id])

        min_movement = 10
        movement_distance = np.linalg.norm(smooth_center - previous_center)
        if movement_distance < min_movement:
            return False

        line_start, line_end = self.get_line_points()
        v1 = line_end - line_start
        v2_previous = previous_center - line_start
        v2_current = smooth_center - line_start

        previous_side = np.cross(v1, v2_previous)
        current_side = np.cross(v1, v2_current)

        if previous_side * current_side < 0:
            self.counted_objects.add(track_id)
            self.previous_centers[track_id] = center
            self.last_count_time[track_id] = frame_count
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

    def check_crossings(self, track_id: int, bbox: tuple, class_name: str, frame_count: int) -> List[int]:
        crossed_lines = []
        for line_number, counter in self.line_counters.items():
            if counter.check_line_crossing(track_id, bbox, class_name, frame_count):
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

def draw_title(frame: np.ndarray):
    title_text = "Augenio Datains Use Case Vehicle Counter"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(title_text, font, font_scale, thickness)

    padding = 5
    rect_x = 10
    rect_y = 10
    rect_width = text_width + (padding * 2)
    rect_height = text_height + (padding * 2)

    cv2.rectangle(frame,
                 (rect_x, rect_y),
                 (rect_x + rect_width, rect_y + rect_height),
                 (0, 0, 0),
                 -1)

    text_x = rect_x + padding
    text_y = rect_y + text_height + padding - 5
    cv2.putText(frame,
                title_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness)

def draw_label(frame, bbox, label, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    padding = 5
    rect_x = int(bbox[0])
    rect_y = int(bbox[1] - text_height - (padding * 2))
    rect_width = text_width + (padding * 2)
    rect_height = text_height + (padding * 2)

    cv2.rectangle(frame,
                 (rect_x, rect_y),
                 (rect_x + rect_width, rect_y + rect_height),
                 (0, 0, 0),
                 -1)

    text_x = rect_x + padding
    text_y = rect_y + text_height + padding - 2
    cv2.putText(frame,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness)

def add_logo(frame, logo_path):
    try:
        # Baca logo
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            print(f"Warning: Could not load logo from {logo_path}")
            return frame

        # Resize logo
        logo_width = 100
        logo_height = 100
        logo_resized = cv2.resize(logo, (logo_width, logo_height))

        y_offset = frame.shape[0] - logo_height - 10

        x_offset = frame.shape[1] - logo_width - 10

        # Region of interest pada frame
        roi = frame[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width]

        # Buat mask dari channel alpha logo
        alpha_channel = logo_resized[:, :, 3] / 255.0
        alpha_3channel = np.stack([alpha_channel] * 3, axis=-1)

        # Hitung komponen foreground dan background
        foreground = logo_resized[:, :, :3] * alpha_3channel
        background = roi * (1 - alpha_3channel)

        # Gabungkan foreground dan background
        result = foreground + background

        # Tempatkan hasil ke frame
        frame[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width] = result

    except Exception as e:
        print(f"Error adding logo: {str(e)}")

    return frame

def main():
    json_path = "british.json"
    video_path = "british_highway_traffic.mp4"
    model_path = "best2_11.pt"
    logo_path = "augenio.png"

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

        draw_title(frame)

        frame = add_logo(frame, logo_path)

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
            draw_label(frame, bbox, label, color)

            multi_counter.check_crossings(track_id, bbox, class_name, frame_count)

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
