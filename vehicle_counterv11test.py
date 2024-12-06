import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from mss import mss

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
        self.object_classes = {}
        self.counts = defaultdict(int)

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
            self.object_classes[track_id] = class_name
            self.counts[class_name] += 1
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

class ScreenCapture:
    def __init__(self, monitor_number=1):
        self.sct = mss()
        self.monitor = self.sct.monitors[monitor_number]

    def capture_screen(self):
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

class LineDrawer:
    def __init__(self):
        self.drawing = False
        self.lines = []
        self.start_point = None
        self.temp_end_point = None
        self.current_line_number = 1

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp_end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                config = LineCounterConfig(
                    start_point=self.start_point,
                    end_point=(x, y),
                    line_number=self.current_line_number
                )
                self.lines.append(config)
                self.current_line_number += 1
                self.start_point = None
                self.temp_end_point = None

def draw_counts(frame: np.ndarray, line_counters: Dict[int, LineCounter]):
    total_height = frame.shape[0]
    x_offset = 20
    y_position = total_height - 60

    for line_number, counter in sorted(line_counters.items()):
        counts = defaultdict(int)
        for obj_id in counter.counted_objects:
            class_name = counter.object_classes.get(obj_id, "unknown")
            counts[class_name] += 1

        count_text = f"Line {line_number}: " + " | ".join([
            f"{class_name}: {count:02d}"
            for class_name, count in counts.items()
        ])

        cv2.putText(
            frame,
            count_text,
            (x_offset, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y_position += 30

def main():
    model_path = "best2_11.pt"  # Sesuaikan dengan path model Anda
    model = YOLO(model_path)
    screen_capture = ScreenCapture()
    line_drawer = LineDrawer()
    tracker = BoundingBoxTracker()

    cv2.namedWindow('Screen Capture')
    cv2.setMouseCallback('Screen Capture', line_drawer.mouse_callback)

    class_colors = {
        'car': (0, 0, 255),
        'motorcycle': (0, 255, 0),
        'truck': (255, 0, 0),
        'bus': (255, 255, 0)
    }

    frame_count = 0
    multi_counter = None

    print("Controls:")
    print("- Left click and drag to draw counting lines")
    print("- Press 'c' to clear all lines")
    print("- Press 'q' to quit")

    while True:
        frame = screen_capture.capture_screen()

        # Update multi_counter jika ada garis baru
        if len(line_drawer.lines) > 0 and (multi_counter is None or len(multi_counter.line_counters) != len(line_drawer.lines)):
            multi_counter = MultiLineCounter(line_drawer.lines)

        # Deteksi objek dan tracking
        if multi_counter:
            results = model(frame, conf=0.5)
            tracked_objects = tracker.update(results, frame_count)

            # Gambar garis counting
            multi_counter.create_lines(frame)

            # Gambar label nomor garis
            for config in line_drawer.lines:
                mid_x = (config.start_point[0] + config.end_point[0]) // 2
                mid_y = (config.start_point[1] + config.end_point[1]) // 2
                cv2.putText(
                    frame,
                    f"Line {config.line_number}",
                    (mid_x - 20, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            # Proses deteksi dan tracking
            for track_id, (bbox, class_id, _) in tracked_objects.items():
                class_name = model.names[int(class_id)]
                color = class_colors.get(class_name, (255, 255, 255))

                # Gambar bounding box
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    2
                )

                # Gambar label kelas
                label = f"{class_name}"
                cv2.putText(
                    frame,
                    label,
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                multi_counter.check_crossings(track_id, bbox, class_name, frame_count)

            # Tampilkan counting
            draw_counts(frame, multi_counter.line_counters)

        # Gambar garis yang sedang dibuat
        if line_drawer.drawing and line_drawer.start_point and line_drawer.temp_end_point:
            cv2.line(frame, line_drawer.start_point, line_drawer.temp_end_point, (255, 0, 0), 2)

        # Tampilkan frame
        cv2.imshow('Screen Capture', frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            line_drawer.lines = []
            multi_counter = None
            line_drawer.current_line_number = 1

        frame_count += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
