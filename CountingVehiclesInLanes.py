import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

class VehicleCounter:
    def __init__(self, video_path, model_path, classes_path):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.classnames = self.load_classes(classes_path)
        self.tracker = Sort()
        self.zones = self.initialize_zones()
        self.counters = {zone: [] for zone in self.zones}

    @staticmethod
    def load_classes(path):
        with open(path, 'r') as f:
            return f.read().splitlines()

    @staticmethod
    def initialize_zones():
        return {
            'A': {'polygon': np.array([[308, 789], [711, 807], [694, 492], [415, 492], [309, 790]], np.int32),
                  'line': np.array([[308, 789], [711, 807]]).reshape(-1),
                  'color': (0, 0, 255)},
            'B': {'polygon': np.array([[727, 797], [1123, 812], [1001, 516], [741, 525], [730, 795]], np.int32),
                  'line': np.array([[727, 797], [1123, 812]]).reshape(-1),
                  'color': (0, 255, 255)},
            'C': {'polygon': np.array([[1116, 701], [1533, 581], [1236, 367], [1009, 442], [1122, 698]], np.int32),
                  'line': np.array([[1116, 701], [1533, 581]]).reshape(-1),
                  'color': (255, 0, 0)}
        }

    def process_frame(self, frame):
        frame = cv2.resize(frame, (1920, 1080))
        results = self.model(frame)
        detections = self.get_detections(results)
        self.draw_zones(frame)
        self.track_and_count(frame, detections)
        self.display_counts(frame)
        return frame

    def get_detections(self, results):
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if self.classnames[cls] in ['car', 'truck', 'bus'] and conf > 0.6:
                    detections.append([x1, y1, x2, y2, conf])
        return np.array(detections)

    def draw_zones(self, frame):
        for zone in self.zones.values():
            cv2.polylines(frame, [zone['polygon']], isClosed=False, color=zone['color'], thickness=8)

    def track_and_count(self, frame, detections):
        tracks = self.tracker.update(detections)
        for x1, y1, x2, y2, track_id in tracks:
            x1, y1, x2, y2, track_id = map(int, [x1, y1, x2, y2, track_id])
            center = (x1 + x2) // 2, (y1 + y2) // 2 - 40
            for zone, data in self.zones.items():
                if self.point_in_line(center, data['line']):
                    if track_id not in self.counters[zone]:
                        self.counters[zone].append(track_id)

    @staticmethod
    def point_in_line(point, line, tolerance=20):
        x, y = point
        x1, y1, x2, y2 = line
        return x1 <= x <= x2 and abs(y - y1) <= tolerance

    def display_counts(self, frame):
        y_offset = 90
        for zone, count in self.counters.items():
            cv2.circle(frame, (970, y_offset), 15, self.zones[zone]['color'], -1)
            cv2.putText(frame, f'LANE {zone} Vehicles = {len(count)}', (1000, y_offset + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 40

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            cv2.imshow('Vehicle Counter', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    counter = VehicleCounter('path_to_video.mp4', 'yolov8n.pt', 'classes.txt')
    counter.run()