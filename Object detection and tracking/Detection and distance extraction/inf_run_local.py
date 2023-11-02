from collections import defaultdict
from time import time
import cv2
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from ultralytics import YOLO
import json

class ObjectDetector:
    def __init__(self, weights_path, classNames):
        self.model = YOLO(weights_path)
        self.classNames = classNames

    def detect_objects(self, frame):
        results = self.model(frame)
        detected_objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = round(box.conf[0].item(), 2)
                cls = int(box.cls[0])
                object_class = self.classNames[cls] if 0 <= cls < len(self.classNames) else "Unknown"
                detected_objects.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': conf, 'class': object_class,
                })
        return detected_objects

class DepthEstimator:
    def __init__(self):
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()

    def calculate_depth_map(self, frame):
        transform = T.Compose([T.Resize(384), T.ToTensor()])
        input_image = transform(Image.fromarray(frame)).unsqueeze(0)
        with torch.no_grad():
            prediction = self.midas(input_image)
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        scaling_factor = 0.01
        depth_map = depth_map * scaling_factor
        return depth_map

class FrameProcessor:
    def __init__(self, camera_center_x, camera_center_y, focal_length, object_detector, depth_estimator):
        self.camera_center_x = camera_center_x
        self.camera_center_y = camera_center_y
        self.focal_length = focal_length
        self.object_detector = object_detector
        self.depth_estimator = depth_estimator

    def process_frame(self, frame):
        depth_map = self.depth_estimator.calculate_depth_map(frame)
        detected_objects = self.object_detector.detect_objects(frame)
        processed_frame = self.draw_objects(frame, detected_objects, depth_map)
        return processed_frame

    def draw_objects(self, frame, detected_objects, depth_map):
        # Drawing code for objects and depth lines
        # ... (your existing code for drawing objects and depth lines)
        return frame

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        if isinstance(obj, np.float32):
            return float(obj)  # Convert np.float32 to regular float
        return super(NumpyEncoder, self).default(obj)

class FrameAnalyzer:
    def __init__(self, output_directory, processing_interval, max_frames_to_process):
        self.output_directory = output_directory
        self.processing_interval = processing_interval
        self.max_frames_to_process = max_frames_to_process
        self.start_time = time()
        self.frame_count = 0
        self.processed_frames = []

    def analyze_frames(self, cap, frame_processor):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1
            processed_frame = frame_processor.process_frame(frame)
            output_image_path = os.path.join(self.output_directory, f"frame_{self.frame_count}.jpg")
            cv2.imwrite(output_image_path, processed_frame)
            self.save_json_data()
            if self.should_break():
                break

    def save_json_data(self):
        elapsed_time = time() - self.start_time
        if elapsed_time >= self.processing_interval:
            json_data = json.dumps(self.processed_frames, indent=4, cls=NumpyEncoder)
            output_json_path = f"C:/Users/Admin/Documents/processed_frames_{self.frame_count}.json"
            with open(output_json_path, "w") as json_file:
                json_file.write(json_data)
            print(f"Success! JSON data saved to {output_json_path}")
            self.processed_frames = []
            self.start_time = time()

    def should_break(self):
        return self.max_frames_to_process and self.frame_count >= self.max_frames_to_process
    
def main():
    output_directory = "C:\\Users\\Admin\\Pictures\\Depth"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    object_detector = ObjectDetector("C://Users//Admin//Downloads//yolo-weights//yolov8l.pt", ["person", "bicycle", "car", "motorbike", "bus", "truck", "tie"])
    depth_estimator = DepthEstimator()
    camera_center_x, camera_center_y, focal_length = 320, 240, 3.0

    frame_processor = FrameProcessor(camera_center_x, camera_center_y, focal_length, object_detector, depth_estimator)

    cap = cv2.VideoCapture(0)
    processing_interval = 10
    max_frames_to_process = 10
    frame_analyzer = FrameAnalyzer(output_directory, processing_interval, max_frames_to_process)
    frame_analyzer.analyze_frames(cap, frame_processor)
    cap.release()
