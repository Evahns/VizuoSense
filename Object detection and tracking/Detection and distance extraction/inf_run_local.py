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
import math

class ObjectDetector:
    def __init__(self, weights_path, classNames):
        self.model = YOLO(weights_path)
        self.classNames = classNames

    def detect_objects(self, frame):
        results = self.model(frame)
        detected_objects = {}
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = round(box.conf[0].item(), 2)
                cls = int(box.cls[0])
                object_class = self.classNames[cls] if 0 <= cls < len(self.classNames) else "Unknown"
                detected_objects[object_class] = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': conf
                }
        return detected_objects

class DepthEstimator:
    def __init__(self):
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()

    def depth(self, frame):
        transform = T.Compose([T.Resize(384), T.ToTensor()])
        input_image = transform(Image.fromarray(frame)).unsqueeze(0)
        with torch.no_grad():
            prediction = self.midas(input_image)
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        scaling_factor = 1.0
        depth_map = depth_map * scaling_factor
        return depth_map
        
class FrameProcessor:
    def __init__(self, camera_center_x, camera_center_y, focal_length, object_detector, depth_estimator):
        self.camera_center_x = camera_center_x
        self.camera_center_y = camera_center_y
        self.focal_length = focal_length
        self.object_detector = object_detector
        self.depth_estimator = depth_estimator
        self.frame_count = 0 
        self.detected_objects_dict = {
            'frame_count': self.frame_count,
            'detected_objects': [],
        }

    def process_frame(self, frame):
        depth_map = self.depth_estimator.depth(frame)
        detected_objects = self.object_detector.detect_objects(frame)

        # Extend the detected_objects_dict with distances and deviations
        for object_class, obj_info in detected_objects.items():
            x1, y1, x2, y2, conf = obj_info['x1'], obj_info['y1'], obj_info['x2'], obj_info['y2'], obj_info['conf']
            distance = self.calculate_distance(x1, y1, x2, y2, depth_map)
            horizontal_deviation, vertical_deviation = self.deviations(x1, x2, y1, y2)
            obj_info['distance'] = distance
            obj_info['horizontal_deviation'] = horizontal_deviation
            obj_info['vertical_deviation'] = vertical_deviation

        return frame, detected_objects

    #def calculate_distance(self, x1, y1, x2, y2, depth_map):
        # Define the region of interest within the bounding box
        #roi = depth_map[y1:y2, x1:x2]
        # Calculate the weighted average depth within the region
        #y, x = np.indices(roi.shape)
        #total_depth = np.sum(roi)
        #weighted_x = np.sum(x * roi)
        #weighted_y = np.sum(y * roi)
        #if total_depth > 0:
            #center_x = x1 + weighted_x / total_depth
            #center_y = y1 + weighted_y / total_depth
            #distance = roi[int(center_y - y1), int(center_x - x1)]
            #return distance
        #else:
            # Handle the case where there is no valid depth information in the region
            #return None

    def deviations(self, x1, x2, y1, y2):
        
        h_deviation = math.degrees(math.atan((x1 + (x2 - x1) / 2 - self.camera_center_x) / self.focal_length))
        v_deviation = math.degrees(math.atan((y1 + (y2 - y1) / 2 - self.camera_center_y) / self.focal_length))
        return h_deviation,  v_deviation

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
        self.features = {}
        

    def analyze_frames(self, cap, frame_processor):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1
            processed_frame, detected_objects = frame_processor.process_frame(frame)
            print(f"Frame {self.frame_count}: Detected objects - {detected_objects}")

            # Extend the detected_objects_dict with the current frame's data
            #self.detected_objects_dict['frame_count'] = self.frame_count
            #self.detected_objects_dict['detected_objects'] = detected_objects

            output_image_path = os.path.join(self.output_directory, f"frame_{self.frame_count}.jpg")
            cv2.imwrite(output_image_path, processed_frame)
            elapsed_time = time() - self.start_time
            if elapsed_time >= self.processing_interval:
                self.features[self.frame_count] = {
                    'frame_count': self.frame_count,
                    'detected_objects': detected_objects
                }

                # Reset the timer
                start_time = time()

                # When saving the JSON data, use the custom encoder
                json_data = json.dumps(self.features, indent=4, cls=NumpyEncoder)
                output_json_path = f"D:/vizuosense_mine/Resources/Saves/processed_frames_{self.frame_count}.json"
                with open(output_json_path, "w") as json_file:
                    json_file.write(json_data)
                print(f"Success! JSON data saved to {output_json_path}")

            # If you want to break the loop after processing a specific number of frames, you can add a condition here
            if self.max_frames_to_process and self.frame_count >= self.max_frames_to_process:
                break

        # Release the VideoCapture and close any open windows
        cap.release()
    
def main():
    output_directory = "D:/vizuosense_mine/Resources/Saves"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    object_detector = ObjectDetector("D:/vizuosense_mine/Resources/yolov8l.pt", ["person", "bicycle", "car", "motorbike", "bus", "truck", "tie"])
    depth_estimator = DepthEstimator()
    camera_center_x, camera_center_y, focal_length = 320, 240, 3.0

    frame_processor = FrameProcessor(camera_center_x, camera_center_y, focal_length, object_detector, depth_estimator)

    cap = cv2.VideoCapture(0)
    processing_interval = 10
    max_frames_to_process = 10
    frame_analyzer = FrameAnalyzer(output_directory, processing_interval, max_frames_to_process)
    frame_analyzer.analyze_frames(cap, frame_processor)
    cap.release()
