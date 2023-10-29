from time import time
import cv2
import math
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pandas as pd 

# Specify the directory where you want to save the frames
output_directory ="C:\\Users\\Admin\\Pictures\\Depth"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load the YOLO model
model = YOLO("C://Users//Admin//Downloads//yolo-weights//yolov8l.pt")

# Load the MiDaS model from PyTorch Hub
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()

# Customized class names for campus navigation
classNames = ["person", "bicycle", "car", "motorbike", "bus", "truck", "tie"]

# Function to calculate depth map using the MiDaS model
def calculate_depth_map(frame):
    # Resize the input image to the required size
    transform = T.Compose([T.Resize(384), T.ToTensor()])
    input_image = transform(Image.fromarray(frame)).unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_image)

    depth_map = prediction.squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # Adjust the scaling factor based on the MiDaS model's output
    scaling_factor = 0.01
    depth_map = depth_map * scaling_factor

    return depth_map

def calculate_distance(x1, y1, x2, y2, depth_map):
    # Define the region of interest within the bounding box
    roi = depth_map[y1:y2, x1:x2]

    # Calculate the weighted average depth within the region
    y, x = np.indices(roi.shape)
    total_depth = np.sum(roi)
    weighted_x = np.sum(x * roi)
    weighted_y = np.sum(y * roi)

    if total_depth > 0:
        center_x = x1 + weighted_x / total_depth
        center_y = y1 + weighted_y / total_depth
        distance = roi[int(center_y - y1), int(center_x - x1)]
        return distance
    else:
        # Handle the case where there is no valid depth information in the region
        return None

# Define camera parameters (example values, replace with your actual camera parameters)
focal_length = 3.0  # Focal length in millimeters
camera_center_x = 320  # Optical center's x-coordinate in the image frame
camera_center_y = 240  # Optical center's y-coordinate in the image frame

def process_image(frame, camera_center_x, camera_center_y, focal_length):
    depth_map = calculate_depth_map(frame)

    results = model(frame)

    detected_objects = {}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if 0 <= cls < len(classNames):
                object_class = classNames[cls]
            else:
                object_class = "Unknown"

            # Calculate the percentage of the image size covered by the bounding box
            image_area = frame.shape[0] * frame.shape[1]
            box_area = w * h
            coverage_percentage = (box_area / image_area) * 100

            

            # Check if we have detected objects of this class before
            if object_class not in detected_objects_per_class:
                detected_objects_per_class[object_class] = 1
            else:
                detected_objects_per_class[object_class] += 1

            # Assign a unique label to this object instance within the class
            object_name = f"{object_class}{detected_objects_per_class[object_class]}"

            # Calculate the distance using the depth map
            distance = calculate_distance(x1, y1, x2, y2, depth_map)

            # Calculate horizontal and vertical deviations in degrees
            horizontal_deviation = math.degrees(math.atan((x1 + w/2 - camera_center_x) / focal_length))
            vertical_deviation = math.degrees(math.atan((y1 + h/2 - camera_center_y) / focal_length))

            detected_objects[object_name] = {
                'distance': distance,
                'horizontal_deviation': horizontal_deviation,
                'vertical_deviation': vertical_deviation
            }
            # Choose the color based on coverage percentage
            if coverage_percentage > 50:
                color = (0, 0, 255)  # Red
                cv2.putText(frame, f"{conf:.2f}) distance: {distance:.2f} :Too Near", (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 2)
            else:
                color = (0, 255, 0)  # Green
            # Draw small dots at x1, y1, x2, y2
            cv2.circle(frame, (x1, y1), 3, color, -1)
            cv2.circle(frame, (x2, y2), 3, color, -1)
            
            # Draw a dotted line that divides the frame into four quadrants
            line_color = (0, 255, 0)  # Green color
            line_thickness = 2
            dot_spacing = 10  # Adjust this value to control the spacing between dots

            # Vertical line
            for y in range(0, frame.shape[0], dot_spacing * 2):
                cv2.line(frame, (camera_center_x, y), (camera_center_x, y + dot_spacing), line_color, line_thickness)

            # Horizontal line
            for x in range(0, frame.shape[1], dot_spacing * 2):
                cv2.line(frame, (x, camera_center_y), (x + dot_spacing, camera_center_y), line_color, line_thickne
                
            #Add "VizuoSense" text to the top left corner
            
            text = "VS engine running..."
            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 0.6
            font_color = (64, 64, 64)  # White color
            cv2.putText(frame, text, (10, 30), font, font_scale, font_color, 1)
            
            

    return frame, detected_objects

# Define a dictionary to keep track of detected objects per class
detected_objects_per_class = {}

# Initialize a dictionary to store processed image data
processed_frames = {}

# Initialize a variable to keep track of time
start_time = time()

# Specify the time interval (in seconds) for processing frames
processing_interval = 10

# Initialize a frame count
frame_count = 0

# Specify the maximum number of frames to process (optional)
max_frames_to_process = 10  # Change this to process a specific number of frames

# Initialize the VideoCapture object (0 for default camera, or provide the video file path)
cap = cv2.VideoCapture(0)  # Change the argument if using a different video source

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Process the frame
    processed_image, detected_objects = process_image(frame, camera_center_x, camera_center_y, focal_length)

    # Save the processed image with bounding boxes
    output_image_path = os.path.join(output_directory, f"frame_{frame_count}.jpg")
    cv2.imwrite(output_image_path, processed_image)

    # Calculate elapsed time
    elapsed_time = time() - start_time

    # If the elapsed time exceeds the processing interval, store the data in the dictionary
    if elapsed_time >= processing_interval:
        processed_frames[frame_count] = {
            'frame_count': frame_count,
            'detected_objects': detected_objects
        }

        # Reset the timer
        start_time = time()
        
        df = pd.DataFrame(processed_frames)
        
        output_csv_path = "C:/Users/Admin/Documents/processed_frames.csv"
        df.to_csv(output_csv_path, index=False)
        print(f"Success !!! DataFrame saved to {output_csv_path}")
        

    # If you want to break the loop after processing a specific number of frames, you can add a condition here
    if max_frames_to_process and frame_count >= max_frames_to_process:
        break

# Release the VideoCapture and close any open windows
cap.release()
