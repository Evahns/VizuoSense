from ultralytics import YOLO
import cv2
import math
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import msvcrt

# Specify the directory where you want to save the frames
output_directory = "C:\\Users\\Admin\\Pictures\\Depth"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('C:\\Users\\Admin\\Downloads\\yolo-weights\\yolov8l.pt')

# Customized class names for campus navigation
classNames = [
    "obstacle",
    "door",
    "staircase",
    "ramp",
    "elevator",
    "exit",
    "restroom",
    "classroom",
    "library",
    "cafeteria",
    "auditorium",
    "office",
    "bench",
    "information board",
    "bus stop",
    "parking area",
    "crosswalk",
    "bike rack",
    "handrail",
    "emergency phone",
    "trash can",
    "water fountain",
    "elevator button panel",
    "curb",
    "building entrance",
    "ATM",
    "vending machine",
    "bus",
    "bike",
    "person"
]

# Load the MiDaS model from PyTorch Hub
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()

frame_counter = 0

# Create an empty dictionary to store detected objects and their distances
object_distances = {}

depth_map = None

# Create a function to process each frame from the camera
def process_frame(frame, frame_count):
    global depth_map

    # Convert the frame to a PIL image
    # image = Image.fromarray(frame)
    frame_pt = torch.Tensor(frame)
    # Preprocess the image for depth estimation
    transform = T.Compose([T.Resize(384), T.ToTensor()])
    input_image = transform(frame_pt).unsqueeze(0)

    # Run depth estimation
    with torch.no_grad():
        prediction = midas(input_image)

    # Post-process the depth map
    depth_map = prediction.squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # Ensure that depth_map and frame have the same dimensions
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Convert depth_map values to meters
    scaling_factor = 0.1  # Adjust this factor based on the MiDaS model's output
    depth_map = depth_map * scaling_factor

    # Call the object detection function
    process_image(frame, frame_count)

# Create a function to calculate distance using the MiDaS depth map
def calculate_distance(x1, y1, x2, y2):
    # Calculate the center coordinates of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Convert the center coordinates to integer
    center_x = int(center_x)
    center_y = int(center_y)

    # Get the depth value at the center coordinates from the depth map
    distance = depth_map[center_y, center_x]

    return distance

# Create a function to process an image and return the detected objects and their distances
def process_image(image, frame_count):
    global depth_map

    results = model(image, stream=True)
    detected_objects = {}
    for r in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = math.ceil((conf * 100)) / 100
        object_name = classNames[int(cls)]

        # Calculate the distance using the depth map (you should replace this with your depth map code)
        distance = calculate_distance(x1, y1, x2, y2)  # Replace with your depth map function

        detected_objects[object_name] = distance

        # Save the processed frame as an image with bounding boxes
        frame_filename = os.path.join(output_directory, f'frame{frame_count}.jpg')
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green bounding box
        cv2.putText(image, f"{object_name} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(frame_filename, image)

    # Store the detected objects and their distances in the dictionary
    object_distances[frame_count] = detected_objects

# Define the frame skip factor (e.g., process every 5th frame)
frame_skip_factor = 4  # Process 1 frame every 5 frames

# Continuously capture frames and process them
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    # Increment the frame counter
    frame_counter += 1

    # Process the frame only if it's a multiple of the frame skip factor
    if frame_counter % frame_skip_factor == 0:
        process_frame(frame, frame_counter)

    # Check for user key press (press 'q' to exit)
    if msvcrt.kbhit():
        key = msvcrt.getch()
        if key == b'q' or key == b'Q':
            break

# Close the camera
cap.release()

# Access the object_distances dictionary to retrieve object and distance information
for frame_number, detected_objects in object_distances.items():
    print(f"Frame {frame_number} - Detected Objects and Distances:")
    for object_name, distance in detected_objects.items():
        print(f"  - {object_name}: {distance} meters")