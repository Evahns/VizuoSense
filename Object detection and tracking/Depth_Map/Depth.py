import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import msvcrt

# Load the MiDaS model from PyTorch Hub
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()

# Initialize the camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera, you can change it if needed

depth_map = None

# Create a function to process each frame from the camera
def process_frame(frame, frame_count):
    global depth_map

    # Convert the frame to a PIL image
    image = Image.fromarray(frame)

    # Preprocess the image for depth estimation
    transform = T.Compose([T.Resize(384), T.ToTensor()])
    input_image = transform(image).unsqueeze(0)

    # Run depth estimation
    with torch.no_grad():
        prediction = midas(input_image)

    # Post-process the depth map
    depth_map = prediction.squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # Ensure that depth_map and frame have the same dimensions
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Save the depth map as an image
    depth_map_image = depth_map.astype("uint8")
    cv2.imwrite(f"C:\\Users\\Admin\\Pictures\\depth_map_{frame_count}.png", depth_map_image)

# Define the frame skip factor (e.g., process every 5th frame)
frame_skip_factor = 4  # Process 1 frame every 5 frames

# Counter to keep track of frames
frame_count = 0

# Continuously capture frames and process them
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    # Increment the frame count
    frame_count += 1

    # Process the frame only if it's a multiple of the frame skip factor
    if frame_count % frame_skip_factor == 0:
        # Call the function to process the frame
        process_frame(frame, frame_count)

    # Check for user key press (press 'q' to exit)
    if msvcrt.kbhit():
        key = msvcrt.getch()
        if key == b'q' or key == b'Q':
            break

# Close the camera
cap.release()