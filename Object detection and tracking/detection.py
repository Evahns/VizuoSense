from ultralytics import YOLO
import cv2
import cvzone  # bounding box
import math
import pyttsx3
import os  # Import the os module to work with file paths

# Specify the directory where you want to save the frames
output_directory = "C:\\Users\\Admin\\Pictures\\Depth"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initializing text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('C:\\Users\\Admin\\Downloads\\yolo-weights\\yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "com", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
              "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
              "pizza", "donut", "cake", "chain", "sofa", "potted-plant", "bed", "table", "toilet", "tv-monitor",
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"]

# Function to process an image and return the detected objects
def process_image(image):
    results = model(image, stream=True)
    detected_objects = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            object_name = classNames[cls]
            detected_objects.append(object_name)
    return detected_objects

frame_counter = 0

while True:
    success, img = cap.read()

    # Check if the frame was successfully captured
    if not success:
        continue  # Skip this iteration and try again

    frame_counter += 1

    # Process every 5 frames
    if frame_counter % 5 == 0:
        detected_objects = process_image(img)

        # Save the processed frame as an image
        if detected_objects:
            frame_filename = os.path.join(output_directory, f'frame{frame_counter}.jpg')
            cv2.imwrite(frame_filename, img)

        # Audio feedback
        if detected_objects:
            audio_feedback = ", ".join(detected_objects)
            engine.say(f" {audio_feedback} detected")
            engine.runAndWait()