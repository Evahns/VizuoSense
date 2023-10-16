# from ultralytics import YOLO
# import cv2
# import cvzone
#
#     cap = cv2.VideoCapture(0)
#     cap.set(3, 1280)  # width
#     cap.set(4, 720)  # height
# while True:
#     success, img = cap.read()
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
###################################################
##################################################

###################################################
##################################################
from ultralytics import YOLO
import cv2
import cvzone  # bounding box
import math
import pyttsx3
# initializing text-to - speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('../yolo-weights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "com", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
              "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
              "pizza", "donut", "cake", "chain", "sofa", "potted-plant", "bed", "table", "toilet", "tv-monitor",
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "nair drier", "toothbrush"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    detected_objects = []  # create a list to store detected objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.9, thickness=1)
            object_name = classNames[cls]
            detected_objects.append(object_name)  # store object name in detected objects list

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # Audio feedback

    if detected_objects:
        audio_feedback = ", ".join(detected_objects)
        engine.say(f" {audio_feedback} detected")
        engine.runAndWait()