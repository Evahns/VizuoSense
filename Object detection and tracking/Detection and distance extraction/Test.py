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

model = YOLO("D:/vizuosense_mine/Resources/yolov8l.pt")
Classnames = model.model.names
names = []
for i in range(len(Classnames)):
    names.append(Classnames[i])
print(names)