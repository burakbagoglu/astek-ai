from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("model.pt")
results = model.predict(source="image.jpg",save=True)
