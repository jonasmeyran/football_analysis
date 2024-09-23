from ultralytics import YOLO
from utils import read_video
import matplotlib.pyplot as plt

model = YOLO('models/best.pt')
"""
results = model.predict('input_videos/08fd33_4.mp4', save=False)
print(results[0])
print("-------------------------")
for box in results[0].boxes:
    print(box)
"""


video_frames = read_video('input_videos/08fd33_4.mp4')
image = video_frames[24]
result = model.predict(image, save=True)
print(result[0])

