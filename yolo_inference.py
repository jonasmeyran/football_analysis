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
image = video_frames[549]
result = model.predict(image, conf=0.09, save=True)
print(result[0])

for box in result[0].boxes:
    if box.cls.item() == 0:
        print(box.xyxy)
        print(box.conf.item())

