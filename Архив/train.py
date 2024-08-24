from ultralytics import YOLO
import torch

model = YOLO('yolov8s.pt')

import torch
print(torch.cuda.is_available())

#results = model.train(data="C:\\Users\\ArgonV\\Documents\\Учеба\\Универ\\Хакатон 2\\TechShtorm_2024\\proj\\data.yaml", epochs=2, imgsz=640, batch=8)

