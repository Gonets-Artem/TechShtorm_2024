from ultralytics import YOLO

model = YOLO('yolov8s-obb.pt')

results = model.train(data='Yolo_dataset/data.yml', epohcs=5, imgsz=640, batch=2)