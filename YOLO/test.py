from ultralytics import YOLO

model = YOLO('yolov8s-obb.pt')

results = model.train(data='Yolo_dataset/data.yaml', epochs=1, imgsz=640, batch=2)

model.save('runs\\obb\\train11\\weights\\best.pt')