from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(data="C:\\Users\\Huawei\\Desktop\\proj\\data.yaml", epochs=2, imgsz=640, batch=8)

