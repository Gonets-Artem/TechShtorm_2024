from ultralytics import YOLO

model = YOLO('yolov5n.pt')

results = model.train(data='D:\\hackatonneft\\TechShtorm_2024\\YOLO\\Yolo_dataset_2\\data.yaml', epochs=1, imgsz=640, batch=2)
print(model.names)
# model.save('runs\\obb\\train11\\weights\\best.pt')