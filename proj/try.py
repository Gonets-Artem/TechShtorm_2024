import torch
from torch import nn
import cv2
from ultralytics import YOLO


class YOLOWithLSTM(nn.Module):
    def __init__(self, yolo_model):
        super(YOLOWithLSTM, self).__init__()
        self.yolo = yolo_model
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=4, batch_first=True)
        self.fc = nn.Linear(64, 128) 

    def forward(self, x):
        features = self.yolo.model.backbone(x)[-1]
        features = features.view(1, -1, features.size(-1))
        lstm_out, _ = self.lstm(features)
        lstm_out = self.fc(lstm_out)
        return self.yolo.detect_head(lstm_out)

standart_path = 'C:\\Users\\ArgonV\\Documents\\Учеба\\Универ\\Хакатон 2\\TechShtorm_2024'
model = YOLO(f'{standart_path}\\runs\\detect\\train15\\weights\\best.pt')

yolo_with_lstm = YOLOWithLSTM(model)

video_path = f"{standart_path}\\proj\\videos\\021_mess_both_s.mp4"
cap = cv2.VideoCapture(video_path)
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    results = yolo_with_lstm(frame_tensor)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow('frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()