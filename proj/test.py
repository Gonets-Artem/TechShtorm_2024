from ultralytics import YOLO
import cv2

standart_path = 'C:\\Users\\ArgonV\\Documents\\Учеба\\Универ\\Хакатон 2\\TechShtorm_2024'

# Загрузка сохранённой модели
model = YOLO(f'{standart_path}\\runs\\detect\\train16\\weights\\best.pt')


# train11 - видео
#C:\Users\ArgonV\Documents\Учеба\Универ\Хакатон 2\TechShtorm_2024\runs\detect\train11\weights\best.pt

# train19 - фотограааафия
#{standart_path}\\runs\\detect\\train11\\weights\\best.pt

# train15 - видео 021_mess
#{standart_path}\\runs\\detect\\train15\\weights\\best.pt

# goal
# keep
# mess
# try

video_path = f"{standart_path}\\proj\\videos\\002_keep_green_s.mp4"
cap = cv2.VideoCapture(video_path)

# Настройка для сохранения результата в виде видео
output_path = 'proj\\output_video_now.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Применение модели на каждом кадре
    results = model(frame, conf=0.5)
    
    # Получение аннотированного изображения

    #boxes = results[0].boxes if hasattr(results[0], 'boxes') else None
    # Проверка и вывод boxes
    #if boxes is not None:
        #print(len(boxes))
        #print("Boxes:", boxes.xywhn)
        #print(boxes.cls)
        #print(boxes.conf)
        #for el in boxes:
            
        #print(boxes[0]['cls'])

    #boxes.
    annotated_frame = results[0].plot()

    # Запись аннотированного кадра в выходное видео
    out.write(annotated_frame)
    
    # Отображение видео (по желанию)
    cv2.imshow('frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()