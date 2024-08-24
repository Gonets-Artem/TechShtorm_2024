from ultralytics import YOLO
import cv2

standart_path = 'C:\\Users\\ArgonV\\Documents\\Учеба\\Универ\\Хакатон 2\\TechShtorm_2024'

# Загрузка сохранённой модели
model = YOLO(f'{standart_path}\\runs\\detect\\train15\\weights\\best.pt')


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

video_path = f"{standart_path}\\proj\\videos\\021_mess_both_s.mp4"
cap = cv2.VideoCapture(video_path)

# Настройка для сохранения результата в виде видео
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    print(frame.shape)
    
    if not ret:
        break
    
    # Применение модели на каждом кадре
    results = model(frame)
    
    # Получение аннотированного изображения
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