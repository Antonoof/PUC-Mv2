import cv2
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path='models/yolov12n-face.pt'):
        """Инициализация модели YOLO"""
        # Загрузка модели
        self.model = YOLO(model_path)
        print(f"Модель {model_path} загружена")
    
    def run_realtime(self, camera_id=0):
        """Запуск детекции в реальном времени"""
        # Открываем видеопоток
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть камеру")
            return
        
        print("Детекция лиц запущена. Нажмите 'q' для выхода.")
        
        while True:
            # Захват кадра
            ret, frame = cap.read()
            if not ret:
                break
            
            # Детекция лиц с помощью YOLO
            results = self.model(frame, verbose=False)[0]
            
            # Отрисовка результатов
            for box in results.boxes:
                # Получаем координаты и confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Рисуем bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Добавляем текст с confidence
                label = f'Face: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Показываем количество лиц
            num_faces = len(results.boxes) if results.boxes is not None else 0
            cv2.putText(frame, f'Faces detected: {num_faces}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Отображаем кадр
            cv2.imshow('Face Detection - YOLOv12n', frame)
            
            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Освобождаем ресурсы
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Создаем детектор
    detector = FaceDetector()
    
    # Запускаем детекцию
    detector.run_realtime()