from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import asyncio
import json
import time
import logging
from typing import Optional, Tuple
import random
from ultralytics import YOLO
import torch

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory=".")

class GameState:
    def __init__(self):
        self.active_game = None
        self.pushups = 0
        self.squats = 0
        self.bomb_time = 8.0
        self.bomb_level = 0
        self.game_start_time = None
        self.debug_mode = False
        self.calibration_step = 0
        self.face_detected = False
        self.last_face_position = None
        self.last_face_center = None
        self.pushup_state = "up"  # Начинаем с "up", так как человек сначала стоит
        self.squat_state = "stand"  # Начинаем с "stand"
        self.last_pushup_time = 0
        self.last_squat_time = 0
        self.level = 1
        self.xp = 0
        self.total_xp = 0
        self.calibration_positions = {
            "pushup": {
                "up_line": 0.3,    # Линия для положения "вверх"
                "down_line": 0.7   # Линия для положения "вниз"
            },
            "squat": {
                "up_line": 0.25,   # Линия для положения "стоя"
                "down_line": 0.65  # Линия для положения "присед"
            }
        }
        self.calibration_complete = False
        self.consecutive_pushups = 0
        self.consecutive_squats = 0
        self.best_streak = 0
        
        # Для игры с бомбой
        self.bomb_position = None
        self.bomb_start_time = None
        self.bomb_active = False
        self.face_in_bomb = False
        self.bomb_exploded = False
        self.bomb_waiting_for_new_face = False
        
        # Таймер потери лица
        self.no_face_start_time = None
        self.face_lost_timer = 3.0
        self.game_over = False
        
        # Для YOLO
        self.last_frame_time = 0
        self.frame_count = 0
        
    def reset(self):
        self.pushups = 0
        self.squats = 0
        self.bomb_time = 8.0
        self.bomb_level = 0
        self.game_start_time = None
        self.calibration_step = 0
        self.face_detected = False
        self.last_face_position = None
        self.last_face_center = None
        self.pushup_state = "up"  # Всегда начинаем с "up"
        self.squat_state = "stand"  # Всегда начинаем с "stand"
        self.level = 1
        self.xp = 0
        self.total_xp = 0
        self.calibration_complete = False
        self.consecutive_pushups = 0
        self.consecutive_squats = 0
        self.best_streak = 0
        
        # Сброс бомбы
        self.bomb_position = None
        self.bomb_start_time = None
        self.bomb_active = False
        self.face_in_bomb = False
        self.bomb_exploded = False
        self.bomb_waiting_for_new_face = False
        
        # Таймер потери лица
        self.no_face_start_time = None
        self.face_lost_timer = 3.0
        self.game_over = False

game_state = GameState()

# Загрузка модели YOLOv12n-face
try:
    logger.info("Loading YOLOv12n-face model...")
    # Попробуем несколько путей к модели
    model_paths = [
        "yolov12n-face.pt",
    ]
    
    face_model = None
    for path in model_paths:
        try:
            logger.info(f"Trying to load model from: {path}")
            face_model = YOLO(path)
            logger.info(f"Model loaded successfully from: {path}")
            break
        except Exception as e:
            logger.warning(f"Failed to load model from {path}: {e}")
            continue
    
    if face_model is None:
        # Если модель не найдена, используем предварительно обученную модель YOLO
        logger.info("YOLOv12n-face not found, using YOLOv8n-face")
        try:
            # Пробуем загрузить стандартную модель YOLO
            face_model = YOLO('yolov8n-face.pt')
        except:
            # Если и это не сработает, используем общую модель YOLO
            logger.info("Using general YOLOv8n model")
            face_model = YOLO('yolov8n.pt')
            
    logger.info("Face detection model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    face_model = None

def detect_face_yolo(frame):
    """Обнаружение лица с помощью YOLO"""
    if face_model is None or frame is None:
        return None
    
    try:
        # Уменьшаем частоту детекции для производительности
        current_time = time.time()
        if current_time - game_state.last_frame_time < 0.1:  # 10 FPS для детекции
            return game_state.last_face_position
        
        # Конвертируем BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Детекция лиц
        results = face_model(rgb_frame, verbose=False)
        
        # Обработка результатов
        faces = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Проверяем класс (если используется общая модель)
                    cls = int(box.cls[0])
                    # Для face модели или если это человек в общей модели
                    if hasattr(face_model, 'names'):
                        class_name = face_model.names.get(cls, '')
                        if 'face' in class_name.lower() or cls == 0:  # 0 = person в COCO
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            faces.append((x1, y1, x2 - x1, y2 - y1))
        
        if len(faces) > 0:
            # Выбираем самое большое лицо
            areas = [w * h for (x, y, w, h) in faces]
            largest_idx = np.argmax(areas)
            game_state.last_frame_time = current_time
            return faces[largest_idx]
        
    except Exception as e:
        logger.error(f"YOLO face detection error: {e}")
    
    return None

def detect_face(frame):
    """Основная функция детекции лица"""
    # Используем YOLO для детекции
    return detect_face_yolo(frame)

def calculate_xp_for_next_level(current_level):
    """Рассчитываем XP для следующего уровня"""
    return current_level * 10

def update_level():
    """Обновляем уровень на основе XP"""
    xp_needed = calculate_xp_for_next_level(game_state.level)
    if game_state.xp >= xp_needed:
        game_state.xp -= xp_needed
        game_state.level += 1
        return True
    return False

def update_bomb_game():
    """Обновление состояния игры с бомбой"""
    if game_state.active_game != "bomb" or game_state.game_over:
        return
    
    current_time = time.time()
    
    # Если бомба активна и прикреплена к лицу
    if game_state.bomb_active and not game_state.bomb_exploded and game_state.bomb_start_time:
        elapsed = current_time - game_state.bomb_start_time
        base_time = max(2.0, 8.0 - game_state.bomb_level * 0.5)
        
        # Если лицо находится в зоне бомбы, замедляем таймер в 2 раза
        if game_state.face_in_bomb and game_state.face_detected:
            game_state.bomb_time = max(0, base_time - elapsed * 0.5)
        else:
            game_state.bomb_time = max(0, base_time - elapsed)
        
        # Если время вышло
        if game_state.bomb_time <= 0 and not game_state.bomb_exploded:
            game_state.bomb_exploded = True
            game_state.bomb_active = False
            game_state.bomb_waiting_for_new_face = True
            game_state.bomb_time = 0.0
            
            # Проверяем, взорвалась ли бомба на лице
            if game_state.face_in_bomb and game_state.face_detected:
                # КОНЕЦ ИГРЫ - бомба взорвалась на лице
                game_state.game_over = True
            else:
                # Просто переходим на следующий уровень
                game_state.bomb_level += 1
    
    # Если бомба взорвалась и мы ждем новое лицо
    if game_state.bomb_waiting_for_new_face and game_state.face_detected:
        # Размещаем новую бомбу на текущем лице
        if game_state.last_face_position:
            x, y, w, h = game_state.last_face_position
            bomb_size = min(w, h) // 2
            game_state.bomb_position = (x + w // 2, y + h // 2, bomb_size)
            game_state.bomb_start_time = current_time
            game_state.bomb_active = True
            game_state.bomb_exploded = False
            game_state.bomb_waiting_for_new_face = False
            game_state.bomb_time = max(2.0, 8.0 - game_state.bomb_level * 0.5)
    
    # Если бомба еще не была размещена и есть лицо
    if not game_state.bomb_active and not game_state.bomb_waiting_for_new_face and game_state.face_detected:
        if game_state.last_face_position:
            x, y, w, h = game_state.last_face_position
            bomb_size = min(w, h) // 2
            game_state.bomb_position = (x + w // 2, y + h // 2, bomb_size)
            game_state.bomb_start_time = current_time
            game_state.bomb_active = True
            game_state.bomb_time = max(2.0, 8.0 - game_state.bomb_level * 0.5)

def check_face_loss():
    """Проверка потери лица для отжиманий и приседаний"""
    if game_state.active_game in ["pushup", "squat"] and game_state.calibration_complete and not game_state.game_over:
        if not game_state.face_detected:
            if game_state.no_face_start_time is None:
                game_state.no_face_start_time = time.time()
            else:
                time_without_face = time.time() - game_state.no_face_start_time
                game_state.face_lost_timer = max(0, 3.0 - time_without_face)
                
                # Если прошло 3 секунды - конец игры
                if time_without_face >= 3.0:
                    game_state.game_over = True
        else:
            game_state.no_face_start_time = None
            game_state.face_lost_timer = 3.0

def draw_calibration_instructions(frame, height, width, game_type):
    """Рисуем инструкции для калибровки"""
    center_x, center_y = width // 2, height // 2
    
    if game_type == "pushup":
        up_line = int(height * game_state.calibration_positions["pushup"]["up_line"])
        down_line = int(height * game_state.calibration_positions["pushup"]["down_line"])
        
        # Рисуем линии
        cv2.line(frame, (0, up_line), (width, up_line), (0, 255, 0), 3)
        cv2.line(frame, (0, down_line), (width, down_line), (0, 0, 255), 3)
        
        # Английский текст инструкций
        cv2.putText(frame, "PUSHUP CALIBRATION", (center_x - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "1. Stand in push-up position before camera", (center_x - 200, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "2. Face should be in the center of frame", (center_x - 200, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "3. Go UP (face above green line)", (center_x - 200, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "4. Go DOWN (face below red line)", (center_x - 200, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "5. Do 1 push-up to calibrate", (center_x - 200, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
    elif game_type == "squat":
        up_line = int(height * game_state.calibration_positions["squat"]["up_line"])
        down_line = int(height * game_state.calibration_positions["squat"]["down_line"])
        
        # Рисуем линии
        cv2.line(frame, (0, up_line), (width, up_line), (0, 255, 0), 3)
        cv2.line(frame, (0, down_line), (width, down_line), (255, 0, 0), 3)
        
        # Английский текст инструкций
        cv2.putText(frame, "SQUAT CALIBRATION", (center_x - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "1. Stand upright before camera", (center_x - 200, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "2. Face should be in the center of frame", (center_x - 200, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "3. SQUAT (face below red line)", (center_x - 200, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "4. STAND (face above green line)", (center_x - 200, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "5. Do 1 squat to calibrate", (center_x - 200, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

def process_frame(frame):
    """Обработка кадра"""
    if frame is None or frame.size == 0:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    height, width = frame.shape[:2]
    
    # Обнаружение лица с помощью YOLO
    face_rect = detect_face(frame)
    
    if face_rect is not None:
        game_state.face_detected = True
        game_state.no_face_start_time = None
        
        x, y, w, h = face_rect
        game_state.last_face_position = (x, y, w, h)
        face_center = (x + w // 2, y + h // 2)
        game_state.last_face_center = face_center
        
        # Для игры с бомбой: проверяем, находится ли лицо в зоне бомбы
        if game_state.active_game == "bomb" and game_state.bomb_position:
            bomb_x, bomb_y, bomb_size = game_state.bomb_position
            distance = np.sqrt((face_center[0] - bomb_x)**2 + (face_center[1] - bomb_y)**2)
            game_state.face_in_bomb = distance < bomb_size
        else:
            game_state.face_in_bomb = False
        
        # Рисуем прямоугольник лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # В зависимости от игры
        if game_state.active_game == "pushup":
            up_line = int(height * game_state.calibration_positions["pushup"]["up_line"])
            down_line = int(height * game_state.calibration_positions["pushup"]["down_line"])
            
            # Рисуем линии
            cv2.line(frame, (0, up_line), (width, up_line), (0, 255, 0), 3)
            cv2.line(frame, (0, down_line), (width, down_line), (0, 0, 255), 3)
            
            # Определение пересечения линий (используем верхнюю и нижнюю точки лица)
            face_top_y = y  # Верхняя граница лица
            face_bottom_y = y + h  # Нижняя граница лица
            current_time = time.time()
            
            # Проверяем пересечение линий - ИСПРАВЛЕННАЯ ЛОГИКА
            # Теперь используем только факт пересечения, а не полное нахождение в зоне
            
            # Флаги пересечения на текущем кадре
            crossed_up_line = False
            crossed_down_line = False
            
            # Проверка пересечения ВЕРХНЕЙ линии
            # Линия пересечена, если верх лица был выше линии, а стал ниже или наоборот
            if hasattr(game_state, 'prev_face_top_y'):
                if (game_state.prev_face_top_y >= up_line and face_top_y < up_line) or \
                   (game_state.prev_face_top_y < up_line and face_top_y >= up_line):
                    crossed_up_line = True
            
            # Проверка пересечения НИЖНЕЙ линии
            if hasattr(game_state, 'prev_face_bottom_y'):
                if (game_state.prev_face_bottom_y <= down_line and face_bottom_y > down_line) or \
                   (game_state.prev_face_bottom_y > down_line and face_bottom_y <= down_line):
                    crossed_down_line = True
            
            # Сохраняем текущие значения для следующего кадра
            game_state.prev_face_top_y = face_top_y
            game_state.prev_face_bottom_y = face_bottom_y
            
            # Определяем текущее состояние для отображения
            if face_top_y < up_line:
                state_text = "ABOVE UP LINE"
                state_color = (0, 255, 0)
            elif face_bottom_y > down_line:
                state_text = "BELOW DOWN LINE"
                state_color = (0, 0, 255)
            else:
                state_text = "BETWEEN LINES"
                state_color = (255, 255, 0)
            
            # Логика подсчета отжиманий - ИСПРАВЛЕННАЯ ВЕРСИЯ
            # Отжимание засчитывается при переходе из "ниже нижней линии" в "выше верхней линии"
            # Но теперь используем только факт пересечения
            
            if game_state.calibration_complete and current_time - game_state.last_pushup_time > 0.5:
                # ОТЖИМАНИЕ: когда пересекаем верхнюю линию ВВЕРХ (идем из нижней позиции)
                if crossed_up_line and face_top_y < up_line:  # Пересекли верхнюю линию вверх
                    # Проверяем, что до этого мы были в нижней позиции
                    # (нижняя граница лица была ниже нижней линии)
                    if hasattr(game_state, 'was_below_down_line') and game_state.was_below_down_line:
                        # ДЕЙСТВИТЕЛЬНОЕ ОТЖИМАНИЕ!
                        game_state.pushups += 1
                        game_state.consecutive_pushups += 1
                        if game_state.consecutive_pushups > game_state.best_streak:
                            game_state.best_streak = game_state.consecutive_pushups
                        
                        # Начисляем XP в зависимости от комбо
                        xp_gain = 1 + min(game_state.consecutive_pushups // 5, 2)
                        game_state.xp += xp_gain
                        game_state.total_xp += xp_gain
                        
                        if update_level():
                            logger.info(f"Level up to {game_state.level}")
                        game_state.last_pushup_time = current_time
                        
                        # Сбрасываем флаг после успешного отжимания
                        game_state.was_below_down_line = False
                
                # Сбрасываем комбо при потере лица
                if not game_state.face_detected:
                    game_state.consecutive_pushups = 0
            
            # Сохраняем состояние для следующего кадра
            # Были ли мы ниже нижней линии (для следующего отжимания)
            game_state.was_below_down_line = (face_bottom_y > down_line)
            
            # Отображаем состояние
            cv2.putText(frame, state_text, (x, y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 3)
            
            # Отображаем факт пересечения (для отладки)
            if game_state.debug_mode:
                if crossed_up_line:
                    cv2.putText(frame, "CROSSED UP!", (x, y - 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if crossed_down_line:
                    cv2.putText(frame, "CROSSED DOWN!", (x, y - 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            
            # Показываем текущую высоту лица
            cv2.putText(frame, f"Top: {face_top_y} | Bottom: {face_bottom_y}", 
                       (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Lines: Up={up_line} | Down={down_line}", 
                       (x, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Показываем счетчик отжиманий
            cv2.putText(frame, f"Push-ups: {game_state.pushups}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Показываем комбо если есть
            if game_state.consecutive_pushups >= 3:
                cv2.putText(frame, f"COMBO x{game_state.consecutive_pushups}", 
                           (width // 2 - 80, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)
            
            # Калибровка
            if game_state.calibration_step:
                draw_calibration_instructions(frame, height, width, "pushup")
                # Проверяем завершение калибровки
                if game_state.pushups >= 1:
                    game_state.calibration_complete = True
                    game_state.calibration_step = 0
            
        elif game_state.active_game == "squat":
            up_line = int(height * game_state.calibration_positions["squat"]["up_line"])
            down_line = int(height * game_state.calibration_positions["squat"]["down_line"])
            
            # Рисуем линии
            cv2.line(frame, (0, up_line), (width, up_line), (0, 255, 0), 3)
            cv2.line(frame, (0, down_line), (width, down_line), (255, 0, 0), 3)
            
            # Определение положения лица (используем нижнюю границу для приседаний)
            face_bottom_y = y + h  # Низ лица
            current_time = time.time()
            
            # Проверяем пересечение линий для приседаний
            crossed_up_line_squat = False
            crossed_down_line_squat = False
            
            if hasattr(game_state, 'prev_face_bottom_y_squat'):
                # Проверка пересечения верхней линии
                if (game_state.prev_face_bottom_y_squat >= up_line and face_bottom_y < up_line) or \
                   (game_state.prev_face_bottom_y_squat < up_line and face_bottom_y >= up_line):
                    crossed_up_line_squat = True
                
                # Проверка пересечения нижней линии
                if (game_state.prev_face_bottom_y_squat <= down_line and face_bottom_y > down_line) or \
                   (game_state.prev_face_bottom_y_squat > down_line and face_bottom_y <= down_line):
                    crossed_down_line_squat = True
            
            game_state.prev_face_bottom_y_squat = face_bottom_y
            
            # Определяем текущее положение
            if face_bottom_y < up_line:  # Низ лица выше верхней линии
                current_position = "stand"
                state_text = "STAND"
                state_color = (0, 255, 0)
            elif face_bottom_y > down_line:  # Низ лица ниже нижней линии
                current_position = "squat"
                state_text = "SQUAT"
                state_color = (255, 0, 0)
            else:
                current_position = "middle"
                state_text = "MIDDLE"
                state_color = (255, 255, 0)
            
            # Проверяем переходы и считаем приседания
            if game_state.calibration_complete and current_time - game_state.last_squat_time > 0.5:
                # ПРИСЕДАНИЕ: когда пересекаем верхнюю линию ВВЕРХ (идем из нижней позиции)
                if crossed_up_line_squat and face_bottom_y < up_line:  # Пересекли верхнюю линию вверх
                    # Проверяем, что до этого мы были ниже нижней линии
                    if hasattr(game_state, 'was_below_down_line_squat') and game_state.was_below_down_line_squat:
                        # ДЕЙСТВИТЕЛЬНОЕ ПРИСЕДАНИЕ!
                        game_state.squats += 1
                        game_state.consecutive_squats += 1
                        if game_state.consecutive_squats > game_state.best_streak:
                            game_state.best_streak = game_state.consecutive_squats
                        
                        # Начисляем XP в зависимости от комбо
                        xp_gain = 1 + min(game_state.consecutive_squats // 5, 2)
                        game_state.xp += xp_gain
                        game_state.total_xp += xp_gain
                        
                        if update_level():
                            logger.info(f"Level up to {game_state.level}")
                        game_state.last_squat_time = current_time
                        
                        # Сбрасываем флаг после успешного приседания
                        game_state.was_below_down_line_squat = False
                
                # Сбрасываем комбо при потере лица
                if not game_state.face_detected:
                    game_state.consecutive_squats = 0
            elif not game_state.calibration_complete:
                # В режиме калибровки
                game_state.squat_state = current_position
            
            # Сохраняем состояние для следующего кадра
            game_state.was_below_down_line_squat = (face_bottom_y > down_line)
            
            # Отображаем состояние
            cv2.putText(frame, state_text, (x, y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 3)
            
            # Показываем счетчик приседаний
            cv2.putText(frame, f"Squats: {game_state.squats}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Показываем комбо если есть
            if game_state.consecutive_squats >= 3:
                cv2.putText(frame, f"COMBO x{game_state.consecutive_squats}", 
                           (width // 2 - 80, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)
            
            # Калибровка
            if game_state.calibration_step:
                draw_calibration_instructions(frame, height, width, "squat")
                # Проверяем завершение калибровки
                if game_state.squats >= 1:
                    game_state.calibration_complete = True
                    game_state.calibration_step = 0
            
        elif game_state.active_game == "bomb":
            # Обновляем состояние бомбы
            update_bomb_game()
            
            # Рисуем бомбу если она активна
            if game_state.bomb_position and game_state.bomb_active and not game_state.bomb_exploded:
                bomb_x, bomb_y, bomb_size = game_state.bomb_position
                
                # Цвет бомбы в зависимости от оставшегося времени
                if game_state.bomb_time > 5.0:
                    bomb_color = (0, 255, 0)  # Зеленый
                elif game_state.bomb_time > 2.0:
                    bomb_color = (0, 255, 255)  # Желтый
                else:
                    bomb_color = (0, 0, 255)  # Красный
                
                # Рисуем бомбу как статичный объект
                cv2.circle(frame, (bomb_x, bomb_y), bomb_size, bomb_color, -1)
                cv2.circle(frame, (bomb_x, bomb_y), bomb_size, (255, 255, 255), 2)
                
                # Фитиль
                fuse_length = int((8.0 - game_state.bomb_time) * 3)
                cv2.line(frame, 
                        (bomb_x, bomb_y - bomb_size),
                        (bomb_x, bomb_y - bomb_size - fuse_length),
                        (0, 0, 0), 3)
                
                # Время
                cv2.putText(frame, f"{game_state.bomb_time:.1f}s", 
                           (bomb_x - 30, bomb_y - bomb_size - fuse_length - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Показываем, если лицо в зоне бомбы
                if game_state.face_in_bomb:
                    cv2.putText(frame, "FACE IN BOMB ZONE!", 
                               (width // 2 - 120, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Если бомба взорвалась
            if game_state.bomb_exploded:
                if game_state.bomb_position:
                    bomb_x, bomb_y, bomb_size = game_state.bomb_position
                    # Анимация взрыва
                    explosion_size = bomb_size * 2
                    cv2.circle(frame, (bomb_x, bomb_y), explosion_size, (0, 0, 255), -1)
                    cv2.putText(frame, "BOOM!", (bomb_x - 40, bomb_y - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                if game_state.game_over:
                    cv2.putText(frame, "GAME OVER - BOMB EXPLODED ON FACE!", 
                               (width // 2 - 200, height // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "NEW BOMB INCOMING...", 
                               (width // 2 - 150, height // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Показываем уровень
            cv2.putText(frame, f"LEVEL: {game_state.bomb_level}", 
                       (width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    else:
        # Лицо не обнаружено
        game_state.face_detected = False
        game_state.face_in_bomb = False
        
        # Очищаем историю позиций при потере лица
        if hasattr(game_state, 'prev_face_top_y'):
            del game_state.prev_face_top_y
        if hasattr(game_state, 'prev_face_bottom_y'):
            del game_state.prev_face_bottom_y
        if hasattr(game_state, 'was_below_down_line'):
            del game_state.was_below_down_line
        if hasattr(game_state, 'prev_face_bottom_y_squat'):
            del game_state.prev_face_bottom_y_squat
        if hasattr(game_state, 'was_below_down_line_squat'):
            del game_state.was_below_down_line_squat
        
        # Проверяем потерю лица для отжиманий и приседаний
        check_face_loss()
        
        # Сбрасываем комбо, если лицо потеряно
        if game_state.active_game == "pushup":
            game_state.consecutive_pushups = 0
        elif game_state.active_game == "squat":
            game_state.consecutive_squats = 0
            
        if game_state.active_game and not game_state.calibration_step:
            cv2.putText(frame, "FACE NOT DETECTED", 
                       (width // 2 - 120, height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Показываем таймер потери лица для отжиманий и приседаний
            if game_state.active_game in ["pushup", "squat"] and game_state.calibration_complete:
                if game_state.face_lost_timer < 3.0:
                    cv2.putText(frame, f"Time left: {game_state.face_lost_timer:.1f}s", 
                               (width // 2 - 100, height // 2 + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    
    # Показываем статистику в углу
    if game_state.active_game and not game_state.game_over:
        # Название игры
        game_name = game_state.active_game.upper()
        cv2.putText(frame, game_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Статистика в зависимости от игры
        if game_state.active_game == "pushup":
            cv2.putText(frame, f"Push-ups: {game_state.pushups}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Level: {game_state.level}", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
        elif game_state.active_game == "squat":
            cv2.putText(frame, f"Squats: {game_state.squats}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Level: {game_state.level}", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
        elif game_state.active_game == "bomb":
            if not game_state.bomb_exploded:
                cv2.putText(frame, f"Time: {game_state.bomb_time:.1f}s", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Level: {game_state.bomb_level}", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Если игра окончена
    if game_state.game_over:
        cv2.putText(frame, "GAME OVER", 
                   (width // 2 - 100, height // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Показываем статистику
        if game_state.active_game == "pushup":
            cv2.putText(frame, f"Final Push-ups: {game_state.pushups}", 
                       (width // 2 - 120, height // 2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        elif game_state.active_game == "squat":
            cv2.putText(frame, f"Final Squats: {game_state.squats}", 
                       (width // 2 - 100, height // 2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        elif game_state.active_game == "bomb":
            cv2.putText(frame, f"Final Level: {game_state.bomb_level}", 
                       (width // 2 - 100, height // 2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    # Debug индикатор
    if game_state.debug_mode:
        cv2.putText(frame, "DEBUG MODE", (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame



async def generate_frames():
    """Генератор кадров"""
    cap = None
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera (attempt {retry_count + 1}/{max_retries})")
                retry_count += 1
                await asyncio.sleep(2)
                continue
            
            # Стандартное разрешение
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera opened successfully")
            retry_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera")
                    break
                
                processed_frame = process_frame(frame)
                
                # Кодирование в JPEG
                ret_encode, buffer = cv2.imencode('.jpg', processed_frame, 
                                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if not ret_encode:
                    logger.warning("Failed to encode frame")
                    continue
                
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                await asyncio.sleep(0.033)
                
        except Exception as e:
            logger.error(f"Error in generate_frames: {str(e)}")
            if cap is not None:
                cap.release()
                cap = None
            
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying camera connection in 2 seconds...")
                await asyncio.sleep(2)
            else:
                logger.error(f"Max retry attempts reached ({max_retries})")
                break
    
    if retry_count >= max_retries:
        logger.error("Failed to connect to camera after all attempts")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, "CAMERA NOT AVAILABLE", 
                   (320 - 150, 240 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(test_frame, "Check your camera connection", 
                   (320 - 180, 240 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        while True:
            _, buffer = cv2.imencode('.jpg', test_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            await asyncio.sleep(1)

# HTML код остается таким же как в предыдущем ответе
# (используйте тот же HTML из предыдущего сообщения)

@app.get("/")
async def home(request: Request):
    """Главная страница"""
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/video_feed")
async def video_feed():
    """Видеопоток"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/test_camera")
async def test_camera():
    """Тест камеры"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return JSONResponse({
            "status": "error", 
            "message": "Camera not available"
        })
    
    success, frame = cap.read()
    cap.release()
    
    if success and frame is not None:
        return JSONResponse({
            "status": "success", 
            "message": "Camera working",
            "resolution": f"{frame.shape[1]}x{frame.shape[0]}"
        })
    else:
        return JSONResponse({
            "status": "error", 
            "message": "Failed to get frame"
        })

@app.post("/start_game/{game_type}")
async def start_game(game_type: str):
    """Запуск игры"""
    if game_type not in ["pushup", "bomb", "squat"]:
        return JSONResponse({
            "status": "error", 
            "message": "Unknown game type"
        })
    
    game_state.active_game = game_type
    game_state.reset()
    game_state.game_start_time = time.time()
    
    # Для бомбы устанавливаем начальную позицию
    if game_type == "bomb":
        game_state.bomb_position = (320, 240, 50)  # Центр кадра
    
    logger.info(f"Game {game_type} started")
    
    return JSONResponse({
        "status": "success",
        "game": game_type,
        "message": f"Game {game_type} started"
    })

@app.post("/stop_game")
async def stop_game():
    """Остановка игры"""
    game_state.active_game = None
    game_state.calibration_step = 0
    game_state.calibration_complete = False
    game_state.game_over = False
    
    return JSONResponse({
        "status": "success",
        "message": "Game stopped"
    })

@app.post("/start_calibration")
async def start_calibration():
    """Начать калибровку"""
    if not game_state.active_game:
        return JSONResponse({
            "status": "error",
            "message": "Select a game first"
        })
    
    game_state.calibration_step = 1
    game_state.calibration_complete = False
    
    return JSONResponse({
        "status": "success",
        "step": 1,
        "message": "Calibration started. Follow on-screen instructions."
    })

@app.post("/complete_calibration")
async def complete_calibration():
    """Завершить калибровку"""
    game_state.calibration_step = 0
    game_state.calibration_complete = True
    
    return JSONResponse({
        "status": "success",
        "message": "Calibration completed"
    })

@app.post("/toggle_debug")
async def toggle_debug():
    """Переключить debug режим"""
    game_state.debug_mode = not game_state.debug_mode
    
    return JSONResponse({
        "status": "success",
        "debug_mode": game_state.debug_mode,
        "message": f"Debug mode: {'ON' if game_state.debug_mode else 'OFF'}"
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    
    try:
        while True:
            # Проверка потери лица
            if game_state.active_game in ["pushup", "squat"] and game_state.calibration_complete:
                check_face_loss()
            
            # Расчет времени игры
            game_time = 0
            if game_state.game_start_time:
                game_time = time.time() - game_state.game_start_time
            
            # Автоматическое увеличение XP со временем
            if game_state.active_game and game_state.calibration_complete and not game_state.game_over:
                # Добавляем 1 XP каждые 30 секунд игры
                current_time = int(time.time())
                if current_time % 30 == 0:
                    game_state.xp += 1
                    game_state.total_xp += 1
                    if update_level():
                        logger.info(f"Level up! Now level {game_state.level}")
            
            stats = {
                "pushups": game_state.pushups,
                "squats": game_state.squats,
                "bomb_time": game_state.bomb_time,
                "bomb_level": game_state.bomb_level,
                "active_game": game_state.active_game,
                "game_time": int(game_time),
                "debug_mode": game_state.debug_mode,
                "calibration_step": game_state.calibration_step,
                "face_detected": game_state.face_detected,
                "level": game_state.level,
                "xp": game_state.xp,
                "total_xp": game_state.total_xp,
                "calibration_complete": game_state.calibration_complete,
                "xp_needed": calculate_xp_for_next_level(game_state.level),
                "consecutive_pushups": game_state.consecutive_pushups,
                "consecutive_squats": game_state.consecutive_squats,
                "best_streak": game_state.best_streak,
                "face_lost_timer": game_state.face_lost_timer,
                "game_over": game_state.game_over,
                "bomb_exploded": game_state.bomb_exploded
            }
            
            await websocket.send_json(stats)
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}")

@app.get("/health")
async def health_check():
    """Проверка здоровья сервера"""
    return JSONResponse({
        "status": "ok",
        "timestamp": time.time(),
        "game_state": game_state.active_game,
        "level": game_state.level,
        "camera_available": cv2.VideoCapture(0).isOpened()
    })

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting PUC-M 2 server...")
    logger.info(f"OpenCV version: {cv2.__version__}")
    
    cap_test = cv2.VideoCapture(0)
    if cap_test.isOpened():
        logger.info("Camera available at startup")
        cap_test.release()
    else:
        logger.warning("Camera not available at startup")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")