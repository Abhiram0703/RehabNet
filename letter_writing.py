import sys
import cv2
import numpy as np
import time
import os
from datetime import datetime
import csv
from collections import defaultdict
import pandas as pd
from scipy.spatial import cKDTree

from HandTrackingModule import HandDetector
import utils

# Emotion detection imports
import mediapipe as mp
import onnxruntime
from torchvision import transforms

# Drawing parameters
DRAWING_THICKNESS = 5
PIXEL_TO_MM = 0.264  # Conversion factor: 1 pixel = 0.264 mm (96 DPI)

# Emotion classes
idx_to_class = {
    0: "Negative", 1: "Negative", 2: "Negative", 3: "Negative",
    4: "Positive", 5: "Neutral", 6: "Negative", 7: "Positive"
}

# CSV Setup
user_details_csv = 'letter_writing_details.csv'
session_log_prefix = './LW_logs/letter_writing_logs_'

# Initialize CSV files
if not os.path.exists(user_details_csv):
    with open(user_details_csv, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Username', 'Timestamp', 'Hand', 'Completion Time', 
                         'Dominant Emotion', 'Avg Movement Speed', 'Avg Error Distance'])

# Game parameters
WIDTH = 640
HEIGHT = 480
CLOSED_LENGTH = 60
DRAWING_THICKNESS = 5
drawColor = (0, 255, 0)

# Letters paths
letters_folder = "Alphabets"
letter_files = ['A.png', 'K.png', 'L.png', 'M.png', 'N.png', 'S.png', 'W.png']
letter_images = []

# Emotion capture parameters
EMOTION_CAPTURE_INTERVAL = 0.05

# Game state
current_letter_index = 0
drawing_started = False
imgCanvas = None
letter_template = None
temp_logs = None
tasks_completed = 0
task_error_distances = []  # Store Avg Error Distances for each task

# Emotion tracking
current_emotion = "Neutral"
emotions = []
last_emotion_capture_time = 0

# Movement tracking
xp, yp = 0, 0
movement_start_time = 0
movement_speeds = []
drawing_pixels = []
template_pixels = []
total_error_distance = 0
last_error_distance = 0

# Save and Next cooldown
last_save_time = 0
SAVE_COOLDOWN = 6

# Hand detection
detector = HandDetector(maxHands=2)
stats_saved = False

# Emotion detection setup
face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
emotion_model_path = 'hsemotion_1280.onnx'
emotion_session = onnxruntime.InferenceSession(emotion_model_path)
emotion_input_name = emotion_session.get_inputs()[0].name

def load_letter_images():
    global letter_images
    letter_size = 200  # Reduced from 250 to 200
    for file in letter_files:
        path = os.path.join(letters_folder, file)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = cv2.resize(img, (letter_size, letter_size), interpolation=cv2.INTER_AREA)
            letter_images.append(img)
        else:
            print(f"Error loading image: {path}")
            blank = np.ones((letter_size, letter_size, 4), dtype=np.uint8) * 255
            cv2.putText(blank, file[0], (letter_size//3, letter_size//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 4)
            letter_images.append(blank)

def detect_emotion(frame):
    global current_emotion, last_emotion_capture_time
    
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected = face_detection.process(rgb_img)
    
    if detected.detections:
        try:
            face_pos = detected.detections[0].location_data.relative_bounding_box
            x = int(rgb_img.shape[1] * max(face_pos.xmin, 0))
            y = int(rgb_img.shape[0] * max(face_pos.ymin, 0))
            w = int(rgb_img.shape[1] * min(face_pos.width, 1))
            h = int(rgb_img.shape[0] * min(face_pos.height, 1))

            face_plus_scalar = 20
            x_end = min(x + w + face_plus_scalar, rgb_img.shape[1])
            y_end = min(y + h + face_plus_scalar, rgb_img.shape[0])
            x = max(0, x - face_plus_scalar)
            y = max(0, y - face_plus_scalar)

            face_img = frame[y:y_end, x:x_end, :]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = face_img.astype(np.float32)
            face_img /= 255
            face_img[:,:,0] = (face_img[:,:,0] - 0.485) / 0.229
            face_img[:,:,1] = (face_img[:,:,1] - 0.456) / 0.224
            face_img[:,:,2] = (face_img[:,:,2] - 0.406) / 0.225
            img_tensor = face_img.transpose(2, 0, 1)
            img_tensor = img_tensor[np.newaxis,...]

            outputs = emotion_session.run(None, {emotion_input_name: img_tensor})
            current_emotion = idx_to_class[np.argmax(outputs[1])]
            valence = outputs[2][0]
            
            return (x, y, x_end, y_end), valence
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return None, 0
    return None, 0

def calculate_dominant_emotion():
    if not emotions:
        return "Neutral"
    emotion_counts = defaultdict(int)
    for e, _ in emotions:
        emotion_counts[e] += 1
    return max(emotion_counts, key=emotion_counts.get)

def calculate_error_distance(hand):
    global drawing_pixels, template_pixels
    if not drawing_pixels or not template_pixels:
        return 0.0
        
    # Adjust template offset based on hand
    if hand.lower() == 'l':
        template_offset_x = int(WIDTH * 0.2)  # 128 pixels, moved toward center
        template_offset_y = int(HEIGHT * 0.1)  # 48 pixels
    else:
        template_offset_x = int(WIDTH * 0.45)  # 288 pixels, moved toward center
        template_offset_y = int(HEIGHT * 0.1)  # 48 pixels
    
    # Transform template pixels
    transformed_template_pixels = np.array([(x + template_offset_x, y + template_offset_y) for x, y in template_pixels])
    drawing_pixels_array = np.array(drawing_pixels)
    
    if len(transformed_template_pixels) == 0 or len(drawing_pixels_array) == 0:
        return 0.0
    
    # Use cKDTree for efficient nearest-neighbor search
    tree = cKDTree(transformed_template_pixels)
    distances, _ = tree.query(drawing_pixels_array, k=1)
    
    total_distance = np.sum(distances)
    count = len(distances)
    
    if count > 0:
        return (total_distance / count) * PIXEL_TO_MM
    else:
        return 0.0

def extract_template_pixels():
    global template_pixels, letter_template
    if letter_template is None:
        return
        
    if letter_template.shape[2] == 4:
        rgb = letter_template[:, :, :3]
        alpha = letter_template[:, :, 3]
        _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        template_points = np.argwhere(mask > 0)
        template_pixels = [(point[1], point[0]) for point in template_points]
    else:
        gray = cv2.cvtColor(letter_template, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        template_points = np.argwhere(binary > 0)
        template_pixels = [(point[1], point[0]) for point in template_points]

def log_frame_data(letter, emotion, speed, error_distance):
    global temp_logs, last_error_distance
    error_to_log = last_error_distance if speed == 0 else error_distance
    new_log = pd.DataFrame([{
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Letter': letter_files[letter][0],
        'Emotion': emotion,
        'Movement Speed': f"{speed:.2f}",
        'Avg Error Distance': f"{error_to_log:.2f}"
    }])
    temp_logs = pd.concat([temp_logs, new_log], ignore_index=True)

def save_logs_to_csv(username, session_timestamp):
    global temp_logs
    if not temp_logs.empty:
        log_filename = f"{session_log_prefix}{username}_{session_timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        temp_logs.to_csv(log_filename, mode='a', header=not os.path.exists(log_filename), index=False, quoting=csv.QUOTE_MINIMAL)

def save_final_stats_to_csv(username, hand, completion_time, session_timestamp):
    global emotions, movement_speeds, task_error_distances, stats_saved
    if stats_saved:
        print("Stats already saved for this session, skipping.")
        return
    try:
        dominant_emotion = calculate_dominant_emotion()
        avg_speed = np.mean(movement_speeds) if movement_speeds else 0
        avg_error_distance = np.mean(task_error_distances) if task_error_distances else 0
        with open(user_details_csv, 'a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                username,
                session_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                hand,
                f"{completion_time:.2f}",
                dominant_emotion,
                f"{avg_speed:.2f}",
                f"{avg_error_distance:.2f}"
            ])
        stats_saved = True
        print(f"Saved stats for {username} at {session_timestamp}")
    except Exception as e:
        print(f"Error saving final stats: {e}")

def reset_task():
    global imgCanvas, drawing_started, xp, yp, drawing_pixels, temp_logs, total_error_distance, last_error_distance
    imgCanvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    drawing_started = False
    xp, yp = 0, 0
    drawing_pixels = []
    total_error_distance = 0
    last_error_distance = 0
    temp_logs = pd.DataFrame(columns=['Timestamp', 'Letter', 'Emotion', 'Movement Speed', 'Avg Error Distance'])

def next_task(username, hand, session_timestamp):
    global current_letter_index, imgCanvas, letter_template, template_pixels
    global drawing_started, task_start_time, drawing_pixels, total_error_distance
    global tasks_completed, last_error_distance, task_error_distances
    task_error_distances.append(total_error_distance)  # Store Avg Error Distance for this task
    save_logs_to_csv(username, session_timestamp)
    tasks_completed += 1
    current_letter_index += 1
    if current_letter_index >= len(letter_files):
        current_letter_index = 0
    reset_task()
    if current_letter_index < len(letter_images):
        letter_template = letter_images[current_letter_index].copy()
        extract_template_pixels()

def start(camera_id, hand, username):
    global current_letter_index, imgCanvas, letter_template, template_pixels
    global drawing_started, task_start_time, task_completion_time
    global xp, yp, current_emotion, emotions, last_emotion_capture_time
    global movement_speeds, drawing_pixels, total_error_distance, stats_saved
    global total_time, last_save_time, temp_logs, tasks_completed, last_error_distance
    global task_error_distances
    
    # Initialize state
    current_letter_index = 0
    imgCanvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    drawing_started = False
    stats_saved = False
    current_emotion = "Neutral"
    emotions = []
    last_emotion_capture_time = 0
    task_start_time = time.time()
    task_completion_time = 0
    xp, yp = 0, 0
    movement_speeds = []
    drawing_pixels = []
    total_error_distance = 0
    last_error_distance = 0
    total_time = 0
    last_save_time = 0
    tasks_completed = 0
    task_error_distances = []
    temp_logs = pd.DataFrame(columns=['Timestamp', 'Letter', 'Emotion', 'Movement Speed', 'Avg Error Distance'])
    
    session_timestamp = datetime.now()
    load_letter_images()
    
    if letter_images and len(letter_images) > 0:
        letter_template = letter_images[current_letter_index].copy()
        extract_template_pixels()
    
    cap = None
    backends = [
        cv2.CAP_AVFOUNDATION,
        cv2.CAP_DSHOW,
        cv2.CAP_V4L2,
        cv2.CAP_ANY
    ]
    
    for backend in backends:
        cap = cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            break
    
    if not cap or not cap.isOpened():
        print("Error: Could not open camera with any backend")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    cv2.namedWindow("Letter Drawing Task", cv2.WINDOW_NORMAL)
    
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from camera")
        sys.exit(1)
    
    letter_completed = False
    total_start_time = time.time()
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        # Resize camera frame to match canvas dimensions
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img = cv2.flip(img, 1)
        
        current_time = time.time()
        face_bbox, valence = detect_emotion(img)
        
        if drawing_started and (current_time - last_emotion_capture_time) >= EMOTION_CAPTURE_INTERVAL:
            emotions.append((current_emotion, valence))
            last_emotion_capture_time = current_time
            
        total_time = current_time - task_start_time
        
        hands, img = detector.findHands(img, flipType=False)
        
        drawing_hand = None
        for h in hands:
            hand_type = h["type"].lower()
            if (hand.lower() == 'l' and hand_type == "left") or (hand.lower() == 'r' and hand_type == "right"):
                drawing_hand = h
                break
        
        can_draw = drawing_hand is not None and len(hands) == 1
        
        current_speed = 0.0
        current_error_distance = 0.0
        
        if drawing_hand is not None:
            lmList = drawing_hand['lmList']
            index_finger_tip = lmList[8]
            x1, y1 = index_finger_tip[0], index_finger_tip[1]
            
            if can_draw and 30 < x1 < WIDTH - 30 and 30 < y1 < HEIGHT - 80:
                if not drawing_started:
                    drawing_started = True
                    if task_start_time == 0:
                        task_start_time = time.time()
                
                if xp != 0 and yp != 0 and drawing_started:
                    distance = np.sqrt((x1 - xp)**2 + (y1 - yp)**2)
                    time_diff = time.time() - movement_start_time
                    
                    if time_diff > 0:
                        current_speed = distance / time_diff
                        movement_speeds.append(current_speed)
                    
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, DRAWING_THICKNESS)
                    drawing_pixels.append((x1, y1))
                    current_error_distance = calculate_error_distance(hand)
                    total_error_distance = current_error_distance
                    last_error_distance = current_error_distance
                
                xp, yp = x1, y1
                movement_start_time = time.time()
                
                log_frame_data(current_letter_index, current_emotion, current_speed, current_error_distance)
            else:
                xp, yp = 0, 0
                log_frame_data(current_letter_index, current_emotion, current_speed, last_error_distance)
        else:
            xp, yp = 0, 0
            log_frame_data(current_letter_index, current_emotion, current_speed, last_error_distance)
        
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        
        if letter_template is not None:
            # Adjust template position based on hand
            if hand.lower() == 'l':
                template_pos_x = int(WIDTH * 0.2)  # 128 pixels, moved toward center
                template_pos_y = int(HEIGHT * 0.1)  # 48 pixels
                ref_pos_x = int(WIDTH * 0.6)  # 384 pixels
                ref_pos_y = int(HEIGHT * 0.9)  # 432 pixels
            else:
                template_pos_x = int(WIDTH * 0.45)  # 288 pixels, moved toward center
                template_pos_y = int(HEIGHT * 0.1)  # 48 pixels
                ref_pos_x = int(WIDTH * 0.05)  # 32 pixels
                ref_pos_y = int(HEIGHT * 0.9)  # 432 pixels
            img = utils.overlayPNG(img, letter_template, pos=(template_pos_x, template_pos_y))
        
        text_scale = 1.0
        text_offset = 5
        # Adjust text position based on hand
        text_x = WIDTH - 250 if hand.lower() == 'l' else 30  # Right side for left hand, left side for right hand
        utils.putTextRect(img, f"Letter: {letter_files[current_letter_index][0]}", [text_x, 30], 
                        colorR=(100, 200, 129), scale=text_scale, thickness=2, offset=text_offset)
        utils.putTextRect(img, f"Hand: {hand}", [text_x, 60], 
                        colorR=(100, 200, 129), scale=text_scale, thickness=2, offset=text_offset)
        utils.putTextRect(img, f"Emotion: {current_emotion}", [text_x, 90], 
                        colorR=(100, 200, 129), scale=text_scale, thickness=2, offset=text_offset)
        utils.putTextRect(img, f"Time: {total_time:.1f}s", [text_x, 120], 
                      colorR=(100, 200, 129), scale=text_scale, thickness=2, offset=text_offset)
        utils.putTextRect(img, f"Avg Error Distance: {total_error_distance:.1f}mm", [text_x, 150], 
                      colorR=(100, 200, 129), scale=text_scale, thickness=2, offset=text_offset)
        utils.putTextRect(img, f"Movement Speed: {current_speed:.1f}px/s", [text_x, 180], 
                      colorR=(100, 200, 129), scale=text_scale, thickness=2, offset=text_offset)
        
        # Add footer instructions
        footer_scale = 0.7
        utils.putTextRect(img, "Press S to save & next task", [WIDTH - 250, HEIGHT - 50], 
                        colorR=(100, 200, 129), scale=footer_scale, thickness=2, offset=text_offset)
        utils.putTextRect(img, "Press R to reset", [WIDTH - 250, HEIGHT - 30], 
                        colorR=(100, 200, 129), scale=footer_scale, thickness=2, offset=text_offset)
        
        if (time.time() - last_save_time) < SAVE_COOLDOWN:
            remaining = SAVE_COOLDOWN - (time.time() - last_save_time)
            utils.putTextRect(img, f"Save Cooldown: {remaining:.1f}s", [WIDTH - 250, 30], 
                           colorR=(200, 100, 100), scale=text_scale, thickness=2, offset=text_offset)
        
        if face_bbox:
            x, y, x_end, y_end = face_bbox
            cv2.rectangle(img, (x, y), (x_end, y_end), (255, 255, 255), 2)
            cv2.putText(img, current_emotion, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        if letter_completed:
            utils.putTextRect(img, "All Letters Completed!", [WIDTH//2 - 150, HEIGHT//2], 
                            colorR=(100, 200, 129), scale=2, thickness=3, offset=10)
            utils.putTextRect(img, "Press 'Q' to exit", [WIDTH//2 - 100, HEIGHT//2 + 50], 
                            colorR=(100, 200, 129), scale=text_scale, thickness=2, offset=text_offset)
        
        cv2.imshow("Letter Drawing Task", img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            if not stats_saved:
                task_completion_time = time.time() - total_start_time
                save_final_stats_to_csv(username, hand, task_completion_time, session_timestamp)
            save_logs_to_csv(username, session_timestamp)
            break
        elif key == ord('r'):
            reset_task()
        elif key == ord('s'):
            current_time = time.time()
            if (current_time - last_save_time) > SAVE_COOLDOWN:
                if drawing_started:
                    next_task(username, hand, session_timestamp)
                    last_save_time = current_time
                    if tasks_completed >= len(letter_files):
                        letter_completed = True
                        task_completion_time = time.time() - total_start_time
                        save_final_stats_to_csv(username, hand, task_completion_time, session_timestamp)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start(0, "R", "test_user")