import sys
import cv2
import numpy as np
import time
import random
import csv
import os
from datetime import datetime
from collections import defaultdict
import pyautogui

from HandTrackingModule import HandDetector
import utils

# Emotion detection imports
import mediapipe as mp
import onnxruntime
from torchvision import transforms

# Emotion classes
idx_to_class = {
    0:"Negative", #'Anger',
    1:"Negative", #'Contempt',
    2: "Negative", #'Disgust',
    3: "Negative",#'Fear',
    4: "Positive",#'Happiness',
    5: "Neutral",#'Neutral',
    6: "Negative",#'Sadness',
    7: "Positive"# 'Surprise'
}

# CSV Setup
user_details_csv = 'BBT_details.csv'
session_log_prefix = './BBT_logs/BBT_logs_'

# Initialize CSV files with headers if they don't exist
if not os.path.exists(user_details_csv):
    with open(user_details_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Username', 'Timestamp', 'Hand', 'Score', 'Tolerance', 
            'Emotion Score', 'Dominant Emotion', 'Avg Movement Speed', 
            'Avg Time per Block', 'Success Rate'
        ])

# Game parameters
screen_width, screen_height = pyautogui.size()
WIDTH = screen_width
HEIGHT = screen_height-10
# WIDTH = 870
# HEIGHT = 500
HAND = "R"
TOTAL_TIME = 60  # seconds to play
CLOSED_LENGTH = 80  # distance between thumb and index
NUM_VISIBLE_MOVABLE_BLOCKS = 1
MAX_VISIBLE_NON_MOVABLE_BLOCKS = 1
TOLERANCE = 3
colorR = (0, 0, 255)
colorG = (0, 255, 0)
colorB = (255, 0, 0)
colorY = (0, 255, 255)

# Emotion capture parameters
EMOTION_CAPTURE_INTERVAL = 0.5  # seconds between emotion captures during movement
DROP_EMOTION_WINDOW = 1.5  # seconds to capture emotions after drop
MOVEMENT_EMOTION_WEIGHT = 0.7
DROP_EMOTION_WEIGHT = 0.3

# Game positions
x1, y1 = WIDTH // 2, HEIGHT // 2 + 80
x2, y2 = WIDTH // 2, HEIGHT
INITIAL_Y_POSITION = HEIGHT - 150
line_thickness = 3

# Game state
score = 0
emotion_score = 0
totalTime = TOTAL_TIME
movableRectList = []
nonMovableRectList = []
colorList = [colorR, colorG, colorB, colorY]


# Emotion tracking
current_emotion = "Neutral"
dominant_emotion = "Neutral"
movement_emotions = []
drop_emotions = []
last_emotion_capture_time = 0
last_drop_time = 0
is_moving_block = False
current_block_number = 0

# Movement tracking
movement_start_time = 0
movement_start_pos = (0, 0)
movement_end_pos = (0, 0)
movement_distance = 0
movement_speed = 0
all_movement_speeds = []
all_block_times = []
successful_blocks = 0

# Hand detection
detector = HandDetector(maxHands=1)
end_test_time = 0
score_saved = False

# Emotion detection setup
face_detection = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.7
)

# Load emotion model
emotion_model_path = 'hsemotion_1280.onnx'
emotion_session = onnxruntime.InferenceSession(emotion_model_path)
emotion_input_name = emotion_session.get_inputs()[0].name

class DragRect:
    def __init__(self, posCenter, size=None):
        if size is None:
            size = [100, 100]
        self.posCenter = posCenter
        self.initialPosX = posCenter[0]
        self.initialPosY = posCenter[1]
        self.size = size
        self.color = random.choice(colorList)

    def update(self, cursor):
        self.posCenter = cursor[0], cursor[1] + 30

    def cursorWithinRegion(self, cursor):
        ox, oy = self.posCenter
        h, w = self.size
        if ox - TOLERANCE * w // 2 < cursor[0] < ox + TOLERANCE * w // 2 and \
                oy - TOLERANCE * h // 2 < cursor[1] < oy + TOLERANCE * h // 2:
            return True
        return False

    def getColor(self):
        return self.color

    def getInitialPosX(self):
        if HAND == 'L':
            return self.initialPosX
        else:
            return WIDTH - self.initialPosX

    def getInitialPosY(self):
        return self.initialPosY

def addRect(removeRect=None, keepRemovedRect=False, x_pos=80, y_pos=INITIAL_Y_POSITION):
    if removeRect is not None:
        movableRectList.remove(removeRect)
        if keepRemovedRect:
            if len(nonMovableRectList) < MAX_VISIBLE_NON_MOVABLE_BLOCKS:
                nonMovableRectList.append(removeRect)
            else:
                nonMovableRectList[:MAX_VISIBLE_NON_MOVABLE_BLOCKS // 2] = []
                nonMovableRectList.append(removeRect)
        if HAND == 'L':
            movableRectList.append(DragRect([x_pos, y_pos]))
        else:
            movableRectList.append(DragRect([WIDTH - x_pos, y_pos]))
    else:
        if HAND == 'L':
            movableRectList.append(DragRect([x_pos, y_pos]))
        else:
            movableRectList.append(DragRect([WIDTH - x_pos, y_pos]))


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

            # Expand the face region slightly
            face_plus_scalar = 20
            x_end = min(x + w + face_plus_scalar, rgb_img.shape[1])
            y_end = min(y + h + face_plus_scalar, rgb_img.shape[0])
            x = max(0, x - face_plus_scalar)
            y = max(0, y - face_plus_scalar)

            # Extract and preprocess face image
            face_img = frame[y:y_end, x:x_end, :]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = face_img.astype(np.float32)
            face_img /= 255
            face_img[:,:,0] = (face_img[:,:,0] - 0.485) / 0.229
            face_img[:,:,1] = (face_img[:,:,1] - 0.456) / 0.224
            face_img[:,:,2] = (face_img[:,:,2] - 0.406) / 0.225
            img_tensor = face_img.transpose(2, 0, 1)
            img_tensor = img_tensor[np.newaxis,...]

            # Run emotion detection
            outputs = emotion_session.run(None, {emotion_input_name: img_tensor})
            
            # Get emotion and valence/arousal
            current_emotion = idx_to_class[np.argmax(outputs[1])]
            valence = outputs[2][0]  # Positive/negative score
            
            return (x, y, x_end, y_end), valence
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return None, 0
    return None, 0

def calculate_emotion_score():
    global emotion_score, dominant_emotion, movement_emotions, drop_emotions
    
    # Calculate average valence for movement and drop phases
    avg_movement = np.mean([v for e, v in movement_emotions]) if movement_emotions else 0
    avg_drop = np.mean([v for e, v in drop_emotions]) if drop_emotions else 0
    
    # Calculate weighted score
    weighted_valence = (avg_movement * MOVEMENT_EMOTION_WEIGHT) + (avg_drop * DROP_EMOTION_WEIGHT)
    
    # Determine emotion category based on valence
    if weighted_valence > 0.33:
        dominant_emotion = "Positive"
    elif weighted_valence < -0.33:
        dominant_emotion = "Negative"
    else:
        dominant_emotion = "Neutral"
    
    # Scale to 0-100 range (50 is neutral)
    final_emotion_score = max(0, min(100, 50 + (weighted_valence * 50)))
    
    return int(final_emotion_score), dominant_emotion


def log_session_data(username, session_timestamp, block_num, event_type, emotion, valence, speed, score):
    log_filename = f"{session_log_prefix}{username}_{session_timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Write header if file doesn't exist
    if not os.path.exists(log_filename):
        with open(log_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Timestamp', 'Block Number', 'Event Type', 
                'Emotion', 'Valence', 'Movement Speed', 'Score'
            ])
    
    # Append the current frame data
    with open(log_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            block_num,
            event_type,
            emotion,
            valence,
            speed,
            score
        ])

def save_final_score_to_csv(username, hand, score, tolerance, session_timestamp):
    global emotion_score, dominant_emotion, all_movement_speeds, all_block_times, successful_blocks
    
    # Calculate averages
    avg_speed = np.mean(all_movement_speeds) if all_movement_speeds else 0
    avg_time_per_block = np.mean(all_block_times) if all_block_times else 0
    success_rate = (successful_blocks / current_block_number * 100) if current_block_number > 0 else 0
    
    with open(user_details_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            username,
            session_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            hand, 
            score, 
            tolerance, 
            emotion_score,
            dominant_emotion,
            f"{avg_speed:.2f}",
            f"{avg_time_per_block:.2f}",
            f"{success_rate:.1f}%"
        ])

def start(camera_id, hand, tolerance, username):
    global totalTime, score, HAND, end_test_time, TOLERANCE, score_saved
    global current_emotion, emotion_score, dominant_emotion
    global movement_emotions, drop_emotions, last_drop_time, last_emotion_capture_time
    global is_moving_block, current_block_number
    global movement_start_time, movement_start_pos, movement_end_pos, movement_distance, movement_speed
    global all_movement_speeds, all_block_times, successful_blocks
    global last_block_increment_time, BLOCK_INCREMENT_COOLDOWN

    # Reset game state
    score = 0
    emotion_score = 0
    movableRectList.clear()
    nonMovableRectList.clear()
    score_saved = False
    HAND = hand
    TOLERANCE = tolerance
    current_emotion = "Neutral"
    dominant_emotion = "Neutral"
    movement_emotions.clear()
    drop_emotions.clear()
    last_drop_time = 0
    last_emotion_capture_time = 0
    is_moving_block = False
    current_block_number = 0
    movement_start_time = 0
    movement_start_pos = (0, 0)
    movement_end_pos = (0, 0)
    movement_distance = 0
    movement_speed = 0
    all_movement_speeds = []
    all_block_times = []
    successful_blocks = 0
    last_block_increment_time = 0
    BLOCK_INCREMENT_COOLDOWN = 0.4  # seconds
    
    # Record session timestamp once at the start
    session_timestamp = datetime.now()

    # Initialize camera with multiple backend attempts
    cap = None
    backends = [
        cv2.CAP_AVFOUNDATION,  # macOS
        cv2.CAP_DSHOW,        # Windows
        cv2.CAP_V4L2,         # Linux
        cv2.CAP_ANY           # Fallback
    ]
    
    for backend in backends:
        cap = cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            break
    
    if not cap or not cap.isOpened():
        print("Error: Could not open camera with any backend")
        exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Set OpenCV window to be resizable and fullscreen
    cv2.namedWindow("Box & Block Test with Emotion Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Box & Block Test with Emotion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    # Verify camera is working
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from camera")
        exit(1)

    # Initialize blocks
    for i in range(NUM_VISIBLE_MOVABLE_BLOCKS):
        if i % 2 != 0:
            addRect(x_pos=80 + i * 230, y_pos=INITIAL_Y_POSITION - 80)
        else:
            addRect(x_pos=80 + i * 230)
            
    triesToMoveBlock = False
    rect = None
    game_start_time = time.time()

    while True:
        timeRemain = int(totalTime - (time.time() - game_start_time))
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Emotion detection - capture at regular intervals during movement
        current_time = time.time()
        face_bbox, valence = detect_emotion(img)
        
        # Inside the main game loop, under emotion detection section
        if is_moving_block:
            # Log movement data per frame
            if movement_start_time > 0:
                current_speed = movement_speed if movement_speed > 0 else 0
                log_session_data(
                    username, session_timestamp, current_block_number, 
                    "MOVEMENT", current_emotion, valence, current_speed, successful_blocks
                )
            
            # Capture emotions at intervals during movement
            if (current_time - last_emotion_capture_time) >= EMOTION_CAPTURE_INTERVAL:
                movement_emotions.append((current_emotion, valence))
                last_emotion_capture_time = current_time
        elif (current_time - last_drop_time) <= DROP_EMOTION_WINDOW:
            # Log drop emotion data
            log_session_data(
                username, session_timestamp, current_block_number, 
                "DROP", current_emotion, valence, 0, successful_blocks
            )
            drop_emotions.append((current_emotion, valence))

        # Calculate emotion score
        emotion_score, dominant_emotion = calculate_emotion_score()

        if timeRemain <= 0:
            # Game over state
            utils.putTextRect(img, "Test Over", [30, 25], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=5)
            utils.putTextRect(img, f'Score: {score}', [30, 50], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(img, f'Emotion Score: {emotion_score}', [30, 75], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(img, f'Dominant Emotion: {dominant_emotion}', [30, 100], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(img, f'Avg Speed: {np.mean(all_movement_speeds):.1f} px/s', [30, 125], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(img, f'Success Rate: {(successful_blocks/current_block_number*100 if current_block_number > 0 else 0):.1f}%', 
                            [30, 150], colorR=(100, 200, 129), scale=2, thickness=2, offset=3)
            utils.putTextRect(img, "Please click 'Q' for analytics", [50, 480], colorR=(124, 154, 199),
                            scale=5, thickness=2, offset=5)

            if not score_saved:
                save_final_score_to_csv(username, HAND, score, TOLERANCE, session_timestamp)
                score_saved = True

            end_test_time = time.time()
            if int(time.time() - end_test_time) >= 2:
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
        else:
            # Main game loop
            hands, img = detector.findHands(img, flipType=False)
            cv2.line(img, (x1, y1), (x2, y2), (255, 50, 100), thickness=line_thickness)

            if hands:
                lmList = hands[0]['lmList']
                length, info = detector.findDistance(lmList[4], lmList[8])

                if rect is None:
                    rect = movableRectList[-1]

                if length < CLOSED_LENGTH:  # Hand is pinched
                    cursor = lmList[8]

                    if not triesToMoveBlock:
                        for r in movableRectList:
                            if r.cursorWithinRegion(cursor):
                                rect = r
                                triesToMoveBlock = True
                                is_moving_block = True
                                movement_start_time = time.time()
                                movement_start_pos = cursor
                                
                                # Only increment if cooldown has passed
                                current_time = time.time()
                                if current_time - last_block_increment_time >= BLOCK_INCREMENT_COOLDOWN:
                                    current_block_number += 1
                                    last_block_increment_time = current_time
                                break

                    if triesToMoveBlock:
                        # Calculate real-time movement metrics
                        movement_duration = time.time() - movement_start_time
                        movement_distance = np.sqrt((cursor[0] - movement_start_pos[0])**2 + 
                                        (cursor[1] - movement_start_pos[1])**2)
                        movement_speed = movement_distance / movement_duration if movement_duration > 0 else 0
                        
                        rect.update(cursor)
                        if rect.posCenter[1] >= y1 - 40 \
                                and x1 - rect.size[0] // 2 <= rect.posCenter[0] <= x1 + rect.size[0] // 2:
                            addRect(removeRect=rect, keepRemovedRect=False, 
                                x_pos=rect.getInitialPosX(), y_pos=rect.getInitialPosY())
                            triesToMoveBlock = False
                            is_moving_block = False
                            all_movement_speeds.append(movement_speed)
                            all_block_times.append(movement_duration)
                else:  # Hand is released
                    if triesToMoveBlock:
                        # Final movement calculations
                        movement_duration = time.time() - movement_start_time
                        movement_distance = np.sqrt((cursor[0] - movement_start_pos[0])**2 + 
                                                (cursor[1] - movement_start_pos[1])**2)
                        movement_speed = movement_distance / movement_duration if movement_duration > 0 else 0
                        
                        # Check if block was successfully moved
                        if (HAND == 'L' and rect.posCenter[0] > x1 + rect.size[0] // 2) or \
                        (HAND == 'R' and rect.posCenter[0] < x1 - rect.size[0] // 2):
                            score += 1
                            successful_blocks += 1
                            last_drop_time = time.time()
                        
                        # Record metrics
                        all_movement_speeds.append(movement_speed)
                        all_block_times.append(movement_duration)
                        
                        # Reset block
                        rect.posCenter = rect.getInitialPosX(), INITIAL_Y_POSITION
                        addRect(removeRect=rect, keepRemovedRect=True, 
                            x_pos=rect.getInitialPosX(), y_pos=rect.getInitialPosY())
                        triesToMoveBlock = False
                        is_moving_block = False
                        rect = None

            # Draw blocks
            imgNew = np.zeros_like(img, np.uint8)
            
            for r in movableRectList:
                cx, cy = r.posCenter
                w, h = r.size
                cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                            (cx + w // 2, cy + h // 2), r.getColor(), cv2.FILLED)
                utils.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 40, rt=0, colorC=(255, 50, 100))

            for r in nonMovableRectList:
                cx, cy = r.posCenter
                w, h = r.size
                cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                            (cx + w // 2, cy + h // 2), r.getColor(), cv2.FILLED)
                utils.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 40, rt=0, colorC=(255, 50, 100))

            # Combine images with transparency
            out = img.copy()
            alpha = 0.1
            mask = imgNew.astype(bool)
            out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

            # Display game info
            utils.putTextRect(out, "Box & Block", [30, 25], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(out, f"Hand: {HAND}", [30, 50], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(out, f'Score: {score}', [30, 75], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(out, f'Current Emotion: {current_emotion}', [30, 100], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(out, f'Emotion Score: {emotion_score}', [30, 125], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(out, f'Time Remaining: {timeRemain}', [30, 150], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(out, f'Current Block: {current_block_number}', [30, 175], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(out, f'Current Speed: {movement_speed:.1f} px/s', [30, 200], colorR=(100, 200, 129),
                            scale=2, thickness=2, offset=3)
            utils.putTextRect(out, f'Success Rate: {(successful_blocks/current_block_number*100 if current_block_number > 0 else 0):.1f}%', 
                            [30, 225], colorR=(100, 200, 129), scale=2, thickness=2, offset=3)

            # Draw face bounding box if detected
            if face_bbox:
                x, y, x_end, y_end = face_bbox
                cv2.rectangle(out, (x, y), (x_end, y_end), (255, 255, 255), 2)
                cv2.putText(out, current_emotion, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            img = out

        # Display and handle input
        # Emotion capture status indicators
        if (current_time - last_drop_time) <= DROP_EMOTION_WINDOW:
            cv2.putText(img, "DROP EMOTION CAPTURE ACTIVE", (WIDTH//2 - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif is_moving_block:
            cv2.putText(img, "MOVEMENT EMOTION CAPTURE ACTIVE", (WIDTH//2 - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Box & Block Test with Emotion Detection", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):  # Reset game
            score = 0
            emotion_score = 0
            current_block_number = 0
            successful_blocks = 0
            movableRectList.clear()
            nonMovableRectList.clear()
            for i in range(NUM_VISIBLE_MOVABLE_BLOCKS):
                if i % 2 != 0:
                    addRect(x_pos=80 + i * 230, y_pos=INITIAL_Y_POSITION - 100)
                else:
                    addRect(x_pos=80 + i * 230)
            rect = None
            game_start_time = time.time()
            totalTime = TOTAL_TIME
            triesToMoveBlock = False
            is_moving_block = False
            score_saved = False
            movement_emotions.clear()
            drop_emotions.clear()
            last_drop_time = 0
            last_emotion_capture_time = 0
            movement_start_time = 0
            movement_distance = 0
            movement_speed = 0
            all_movement_speeds = []
            all_block_times = []
            # Generate new session timestamp on reset
            session_timestamp = datetime.now()
        elif key == ord('q'):  # Quit
            if not score_saved:
                save_final_score_to_csv(username, HAND, score, TOLERANCE, session_timestamp)
                score_saved = True
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Example usage - username would come from frontend
    start(0, "R", 3, "test_user")