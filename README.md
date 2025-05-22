# RehabNet: AI-Driven Multimodal Vision Therapy for Post-Stroke Hand Recovery

RehabNet is an innovative AI-powered rehabilitation system designed to support post-stroke hand recovery through vision-based assessments. It integrates real-time hand tracking, emotion detection, and performance analytics to provide a holistic evaluation of motor skills and emotional states. The system features two interactive tests: the **Letter Writing Test** and the **Box-and-Block Test**, which assess fine motor skills and gross manual dexterity, respectively. Built with computer vision (OpenCV, MediaPipe) and machine learning (ONNX), RehabNet offers clinicians detailed analytics and visualizations to personalize therapy plans.

## 🚀 Features

- **Two Assessment Modules:**
  - **Letter Writing Test**: Users trace letters (A, K, L, M, N, S, W) using their index finger to assess fine motor control and emotional responses.
  - **Box-and-Block Test**: Users move virtual blocks between compartments to evaluate gross motor dexterity and emotional states.
- **Real-Time Tracking:**
  - Hand tracking via MediaPipe, supporting left or right hand selection.
  - Emotion detection using the `hsemotion_1280.onnx` model, classifying emotions as Positive, Negative, or Neutral.
- **Performance Metrics:**
  - **Letter Writing**: Completion time, movement speed, average error distance, dominant emotion.
  - **Box-and-Block**: Score (successful moves), success rate, average movement speed, average time per block, emotion score.
- **Interactive GUI**: Tkinter-based interface for test selection, configuration, and analytics dashboards.
- **Data Visualization**: Matplotlib-powered charts for performance trends, emotion variation, and per-task analysis.
- **Customizable Settings**: Adjustable camera, hand choice, and tolerance (Box-and-Block: 1.0–4.5).
- **Cross-Platform**: Compatible with Windows, macOS, and Linux.
- **Data Logging**: Detailed per-frame logs and summary statistics saved in CSV files for analysis.

## 🧰 Prerequisites

### Hardware
- Webcam or built-in camera
- Modern CPU with ≥4GB RAM

### Software
- Python 3.8+
- pip package installer

### Dependencies
Install via `requirements.txt`:
```
opencv-python
mediapipe
onnxruntime
torchvision
pandas
numpy
scipy
matplotlib
tkinter
```

## 📦 Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Abhiram0703/RehabNet.git
   cd RehabNet
   ```

2. **Set Up Virtual Environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 File Structure

```
RehabNet/
├── Alphabets/
│   ├── A.png
│   ├── K.png
│   ├── L.png
│   ├── M.png
│   ├── N.png
│   ├── S.png
│   └── W.png
├── BBT_details.csv
├── BBT_logs/
│   ├── BBT_logs_AR_20250516_173118.csv
│   ├── BBT_logs_Abhiram_20250521_203440.csv
│   ├── BBT_logs_Abhiram_20250521_203655.csv
│   ├── BBT_logs_Abhiram_20250521_204045.csv
│   └── BBT_logs_Abhiram_20250522_215022.csv
├── HandTrackingModule.py
├── LW_logs/
│   ├── letter_writing_logs_AR_20250516_175122.csv
│   ├── letter_writing_logs_AR_20250516_175410.csv
│   ├── letter_writing_logs_AR_20250516_180309.csv
│   ├── letter_writing_logs_Abhiram_20250521_204631.csv
│   ├── letter_writing_logs_Abhiram_20250521_204943.csv
│   ├── letter_writing_logs_Abhiram_20250521_205118.csv
│   ├── letter_writing_logs_Abhiram_20250522_215230.csv
│   ├── letter_writing_logs_test_user_20250516_180935.csv
│   └── letter_writing_logs_test_user_20250516_221201.csv
├── README.md
├── box_and_block.py
├── hsemotion_1280.onnx
├── letter_writing.py
├── letter_writing_details.csv
├── main.py
├── main_box_and_block.py
├── requirements.txt
├── utils.py
└── writing_letters_main.py
```

## ▶️ Usage

### Launching the Application
Run the main script to open the Tkinter GUI:
```bash
python main.py
```
The GUI provides options to select either the Letter Writing or Box-and-Block Test, configure settings (username, camera, hand, tolerance), and view analytics.

---

## 📋 Letter Writing Test

### Overview
The Letter Writing Test evaluates fine motor skills and emotional responses by having users trace a sequence of letters (A, K, L, M, N, S, W) using their index finger on a virtual canvas. The system tracks hand movements, calculates error distances, monitors emotions, and logs performance metrics.

### Objective
Trace each letter accurately while following an on-screen template, with real-time feedback on accuracy and emotional state.

### Game Rules
1. **Drawing Mechanism**:
   - Use the index finger tip of the selected hand (left or right) to draw.
   - Drawing occurs within canvas bounds (30 < x < 610, 30 < y < 400).
   - Drawing pauses if:
     - The non-selected hand is detected (implicit pause feature).
     - The index finger moves outside the canvas bounds.
     - No hand is detected.
2. **Letter Templates**:
   - Letters (A, K, L, M, N, S, W) are loaded as 200×200 PNG images from the `Alphabets` folder.
   - Positioned at (128, 48) for the left hand or (288, 48) for the right hand.
   - If a template fails to load, a blank image with the letter’s character is used.
3. **Controls**:
   - **S**: Save current task data and proceed to the next letter (6-second cooldown).
   - **R**: Reset the current letter, clearing the canvas and metrics.
   - **Q**: Save final stats and exit.
4. **Task Progression**:
   - The test includes 7 letters, cycled in order.
   - After completing all letters, the system displays “All Letters Completed!” and prompts to exit with ‘Q’.

### Technical Details
- **Camera Setup**:
  - Initializes webcam at 640×480 resolution, 60 FPS.
  - Supports multiple backends (CAP_AVFOUNDATION, CAP_DSHOW, CAP_V4L2, CAP_ANY).
- **Hand Tracking**:
  - Uses `HandDetector` from `HandTrackingModule` to track the index finger tip (landmark 8).
  - Draws green lines (RGB: 0, 255, 0, thickness: 5 pixels) on a 640×480×3 canvas (`imgCanvas`).
- **Emotion Detection**:
  - Captures emotions every 0.05 seconds during active drawing.
  - Uses MediaPipe for face detection and `hsemotion_1280.onnx` for classifying emotions (Positive, Negative, Neutral).
  - Emotions are not saved during resets; only successful attempts (via ‘S’) are logged.
- **Error Calculation**:
  - Computes average error distance between drawn pixels and template pixels using `cKDTree` for nearest-neighbor search.
  - Converts to millimeters (1 pixel = 0.264 mm at 96 DPI).
  - Formula:  
    \[
    \text{Avg Error Distance} = \left( \frac{\sum \text{minimum distances}}{\text{number of drawn pixels}} \right) \times 0.264 \, \text{mm/pixel}
    \]
- **Movement Speed**:
  - Calculates speed as the Euclidean distance between consecutive finger positions divided by time elapsed (pixels/second).
  - Logged as 0 when not drawing.
- **Data Logging**:
  - **Per-Frame Logs**: Stored in `LW_logs/letter_writing_logs_<username>_<timestamp>.csv` with columns: Timestamp, Letter, Emotion, Movement Speed, Avg Error Distance.
  - **Summary Stats**: Saved in `letter_writing_details.csv` with columns: Username, Timestamp, Hand, Completion Time, Dominant Emotion, Avg Movement Speed, Avg Error Distance.
- **User Interface**:
  - Displays real-time video feed with overlaid template, drawn lines, and metrics (letter, hand, emotion, time, error distance, movement speed).
  - Footer instructions: “Press S to save & next task”, “Press R to reset”.
  - Shows cooldown timer for ‘S’ key and face bounding box with emotion label.

### Analysis Dashboard
The analytics dashboard, accessible post-test, includes:
1. **Summary Tab**:
   - Displays username, session timestamp, hand used, and metrics: Total Letters, Completion Time, Avg Speed, Avg Error, Dominant Emotion.
   - Lists previous sessions in a Treeview widget for selection.
2. **Emotion Variation Tab**:
   - Step plot of emotions (Positive=1, Neutral=0, Negative=-1) over time for a selected letter.
   - Filters for active tracing (Movement Speed > 0).
3. **Movement Analysis Tab**:
   - Bar plot of average movement speed per letter.
4. **Error Analysis Tab**:
   - Bar plot of average error distance per letter.
5. **Emotion vs Performance Tab**:
   - Grouped bar plot comparing Movement Speed or Avg Error Distance across emotions per letter.
6. **Trends Tab**:
   - Line plots for Completion Time, Avg Movement Speed, Avg Error Distance, or Emotion Stability across sessions.

### Clinical Utility
- **Motor Skills**: Tracks precision and speed, identifying fine motor impairments (e.g., tremors, coordination issues).
- **Emotional State**: Monitors emotional responses to assess stress or engagement, aiding psychological evaluations.
- **Progress Tracking**: Compares performance across sessions to evaluate therapy effectiveness.

---

## 📋 Box-and-Block Test

### Overview
The Box-and-Block Test assesses gross manual dexterity and emotional states by having users move virtual blocks between screen compartments within a 60-second time limit. The system tracks pinch gestures, block movements, and emotions.

### Objective
Move blocks from one compartment to another without touching the dividing line, while monitoring performance and emotional metrics.

### Game Rules
1. **Compartments**:
   - Screen divided by a vertical line at x = WIDTH/2.
   - Left hand: Move blocks from left to right compartment.
   - Right hand: Move blocks from right to left.
2. **Block Movement**:
   - Pinch thumb and index finger (distance < 80 pixels) to grab a block within its tolerance region.
   - Move the block to the opposite compartment without crossing the dividing line.
   - Release the pinch to drop the block.
3. **Scoring**:
   - **Success**: Block moved to the correct compartment without touching the line (x > WIDTH/2 + block_width/2 for left hand; x < WIDTH/2 - block_width/2 for right hand).
   - **Attempt**: Increments current block number when a block is grabbed (0.4-second cooldown).
   - **Success Rate**:  
     \[
     \text{Success Rate} = \left( \frac{\text{successful moves}}{\text{total attempts}} \right) \times 100\%
     \]
4. **Time Limit**: 60 seconds.
5. **Controls**:
   - **R**: Reset game state, blocks, and timers.
   - **Q**: Save stats and exit.

### Technical Details
- **Camera Setup**:
  - Matches screen resolution (via `pyautogui.size()`), 60 FPS.
  - Uses multiple backends for compatibility.
- **Hand Tracking**:
  - Detects one hand (left or right) using `HandDetector` (maxHands=1).
  - Tracks thumb tip (landmark 4) and index finger tip (landmark 8) for pinch detection.
  - Cursor position is the index finger tip.
- **Emotion Detection**:
  - Captures emotions every 0.5 seconds during movement and for 1.5 seconds after dropping a block.
  - Uses MediaPipe for face detection and `hsemotion_1280.onnx` for emotion classification.
  - Valence scores (-1 to 1) map to: Positive (>0.33), Neutral (-0.33 to 0.33), Negative (<-0.33).
  - **Emotion Score**:  
    \[
    \text{Emotion Score} = 50 + (\text{weighted_valence} \times 50)
    \]
    where \(\text{weighted_valence} = (\text{avg_movement} \times 0.7) + (\text{avg_drop} \times 0.3)\).
- **Block Management**:
  - One movable block (100×100 pixels, random color: red, green, blue, yellow) at (80, HEIGHT-150) for left hand or (WIDTH-80, HEIGHT-150) for right hand.
  - Up to one non-movable block stored after a drop.
- **Performance Metrics**:
  - Score: Number of successful block moves.
  - Avg Movement Speed: Distance moved ÷ time taken (pixels/second).
  - Avg Time per Block: Mean duration per block move.
  - Success Rate: Percentage of successful moves.
- **Data Logging**:
  - **Per-Frame Logs**: Stored in `BBT_logs/BBT_logs_<username>_<timestamp>.csv` with columns: Timestamp, Block Number, Event Type (MOVEMENT/DROP), Emotion, Valence, Movement Speed, Score.
  - **Summary Stats**: Saved in `BBT_details.csv` with columns: Username, Timestamp, Hand, Score, Tolerance, Emotion Score, Dominant Emotion, Avg Movement Speed, Avg Time per Block, Success Rate.
- **User Interface**:
  - Fullscreen OpenCV window with blocks, dividing line (pink, 3 pixels), and semi-transparent overlay (10% opacity).
  - Displays metrics: Hand, Score, Current Emotion, Emotion Score, Time Remaining, Current Block, Current Speed, Success Rate.
  - Shows “MOVEMENT EMOTION CAPTURE ACTIVE” (yellow) or “DROP EMOTION CAPTURE ACTIVE” (green) indicators.
  - Face bounding box with emotion label.

### Analysis Dashboard
The post-test dashboard includes:
1. **Summary Tab**:
   - Displays 7 metrics: Score, Emotion Score, Dominant Emotion, Avg Movement Speed, Avg Time per Block, Success Rate, Tolerance.
   - Lists previous sessions for selection.
2. **Emotion Distribution Tab**:
   - Bar plot of emotion frequencies (Positive, Negative, Neutral) for successful block moves.
3. **Movement Analysis Tab**:
   - Line plot of average movement speed per successful block.
4. **Time Analysis Tab**:
   - Bar plot of time taken per successful block.
5. **Emotion Timeline Tab**:
   - Stepped line plot of emotion states (Positive=1, Neutral=0, Negative=-1) per 1-second bin.
6. **Emotion Valence Tab**:
   - Line plot of weighted valence per successful block.
7. **Trends Tab**:
   - Line plots for Total Blocks Moved, Avg Movement Speed, Emotion Score, or Avg Time per Block across sessions.

### Clinical Utility
- **Motor Dexterity**: Measures gross motor skills, identifying issues like hesitancy or fatigue.
- **Emotional Response**: Tracks emotional stability, detecting stress or engagement levels.
- **Therapy Monitoring**: Trends analysis supports longitudinal progress tracking.

---

## 📝 Output Files

| **Test**          | **Summary CSV**                     | **Detailed Logs**                                      |
|-------------------|-------------------------------------|-------------------------------------------------------|
| Letter Writing    | `letter_writing_details.csv`        | `LW_logs/letter_writing_logs_<user>_<time>.csv`       |
| Box-and-Block     | `BBT_details.csv`                   | `BBT_logs/BBT_logs_<user>_<time>.csv`                 |

