RehabNet: AI-Driven Multimodal Vision Therapy for Post-Stroke Hand Recovery
RehabNet is a computerized system designed to assess fine motor skills and manual dexterity through two interactive tests: the Letter Writing Test and the Box and Block Test. Using computer vision and hand-tracking technology, RehabNet evaluates user performance while monitoring emotional responses via facial expression analysis. The system provides detailed performance analytics through a user-friendly Tkinter-based interface, making it suitable for rehabilitation, research, and clinical applications.


Features

Two Assessment Tests:
Letter Writing Test: Users trace letters (A, K, L, M, N, S, W) using hand movements, assessing fine motor skills and emotional responses.
Box and Block Test: Users move virtual blocks between compartments, measuring gross manual dexterity.


Hand Tracking: Utilizes MediaPipe for real-time hand detection, supporting left or right hand usage.
Emotion Detection: Analyzes facial expressions using an ONNX-based model (hsemotion_1280.onnx) to track emotions (Positive, Neutral, Negative).
Performance Analytics: Generates detailed metrics (e.g., completion time, movement speed, error distance, emotion scores) saved in CSV files.
Interactive UI: Tkinter-based interface with a main menu to select tests, configure settings, and view analytics dashboards.
Data Visualization: Matplotlib-powered charts for performance trends, emotion variation, and per-task analysis.
Customizable: Adjustable parameters (e.g., camera selection, hand choice, tolerance for Box and Block Test).
Cross-Platform: Compatible with Windows, macOS, and Linux (with appropriate camera support).

Prerequisites

Hardware:
Webcam or built-in camera for hand and face tracking.
Computer with at least 4GB RAM and a modern CPU.(preferrable)


Software:
Python 3.8 or higher.
pip for package installation.


Dependencies (installed via requirements.txt):
opencv-python: Computer vision for camera input and image processing.
mediapipe: Hand and face detection.
onnxruntime: Emotion detection model inference.
torchvision: Image preprocessing for emotion detection.
pandas: Data handling and CSV processing.
numpy: Numerical computations.
scipy: Optimized distance calculations (cKDTree).
matplotlib: Data visualization.
tkinter: GUI framework (usually included with Python).



Installation

Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Then install the dependencies:
pip install -r requirements.txt




File Structure
rehabnet/
├── main.py                     # Main entry point to select between tests
├── writing_letters_main.py     # UI for Letter Writing Test
├── main_box_and_block.py       # UI for Box and Block Test
├── letter_writing.py           # Core logic for Letter Writing Test
├── box_and_block.py            # Core logic for Box and Block Test
├── utils.py                    # Utility functions (e.g., camera detection, text overlay)
├── HandTrackingModule.py       # MediaPipe hand tracking module
├── hsemotion_1280.onnx         # Emotion detection model
├── Alphabets/                  # Folder containing letter PNGs
│   ├── A.png
│   ├── K.png
│   ├── L.png
│   ├── M.png
│   ├── N.png
│   ├── S.png
│   └── W.png
├── letter_writing_details.csv  # Output: Summary stats for Letter Writing Test
├── letter_writing_logs_*.csv   # Output: Per-session logs for Letter Writing Test
├── BBT_details.csv             # Output: Summary stats for Box and Block Test
├── BBT_logs_*.csv              # Output: Per-session logs for Box and Block Test
└── README.md                   # This file

Usage

Run the Application:
python main.py


A Tkinter window opens with options to select either the Letter Writing Test or Box and Block Test.


Select a Test:

Click Letter Writing Test to launch its configuration UI.
Click Box and Block Test to launch its configuration UI.


Configure and Start a Test:

Letter Writing Test:
Enter a username (no commas allowed).
Select a camera (e.g., "Camera 0") and hand (Left or Right).
Click START TEST.
Trace letters using hand movements, following on-screen templates.
(If the camera detects the other hand which is not chosen then it pauses writing you can take advantage of it while tracing the letters)
Press S to save and move to the next letter, R to reset, Q to quit.


Box and Block Test:
Enter a username.
Select a camera, hand, and tolerance (1.0 to 4.5).
Click START TEST.
Move virtual blocks between compartments within the time limit.
(If you choose right hand. then move the block from right compartment to left compartment else reverse)
Specific controls depend on box_and_block.py implementation.




View Analytics:

After completing a test, an analytics dashboard displays:
Summary statistics (e.g., completion time, score, average speed).
Charts for movement speed, error distance (Letter Writing), emotion variation, and trends across sessions.
Previous sessions list to review past performance.


Use dropdowns to filter data (e.g., by letter or metric).
Click View Test Analysis to switch sessions or New Session to start a new test.


Output Files:

Letter Writing Test:
letter_writing_details.csv: Session summary (username, timestamp, hand, completion time, etc.).
letter_writing_logs_[username]_[timestamp].csv: Frame-by-frame logs (letter, emotion, speed, error).


Box and Block Test:
BBT_details.csv: Session summary (score, emotion score, etc.).
BBT_logs_[username]_[timestamp].csv: Event logs (block moves, valence, speed).





Test Descriptions
Letter Writing Test

Purpose: Assesses fine motor skills by tracking hand movements to trace letters.
Mechanics:
Users trace letters displayed as templates (200x200 pixels) using their index finger.
Templates are positioned based on hand choice (left: x=128, right: x=288).
Reference templates show the target letter (left hand: x=384, right hand: x=32).
Key controls:
S: Save current drawing and proceed to the next letter (6-second cooldown).
R: Reset the current letter.
Q: Save stats and exit.


Metrics: Completion time, movement speed (px/s), error distance (mm), dominant emotion.


Emotion Detection: Captures facial expressions every 0.05 seconds, mapping to Positive, Neutral, or Negative.
Output: CSV files with session stats and detailed logs.

Box and Block Test

Purpose: Measures gross manual dexterity by simulating block-moving tasks.
Mechanics:
Users move virtual blocks from one compartment to another within a time limit.
Configurable tolerance (1.0 to 4.5) adjusts task difficulty.
Hand tracking detects block interactions (specifics depend on box_and_block.py).
Metrics: Score (successful blocks), success rate, average speed, emotion score.


Emotion Detection: Tracks valence during movement and drop phases, weighted (70% movement, 30% drop).
Output: CSV files with session stats and event logs.

Analytics Dashboard

Common Features:
Summary Tab: Displays key metrics (e.g., score, completion time, emotion) in card-based layouts.
Previous Sessions: Table to view and switch between past sessions.
Navigation: Buttons for viewing other sessions or starting a new test.


Letter Writing Test:
Tabs: Emotion Variation, Movement Analysis, Error Analysis, Emotion vs Performance, Trends.
Charts:
Emotion over time per letter (dropdown).
Average speed and error distance per letter.
Emotion correlation with speed or error.
Trends (completion time, speed, error, emotion stability) across sessions.




Box and Block Test:
Tabs: Emotion Analysis, Movement Analysis, Time Analysis, Emotion Timeline, Emotion Score Analysis, Trends.
Charts:
Emotion distribution by successful block.
Speed and time per block.
Emotion timeline (mode per second).
Valence scores per block.
Trends (blocks moved, speed, emotion score, time per block) across sessions.





Troubleshooting

Camera Not Detected:
Ensure a webcam is connected and drivers are installed.
Run utils.returnCameraIndexes() to verify available cameras.
Try different camera indices in the UI.


Module Import Errors:
Verify all dependencies are installed (pip install -r requirements.txt).
Check that utils.py and HandTrackingModule.py are in the project root.


Missing ONNX Model:
Download hsemotion_1280.onnx and place it in the project root.


Letter Images Missing:
Ensure Alphabets/ contains all required PNGs (A.png, K.png, etc.).
Create placeholder images if needed (white background, black letter, 200x200).


CSV Errors:
Check write permissions in the project directory.
Ensure CSV files are not open in other applications during execution.


Performance Issues:
Reduce camera resolution in letter_writing.py (e.g., WIDTH=320, HEIGHT=240).
Close background applications to free up CPU resources.


UI Issues:
If buttons are unresponsive, ensure Tkinter is properly installed (python -m tkinter).
Verify Python version compatibility (3.8+ recommended).

