# 🚗 Real-Time Driver Drowsiness Detection System

An intelligent computer vision and machine learning system built with **Python** and **OpenCV** to monitor driver alertness in real-time. The system detects the driver's face and eyes, evaluates eye openness/closure across video frames, differentiates natural blinks from micro-sleeps, and triggers an urgent multi-threaded audible alarm whenever eyes remain closed beyond a safety threshold.

---

## 🌟 Key Features

- **Dual-Engine Computer Vision Architecture**:
  - **Primary Engine**: Google MediaPipe 478-point 3D Landmark Mesh calculating the exact mathematical **Eye Aspect Ratio (EAR)**:
    $$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$
  - **Fallback Engine**: OpenCV Haar Cascades with anatomical eye zoning and CLAHE contrast enhancement (runs with zero external model dependencies).
- **Yawn Detection & Fatigue Warning**: Calculates **Mouth Aspect Ratio (MAR)** to detect repeated yawning as an early sign of driver fatigue before micro-sleeps occur.
- **Micro-Sleep vs. Blink Discrimination**: Intelligent temporal state machine that counts natural blinks (~100–400ms) without triggering false alarms, while catching prolonged eye closures (≥ 1.8s).
- **Asynchronous Audio Alert System**: Background multi-threaded Windows sound manager (`winsound.Beep`) emits urgent alternating alarm pulses without dropping camera framerate or causing UI lag.
- **Automotive Cockpit HUD Overlay**:
  - Live driver alertness status banner (Green = Awake, Yellow = Caution / Yawning, Flashing Red = Drowsiness Detected).
  - Highlighting bounding boxes with futuristic corner brackets and polygonal eye/mouth contour tracking.
  - Real-time telemetry dashboard showing EAR, MAR, closure timer gauge, blink counter, blinks/min rate, yawn count, drowsiness episodes, alarm audio mute status, and FPS.
- **Interactive Keyboard Controls**:
  - Adjust sensitivity / threshold on the fly (`+` / `-`).
  - Mute/unmute alarm (`m`).
  - Test alarm sound with one keypress (`t`).
  - Capture frame snapshots (`s`).
- **Simulation & Testing Suite**: Includes automated unit tests and a synthetic video generator for headless environments or users without a webcam.

---

## 📁 Project Structure

```
driver-drowsiness-detection/
├── app.py                   # Main interactive real-time application with HUD
├── drowsiness_detector.py   # Core vision detector, temporal tracker & audio alert manager
├── download_assets.py       # Auto-downloader for OpenCV Haar Cascade XML models
├── test_system.py           # Automated unit test suite & synthetic driver simulation
├── simulation_driver.mp4    # Generated synthetic test video
├── requirements.txt         # Project dependencies (opencv-python, numpy, scipy)
├── cascades/                # Downloaded cascade models (offline-ready)
└── README.md                # Documentation and usage guide
```

---

## 🚀 Quick Start

### 1. Set Active Workspace (Recommended)
Open this project folder in Antigravity or VS Code:
```
C:\Users\SARANG\.gemini\antigravity-ide\scratch\driver-drowsiness-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Real-Time Webcam Monitor
Ensure your webcam is connected, then run:
```bash
python app.py
```

### 4. Run with Custom Options
- Change webcam device index (e.g., camera 1):
  ```bash
  python app.py --camera 1
  ```
- Adjust closure threshold (e.g., 2.0 seconds):
  ```bash
  python app.py --threshold 2.0
  ```
- Run with a pre-recorded or synthetic video:
  ```bash
  python app.py --video simulation_driver.mp4
  ```

---

## 🎮 Interactive Keyboard Controls

While the camera feed window is focused:

| Key | Action |
|:---:|:---|
| **`q`** or **`ESC`** | Exit the application safely. |
| **`+`** / **`=`** | Increase drowsiness time threshold (+0.2s) for less sensitive alerts. |
| **`-`** / **`_`** | Decrease drowsiness time threshold (-0.2s) for faster alerts. |
| **`m`** | Toggle audio alarm **MUTE / UNMUTE**. |
| **`t`** | Play a one-off **Test Beep** to verify speaker output. |
| **`r`** | Reset session statistics (blinks and drowsy episode count). |
| **`s`** | Save a high-resolution snapshot screenshot to disk. |

---

## 🧪 Running Automated Tests

Run the included test suite to verify vision models, state transitions, and audio threads:
```bash
python test_system.py
```
This will run all 5 unit tests and generate the `simulation_driver.mp4` video.

---

## 🧠 How the Machine Learning & CV Concept Works

1. **Face Localization**: OpenCV detects the driver's face bounding box $(x, y, w, h)$ at multi-scale resolution.
2. **Anatomical Eye Isolation**: Human eyes reside in the upper region of the face ($y \in [0.22h, 0.54h]$). Constraining the search space to these anatomical zones prevents false detections on mouths, nostrils, or background textures.
3. **Contrast & Pupil Analysis**: Histogram equalization (CLAHE) is applied to handle uneven cabin lighting. An open eye displays high-contrast dark iris/pupil features against the white sclera. When eyelids close, the eye becomes skin-toned and uniform, dropping the darkness gradient and aspect ratio below the detection threshold.
4. **Temporal State Tracking**:
   - $\Delta t < 0.45\text{s}$: Registered as a natural blink (increments blink counter).
   - $0.8\text{s} \le \Delta t < 1.8\text{s}$: Enters `DROWSY_WARNING` caution state.
   - $\Delta t \ge 1.8\text{s}$: Enters `ALARM` state, initiating non-blocking background sound beeps and full-screen flashing alerts.
   - Reopening eyes instantly resets the closure timer and stops the alarm.
