# 🚗 Real-Time Driver Drowsiness & Fatigue Detection System

An intelligent computer vision and machine learning system built with **Python**, **OpenCV**, and **Google MediaPipe** to monitor driver alertness in real-time. The system detects the driver's face, eyes, and mouth, evaluates eye openness/closure using precise Eye Aspect Ratio (EAR) math, detects yawning, tracks 3D head pose, and triggers an urgent audible alarm through headphones/speakers whenever eyes remain closed beyond a safety threshold.

---

## 🌟 Key Features

- **Dual-Engine Computer Vision Architecture**:
  - **Primary Engine**: Google MediaPipe 478-point 3D Landmark Mesh calculating the exact mathematical **Eye Aspect Ratio (EAR)**:
    $$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$
  - **Fallback Engine**: OpenCV Haar Cascades with anatomical eye zoning and CLAHE contrast enhancement (runs with zero external model dependencies, use `--opencv`).
- **Yawn Detection & Fatigue Warning**: Calculates **Mouth Aspect Ratio (MAR)** to detect repeated yawning as an early sign of driver fatigue before micro-sleeps occur.
- **3D Head Pose Estimation**: Detects head nodding forward (`PITCH`) and looking away from the road (`YAW > 28°`) using `cv2.solvePnP`.
- **Micro-Sleep vs. Blink Discrimination**: Intelligent temporal state machine that counts natural blinks (~100–400ms) without triggering false alarms, while catching prolonged eye closures (≥ 1.8s).
- **Multi-Engine Audio Alert System**:
  - Loud 16-bit PCM emergency siren via **pygame.mixer** → routes directly to headphones & speakers.
  - Instant voice alert via **Windows SAPI**: *"Wake up! Drowsiness detected!"*
  - Falls back to **sounddevice** PCM streaming or **winsound** if needed.
- **Automotive Cockpit HUD Overlay**:
  - Live driver alertness status banner (Green = Awake, Yellow = Caution / Yawning, Flashing Red = Drowsiness Detected).
  - Futuristic corner brackets on face, polygonal eye/mouth contour tracking.
  - Real-time telemetry dashboard: EAR, MAR, Head Pitch/Yaw, closure timer gauge, blink counter, blinks/min rate, yawn count, drowsiness incidents, alarm status, FPS.
- **Interactive Keyboard Controls** (`q`, `+`, `-`, `m`, `t`, `r`, `s`).
- **Session Trip Logger & HTML Safety Report**: Auto-generates a dark-mode Chart.js interactive HTML safety report with a Driver Alertness Score (0–100) and events timeline upon exit.
- **Simulation & Testing Suite**: 5 automated unit tests and a synthetic video generator for headless/no-webcam testing.

---

## 📁 Project Structure

```
driver-drowsiness-detection/
├── app.py                   # Main interactive real-time application with HUD
├── drowsiness_detector.py   # Dual-engine CV core, EAR/MAR/PnP, AlertManager, DrowsinessTracker
├── session_logger.py        # Trip telemetry logger and HTML safety report generator
├── create_alarm_sound.py    # Procedural WAV alarm & chime asset generator
├── download_assets.py       # Auto-downloader for OpenCV Haar Cascades + MediaPipe model
├── test_system.py           # Automated unit test suite & synthetic driver simulation video
├── alarm.wav                # 16-bit PCM emergency siren audio asset
├── warning.wav              # Caution chime audio asset
├── simulation_driver.mp4    # Generated synthetic test video
├── requirements.txt         # Project dependencies
├── cascades/                # Downloaded cascade models & MediaPipe task (offline-ready)
└── README.md                # Documentation and usage guide
```

---

## 🚀 Quick Start

### 1. Set Active Workspace (Recommended)
Open this project folder in your IDE:
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
- Force the OpenCV Haar Cascade engine (no MediaPipe model needed):
  ```bash
  python app.py --opencv
  ```
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
| **`q`** or **`ESC`** | Exit and generate HTML Driver Safety Report. |
| **`+`** / **`=`** | Increase drowsiness time threshold (+0.2s) — less sensitive. |
| **`-`** / **`_`** | Decrease drowsiness time threshold (-0.2s) — more sensitive. |
| **`m`** | Toggle audio alarm **MUTE / UNMUTE**. |
| **`t`** | Play a one-off **Test Beep** through headphones/speakers. |
| **`r`** | Reset session statistics (blinks, yawns, incidents). |
| **`s`** | Save a high-resolution snapshot screenshot to disk. |

---

## 🔊 Audio Alert System

The alarm outputs directly to your **headphones or speakers** (not the internal PC speaker):

1. **Emergency Siren** (`alarm.wav`): Loud alternating 2500 Hz / 1900 Hz pulse played via `pygame.mixer` at full volume, looping continuously until the driver's eyes reopen.
2. **Voice Wake-up**: Windows SAPI synthesizes *"Wake up! Drowsiness detected!"* asynchronously.
3. **Fallback chain**: `sounddevice` → `winsound` if pygame is unavailable.

> **Press `t`** at any time in the camera window to immediately test audio output through your current headphones/speakers.

---

## 📊 Session Safety Reports

When you press `q` to exit, the system automatically generates an interactive HTML report in the `logs/` directory:

```
logs/trip_YYYYMMDD_HHMMSS_report.html
```

The report includes:
- **Driver Alertness Score** (0–100, Grade A/B/C)
- **Chart.js telemetry graphs**: EAR and MAR over time
- **Safety Events Timeline**: All drowsiness alarms, yawns, head nods, and distraction alerts

---

## 🧪 Running Automated Tests

Run the included test suite to verify vision models, state transitions, and audio lifecycle:
```bash
python test_system.py
```
This runs all 5 unit tests and generates the `simulation_driver.mp4` test video.

---

## 🧠 How It Works (ML & CV Concepts)

1. **Face Localization**: Multi-scale detection via MediaPipe (478 3D landmarks) or OpenCV Haar Cascades.
2. **EAR — Eye Aspect Ratio**: Open eyes maintain a stable EAR ≥ 0.22. Closed eyes collapse the vertical axis, dropping EAR sharply.
3. **MAR — Mouth Aspect Ratio**: A yawn stretches the vertical mouth distance while horizontal remains constant, raising MAR > 0.58.
4. **3D Head Pose (`cv2.solvePnP`)**: Maps 6 facial anchor points to a known 3D face model to compute Pitch/Yaw/Roll angles in real-time.
5. **Temporal State Tracking**:
   - $\Delta t < 0.45\text{s}$: Natural blink (counter increments, no alarm).
   - $0.8\text{s} \le \Delta t < 1.8\text{s}$: `DROWSY_WARNING` caution banner.
   - $\Delta t \ge 1.8\text{s}$: `ALARM` — loud siren + voice alert + flashing red screen.
   - Reopening eyes instantly resets the closure timer and silences the alarm.
6. **Distraction Detection**: `|Yaw| > 28°` for > 2 seconds triggers `DISTRACTED` alert.
7. **Head Nodding**: `|Pitch| > 18°` triggers `HEAD_NOD` micro-sleep warning.
