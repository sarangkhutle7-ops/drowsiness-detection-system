"""
Driver Drowsiness, Fatigue, and Distraction Detection Core Engine.
Features:
 - Dual-Engine Vision: MediaPipe 478-point 3D Landmark Mesh
   with automatic OpenCV Haar Cascade fallback.
 - Eye Aspect Ratio (EAR) calculation for eye closure tracking.
 - Mouth Aspect Ratio (MAR) calculation for yawn detection.
 - Head Pose Estimation (Pitch, Yaw, Roll) for head nod & distraction detection.
 - Non-blocking asynchronous audio alerts using Windows winsound.
 - Multi-factor driver state tracking (Awake, Blinking, Drowsy, Yawning, Nodding, Distracted).
"""

import math
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from download_assets import ensure_cascades

# Sound support for Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# MediaPipe support
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


def euclidean_dist(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


def calculate_ear(landmarks: List[Tuple[float, float]], indices: List[int]) -> float:
    """
    Eye Aspect Ratio (EAR) based on Soukupova & Cech (2016).
    indices: [p1, p2, p3, p4, p5, p6]
    EAR = (|p2 - p6| + |p3 - p5|) / (2.0 * |p1 - p4|)
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)

    if horizontal <= 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks: List[Tuple[float, float]]) -> float:
    """Mouth Aspect Ratio (MAR) for yawn detection."""
    p_left = landmarks[61]
    p_right = landmarks[291]
    p_top = landmarks[13]
    p_bottom = landmarks[14]
    p_v1_top, p_v1_bot = landmarks[81], landmarks[178]
    p_v2_top, p_v2_bot = landmarks[311], landmarks[402]

    horizontal = euclidean_dist(p_left, p_right)
    if horizontal <= 1e-6:
        return 0.0

    v1 = euclidean_dist(p_top, p_bottom)
    v2 = euclidean_dist(p_v1_top, p_v1_bot)
    v3 = euclidean_dist(p_v2_top, p_v2_bot)

    return (v1 + v2 + v3) / (3.0 * horizontal)


# 3D Facial Model Reference Points for solvePnP
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (index 1)
    (0.0, -330.0, -65.0),        # Chin (index 152)
    (-225.0, 170.0, -135.0),     # Right eye corner (index 33)
    (225.0, 170.0, -135.0),      # Left eye corner (index 263)
    (-150.0, -150.0, -125.0),    # Right mouth corner (index 61)
    (150.0, -150.0, -125.0)      # Left mouth corner (index 291)
], dtype=np.float64)


# Audio support
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

try:
    import win32com.client
    HAS_SAPI = True
except Exception:
    HAS_SAPI = False


class AlertManager:
    """
    Robust Multi-Engine Audio Alert System.
    Plays loud emergency siren alarms via pygame.mixer directly to headphones/speakers,
    with instant stop, voice wake-up synthesis, and sounddevice fallback.
    """

    def __init__(self, alarm_wav: str = "alarm.wav", warning_wav: str = "warning.wav", *args, **kwargs):
        self.alarm_wav = alarm_wav
        self.warning_wav = warning_wav
        self.frequency = kwargs.get("frequency", 2200)
        self.duration_ms = kwargs.get("duration_ms", 140)
        self.is_active = False
        self.is_muted = False
        self.volume = 1.0

        self.alarm_sound = None
        self.warning_sound = None
        self.voice_engine = None
        self.last_voice_time = 0.0

        # Ensure WAV files exist
        if not os.path.exists(self.alarm_wav) or not os.path.exists(self.warning_wav):
            try:
                from create_alarm_sound import generate_audio_assets
                generate_audio_assets(os.path.dirname(os.path.abspath(self.alarm_wav)) or ".")
            except Exception:
                pass

        # Load pygame sounds
        if HAS_PYGAME:
            try:
                if os.path.exists(self.alarm_wav):
                    self.alarm_sound = pygame.mixer.Sound(self.alarm_wav)
                    self.alarm_sound.set_volume(self.volume)
                if os.path.exists(self.warning_wav):
                    self.warning_sound = pygame.mixer.Sound(self.warning_wav)
                    self.warning_sound.set_volume(self.volume * 0.7)
            except Exception as e:
                print(f"[AlertManager] Pygame sound load error: {e}", file=sys.stderr)

        # Initialize Windows Voice synthesizer
        if HAS_SAPI:
            try:
                self.voice_engine = win32com.client.Dispatch("SAPI.SpVoice")
                self.voice_engine.Rate = 2  # Urgent speaking speed
                self.voice_engine.Volume = 100
            except Exception:
                self.voice_engine = None

        # Background worker thread for fallback sound / voice monitoring
        self._stop_event = threading.Event()
        self._alarm_event = threading.Event()
        self._thread = threading.Thread(target=self._fallback_sound_loop, daemon=True)
        if not HAS_PYGAME:
            self._thread.start()

    def _fallback_sound_loop(self):
        """Emergency fallback loop using sounddevice or winsound."""
        sample_rate = 44100
        duration = 0.2
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = (0.7 * np.sin(2 * np.pi * 2200 * t)).astype(np.float32)

        while not self._stop_event.is_set():
            if self._alarm_event.is_set() and not self.is_muted:
                try:
                    if HAS_SOUNDDEVICE:
                        sd.play(tone, sample_rate)
                        sd.wait()
                        time.sleep(0.05)
                    elif HAS_WINSOUND:
                        winsound.Beep(2200, 180)
                        time.sleep(0.05)
                except Exception:
                    time.sleep(0.1)
            else:
                time.sleep(0.04)

    def trigger(self):
        """Fires the loud emergency alarm instantly."""
        if self.is_muted or self.is_active:
            return

        self.is_active = True

        if HAS_PYGAME and self.alarm_sound is not None:
            try:
                # -1 means loop indefinitely until stopped
                self.alarm_sound.play(loops=-1)
            except Exception as e:
                print(f"[AlertManager] Pygame play error: {e}", file=sys.stderr)
        else:
            self._alarm_event.set()

        # Speak voice alert asynchronously if driver has not reacted
        now = time.time()
        if self.voice_engine and (now - self.last_voice_time > 3.0):
            self.last_voice_time = now
            threading.Thread(target=self._speak_alert, daemon=True).start()

    def stop(self):
        """Instantly turns off the alarm."""
        if not self.is_active:
            return

        self.is_active = False

        if HAS_PYGAME and self.alarm_sound is not None:
            try:
                self.alarm_sound.stop()
            except Exception:
                pass
        else:
            self._alarm_event.clear()

    def play_warning_chime(self):
        """Plays a gentle reminder chime for caution state."""
        if self.is_muted:
            return
        if HAS_PYGAME and self.warning_sound is not None:
            try:
                self.warning_sound.play()
            except Exception:
                pass

    def _speak_alert(self):
        """Speaks an urgent vocal command to wake up the driver."""
        if not self.is_muted and self.voice_engine and self.is_active:
            try:
                # 1 = SVSFlagsAsync (non-blocking)
                self.voice_engine.Speak("Wake up! Drowsiness detected!", 1)
            except Exception:
                pass

    def toggle_mute(self) -> bool:
        """Toggles mute state and halts any active audio immediately."""
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.stop()
        return self.is_muted

    def play_test_beep(self):
        """Plays a test tone to confirm speaker/headphone volume."""
        if self.is_muted:
            return
        if HAS_PYGAME and self.alarm_sound is not None:
            self.alarm_sound.play()
            threading.Timer(0.4, self.alarm_sound.stop).start()
        elif HAS_SOUNDDEVICE:
            duration = 0.3
            sr = 44100
            t = np.linspace(0, duration, int(sr * duration), False)
            tone = (0.6 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
            sd.play(tone, sr)
        elif HAS_WINSOUND:
            threading.Thread(target=lambda: winsound.Beep(2000, 250), daemon=True).start()

    def shutdown(self):
        """Shuts down all audio streams."""
        self.stop()
        self._stop_event.set()
        if HAS_PYGAME:
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)


class MediaPipeMeshEngine:
    """
    Advanced 478-point 3D Landmark Mesh detector using Google MediaPipe.
    Computes EAR, MAR, and 3D Head Pose (Pitch, Yaw, Roll).
    """

    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    MOUTH_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model file not found: {model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> Optional[dict]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        face_landmarks = results.face_landmarks[0]
        pts = [(lm.x * w, lm.y * h) for lm in face_landmarks]

        # 1. EAR
        left_ear = calculate_ear(pts, self.LEFT_EYE_INDICES)
        right_ear = calculate_ear(pts, self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0

        # 2. MAR
        mar = calculate_mar(pts)

        # 3. Head Pose Estimation (Pitch, Yaw, Roll) via solvePnP
        image_points = np.array([
            pts[1],     # Nose tip
            pts[152],   # Chin
            pts[33],    # Right eye outer
            pts[263],   # Left eye outer
            pts[61],    # Right mouth
            pts[291]    # Left mouth
        ], dtype=np.float64)

        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        pitch, yaw, roll = 0.0, 0.0, 0.0
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            # Adjust angles for intuitive reading
            pitch = float(angles[0])
            yaw = float(angles[1])
            roll = float(angles[2])

        # Face bounding box
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fx = int(max(0, min(xs) - 10))
        fy = int(max(0, min(ys) - 10))
        fw = int(min(w - fx, max(xs) - min(xs) + 20))
        fh = int(min(h - fy, max(ys) - min(ys) + 20))

        # Polygons
        left_eye_poly = np.array([pts[i] for i in self.LEFT_EYE_INDICES], dtype=np.int32)
        right_eye_poly = np.array([pts[i] for i in self.RIGHT_EYE_INDICES], dtype=np.int32)
        mouth_poly = np.array([pts[i] for i in self.MOUTH_OUTLINE], dtype=np.int32)

        # Head Nodding (dropping head forward) and Distraction (looking left/right)
        is_nodding = pitch > 18.0 or pitch < -18.0
        is_distracted = abs(yaw) > 28.0

        return {
            "engine": "MediaPipe Mesh",
            "face_box": (fx, fy, fw, fh),
            "left_ear": float(left_ear),
            "right_ear": float(right_ear),
            "avg_ear": float(avg_ear),
            "mar": float(mar),
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "is_nodding": is_nodding,
            "is_distracted": is_distracted,
            "left_poly": left_eye_poly,
            "right_poly": right_eye_poly,
            "mouth_poly": mouth_poly,
            "eyes_closed": avg_ear < 0.22,
            "is_yawning": mar > 0.58,
        }


class OpenCVCascadeEngine:
    """Fast, reliable OpenCV Haar Cascade detector."""

    def __init__(self, cascades_dir: Optional[str] = None):
        paths = ensure_cascades(cascades_dir)
        self.face_cascade = cv2.CascadeClassifier(paths["haarcascade_frontalface_default.xml"])
        self.eye_cascade = cv2.CascadeClassifier(paths["haarcascade_eye.xml"])
        self.eye_glasses_cascade = cv2.CascadeClassifier(paths["haarcascade_eye_tree_eyeglasses.xml"])

    def process(self, frame_bgr: np.ndarray) -> Optional[dict]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(120, 120))
        if len(faces) == 0:
            return None

        fx, fy, fw, fh = tuple(map(int, max(faces, key=lambda b: b[2] * b[3])))

        zy = int(fy + fh * 0.22)
        zh = int(fh * 0.32)
        lz = (int(fx + fw * 0.10), zy, int(fw * 0.38), zh)
        rz = (int(fx + fw * 0.52), zy, int(fw * 0.38), zh)

        left_open, left_box = self._check_zone(gray, lz)
        right_open, right_box = self._check_zone(gray, rz)

        eyes_closed = not (left_open or right_open)
        ear_approx = 0.32 if (left_open and right_open) else (0.24 if (left_open or right_open) else 0.14)

        return {
            "engine": "OpenCV Cascade",
            "face_box": (fx, fy, fw, fh),
            "left_ear": ear_approx,
            "right_ear": ear_approx,
            "avg_ear": ear_approx,
            "mar": 0.2,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "is_nodding": False,
            "is_distracted": False,
            "left_poly": None,
            "right_poly": None,
            "mouth_poly": None,
            "left_box": left_box,
            "right_box": right_box,
            "left_zone": lz,
            "right_zone": rz,
            "eyes_closed": eyes_closed,
            "is_yawning": False,
        }

    def _check_zone(self, gray: np.ndarray, zone: Tuple[int, int, int, int]):
        zx, zy, zw, zh = zone
        h_img, w_img = gray.shape[:2]
        x1, y1 = max(0, zx), max(0, zy)
        x2, y2 = min(w_img, zx + zw), min(h_img, zy + zh)
        if x2 <= x1 or y2 <= y1:
            return False, None

        roi = gray[y1:y2, x1:x2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(roi)

        eyes = self.eye_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=4, minSize=(int(zw * 0.25), int(zh * 0.25)))
        if len(eyes) == 0 and not self.eye_glasses_cascade.empty():
            eyes = self.eye_glasses_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=3, minSize=(int(zw * 0.25), int(zh * 0.25)))

        if len(eyes) > 0:
            ex, ey, ew, eh = eyes[0]
            return True, (x1 + ex, y1 + ey, ew, eh)

        mean_val = np.mean(enhanced)
        dark_ratio = float(np.sum(enhanced < max(35.0, mean_val * 0.65))) / float(enhanced.size)
        return dark_ratio > 0.18, None


class EyeDetector:
    """Unified detector deploying MediaPipe Mesh or OpenCV Haar Cascades."""

    def __init__(self, cascades_dir: Optional[str] = None, force_opencv: bool = False):
        self.engine_name = "None"
        self.mp_engine: Optional[MediaPipeMeshEngine] = None
        self.cv_engine: Optional[OpenCVCascadeEngine] = None

        cascades = ensure_cascades(cascades_dir)
        task_model = cascades.get("face_landmarker.task", "")

        if HAS_MEDIAPIPE and os.path.exists(task_model) and not force_opencv:
            try:
                self.mp_engine = MediaPipeMeshEngine(task_model)
                self.engine_name = "MediaPipe FaceMesh (478 Landmarks)"
                print(f"[EyeDetector] Loaded primary engine: {self.engine_name}")
            except Exception as e:
                print(f"[EyeDetector] MediaPipe load failed: {e}. Falling back to OpenCV.", file=sys.stderr)

        if self.mp_engine is None:
            self.cv_engine = OpenCVCascadeEngine(cascades_dir)
            self.engine_name = "OpenCV Haar Cascades"
            print(f"[EyeDetector] Loaded fallback engine: {self.engine_name}")

        if self.cv_engine is None:
            try:
                self.cv_engine = OpenCVCascadeEngine(cascades_dir)
            except Exception:
                pass

        self.face_cascade = self.cv_engine.face_cascade if self.cv_engine else None
        self.eye_cascade = self.cv_engine.eye_cascade if self.cv_engine else None

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[dict]:
        if self.mp_engine:
            try:
                res = self.mp_engine.process(frame_bgr)
                if res is not None:
                    return res
            except Exception:
                pass
        if self.cv_engine:
            return self.cv_engine.process(frame_bgr)
        return None

    def detect_face(self, gray_frame: np.ndarray):
        if self.cv_engine:
            faces = self.cv_engine.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.15, minNeighbors=5, minSize=(120, 120))
            if len(faces) > 0:
                return tuple(map(int, max(faces, key=lambda b: b[2] * b[3])))
        return None

    def extract_eye_zones(self, face_box: Tuple[int, int, int, int]):
        fx, fy, fw, fh = face_box
        zy = int(fy + fh * 0.22)
        zh = int(fh * 0.32)
        return (int(fx + fw * 0.10), zy, int(fw * 0.38), zh), (int(fx + fw * 0.52), zy, int(fw * 0.38), zh)

    def analyze_eye_openness(self, gray_frame: np.ndarray, zone: Tuple[int, int, int, int]):
        if self.cv_engine:
            is_open, box = self.cv_engine._check_zone(gray_frame, zone)
            return is_open, (0.35 if is_open else 0.15), box
        return False, 0.0, None


class DrowsinessTracker:
    """
    Temporal state machine tracking eye closure, natural blinks,
    yawning, head nodding, and distraction.
    """

    STATE_AWAKE = "AWAKE"
    STATE_BLINKING = "BLINKING"
    STATE_DROWSY_WARNING = "DROWSY_WARNING"
    STATE_ALARM = "ALARM"
    STATE_YAWNING = "YAWNING"
    STATE_HEAD_NOD = "HEAD_NOD"
    STATE_DISTRACTED = "DISTRACTED"

    def __init__(
        self,
        drowsy_time_threshold: float = 1.8,
        warning_time_threshold: float = 0.8,
        blink_max_duration: float = 0.45,
        yawn_time_threshold: float = 1.5,
        distraction_time_threshold: float = 2.0,
    ):
        self.drowsy_time_threshold = drowsy_time_threshold
        self.warning_time_threshold = warning_time_threshold
        self.blink_max_duration = blink_max_duration
        self.yawn_time_threshold = yawn_time_threshold
        self.distraction_time_threshold = distraction_time_threshold

        self.state = self.STATE_AWAKE
        self.closure_start_time: Optional[float] = None
        self.current_closure_duration: float = 0.0
        self.total_blinks: int = 0
        self.total_drowsy_episodes: int = 0
        self.total_yawns: int = 0
        self.total_head_nods: int = 0
        self.total_distractions: int = 0

        self.yawn_start_time: Optional[float] = None
        self.current_yawn_duration: float = 0.0
        self.distraction_start_time: Optional[float] = None
        self.current_distraction_duration: float = 0.0
        self.start_time: float = time.time()

        self.consecutive_closed_frames: int = 0
        self.consecutive_open_frames: int = 0

    def update(
        self,
        eyes_closed: bool,
        is_yawning: bool = False,
        is_nodding: bool = False,
        is_distracted: bool = False,
    ) -> Tuple[str, float, bool]:
        now = time.time()
        alarm_triggered = False

        # 1. Distraction Tracking (Looking away)
        if is_distracted:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.current_distraction_duration = now - self.distraction_start_time
            if self.current_distraction_duration >= self.distraction_time_threshold:
                if self.state != self.STATE_DISTRACTED:
                    self.total_distractions += 1
                if self.state not in (self.STATE_ALARM, self.STATE_DROWSY_WARNING):
                    self.state = self.STATE_DISTRACTED
        else:
            self.distraction_start_time = None
            self.current_distraction_duration = 0.0

        # 2. Head Nodding Tracking
        if is_nodding and not eyes_closed and self.state not in (self.STATE_ALARM, self.STATE_DROWSY_WARNING):
            self.state = self.STATE_HEAD_NOD

        # 3. Yawn Tracking
        if is_yawning:
            if self.yawn_start_time is None:
                self.yawn_start_time = now
            self.current_yawn_duration = now - self.yawn_start_time
            if self.current_yawn_duration >= self.yawn_time_threshold and self.state != self.STATE_ALARM:
                self.state = self.STATE_YAWNING
        else:
            if self.yawn_start_time is not None:
                if self.current_yawn_duration >= self.yawn_time_threshold:
                    self.total_yawns += 1
            self.yawn_start_time = None
            self.current_yawn_duration = 0.0

        # 4. Eye Closure Tracking (Highest Priority)
        if eyes_closed:
            self.consecutive_closed_frames += 1
            self.consecutive_open_frames = 0

            if self.closure_start_time is None:
                self.closure_start_time = now
            self.current_closure_duration = now - self.closure_start_time

            if self.current_closure_duration >= self.drowsy_time_threshold:
                if self.state != self.STATE_ALARM:
                    self.total_drowsy_episodes += 1
                self.state = self.STATE_ALARM
                alarm_triggered = True
            elif self.current_closure_duration >= self.warning_time_threshold:
                self.state = self.STATE_DROWSY_WARNING
            else:
                self.state = self.STATE_BLINKING
        else:
            self.consecutive_open_frames += 1
            self.consecutive_closed_frames = 0

            if self.closure_start_time is not None:
                duration = now - self.closure_start_time
                if 0.05 <= duration <= self.blink_max_duration:
                    self.total_blinks += 1

            self.closure_start_time = None
            self.current_closure_duration = 0.0

            if (not is_yawning or self.current_yawn_duration < self.yawn_time_threshold) and not is_nodding and not is_distracted:
                self.state = self.STATE_AWAKE

        return self.state, self.current_closure_duration, alarm_triggered

    def get_blinks_per_minute(self) -> float:
        elapsed_min = max(0.05, (time.time() - self.start_time) / 60.0)
        return round(self.total_blinks / elapsed_min, 1)

    def reset_stats(self):
        self.closure_start_time = None
        self.current_closure_duration = 0.0
        self.total_blinks = 0
        self.total_drowsy_episodes = 0
        self.total_yawns = 0
        self.total_head_nods = 0
        self.total_distractions = 0
        self.yawn_start_time = None
        self.current_yawn_duration = 0.0
        self.distraction_start_time = None
        self.current_distraction_duration = 0.0
        self.start_time = time.time()
        self.state = self.STATE_AWAKE
