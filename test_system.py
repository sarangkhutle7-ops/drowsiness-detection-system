"""
Automated Test Suite for Driver Drowsiness Detection System.
Validates vision detectors, temporal tracking state machine,
audio alert manager, and generates a simulation video.
"""

import os
import sys
import time
import unittest
import numpy as np
import cv2

from drowsiness_detector import EyeDetector, DrowsinessTracker, AlertManager


class TestDrowsinessSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = EyeDetector()

    def test_01_cascades_loaded(self):
        """Verify cascade classifiers are successfully initialized."""
        self.assertFalse(self.detector.face_cascade.empty())
        self.assertFalse(self.detector.eye_cascade.empty())

    def test_02_alert_manager_lifecycle(self):
        """Verify AlertManager triggers, stops, mutes, and shuts down safely."""
        alert = AlertManager(frequency=2000, duration_ms=50)
        self.assertFalse(alert.is_active)
        self.assertFalse(alert.is_muted)

        alert.trigger()
        self.assertTrue(alert.is_active)

        alert.stop()
        self.assertFalse(alert.is_active)

        muted = alert.toggle_mute()
        self.assertTrue(muted)
        self.assertTrue(alert.is_muted)

        alert.toggle_mute()
        self.assertFalse(alert.is_muted)

        alert.shutdown()
        self.assertFalse(alert._thread.is_alive())

    def test_03_drowsiness_tracker_awake_and_blink(self):
        """Verify awake state and that natural blinks do not trigger alarms."""
        tracker = DrowsinessTracker(drowsy_time_threshold=1.0, warning_time_threshold=0.5)

        # 1. Awake state with eyes open
        state, duration, alarm = tracker.update(eyes_closed=False)
        self.assertEqual(state, DrowsinessTracker.STATE_AWAKE)
        self.assertEqual(duration, 0.0)
        self.assertFalse(alarm)

        # 2. Simulate short natural blink (0.2 seconds)
        for _ in range(3):
            tracker.update(eyes_closed=True)
            time.sleep(0.06)

        # Eyes reopen
        state, duration, alarm = tracker.update(eyes_closed=False)
        self.assertEqual(state, DrowsinessTracker.STATE_AWAKE)
        self.assertFalse(alarm)
        self.assertGreaterEqual(tracker.total_blinks, 1)
        self.assertEqual(tracker.total_drowsy_episodes, 0)

    def test_04_drowsiness_tracker_alarm_trigger(self):
        """Verify prolonged eye closure triggers warning and alarm."""
        tracker = DrowsinessTracker(drowsy_time_threshold=0.6, warning_time_threshold=0.3)

        # Prolonged closed eyes simulation
        alarm_fired = False
        warning_fired = False
        start = time.time()
        while time.time() - start < 1.0:
            state, duration, alarm = tracker.update(eyes_closed=True)
            if state == DrowsinessTracker.STATE_DROWSY_WARNING:
                warning_fired = True
            if alarm:
                alarm_fired = True
            time.sleep(0.05)

        self.assertTrue(warning_fired, "Warning state was not entered during prolonged closure")
        self.assertTrue(alarm_fired, "Alarm was not triggered after threshold exceeded")
        self.assertEqual(tracker.state, DrowsinessTracker.STATE_ALARM)
        self.assertGreaterEqual(tracker.total_drowsy_episodes, 1)

        # Now driver wakes up and opens eyes
        state, duration, alarm = tracker.update(eyes_closed=False)
        for _ in range(5):
            state, duration, alarm = tracker.update(eyes_closed=False)

        self.assertEqual(state, DrowsinessTracker.STATE_AWAKE)
        self.assertFalse(alarm, "Alarm failed to stop after eyes reopened")

    def test_05_eye_zone_extraction(self):
        """Verify anatomical eye zone calculations from face box."""
        face_box = (100, 100, 200, 200)
        left_zone, right_zone = self.detector.extract_eye_zones(face_box)

        # Left zone (viewer left)
        self.assertGreater(left_zone[0], face_box[0])
        self.assertGreater(left_zone[1], face_box[1])
        # Right zone should be to the right of left zone
        self.assertGreater(right_zone[0], left_zone[0])


def generate_simulation_video(output_path: str = "simulation_driver.mp4", duration_sec: int = 8):
    """
    Generates a synthetic driver video for testing without a live webcam.
    Simulates:
      - Seconds 0-3: Awake driver (eyes open)
      - Seconds 3-4: Driver blinks
      - Seconds 4-8: Driver falls asleep (eyes closed), triggering alarm
    """
    print(f"[Simulation] Generating test video: {output_path} ({duration_sec}s)...")
    fps = 20
    total_frames = duration_sec * fps
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Face center
    cx, cy = width // 2, height // 2
    face_w, face_h = 220, 260

    for f in range(total_frames):
        t = f / float(fps)
        frame = np.full((height, width, 3), (35, 35, 40), dtype=np.uint8)

        # Background car interior silhouette
        cv2.rectangle(frame, (0, height - 80), (width, height), (20, 20, 25), -1)

        # Draw Face oval
        cv2.ellipse(frame, (cx, cy), (face_w // 2, face_h // 2), 0, 0, 360, (190, 160, 140), -1)

        # Determine eye state based on timestamp
        is_eyes_open = True
        if 3.0 <= t < 3.3:
            is_eyes_open = False  # Quick blink
        elif t >= 4.5:
            is_eyes_open = False  # Prolonged sleep / drowsy!

        # Eye positions
        eye_y = cy - 30
        left_eye_x = cx - 45
        right_eye_x = cx + 45

        for ex in (left_eye_x, right_eye_x):
            if is_eyes_open:
                # Sclera (white)
                cv2.ellipse(frame, (ex, eye_y), (22, 14), 0, 0, 360, (245, 245, 245), -1)
                # Iris (dark)
                cv2.circle(frame, (ex, eye_y), 8, (45, 30, 20), -1)
                # Pupil
                cv2.circle(frame, (ex, eye_y), 3, (10, 10, 10), -1)
            else:
                # Closed eyelid (thin arc)
                cv2.ellipse(frame, (ex, eye_y + 2), (22, 5), 0, 0, 180, (120, 90, 80), 3)

        # Mouth
        cv2.ellipse(frame, (cx, cy + 65), (25, 8), 0, 0, 360, (120, 80, 80), -1)

        # Annotation watermark on synthetic video
        phase = "AWAKE" if t < 3.0 else ("BLINK" if t < 4.5 else "FALLING ASLEEP (DROWSY)")
        cv2.putText(frame, f"TEST FEED: {phase}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out.write(frame)

    out.release()
    print(f"[Simulation] Generated video saved to {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    # 1. Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDrowsinessSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 2. Generate simulation video
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_video_path = os.path.join(script_dir, "simulation_driver.mp4")
    generate_simulation_video(sim_video_path, duration_sec=8)

    if not result.wasSuccessful():
        sys.exit(1)
