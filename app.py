"""
Real-Time Driver Drowsiness, Fatigue & Distraction Detection Application.
Runs live camera stream, computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
and Head Pose (Pitch/Yaw), triggers audible alerts on drowsiness, and outputs
an interactive HTML trip safety report on session completion.
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from drowsiness_detector import AlertManager, DrowsinessTracker, EyeDetector
from session_logger import SessionLogger


class DrowsinessApp:
    """
    Interactive real-time driver drowsiness monitoring application.
    """

    def __init__(
        self,
        camera_id: int = 0,
        video_source: Optional[str] = None,
        drowsy_threshold: float = 1.8,
        warning_threshold: float = 0.8,
        force_opencv: bool = False,
    ):
        self.camera_id = camera_id
        self.video_source = video_source
        self.drowsy_threshold = drowsy_threshold
        self.warning_threshold = warning_threshold

        print("[App] Initializing Vision Models, Audio Subsystem, and Session Logger...")
        self.eye_detector = EyeDetector(force_opencv=force_opencv)
        self.alert_manager = AlertManager()
        self.tracker = DrowsinessTracker(
            drowsy_time_threshold=self.drowsy_threshold,
            warning_time_threshold=self.warning_threshold,
        )
        self.logger = SessionLogger(output_dir="logs")

        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.previous_state = "AWAKE"

    def run(self):
        source = self.video_source if self.video_source else self.camera_id
        print(f"[App] Opening video source: {source}")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[App] ERROR: Unable to open video source {source}.", file=sys.stderr)
            print("[App] Please ensure your webcam is connected or specify a video file path.", file=sys.stderr)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        window_name = "AI Driver Drowsiness & Fatigue Monitor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1080, 680)

        print("\n" + "=" * 68)
        print("  AI DRIVER MONITORING SYSTEM (DROWSINESS, FATIGUE, DISTRACTION)")
        print(f"  Active Engine: {self.eye_detector.engine_name}")
        print("=" * 68)
        print("  Controls:")
        print("    [q] : Quit and generate HTML Driver Safety Report")
        print("    [+] : Increase drowsiness time threshold (+0.2s)")
        print("    [-] : Decrease drowsiness time threshold (-0.2s)")
        print("    [m] : Toggle sound alarm MUTE")
        print("    [t] : Test alarm beep")
        print("    [r] : Reset statistics (blinks, yawns, episodes)")
        print("    [s] : Save snapshot frame")
        print("=" * 68 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[App] Video stream ended or frame unavailable.")
                    break

                # Mirror frame horizontally for webcam, or pace video playback
                if self.video_source is None:
                    frame = cv2.flip(frame, 1)
                else:
                    # Pace video file playback to real-time (25-30 FPS)
                    time.sleep(0.035)

                h, w = frame.shape[:2]

                # 1. Process Frame via Unified Detector
                data = self.eye_detector.process_frame(frame)

                if data is not None:
                    eyes_closed = data.get("eyes_closed", False)
                    is_yawning = data.get("is_yawning", False)
                    is_nodding = data.get("is_nodding", False)
                    is_distracted = data.get("is_distracted", False)
                    ear_val = data.get("avg_ear", 0.0)
                    mar_val = data.get("mar", 0.0)
                    pitch_val = data.get("pitch", 0.0)
                    yaw_val = data.get("yaw", 0.0)
                else:
                    eyes_closed = False
                    is_yawning = False
                    is_nodding = False
                    is_distracted = False
                    ear_val, mar_val, pitch_val, yaw_val = 0.0, 0.0, 0.0, 0.0

                # 2. Update Temporal State Machine
                state, duration, alarm_active = self.tracker.update(
                    eyes_closed=eyes_closed,
                    is_yawning=is_yawning,
                    is_nodding=is_nodding,
                    is_distracted=is_distracted,
                )

                # 3. Log State Transitions
                if state != self.previous_state:
                    if state == DrowsinessTracker.STATE_ALARM:
                        self.logger.log_event("ALARM_DROWSY", f"Driver eyes closed for {duration:.1f}s")
                    elif state == DrowsinessTracker.STATE_DROWSY_WARNING:
                        self.logger.log_event("WARNING_DROWSY", f"Drowsiness warning ({duration:.1f}s)")
                    elif state == DrowsinessTracker.STATE_YAWNING:
                        self.logger.log_event("YAWN_DETECTED", f"Driver yawning (MAR: {mar_val:.2f})")
                    elif state == DrowsinessTracker.STATE_HEAD_NOD:
                        self.logger.log_event("HEAD_NOD_ALERT", f"Head nodding pitch: {pitch_val:.1f} deg")
                    elif state == DrowsinessTracker.STATE_DISTRACTED:
                        self.logger.log_event("DISTRACTION_ALERT", f"Driver looking away yaw: {yaw_val:.1f} deg")
                    self.previous_state = state

                # Sample Telemetry for Reporting
                self.logger.sample_telemetry(
                    ear=ear_val,
                    mar=mar_val,
                    pitch=pitch_val,
                    yaw=yaw_val,
                    state=state,
                    fps=self.fps,
                )

                # 4. Sound Alarm Management
                if alarm_active:
                    self.alert_manager.trigger()
                else:
                    self.alert_manager.stop()

                # 5. Render HUD Overlay
                self._draw_hud(frame=frame, data=data, state=state, duration=duration, alarm_active=alarm_active)

                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 15 == 0:
                    elapsed = time.time() - self.fps_start_time
                    self.fps = round(15.0 / max(0.001, elapsed), 1)
                    self.fps_start_time = time.time()

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("+") or key == ord("="):
                    self.tracker.drowsy_time_threshold = round(self.tracker.drowsy_time_threshold + 0.2, 1)
                    print(f"[App] Drowsy threshold: {self.tracker.drowsy_time_threshold}s")
                elif key == ord("-") or key == ord("_"):
                    self.tracker.drowsy_time_threshold = max(0.6, round(self.tracker.drowsy_time_threshold - 0.2, 1))
                    print(f"[App] Drowsy threshold: {self.tracker.drowsy_time_threshold}s")
                elif key == ord("m"):
                    is_muted = self.alert_manager.toggle_mute()
                    print(f"[App] Alarm Audio: {'MUTED' if is_muted else 'ACTIVE'}")
                elif key == ord("t"):
                    print("[App] Test alarm sound triggered.")
                    self.alert_manager.play_test_beep()
                elif key == ord("r"):
                    self.tracker.reset_stats()
                    print("[App] Statistics reset.")
                elif key == ord("s"):
                    filename = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[App] Snapshot saved: {filename}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.alert_manager.shutdown()

            # Generate and print trip report
            self.logger.total_blinks = self.tracker.total_blinks
            self.logger.log_event("SESSION_END", "Monitoring session ended.")
            report_path = self.logger.generate_html_report()
            score = self.logger.calculate_safety_score()

            print("\n" + "=" * 68)
            print("  DRIVING SESSION COMPLETED")
            print(f"  Safety Score: {score}/100")
            print(f"  Report Generated: {os.path.abspath(report_path)}")
            print("=" * 68 + "\n")

    def _draw_hud(
        self,
        frame: np.ndarray,
        data: Optional[dict],
        state: str,
        duration: float,
        alarm_active: bool,
    ):
        h, w = frame.shape[:2]

        # 1. Visual Overlays on Face / Eyes / Mouth
        if data is not None:
            fx, fy, fw, fh = data["face_box"]
            self._draw_corner_brackets(frame, fx, fy, fw, fh, color=(0, 230, 115), length=25, thickness=2)

            if data.get("left_poly") is not None and data.get("right_poly") is not None:
                eye_color = (0, 70, 255) if data["eyes_closed"] else (0, 255, 120)
                cv2.polylines(frame, [data["left_poly"]], isClosed=True, color=eye_color, thickness=2)
                cv2.polylines(frame, [data["right_poly"]], isClosed=True, color=eye_color, thickness=2)

                if data.get("mouth_poly") is not None:
                    mouth_color = (0, 140, 255) if data["is_yawning"] else (180, 180, 180)
                    cv2.polylines(frame, [data["mouth_poly"]], isClosed=True, color=mouth_color, thickness=1)

            elif data.get("left_box") is not None or data.get("right_box") is not None:
                for box, is_open, lbl in [(data.get("left_box"), not data["eyes_closed"], "R.Eye"),
                                          (data.get("right_box"), not data["eyes_closed"], "L.Eye")]:
                    if box is not None:
                        bx, by, bw, bh = box
                        col = (0, 255, 120) if is_open else (0, 70, 255)
                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), col, 2)
                        cv2.putText(frame, lbl, (bx, max(20, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # 2. Top Banner Status Bar
        banner_h = 55
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        if alarm_active:
            flash = int(time.time() * 6) % 2 == 0
            banner_color = (0, 0, 230) if flash else (0, 165, 255)
            cv2.rectangle(frame, (0, 0), (w, banner_h), banner_color, -1)
            cv2.putText(
                frame,
                "*** DROWSINESS DETECTED! WAKE UP! ***",
                (w // 2 - 280, 38),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        elif state == DrowsinessTracker.STATE_DROWSY_WARNING:
            cv2.putText(
                frame,
                f"WARNING: EYES CLOSED ({duration:.1f}s / {self.tracker.drowsy_time_threshold}s)",
                (w // 2 - 260, 36),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 190, 255),
                2,
                cv2.LINE_AA,
            )
        elif state == DrowsinessTracker.STATE_HEAD_NOD:
            cv2.putText(
                frame,
                "ALERT: HEAD NODDING DETECTED (MICRO-SLEEP RISK)",
                (w // 2 - 290, 36),
                cv2.FONT_HERSHEY_DUPLEX,
                0.78,
                (0, 120, 255),
                2,
                cv2.LINE_AA,
            )
        elif state == DrowsinessTracker.STATE_DISTRACTED:
            cv2.putText(
                frame,
                "CAUTION: DRIVER LOOKING AWAY FROM ROAD",
                (w // 2 - 260, 36),
                cv2.FONT_HERSHEY_DUPLEX,
                0.78,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        elif state == DrowsinessTracker.STATE_YAWNING:
            cv2.putText(
                frame,
                "FATIGUE WARNING: YAWNING DETECTED",
                (w // 2 - 220, 36),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
        elif state == DrowsinessTracker.STATE_BLINKING:
            cv2.putText(
                frame,
                f"EYE BLINK DETECTED ({duration:.1f}s)",
                (w // 2 - 160, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 215, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            status_text = "DRIVER AWAKE & ATTENTIVE" if data is not None else "SEARCHING FOR DRIVER FACE..."
            status_color = (0, 230, 115) if data is not None else (180, 180, 180)
            cv2.putText(frame, status_text, (w // 2 - 180, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        # 3. Left Telemetry Dashboard
        panel_w = 330
        panel_h = 315
        panel_x = 15
        panel_y = banner_h + 15

        dash_overlay = frame.copy()
        cv2.rectangle(dash_overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 18), -1)
        cv2.addWeighted(dash_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (60, 65, 75), 1)

        ear_val = data.get("avg_ear", 0.0) if data else 0.0
        mar_val = data.get("mar", 0.0) if data else 0.0
        pitch_val = data.get("pitch", 0.0) if data else 0.0
        yaw_val = data.get("yaw", 0.0) if data else 0.0

        cv2.putText(frame, "TELEMETRY DASHBOARD", (panel_x + 12, panel_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

        y_cursor = panel_y + 48
        metrics = [
            ("Driver Status:", state, (0, 0, 255) if alarm_active else (0, 230, 115)),
            ("Eye Aspect Ratio (EAR):", f"{ear_val:.2f} (Thresh: 0.22)", (0, 255, 120) if ear_val >= 0.22 else (0, 80, 255)),
            ("Mouth Aspect (MAR):", f"{mar_val:.2f} (Yawn: >0.58)", (0, 140, 255) if mar_val >= 0.58 else (200, 200, 200)),
            ("Head Pose (Pitch/Yaw):", f"{pitch_val:.0f}° / {yaw_val:.0f}°", (0, 220, 255)),
            ("Closure Timer:", f"{duration:.2f}s / {self.tracker.drowsy_time_threshold:.1f}s", (255, 255, 255)),
            ("Total Blinks:", f"{self.tracker.total_blinks} ({self.tracker.get_blinks_per_minute()}/min)", (200, 230, 255)),
            ("Total Yawns:", f"{self.tracker.total_yawns}", (255, 200, 100)),
            ("Drowsy Incidents:", str(self.tracker.total_drowsy_episodes), (0, 160, 255) if self.tracker.total_drowsy_episodes > 0 else (180, 180, 180)),
            ("Alarm Audio:", "MUTED [m]" if self.alert_manager.is_muted else "ACTIVE [m]", (100, 100, 255) if self.alert_manager.is_muted else (0, 255, 100)),
            ("Vision Engine:", self.eye_detector.engine_name[:20], (180, 180, 180)),
            ("Processing FPS:", f"{self.fps}", (180, 180, 180)),
        ]

        for label, val, col in metrics:
            cv2.putText(frame, label, (panel_x + 12, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(frame, val, (panel_x + 175, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            y_cursor += 19

        # Closure Duration Progress Bar
        bar_x = panel_x + 12
        bar_y = y_cursor + 4
        bar_w = panel_w - 24
        bar_h = 10
        progress = min(1.0, duration / max(0.1, self.tracker.drowsy_time_threshold))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        bar_col = (0, 230, 115) if progress < 0.45 else ((0, 200, 255) if progress < 0.9 else (0, 0, 255))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), bar_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 90), 1)

        # 4. Bottom Keybind Guide
        guide_text = "[q] Exit & Report  |  [+/-] Sensitivity  |  [m] Mute  |  [t] Test  |  [r] Reset"
        cv2.putText(frame, guide_text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_corner_brackets(img, x, y, w, h, color, length=20, thickness=2):
        cv2.line(img, (x, y), (x + length, y), color, thickness)
        cv2.line(img, (x, y), (x, y + length), color, thickness)
        cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
        cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Driver Drowsiness Detection System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional video file path for testing")
    parser.add_argument("--threshold", type=float, default=1.8, help="Drowsiness closure threshold in seconds (default: 1.8)")
    parser.add_argument("--opencv", action="store_true", help="Force OpenCV Haar Cascade engine instead of MediaPipe")
    args = parser.parse_args()

    app = DrowsinessApp(
        camera_id=args.camera,
        video_source=args.video,
        drowsy_threshold=args.threshold,
        force_opencv=args.opencv,
    )
    app.run()


if __name__ == "__main__":
    main()
