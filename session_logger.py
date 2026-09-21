"""
Trip & Safety Session Logger for Driver Drowsiness Detection System.
Logs real-time driving telemetry, tracks safety infractions (drowsiness, yawns, distraction),
and exports a modern, interactive HTML Driver Safety Report.
"""

import datetime
import json
import os
import time
from typing import Dict, List, Optional


class SessionLogger:
    """
    Records driver monitoring telemetry during a driving session
    and generates an interactive HTML trip analytics report.
    """

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.session_id = datetime.datetime.now().strftime("trip_%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.start_datetime = datetime.datetime.now()

        self.events: List[Dict] = []
        self.telemetry_samples: List[Dict] = []
        self.last_sample_time = 0.0

        # Totals
        self.total_blinks = 0
        self.total_yawns = 0
        self.total_drowsy_alarms = 0
        self.total_distractions = 0
        self.total_head_nods = 0

        self.log_event("SESSION_START", "Driver monitoring session initiated.")

    def log_event(self, event_type: str, details: str):
        """Records a timestamped safety event."""
        now = time.time()
        rel_time = round(now - self.start_time, 1)
        timestamp_str = datetime.datetime.now().strftime("%H:%M:%S")

        event = {
            "time_sec": rel_time,
            "timestamp": timestamp_str,
            "type": event_type,
            "details": details,
        }
        self.events.append(event)

        if event_type == "ALARM_DROWSY":
            self.total_drowsy_alarms += 1
        elif event_type == "YAWN_DETECTED":
            self.total_yawns += 1
        elif event_type == "DISTRACTION_ALERT":
            self.total_distractions += 1
        elif event_type == "HEAD_NOD_ALERT":
            self.total_head_nods += 1

    def sample_telemetry(
        self,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        state: str,
        fps: float,
    ):
        """Periodically samples continuous telemetry (e.g. every 0.5 seconds)."""
        now = time.time()
        if now - self.last_sample_time < 0.4:
            return

        self.last_sample_time = now
        rel_time = round(now - self.start_time, 1)

        self.telemetry_samples.append({
            "t": rel_time,
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "pitch": round(pitch, 1),
            "yaw": round(yaw, 1),
            "state": state,
            "fps": round(fps, 1),
        })

    def calculate_safety_score(self) -> int:
        """
        Calculates driver alertness safety rating [0 - 100].
        Deductions:
          - Drowsy alarms: -25 each
          - Head nods: -15 each
          - Distraction: -10 each
          - Yawns: -5 each
        """
        score = 100
        score -= self.total_drowsy_alarms * 25
        score -= self.total_head_nods * 15
        score -= self.total_distractions * 10
        score -= self.total_yawns * 5
        return max(0, min(100, score))

    def generate_html_report(self, filename: Optional[str] = None) -> str:
        """Generates a rich, interactive HTML Driver Safety Report."""
        duration_sec = max(1.0, time.time() - self.start_time)
        duration_str = str(datetime.timedelta(seconds=int(duration_sec)))
        score = self.calculate_safety_score()

        if score >= 85:
            score_badge = "Excellent Alertness"
            score_color = "#10b981"  # Emerald
            grade = "A"
        elif score >= 70:
            score_badge = "Moderate Fatigue Risk"
            score_color = "#f59e0b"  # Amber
            grade = "B"
        else:
            score_badge = "High Drowsiness Hazard"
            score_color = "#ef4444"  # Red
            grade = "C"

        if filename is None:
            filename = os.path.join(self.output_dir, f"{self.session_id}_report.html")

        # Telemetry arrays for Chart.js
        sample_times = [s["t"] for s in self.telemetry_samples]
        ear_data = [s["ear"] for s in self.telemetry_samples]
        mar_data = [s["mar"] for s in self.telemetry_samples]

        events_html = ""
        for ev in reversed(self.events[-30:]):  # Last 30 events
            ev_class = "event-normal"
            if "ALARM" in ev["type"]:
                ev_class = "event-danger"
            elif "WARNING" in ev["type"] or "YAWN" in ev["type"] or "NOD" in ev["type"]:
                ev_class = "event-warning"

            events_html += f"""
            <tr class="{ev_class}">
                <td><code>{ev['timestamp']}</code></td>
                <td>+{ev['time_sec']}s</td>
                <td><span class="badge">{ev['type']}</span></td>
                <td>{ev['details']}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driver Safety & Drowsiness Report - {self.session_id}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --surface-accent: #21262d;
            --border: #30363d;
            --text: #f0f6fc;
            --text-muted: #8b949e;
            --primary: #58a6ff;
            --danger: #f85149;
            --warning: #e3b341;
            --success: #3fb950;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background: var(--bg); color: var(--text); padding: 30px 20px; line-height: 1.5; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 30px; }}
        h1 {{ font-size: 1.8rem; font-weight: 700; color: #fff; }}
        .meta {{ color: var(--text-muted); font-size: 0.9rem; margin-top: 5px; }}

        /* Score & Stat Cards */
        .grid-stats {{ display: grid; grid-template-columns: 280px repeat(auto-fit, minmax(170px, 1fr)); gap: 18px; margin-bottom: 30px; }}
        .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
        .score-card {{ text-align: center; border-color: {score_color}; background: linear-gradient(180deg, rgba({score_color},0.15) 0%, var(--surface) 100%); }}
        .score-val {{ font-size: 3.5rem; font-weight: 800; color: {score_color}; line-height: 1; margin: 10px 0; }}
        .stat-label {{ font-size: 0.82rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
        .stat-val {{ font-size: 1.8rem; font-weight: 700; color: #fff; margin-top: 8px; }}

        /* Chart Section */
        .chart-container {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 22px; margin-bottom: 30px; }}
        .chart-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 15px; color: var(--text); }}

        /* Table */
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9rem; }}
        th {{ background: var(--surface-accent); padding: 12px 14px; text-align: left; color: var(--text-muted); border-bottom: 1px solid var(--border); }}
        td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); }}
        tr:hover td {{ background: rgba(255,255,255,0.02); }}
        .badge {{ background: var(--surface-accent); padding: 3px 8px; border-radius: 6px; font-size: 0.78rem; font-weight: 600; }}
        .event-danger .badge {{ background: rgba(248, 81, 73, 0.2); color: var(--danger); }}
        .event-warning .badge {{ background: rgba(227, 179, 65, 0.2); color: var(--warning); }}

        footer {{ text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 40px; border-top: 1px solid var(--border); padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Automotive Driver Safety Report</h1>
                <div class="meta">Session ID: <code>{self.session_id}</code> | Started: {self.start_datetime.strftime("%B %d, %Y - %H:%M:%S")}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.85rem; color: var(--text-muted);">Session Duration</div>
                <div style="font-size: 1.3rem; font-weight: 700;">{duration_str}</div>
            </div>
        </header>

        <div class="grid-stats">
            <div class="card score-card">
                <div class="stat-label">Driver Alertness Score</div>
                <div class="score-val">{score}<span style="font-size: 1.5rem; font-weight: 400;">/100</span></div>
                <div style="font-size: 0.95rem; font-weight: 600; color: {score_color};">{score_badge} (Grade {grade})</div>
            </div>
            <div class="card">
                <div class="stat-label">Drowsy Alarms</div>
                <div class="stat-val" style="color: {'#ef4444' if self.total_drowsy_alarms > 0 else '#fff'};">{self.total_drowsy_alarms}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Prolonged eye closure</div>
            </div>
            <div class="card">
                <div class="stat-label">Total Yawns</div>
                <div class="stat-val" style="color: {'#f59e0b' if self.total_yawns > 0 else '#fff'};">{self.total_yawns}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Fatigue indicators</div>
            </div>
            <div class="card">
                <div class="stat-label">Head Nods</div>
                <div class="stat-val" style="color: {'#f59e0b' if self.total_head_nods > 0 else '#fff'};">{self.total_head_nods}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Micro-sleep gestures</div>
            </div>
            <div class="card">
                <div class="stat-label">Total Blinks</div>
                <div class="stat-val">{self.total_blinks}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Natural eye blinks</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Continuous Telemetry: Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR) Over Time</div>
            <canvas id="telemetryChart" height="90"></canvas>
        </div>

        <div class="card">
            <div class="chart-title" style="margin-bottom: 10px;">Recent Safety Events Timeline</div>
            <table>
                <thead>
                    <tr>
                        <th>Clock</th>
                        <th>Offset</th>
                        <th>Event Type</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    {events_html if events_html else '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No infractions recorded. Clean driving session!</td></tr>'}
                </tbody>
            </table>
        </div>

        <footer>
            AI Driver Monitoring System &bull; Computer Vision & Machine Learning Powered &bull; Generated automatically upon trip termination.
        </footer>
    </div>

    <script>
        const ctx = document.getElementById('telemetryChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(sample_times)},
                datasets: [
                    {{
                        label: 'Eye Aspect Ratio (EAR)',
                        data: {json.dumps(ear_data)},
                        borderColor: '#3fb950',
                        backgroundColor: 'rgba(63, 185, 80, 0.1)',
                        borderWidth: 2,
                        tension: 0.2,
                        pointRadius: 1,
                        fill: true
                    }},
                    {{
                        label: 'Mouth Aspect Ratio (MAR)',
                        data: {json.dumps(mar_data)},
                        borderColor: '#58a6ff',
                        backgroundColor: 'rgba(88, 166, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.2,
                        pointRadius: 1,
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Trip Time (Seconds)', color: '#8b949e' }},
                        grid: {{ color: '#21262d' }},
                        ticks: {{ color: '#8b949e' }}
                    }},
                    y: {{
                        min: 0,
                        max: 1.0,
                        title: {{ display: true, text: 'Aspect Ratio Metric', color: '#8b949e' }},
                        grid: {{ color: '#21262d' }},
                        ticks: {{ color: '#8b949e' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#f0f6fc' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[SessionLogger] Generated safety report: {filename}")
        return filename
