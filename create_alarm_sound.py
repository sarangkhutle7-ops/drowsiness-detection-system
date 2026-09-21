"""
Audio Asset Generator for Driver Drowsiness Detection System.
Generates loud, high-visibility vehicle alarm WAV files for headphones & speakers.
"""

import os
import numpy as np
from scipy.io import wavfile


def generate_audio_assets(target_dir: str = "."):
    os.makedirs(target_dir, exist_ok=True)
    sample_rate = 44100

    # 1. Loud Emergency Siren / Alarm (alarm.wav)
    # High-intensity alternating frequencies: 2400 Hz and 1800 Hz pulses
    alarm_path = os.path.join(target_dir, "alarm.wav")
    if not os.path.exists(alarm_path):
        duration = 1.0  # 1 second seamless looping alarm
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.zeros_like(t)

        for i in range(4):
            start = i * 0.25
            end = start + 0.19
            mask = (t >= start) & (t < end)
            freq = 2500 if (i % 2 == 0) else 1900
            t_pulse = t[mask] - start

            # Sharp envelope to maximize loudness without clicking
            env = np.sin(np.pi * (t_pulse / 0.19)) ** 0.4
            wave = 0.65 * np.sin(2 * np.pi * freq * t_pulse) + 0.35 * np.sin(2 * np.pi * (freq * 1.5) * t_pulse)
            signal[mask] = wave * env

        # 16-bit PCM at 95% volume ceiling
        audio_16bit = np.int16(signal * 32767 * 0.95)
        wavfile.write(alarm_path, sample_rate, audio_16bit)
        print(f"[SoundGen] Created {alarm_path} ({len(audio_16bit)} samples)")

    # 2. Caution / Warning Chime (warning.wav)
    warning_path = os.path.join(target_dir, "warning.wav")
    if not os.path.exists(warning_path):
        dur_w = 0.4
        tw = np.linspace(0, dur_w, int(sample_rate * dur_w), endpoint=False)
        sig_w = np.zeros_like(tw)

        # Two gentle notification pings (1200 Hz -> 1600 Hz)
        for i, f in enumerate([1200, 1600]):
            st = i * 0.18
            en = st + 0.15
            m = (tw >= st) & (tw < en)
            tp = tw[m] - st
            env = np.exp(-15 * tp)
            sig_w[m] = 0.7 * np.sin(2 * np.pi * f * tp) * env

        audio_w_16bit = np.int16(sig_w * 32767 * 0.85)
        wavfile.write(warning_path, sample_rate, audio_w_16bit)
        print(f"[SoundGen] Created {warning_path} ({len(audio_w_16bit)} samples)")


if __name__ == "__main__":
    generate_audio_assets()
