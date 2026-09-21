"""
Asset downloader for Driver Drowsiness Detection System.
Downloads required OpenCV Haar Cascades and MediaPipe Face Landmarker models.
"""

import os
import sys
import urllib.request

CASCADES = {
    "haarcascade_frontalface_default.xml": (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    ),
    "haarcascade_eye.xml": (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    ),
    "haarcascade_eye_tree_eyeglasses.xml": (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}

def ensure_cascades(target_dir: str = None) -> dict:
    if target_dir is None:
        target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades")
    os.makedirs(target_dir, exist_ok=True)

    paths = {}
    for filename, url in CASCADES.items():
        dest = os.path.join(target_dir, filename)
        if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
            print(f"[AssetManager] Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"[AssetManager] Successfully saved {filename} ({os.path.getsize(dest)} bytes)")
            except Exception as e:
                print(f"[AssetManager] Warning: Could not download {filename}: {e}", file=sys.stderr)
        paths[filename] = dest

    return paths

if __name__ == "__main__":
    downloaded = ensure_cascades()
    print("All assets ready:")
    for k, v in downloaded.items():
        size = os.path.getsize(v) if os.path.exists(v) else 0
        print(f" - {k}: {v} ({size} bytes)")
