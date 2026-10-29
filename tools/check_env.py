"""Environment and hardware capability verification tool.

Checks installed package versions, CPU threading, PyTorch/YOLO inference on CPU,
benchmarks inference latency at imgsz 640 and 480, and checks webcam and RTSP connectivity.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> int:
    print("=" * 60)
    print("Environment Verification (check_env.py)")
    print("=" * 60)

    # 1. Package versions
    import numpy as np
    import cv2
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO

    print(f"Python:       {sys.version.split()[0]} ({sys.executable})")
    print(f"NumPy:        {np.__version__}")
    print(f"OpenCV:       {cv2.__version__}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Torchvision:  {torchvision.__version__}")
    print(f"Ultralytics:  {ultralytics.__version__}")

    # 2. Hardware / CPU info
    cuda_available = torch.cuda.is_available()
    num_threads = torch.get_num_threads()
    print(f"CUDA available: {cuda_available} (expected: False)")
    print(f"PyTorch CPU threads: {num_threads}")
    assert not cuda_available, f"Expected CPU-only mode, but CUDA is available."

    # 3. Model loading and inference on bus.jpg
    model_path = Path("models/yolo11n.pt")
    if model_path.is_file():
        model = YOLO(str(model_path))
    else:
        # Fallback to auto-download name and save to models/
        model = YOLO("yolo11n.pt")
        os.makedirs("models", exist_ok=True)
        if Path("yolo11n.pt").is_file():
            Path("yolo11n.pt").rename(model_path)
            model = YOLO(str(model_path))

    # Test image: check local fixture first, otherwise URL
    fixture_path = Path("data/fixtures/bus.jpg")
    if fixture_path.is_file():
        img_source = str(fixture_path)
    else:
        img_source = "https://ultralytics.com/images/bus.jpg"

    print(f"\nRunning YOLO11n on: {img_source}")
    results = model(img_source, device="cpu", verbose=False)
    boxes = results[0].boxes
    cls_ids = boxes.cls.int().cpu().tolist()
    names = results[0].names
    detected_classes = [names[c] for c in cls_ids]
    print(f"Detected {len(cls_ids)} objects: {detected_classes}")

    person_count = sum(1 for c in detected_classes if c == "person")
    print(f"Detected persons: {person_count}")
    assert person_count >= 3, f"Expected at least 3 persons on bus.jpg, found {person_count}"
    print("[PASS] Detector sanity check passed.")

    # 4. Latency benchmark over 30 runs for imgsz 640 and 480
    print("\nBenchmarking CPU inference latency over 30 runs:")
    # Use the first result's image array for consistent timing
    img_array = results[0].orig_img
    for imgsz in (640, 480):
        # Warmup (3 runs)
        for _ in range(3):
            model.predict(img_array, imgsz=imgsz, device="cpu", verbose=False)

        times_ms = []
        for _ in range(30):
            t0 = time.perf_counter()
            model.predict(img_array, imgsz=imgsz, device="cpu", verbose=False)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        avg_ms = sum(times_ms) / len(times_ms)
        min_ms = min(times_ms)
        max_ms = max(times_ms)
        approx_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"  imgsz={imgsz}: avg = {avg_ms:.1f} ms ({approx_fps:.1f} FPS) [min: {min_ms:.1f} ms, max: {max_ms:.1f} ms]")

    # 5. Webcam test (index 0)
    print("\nTesting webcam index 0:")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  Webcam 0 opened successfully. Resolution: {w}x{h}")
        else:
            print("  Webcam 0 opened but failed to grab a frame.")
        cap.release()
    else:
        print("  Webcam index 0 could not be opened (no camera attached or permission denied). Skipping gracefully.")

    # 6. RTSP stream test (if RTSP_URL is set)
    rtsp_url = os.environ.get("RTSP_URL")
    if rtsp_url:
        print(f"\nTesting RTSP stream from RTSP_URL: {rtsp_url}")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        rtsp_cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if rtsp_cap.isOpened():
            ret, frame = rtsp_cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  RTSP stream opened successfully. Resolution: {w}x{h}")
            else:
                print("  RTSP stream opened but failed to grab a frame.")
            rtsp_cap.release()
        else:
            print("  Failed to open RTSP stream.")
    else:
        print("\nRTSP_URL environment variable not set. Skipping RTSP test.")

    print("\n" + "=" * 60)
    print("check_env complete: All assertions passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

