# Minor Project (MCA461P)

**Real-Time Camera-Based Distance Estimation, Proximity Detection and Speed Classification of Pedestrians and Vehicles**

A fixed camera (phone over RTSP or laptop webcam) monitors a flat, calibrated ground area. The pipeline detects and tracks objects, merges riders with two-wheelers, estimates ground distance via a calibrated planar homography, detects proximity threshold violations with hysteresis, and estimates velocity to classify speed into 5 levels.

## Pipeline Overview

```
Live video -> Detect (YOLO11) -> Track (ByteTrack) -> Merge rider+bike
          -> Foot point (pixels) -> Homography H -> Ground position (X, Y) metres
          -> Distance to camera -> Proximity flag
          -> Position history -> Speed (m/s) -> 5-level speed class
          -> Flask + WebSocket dashboard
```

## Setup

1. System requirements: Arch Linux (or modern Linux), CPU only.
2. Ensure `uv` is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
3. Setup virtual environment and dependencies:
   ```bash
   uv venv --python 3.12 --seed
   source .venv/bin/activate
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   uv pip install ultralytics lapx numpy scipy pyyaml pandas matplotlib flask flask-socketio simple-websocket pytest
   ```
4. Check environment:
   ```bash
   python tools/check_env.py
   ```

## Configuration

Configuration values live in `config/default.yaml`.
Machine-specific overrides (such as local RTSP URLs or camera index) can be placed in `config/local.yaml` (gitignored).

