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

## Known Limitations

- **ByteTrack Appearance Model:** ByteTrack relies strictly on motion prediction (Kalman filtering) and bounding box IoU association to ensure lightweight, real-time performance on CPU architectures without deep ReID appearance feature extractors. Consequently:
  - If an object is fully hidden from view (e.g., completely occluded by another person or walking behind an obstacle) for longer than `track_buffer` frames, it will be assigned a new track ID upon reappearance.
  - **Effect on Distance Estimation:** **None.** Ground position and distance calculations are instantaneous geometric projections from the bounding box foot point via the planar homography matrix $H$; distance does not depend on tracking history or track IDs.
  - **Effect on Speed Estimation:** A new track ID restarts its position deque history, temporarily returning `ESTIMATING` until `min_points` accumulate. The velocity spike guard (`max_plausible_mps`) rejects abrupt position jumps, preventing spurious velocity spikes during ID swaps.


