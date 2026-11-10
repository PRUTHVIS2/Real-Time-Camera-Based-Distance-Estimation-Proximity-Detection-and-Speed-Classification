# Project Implementation Progress

**Project:** Real-Time Camera-Based Distance Estimation, Proximity Detection and Speed Classification of Pedestrians and Vehicles (MCA461P)  
**Current Active Branch:** `phase-3-calibration-homography`  
**Latest Tag:** `phase-3-done`  
**Status Date:** 2026-11-10  

---

## 1. Executive Summary

| Phase | Phase Name | Status | Acceptance Criteria / Tests |
|---|---|---|---|
| **Phase 0** | Environment and skeleton | **COMPLETE** | CPU PyTorch, pinned Python 3.12, strict typed config, `check_env.py` benchmark passed, RTSP `ffplay` & `check_env.py` passed. Tag: `phase-0-done`. |
| **Phase 1** | Capture and detection | **COMPLETE** | `FrameSource` (Webcam/RTSP/File), YOLO11 detector wrapper, COCO class filter, monotonic timestamps, `record.py`, `run_live.py`, RTSP reconnect & 2-min lag tests passed. Tag: `phase-1-done`. |
| **Phase 2** | Tracking & rider+bike merge | **COMPLETE** | ByteTrack tracking, pure geometric containment rider merger, pillion support, time-based flicker memory suppression, edge truncation tags. Tag: `phase-2-done`. |
| **Phase 3** | Calibration & homography | **COMPLETE** | Pure planar homography $H$ mathematics, reprojection errors (metres/px), leave-one-out CV, `tools/calibrate.py` with freeze-frame & 4x magnifying loupe, resolution guard. Synthetic tests (P3-A1–P3-A6) and human RTSP phone calibration tests (P3-H1–P3-H4) all passed (`0.53 cm` mean reprojection error, `0.75 cm` mean held-out validation error). Tag: `phase-3-done`. |
| **Phase 4** | Distance estimation | *In Progress* | Skeleton in place (`src/vision/geometry.py`, `tools/eval_distance.py`, `tests/test_geometry.py`). |
| **Phase 5** | Proximity detection | *Upcoming* | Skeleton in place (`src/vision/proximity.py`, `tests/test_proximity.py`). |
| **Phase 6** | Speed estimation & 5-level classification | *Upcoming* | Skeleton in place (`src/vision/speed.py`, `tools/eval_speed.py`, `tests/test_speed.py`). |
| **Phase 7** | Flask + WebSocket dashboard | *Upcoming* | Skeleton in place (`src/vision/dashboard/app.py`, `tests/test_dashboard.py`). |
| **Phase 8** | Full evaluation, benchmarks & paper figures | *Upcoming* | Skeleton in place (`tools/bench_fps.py`, `tools/eval_detection.py`). |

---

## 2. Detailed Phase-by-Phase Progress vs. `plan.md` Specification

### Phase 0: Environment and Skeleton
- **Goal:** Working Python environment, repo skeleton, config loader, and proof of detector capability.
- **Specification vs Actual:**
  - [x] Repository layout: sections, modules, tools, and test suites established.
  - [x] Pinned Python 3.12 environment via `uv` with CPU-only PyTorch and pinned `requirements.lock`.
  - [x] `config/default.yaml` and `src/vision/config.py`: typed dataclass configuration (`AppConfig`), deep-merging for `local.yaml`, strict unknown-key validation.
  - [x] `tools/check_env.py`: checks environment, CPU threads (8 cores), verifies YOLO11n inference on `bus.jpg` (4 persons detected), benchmarks latency (19.8 ms at 640px = ~50.6 FPS; 12.7 ms at 480px = ~78.7 FPS).
  - [x] Automated tests (`tests/test_config.py`): 5/5 passing (P0-A1, P0-A2).

### Phase 1: Capture and Detection
- **Goal:** Video capture from webcam or RTSP, run YOLO11, draw boxes for 3 target classes, show rolling FPS.
- **Specification vs Actual:**
  - [x] `src/vision/capture.py`: `FrameSource` interface with `WebcamSource`, threaded `RtspSource` (auto-reconnect + freshest frame drop queue), and `FileSource` with deterministic timestamps.
  - [x] Capture timestamping: `time.monotonic()` recorded at the moment of frame read, never after inference.
  - [x] `src/vision/detector.py`: Ultralytics YOLO11 wrapper mapping COCO classes 0, 1, 2, 3 (`person`, `bicycle`, `car`, `motorcycle`) and discarding all non-target classes (bus, truck, dog, etc.).
  - [x] `tools/run_live.py`: Real-time OpenCV viewer with class colors, confidence tags, and rolling FPS.
  - [x] `tools/record.py`: Records clean raw video and CSV file with monotonic capture timestamps (`frame_idx, t_capture`).
  - [x] Automated tests (`tests/test_capture.py`, `tests/test_detector.py`): 5/5 passing (P1-A1, P1-A2, P1-A3).

### Phase 2: Tracking and Rider+Bike Merge
- **Goal:** Persistent IDs with ByteTrack and one unified object per rider+bike.
- **Specification vs Actual:**
  - [x] ByteTrack integration: `model.track(persist=True, tracker="config/bytetrack.yaml")` with persistent IDs.
  - [x] `src/vision/merge.py`: Pure geometric containment `area(person ∩ bike) / area(person) >= 0.5`, horizontal center check, bottom edge alignment check.
  - [x] Merged representation: Output as single `two_wheeler` carrying the bike's track ID, bike bounding box (for ground wheel contact point), and accurate `rider_count` (supports single and pillion riders).
  - [x] Rider memory suppression: Suppresses flicker where a bike detection temporarily drops, preventing the rider from flashing as a pedestrian. Converted from frame count to time-based `rider_memory_s: 1.0` to be independent of camera FPS.
  - [x] Edge truncation: Marks `truncated = True` if the bounding box touches any frame margin within `truncation_margin_px`.
  - [x] Added `--save-video` feature to `tools/run_live.py` to record annotated video with burned-in boxes and `#ID` tags for live demonstration.
  - [x] Automated tests (`tests/test_merge.py`): 8/8 passing (P2-A1 through P2-A8).

### Phase 3: Calibration and Homography
- **Goal:** Tool to compute, validate, and save planar homography $H$ mapping pixels $(u, v)$ to ground world coordinates $(X, Y)$ in metres.
- **Specification vs Actual:**
  - [x] Webcam hardware & format follow-ups (Part A): Negotiated `MJPG` fourcc so the Linux UVC driver delivers native 1280x720 @ 30 FPS without downscaling to 640x480 YUYV. Identified `exposure_dynamic_framerate` as the cause of low-light frame-rate drops.
  - [x] Tracker tuning (Part A): Localized `config/bytetrack.yaml`, set `track_buffer: 60` for better occlusion recovery, and documented tracker motion-model limitations in `README.md`.
  - [x] `src/vision/calibration.py`: Pure mathematics for `compute_homography`, `pixel_to_world`, `world_to_pixel`, `reprojection_errors` (in metres and pixels), `leave_one_out_errors`, and resolution-validated `CalibrationData` JSON serialization.
  - [x] Collinearity & safety guards: SVD singularity checks enforce at least 4 non-collinear correspondence points.
  - [x] `tools/calibrate.py`: Interactive calibration GUI featuring live freeze-frame (`Space`/`f`), 4x magnifying loupe for sub-pixel clicking accuracy, undo (`u`), reset (`r`), compute $H$ (`Enter`), projected 1 m metric grid overlay, held-out validation (`v`), and JSON save (`s`).
  - [x] Grid specification: 15-point laptop grid (`data/calibration/laptop_webcam_points.csv`) and 9-point indoor RTSP phone grid (`data/calibration/phone_main_points.csv`, 1 m × 1 m at 0.5 m spacing) + 4 validation points (`data/calibration/phone_main_val_points.csv`).
  - [x] Automated tests (`tests/test_calibration.py`): 7/7 passing (P3-A1 through P3-A6 + LOO test).
  - [x] **Human Verification (Phone RTSP `phone_main`, 1920×1080):**
    - `P3-H1`: Saved `data/calibration/phone_main.json` (`camera_xy=[1.0, 0.0]`, `camera_height_m=1.47`). Mean reprojection error `0.53 cm` (max `1.10 cm`, mean LOO `1.01 cm`).
    - `P3-H2`: Held-out validation on `V0`–`V3`: mean error `0.75 cm`, max `1.03 cm`.
    - `P3-H3`: Visual metric grid aligned with physical floor tape lines.
    - `P3-H4`: Camera bump sensitivity confirmed (grid misaligns on camera movement; recalibrated afterward).

---

## 3. Test Suite Summary

Total automated unit tests currently passing: **25 / 25**

```
============================== 25 passed in 1.61s ==============================
tests/test_calibration.py .......                                        [ 28%]
tests/test_capture.py ...                                                [ 40%]
tests/test_config.py .....                                               [ 60%]
tests/test_detector.py ..                                                [ 68%]
tests/test_merge.py ........                                             [100%]
```

---

## 4. Git Origin and Remote Status

- **Configured Remote (`origin`):** `git@github.com:PRUTHVIS2/Real-Time-Camera-Based-Distance-Estimation-Proximity-Detection-and-Speed-Classification.git`
- **Current Branch:** `phase-3-calibration-homography` (moving to `phase-4-distance-estimation`)
- **Latest Tag:** `phase-3-done`

