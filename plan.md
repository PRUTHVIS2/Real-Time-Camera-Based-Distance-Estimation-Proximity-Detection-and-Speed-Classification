# Minor Project Implementation Plan

**Project:** Real-Time Camera-Based Distance Estimation, Proximity Detection and Speed Classification of Pedestrians and Vehicles (MCA461P)

**Audience:** an AI coding agent (and the human reviewer, Pruthvi). Build the project **one phase at a time**.

---

## 0. Rules for the agent (read first, obey always)

1. **Work on exactly one phase, then STOP.** When a phase's acceptance criteria are met, write the *Phase Report* (format in section 0.1) and stop. Do not start the next phase until the human replies with `approved, start phase N`.
2. **Never invent results.** Run the tests and paste the real output. If a test needs the physical world (camera, tape measure, a person walking), do not pretend to run it. List it under "Human tests pending" with exact instructions and stop.
3. **Stay in scope.** Do NOT: train or fine-tune a detector, use depth models (Depth Anything, MiDaS), use LiDAR/stereo, target moving cameras, deploy to a campus gate or a vehicle, or add GPU-only code. CPU only.
4. **Minimal dependencies.** Use only the packages in section 3. If you need another, say why in the Phase Report and ask first.
5. **Keep geometry code pure.** Anything that does maths (homography, distance, speed, classification, proximity, merging) must be plain functions with no camera, network or file I/O, so it can be unit tested with synthetic data.
6. **Everything tunable lives in `config/default.yaml`.** No magic numbers in code. Thresholds are frozen before final evaluation (Phase 8).
7. **Units:** metres, seconds, m/s internally. Convert to km/h only for display and for the vehicle speed classes. Pixel coordinates are `(u, v)` with origin top-left.
8. **Timestamps:** every frame is stamped with `time.monotonic()` at the moment it is *read from the camera*, never after inference.
9. **Git discipline:** one branch per phase (`phase-N-<name>`), small commits, tag `phase-N-done` at the end. Do not merge or delete branches. Do not commit weights, videos or recordings (`.gitignore` them).
10. **Code quality:** type hints, short functions, docstrings stating units and coordinate frames, `logging` instead of `print` (except in CLI tools), no global mutable state.

### 0.1 Phase Report format (output this at the end of every phase, then stop)

```
## Phase N Report
- Summary: what was built (3-6 lines)
- Files added/changed: list
- How to run: exact commands
- Automated tests: command + pasted pass/fail summary
- Human tests pending: list with IDs from this plan (or "none")
- Deviations from plan: and why (or "none")
- Known issues / risks
- Questions for the human
STOPPED. Waiting for: "approved, start phase N+1"
```

---

## 1. Project in one page

A fixed camera (phone over RTSP, or laptop webcam) watches a flat, measured area of ground. The pipeline:

```
Live video -> Detect (YOLO11) -> Track (ByteTrack) -> Merge rider+bike
          -> Foot point (pixels) -> Homography H -> Ground position (X, Y) metres
          -> Distance to camera -> Proximity flag
          -> Position history -> Speed (m/s) -> 5-level speed class
          -> Flask + WebSocket dashboard
```

Key idea: a **homography H** maps pixels that lie on the ground plane to real ground coordinates in metres. It is computed once from a calibration grid measured with a tape. It is valid only for **one fixed camera pose** on **flat ground**.

### Objects and classes

| Our class | COCO classes used | Foot point used for ground position |
|---|---|---|
| `person` | person (0) | bottom-centre of the person box |
| `two_wheeler` | bicycle (1), motorcycle (3) (+ the riders merged into it) | bottom-centre of the **bike** box (wheel contact) |
| `car` | car (2) | bottom-centre of the car box |

Ignore every other COCO class (bus, truck, etc.).

### Targets from the review deck (to be measured, not assumed)
Distance accuracy 90-95%, speed-class accuracy 85-90%, per-class detection accuracy 90-95%, and a reported FPS figure.

---

## 2. Fixed design decisions

| Topic | Decision |
|---|---|
| Detector | Ultralytics YOLO11, COCO-pretrained, `yolo11n.pt` first, `yolo11s.pt` as comparison |
| Tracker | ByteTrack through Ultralytics (`tracker="bytetrack.yaml"`, `persist=True`) |
| Rider + bike | Merge **after** tracking: any person track that is mostly inside a bike box is absorbed into that bike. The merged object keeps the **bike's track ID** (see Phase 2) |
| Distance | Camera-to-object ground distance from homography. Pinhole formula implemented only as a baseline for comparison |
| Proximity | Camera-to-object distance compared with a threshold (per class), with hysteresis. Object-to-object is an optional extra, off by default |
| Speed | Ground-plane velocity from a windowed linear fit over real timestamps |
| Speed classes | 5 levels. Person scale in m/s, vehicle scale in km/h (shared by `two_wheeler` and `car`). Defaults in section 5 |
| Dashboard | Flask + Flask-SocketIO (threading mode) |
| Live vs recorded | The product runs on **live** feeds. Recorded clips are allowed only as dev fixtures and for post-hoc ground truth (frame counting), never as the demonstrated input |

---

## 3. Environment setup (Arch Linux, Ryzen 7, 32 GB RAM, no dedicated GPU)

Everything runs on the **CPU**. Do not try ROCm or any GPU path for the integrated graphics.

### 3.1 System packages

```bash
sudo pacman -Syu
sudo pacman -S --needed base-devel git uv ffmpeg v4l-utils
```

- `uv` manages Python versions and virtual environments.
- `ffmpeg` provides `ffplay`/`ffprobe`, which are used to test the RTSP stream *outside* Python.
- `v4l-utils` provides `v4l2-ctl` to list webcams.

Optional, only if `cv2.imshow` fails with a Qt/xcb error (see Troubleshooting):

```bash
sudo pacman -S --needed xcb-util-cursor
```

### 3.2 Python environment (use a pinned Python, not the rolling system one)

Arch's system Python is very new and PyTorch wheels can lag behind it, so use a pinned interpreter via `uv`.

```bash
mkdir -p ~/minor-project && cd ~/minor-project
git init
uv venv --python 3.12 --seed
source .venv/bin/activate

# 1) CPU-only PyTorch first (avoids pulling a multi-GB CUDA build)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else
uv pip install ultralytics lapx numpy scipy pyyaml pandas matplotlib \
               flask flask-socketio simple-websocket pytest
```

Notes:
- `--seed` puts `pip` in the venv. Ultralytics tries to auto-install missing tracking requirements with pip.
- `lapx` is installed up front because ByteTrack needs a linear-assignment solver (`lap`). If `lapx` fails to install, try `uv pip install lap`.
- OpenCV arrives as a dependency of Ultralytics (`opencv-python`). Do not also install `opencv-python-headless`.
- After everything works once, freeze versions: `uv pip freeze > requirements.lock` and commit it.

### 3.3 Webcam and RTSP checks (run by the human, outside Python)

```bash
# Laptop webcam
v4l2-ctl --list-devices          # note the /dev/videoN of the webcam
ls -l /dev/video*                # check you have access (if not: add yourself to the 'video' group)

# Phone RTSP stream (phone and laptop on the SAME Wi-Fi network)
ffplay -rtsp_transport tcp rtsp://<PHONE_IP>:<PORT>/<PATH>
```

If `ffplay` shows the phone's video, the stream is fine. Write the exact URL in `config/local.yaml` (never commit it, it is gitignored). If it does not work here, it will not work in Python either, so fix the app/network first.

### 3.4 Model weights

Ultralytics downloads `yolo11n.pt` automatically on first use. Store weights in `models/` (gitignored). Download `yolo11s.pt` the same way for the comparison later.

### 3.5 Performance expectations (be honest)

On a CPU, YOLO11n at 640 px may reach roughly 8-25 FPS and YOLO11s noticeably less. Those are rough guesses, so **measure** in Phase 0 and report the real numbers. If FPS is too low, options in order: smaller `imgsz` (480/416), `yolo11n`, skip every 2nd frame for detection while keeping tracker updates, export to ONNX (`model.export(format="onnx")`) and run with ONNX Runtime. Do the cheap options first.

---

## 4. Repository layout

```
minor-project/
├── plan.md
├── README.md
├── pyproject.toml            # or requirements.txt + requirements.lock
├── .gitignore                # models/, data/recordings/, config/local.yaml, results/*.mp4
├── config/
│   ├── default.yaml
│   └── local.yaml            # gitignored: RTSP URL, camera index, per-machine overrides
├── models/                   # yolo weights (gitignored)
├── data/
│   ├── calibration/          # <camera_id>.json (committed, small)
│   ├── recordings/           # raw clips + timestamp CSVs (gitignored)
│   └── ground_truth/         # CSV run sheets filled in by hand
├── src/vision/
│   ├── config.py             # load/merge yaml, dataclasses
│   ├── types.py              # Detection, TrackedObject, Measurement
│   ├── capture.py            # FrameSource: Webcam, Rtsp (threaded, latest-frame), File (dev only)
│   ├── detector.py           # YOLO wrapper + class mapping
│   ├── merge.py              # rider+bike merge (pure)
│   ├── geometry.py           # homography apply, foot point, distance, pinhole baseline (pure)
│   ├── calibration.py        # compute H, reprojection error, save/load JSON (pure + small IO)
│   ├── proximity.py          # threshold + hysteresis state machine (pure)
│   ├── speed.py              # windowed fit, guards, classification, voting (pure)
│   ├── pipeline.py           # wires everything, one process_frame()
│   └── dashboard/
│       ├── app.py            # Flask + SocketIO
│       └── static/ + templates/
├── tools/
│   ├── check_env.py
│   ├── run_live.py           # OpenCV window, overlay
│   ├── calibrate.py          # interactive point clicking
│   ├── record.py             # raw video + per-frame timestamps CSV
│   ├── eval_distance.py
│   ├── eval_speed.py
│   ├── eval_detection.py
│   └── bench_fps.py
├── tests/
│   ├── test_merge.py ... test_speed.py ...
└── results/                  # CSVs, plots, tables for the paper
```

---

## 5. Config reference (`config/default.yaml`, values are starting points)

```yaml
camera:
  camera_id: phone_main        # selects data/calibration/phone_main.json
  source: 0                    # int = webcam index | "rtsp://..." | "clip.mp4" (dev only)
  width: 1280
  height: 720
  rtsp_transport: tcp          # tcp avoids packet-loss artefacts over Wi-Fi

detector:
  model: models/yolo11n.pt
  imgsz: 640
  conf: 0.25                   # try 0.10-0.15 in Phase 2 so ByteTrack can use low-score boxes
  iou: 0.5
  device: cpu
  num_threads: 8               # set to physical cores

merge:
  min_containment: 0.5         # share of the person box area that must lie inside the bike box
  rider_memory_frames: 15      # keep suppressing a recently merged rider if the bike detection flickers

geometry:
  max_range_m: 15              # beyond this, distance is OUT_OF_RANGE
  distance_ema_alpha: 0.4
  truncation_margin_px: 3      # box touching frame edge => foot point unreliable

pinhole_baseline:              # ASSUMED real heights, used only for the comparison baseline
  person_height_m: 1.7
  two_wheeler_height_m: 1.1
  car_height_m: 1.5

proximity:
  threshold_m: {person: 3.0, two_wheeler: 5.0, car: 8.0}   # placeholders, confirm with guide
  hysteresis_m: 0.3            # enter at d <= T, leave at d > T + hysteresis

speed:
  window_s: 0.7
  min_points: 5
  class_vote_len: 5
  track_forget_s: 2.0
  dead_zone_mps: {person: 0.3, two_wheeler: 0.5, car: 0.5}
  max_plausible_mps: {person: 12, two_wheeler: 40, car: 45}   # reject spikes above this (ID switches)
  # Upper bounds of the first four classes; anything above the last is the fifth class.
  person_upper_mps: [0.3, 1.0, 1.8, 3.0]      # Stationary / Slow / Moderate / Fast / Very fast
  vehicle_upper_kmh: [2, 30, 60, 90]          # shared by two_wheeler and car

dashboard:
  host: 0.0.0.0
  port: 5000
  max_send_fps: 15
  jpeg_quality: 70
```

**Speed class names:** `STATIONARY, SLOW, MODERATE, FAST, VERY_FAST`.

---

## 6. Shared data types and conventions

```python
@dataclass
class Detection:            # raw YOLO output after class mapping
    xyxy: tuple[float, float, float, float]
    conf: float
    cls: str                # 'person' | 'bicycle' | 'motorcycle' | 'car'
    track_id: int | None

@dataclass
class TrackedObject:        # after merge
    track_id: int
    cls: str                # 'person' | 'two_wheeler' | 'car'
    box_xyxy: tuple[float, float, float, float]   # box used for the foot point
    conf: float
    rider_count: int        # 0 for non-two-wheelers or empty bikes
    truncated: bool
    t: float                # time.monotonic() at capture

@dataclass
class Measurement:
    obj: TrackedObject
    foot_px: tuple[float, float]
    world_xy: tuple[float, float] | None     # metres in the calibration frame
    distance_m: float | None
    in_range: bool
    proximity: str          # 'SAFE' | 'TOO_CLOSE' | 'OUT_OF_RANGE' | 'UNKNOWN'
    speed_mps: float | None
    speed_kmh: float | None
    speed_class: str | None # one of the 5 names, or 'ESTIMATING'
```

**World frame:** origin = one marked corner of the calibration grid, +X to the right, +Y away from the camera (forward), metres, Z = 0 is the ground. The camera's own ground position `(Xc, Yc)` (the point on the ground directly below the camera lens, found with a plumb line and tape) is stored in the calibration file.

---

# PHASES

---

## Phase 0: Environment and skeleton

**Goal:** a working Python environment, repo skeleton, config loader, and proof that the laptop can run the detector and open both camera types.

### Instructions
1. Create the repository layout from section 4 (empty modules with docstrings are fine), `.gitignore`, `pyproject.toml`/requirements, and a short `README.md`.
2. Implement `config.py`: load `default.yaml`, deep-merge `local.yaml` if present, expose a typed object, and fail loudly on unknown keys.
3. Implement `tools/check_env.py` which prints and asserts:
   - Python, torch, torchvision, ultralytics, opencv, numpy versions
   - `torch.cuda.is_available()` (expected False) and the number of CPU threads in use
   - loads `yolo11n.pt`, runs it on `https://ultralytics.com/images/bus.jpg`, and reports the detections
   - tries to open webcam index 0 and grab one frame (report resolution), skipping gracefully if no camera
   - if the env var `RTSP_URL` is set, opens it with OpenCV (FFmpeg backend, TCP transport via `OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"`) and grabs a frame
   - prints average inference time (ms) over 30 runs for `imgsz` 640 and 480
4. Create `tests/test_config.py`.

### Automated tests
| ID | Test | Expected |
|---|---|---|
| P0-A1 | `pytest tests/test_config.py` | default config loads; `local.yaml` overrides a value; an unknown key raises an error |
| P0-A2 | `python tools/check_env.py` | exits 0; at least 3 `person` detections on bus.jpg |

### Human tests
| ID | Test | Expected |
|---|---|---|
| P0-H1 | Run the `ffplay` command from 3.3 | phone video appears |
| P0-H2 | `RTSP_URL=<url> python tools/check_env.py` | a frame from the phone is grabbed, resolution printed |
| P0-H3 | Run `check_env.py` with the webcam | webcam frame grabbed |

### Acceptance
- [ ] All tests above pass or are listed as pending with a reason.
- [ ] Inference times recorded in the report (they set expectations for everything later).

### Review focus for the human
Config design, package choices, whether the inference times are workable.

**STOP after the Phase Report.**

---

## Phase 1: Capture and detection

**Goal:** live video from webcam or RTSP, run YOLO11, draw boxes for the three classes, show FPS.

### Instructions
1. `capture.py`: a `FrameSource` interface with `read() -> (frame, t_capture)` and `close()`.
   - `WebcamSource(index, width, height)`.
   - `RtspSource(url)`: reads in a **background thread** and keeps only the **latest** frame so the pipeline never processes stale frames. Uses TCP. Auto-reconnects (retry every 2 s) and exposes `is_connected`.
   - `FileSource(path)` for tests only, with timestamps taken from the video's frame times.
   - `t_capture` is `time.monotonic()` taken when the frame is read.
2. `detector.py`: wrapper around Ultralytics. Map COCO ids 0,1,2,3 to `person, bicycle, car, motorcycle`, drop everything else. Respect `imgsz`, `conf`, `iou`, `device=cpu`, `torch.set_num_threads(cfg)`.
3. `tools/run_live.py`: OpenCV window showing boxes with class and confidence, an FPS counter (rolling average over the last 30 frames), and `q` to quit. Choose the source with `--source`.
4. `tools/record.py`: save raw video and a CSV `frame_idx, t_capture` (needed later for ground truth by frame counting).

### Automated tests
| ID | Test | Expected |
|---|---|---|
| P1-A1 | `FileSource` on a short fixture clip | frames returned in order, timestamps strictly increasing |
| P1-A2 | Class mapping unit test with fake YOLO output | only person/bicycle/car/motorcycle survive, bus/truck dropped |
| P1-A3 | `RtspSource` with a bad URL | does not crash; `is_connected` is False; retries |

### Human tests
| ID | Test | Expected |
|---|---|---|
| P1-H1 | `run_live.py --source 0` with yourself in view | `person` box, stable, FPS shown |
| P1-H2 | `run_live.py --source <rtsp url>` | same, from the phone; compare perceived lag to the webcam |
| P1-H3 | Show a bike, a motorbike or a toy car/photo in view | correct class labels |
| P1-H4 | Unplug/stop the phone stream, then restart it | window recovers without restarting the program |
| P1-H5 | Wave your hand fast, compare against displayed video | lag does not grow over 2 minutes (no frame backlog) |

### Acceptance
- [ ] Live detections from both sources.
- [ ] Measured FPS for webcam and RTSP is written in the report.
- [ ] No growing latency (P1-H5).

### Review focus
The `RtspSource` threading and reconnect logic, and the timestamp placement.

**STOP after the Phase Report.**

---

## Phase 2: Tracking and rider+bike merge

**Goal:** persistent IDs with ByteTrack, and one object per rider+bike.

### Instructions
1. Use `model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0,1,2,3], ...)` so each detection carries a `track_id`. Convert results to `Detection`.
2. `merge.py` (pure function) `merge_riders(dets, min_containment) -> list[TrackedObject]`:
   - For every `person` and every `bicycle`/`motorcycle` box compute `containment = area(person ∩ bike) / area(person)`.
   - A person is a **rider** of a bike if `containment >= min_containment` AND the person's horizontal centre lies within the bike box's x-range AND the person's bottom edge is not far below the bike's bottom edge. If one person qualifies for several bikes, pick the largest containment.
   - Merged output: one `two_wheeler` with `track_id` = the **bike's** track ID, `box_xyxy` = the **bike box** (used for the foot point), and `rider_count` = number of riders (supports pillion).
   - Unmerged persons stay `person`; unmerged bikes stay `two_wheeler` with `rider_count = 0`.
   - `rider_memory_frames`: if a person track was merged recently and its bike is missing for a few frames (detector flicker), keep suppressing that person instead of letting it flash up as a pedestrian. Keep this simple and tested.
3. Mark `truncated = True` when the box touches the frame edge within `truncation_margin_px`.
4. Extend `run_live.py` to draw `#ID class` and the merged boxes (rider and bike drawn as one object, say with a small "rider(s): n" tag).
5. Add an experiment flag `--conf` so `conf = 0.25` and `conf = 0.10` can be compared for ID stability.

### Automated tests (synthetic boxes, `tests/test_merge.py`)
| ID | Case | Expected |
|---|---|---|
| P2-A1 | One person box well inside a motorcycle box | 1 `two_wheeler`, rider_count 1, ID = bike's ID, no `person` left |
| P2-A2 | Person standing beside a motorcycle (small overlap) | 1 person + 1 two_wheeler (rider_count 0) |
| P2-A3 | Two persons on one motorcycle (pillion) | 1 two_wheeler, rider_count 2 |
| P2-A4 | Person on a bicycle | merged as two_wheeler |
| P2-A5 | Person overlapping two bikes, more contained in bike B | assigned to B only |
| P2-A6 | Rider memory: 5 frames merged, then 3 frames with no bike box but the same person track | person stays suppressed for those frames, reappears as person after `rider_memory_frames` |
| P2-A7 | Box touching frame edge | `truncated = True` |
| P2-A8 | Empty input | empty output, no crash |

### Human tests
| ID | Test | Expected |
|---|---|---|
| P2-H1 | Walk across the view alone, 5 times | same ID for the whole walk in at least 4 of 5 runs |
| P2-H2 | Two people cross paths (brief occlusion) | IDs preserved after crossing in most runs; note how many swaps occur |
| P2-H3 | Person walking a bicycle / sitting on a parked motorcycle | one merged object, bike ID |
| P2-H4 | Ride a bike/motorcycle through (safely) | single `two_wheeler` with a stable ID |
| P2-H5 | Compare `conf 0.25` vs `0.10` over the same scenes | write the ID-switch counts in the report |

### Acceptance
- [ ] All synthetic merge tests pass.
- [ ] Tracking observed live; ID-switch counts recorded.

### Review focus
The merge thresholds and the rider-memory logic. Check that no information needed later is thrown away (rider_count, truncated).

**STOP after the Phase Report.**

---

## Phase 3: Calibration and homography

**Goal:** a tool to build and save the homography from a measured ground grid, with proper error reporting.

### Physical setup (human, do this before testing)
1. Mount the camera on a tripod/stand **1.5-3 m high**, tilted down at the ground area. It must not move after calibration. Lock the phone's focus/exposure/zoom if the app allows it, and keep the same resolution as `config.camera`.
2. Mark a grid on flat ground with tape/chalk, e.g. a 4 m × 8 m area with marks every 2 m (15 points) or at least 6-8 well-spread points. The points must not all be in a line.
3. Choose the origin at one marked corner. +X right, +Y forward. Measure each mark's `(X, Y)` in metres with the tape and write them in `data/calibration/<camera_id>_points.csv` (`label,X,Y`).
4. Measure the camera's ground position `(Xc, Yc)`: hang a plumb line from the lens, mark where it touches the ground, measure it relative to the origin. Also record the lens height in metres.
5. Mark **extra test points** (at least 4) that are NOT used for calibration, for validation.

### Instructions
1. `calibration.py` (pure maths + small JSON IO):
   - `compute_homography(pixel_pts, world_pts)` using `cv2.findHomography(..., method=0)` (least squares). Require at least 4 points; warn if they are nearly collinear.
   - `pixel_to_world(H, uv)`, `world_to_pixel(H, XY)` (inverse).
   - `reprojection_errors(H, pixel_pts, world_pts)`: per-point error in **metres** (pixel→world vs measured) and in pixels (world→pixel).
   - `leave_one_out_errors(...)`: recompute H without each point and measure that point's error in metres.
   - Save/load JSON: `camera_id, image_width, image_height, H, pixel_pts, world_pts, camera_xy, camera_height_m, date, notes`. On load, refuse to use it if the live frame size differs from `image_width/height`.
2. `tools/calibrate.py` (OpenCV window):
   - Grab a still frame from the chosen source (or load a saved still).
   - Click the marks in the order given by the CSV; show numbered dots; keys: `u` undo, `r` reset, `Enter` compute, `s` save, `v` validate with extra points, `q` quit. Support zoom for precise clicking (e.g. `+`/`-` or mouse wheel).
   - After computing H, overlay a metric grid (lines every 1 m) on the image and print the reprojection and leave-one-out errors.
3. Optional (do only if lens distortion is visible at the frame edges, i.e. straight lines look curved): add `tools/calibrate_lens.py` that uses a printed chessboard and `cv2.calibrateCamera`, saves intrinsics, and undistorts frames **before** detection. Ask the human before doing this.

### Automated tests (`tests/test_calibration.py`)
| ID | Test | Expected |
|---|---|---|
| P3-A1 | Build a synthetic camera (known H_true), generate pixels from 12 known world points, recover H | mapping error < 1e-6 m |
| P3-A2 | Add Gaussian pixel noise (σ = 1.5 px) to the clicks | mean error in metres stays small and is reported; error grows with distance from the camera (assert the trend) |
| P3-A3 | `pixel_to_world` then `world_to_pixel` | round trip within 1e-6 |
| P3-A4 | Fewer than 4 points | clear error |
| P3-A5 | Collinear points | warning/error |
| P3-A6 | JSON save → load | identical H; loading with a different frame size is refused |

### Human tests
| ID | Test | Expected |
|---|---|---|
| P3-H1 | Click 8-15 calibration points, compute H | mean reprojection error ≤ 3 cm (starting target; report real value) |
| P3-H2 | Press `v` with the extra held-out points | error ≤ 10 cm for points within 6 m (report actual values, including the worst point) |
| P3-H3 | Look at the overlaid 1 m grid on the live image | grid lines run parallel to the tape lines on the ground |
| P3-H4 | Bump or tilt the camera slightly, view the overlay again | grid visibly misaligned (proves why the camera must not move). Then recalibrate |

### Acceptance
- [ ] Synthetic tests pass.
- [ ] A saved `data/calibration/<camera_id>.json` exists with errors documented in the report.

### Review focus
The coordinate conventions, the error metrics (metres, not just pixels), and the validation workflow.

**STOP after the Phase Report.**

---

## Phase 4: Distance estimation

**Goal:** for each tracked object compute its ground position and its distance from the camera, validated against tape measurements. Implement the pinhole baseline for comparison.

### Instructions
1. `geometry.py` (pure):
   - `foot_point(obj) -> (u, v)`: bottom-centre of `box_xyxy`.
   - `distance_from_camera(H, foot_px, camera_xy) -> (X, Y, d)` with `d = hypot(X - Xc, Y - Yc)`.
   - `is_in_range(...)`: false if the ground point is far outside the calibrated grid's convex hull (with a tolerance) or farther than `max_range_m`.
   - `pinhole_distance(h_px, real_height_m, focal_px)`: baseline `d = f * H_real / h_px`. The focal length in pixels is estimated from one tape-measured reference (a person of known height at a known distance) and stored in the calibration JSON as `focal_px`.
   - An EMA smoother for distance (`distance_ema_alpha`).
2. Truncated objects (`truncated = True` at the bottom edge) → distance is `None` (foot not visible).
3. `pipeline.py`: a `process_frame()` that returns a list of `Measurement` (speed/proximity still empty). Add `--log-csv` to write per frame: `t, id, cls, X, Y, distance, in_range, truncated`.
4. `run_live.py`: overlay `d = 4.2 m` next to each box and draw a small top-down map in a corner (optional but useful for debugging).
5. `tools/eval_distance.py`: reads a ground-truth CSV (`point_label, X_true, Y_true, d_true`) and a log CSV; for each test position takes the **median** over the stable frames and reports per-position error, MAE (m), mean relative error (MRE), accuracy `= 1 - MRE`, error vs range bucket (0-4, 4-8, 8-12 m), and homography-vs-pinhole side by side. Outputs a CSV and a plot to `results/`.

### Automated tests
| ID | Test | Expected |
|---|---|---|
| P4-A1 | Synthetic camera: objects at known ground points, pixels generated from H_true | computed distance equals true distance within 1e-6 |
| P4-A2 | Object box at the frame bottom edge | `truncated`, distance None |
| P4-A3 | Foot point of a merged two_wheeler | uses the bike box, not the rider box |
| P4-A4 | Point outside the calibrated area | `in_range = False` |
| P4-A5 | EMA smoother | step input converges, no overshoot |
| P4-A6 | Pinhole function | `d` scales inversely with `h_px` |

### Human tests (distance protocol, person first)
Mark 12 test positions with tape, e.g. Y = 3, 5, 7, 9 m and X = -2, 0, +2 m, none of them calibration points. Measure each position's true camera distance with the tape.
| ID | Test | Expected |
|---|---|---|
| P4-H1 | A person stands (feet on the mark) at each position for ~5 s, 3 trials, logging on | CSV logs for all positions |
| P4-H2 | `eval_distance.py` on the person logs | report MAE, MRE, error vs range, homography vs pinhole. Starting target: MRE ≤ 10% within 8 m. Report honestly what you get |
| P4-H3 | Person walks toward and away from the camera along the centre line | distance decreases/increases smoothly on the overlay |
| P4-H4 | Person partly out of frame at the bottom | distance hidden, not wrong |
| P4-H5 | Repeat P4-H1/H2 for a **two-wheeler** (parked bike with the foot point on the wheel contact) | separate result table |
| P4-H6 | Repeat for a **car** (parked at several marks) | separate result table |

### Acceptance
- [ ] Synthetic tests pass.
- [ ] Person distance results table produced; two-wheeler and car tables produced or listed as pending.
- [ ] Comparison table homography vs pinhole produced.

### Review focus
Which point of the box is the foot point; how range limits and truncation are handled; the evaluation maths.

**STOP after the Phase Report.**

---

## Phase 5: Proximity detection

**Goal:** flag objects that are within a threshold distance of the camera, without flicker.

### Instructions
1. `proximity.py` (pure): a small per-track state machine.
   - States: `SAFE`, `TOO_CLOSE`, `OUT_OF_RANGE`, `UNKNOWN` (no valid distance).
   - Enter `TOO_CLOSE` when `distance <= threshold[cls]`. Leave it only when `distance > threshold[cls] + hysteresis`.
   - `OUT_OF_RANGE` when `in_range` is false; `UNKNOWN` when distance is None.
   - State is keyed by `track_id` and forgotten when the track disappears for `track_forget_s`.
2. Overlay: red box and "TOO CLOSE" label; yellow "out of range". Add counters `objects_total`, `objects_too_close`.
3. Optional, off by default (`proximity.pairwise: false`): object-to-object distance from the ground positions. Implement only if the human asks.

### Automated tests (`tests/test_proximity.py`)
| ID | Test | Expected |
|---|---|---|
| P5-A1 | Distance sequence 5.0, 4.0, 3.1, 2.9, 3.1, 3.2, 3.4 with threshold 3.0 and hysteresis 0.3 | SAFE, SAFE, SAFE, TOO_CLOSE, TOO_CLOSE, TOO_CLOSE, SAFE |
| P5-A2 | Distance jittering 2.95 ↔ 3.05 for 60 frames | no more than 1 state change |
| P5-A3 | Different thresholds per class | person at 4 m SAFE, car at 4 m TOO_CLOSE (with the default placeholder values 3 m / 8 m) |
| P5-A4 | `in_range = False` | OUT_OF_RANGE regardless of distance |
| P5-A5 | Track disappears longer than `track_forget_s` | state cleared |

### Human tests
| ID | Test | Expected |
|---|---|---|
| P5-H1 | Walk slowly toward the camera | label flips to TOO_CLOSE once near the threshold and stays steady |
| P5-H2 | Stand exactly at the threshold line for 10 s | at most one flip |
| P5-H3 | Two people at different distances | only the near one is flagged |
| P5-H4 | Two-wheeler / car pass | thresholds per class applied |

### Acceptance
- [ ] All tests pass; no visible flicker at the threshold.

### Review focus
Whether the thresholds make sense per class, and whether hysteresis feels right on screen.

**STOP after the Phase Report.**

---

## Phase 6: Speed estimation and 5-level classification

**Goal:** per-track ground speed from real timestamps, smoothed, guarded, and classified into 5 levels.

### Instructions
1. `speed.py` (pure):
   - `SpeedEstimator` keeps, per `track_id`, a deque of `(t, X, Y)` using `t_capture` (never frame counts).
   - Add a point only if the foot point is valid and in range. Drop points older than `window_s`.
   - If fewer than `min_points`, or the window spans less than ~60% of `window_s`, return `ESTIMATING`.
   - Fit a line of X against t and Y against t (`numpy.polyfit(..., 1)`); speed = `hypot(vx, vy)` in m/s.
   - **Guards:** if the speed exceeds `max_plausible_mps[cls]` treat it as a tracking glitch, keep the previous value and do not store that point; apply the dead zone (below `dead_zone_mps[cls]` → 0).
   - Forget tracks not seen for `track_forget_s`.
   - `classify_speed(speed_mps, cls, cfg)`: persons use `person_upper_mps`; `two_wheeler` and `car` use `vehicle_upper_kmh` after converting m/s → km/h. Boundary rule: a value exactly equal to an upper bound belongs to the **lower** class.
   - `ClassVoter`: majority of the last `class_vote_len` classes, to stop boundary flicker.
2. Overlay and logging: add `speed_kmh` and `speed_class`; log columns `speed_mps, speed_class`.
3. `tools/eval_speed.py`: reads a ground-truth CSV (`run_id, cls, true_speed, unit, true_class`) and the logs, computes the confusion matrix, exact accuracy, **within-one-class accuracy**, per-class precision/recall, and saves them to `results/`.
4. `tools/record.py` should also work during live runs, so ground truth can be derived by **frame counting** between two ground lines from the saved raw video and timestamps (more accurate than a stopwatch).

### Automated tests (`tests/test_speed.py`)
| ID | Test | Expected |
|---|---|---|
| P6-A1 | Synthetic constant-velocity track, 1.4 m/s, 30 Hz, no noise | estimated 1.4 m/s (±0.01) |
| P6-A2 | Same with σ = 5 cm position noise | within ±0.3 m/s after the window fills |
| P6-A3 | Stationary object with 5 cm noise | below the dead zone → 0 → `STATIONARY` |
| P6-A4 | Irregular timestamps (random dropped frames, gaps up to 200 ms) | speed still correct, because it uses timestamps |
| P6-A5 | A sudden 10 m position jump (ID switch) | spike rejected, previous speed kept |
| P6-A6 | New track with < `min_points` | `ESTIMATING` |
| P6-A7 | Class boundaries: persons at 0.3, 0.31, 1.0, 1.81, 3.01 m/s | STATIONARY, SLOW, SLOW, FAST, VERY_FAST |
| P6-A8 | Vehicles at 2, 30, 30.1, 60, 90.1 km/h | STATIONARY, SLOW, MODERATE, MODERATE, VERY_FAST |
| P6-A9 | Speed oscillating across a boundary | voter limits flips |
| P6-A10 | Track unseen for > `track_forget_s` | history cleared |

### Human tests (speed protocol)
Set two ground lines a known distance D apart inside the calibrated area (D = 6 m for people/bikes). The runs must take place inside the calibrated region and the camera must see the whole lane.
| ID | Test | Expected |
|---|---|---|
| P6-H1 | Person: 5 runs each for standing still, slow walk, normal walk, brisk/jog, run. True speed = D / time (frame counting or stopwatch) | CSV run sheet filled in |
| P6-H2 | `eval_speed.py` on the person runs | confusion matrix + accuracy + within-one-class accuracy (target 85-90%, report actual) |
| P6-H3 | Boundary check: a few runs deliberately close to class boundaries | listed separately, not mixed into the headline accuracy |
| P6-H4 | Person walks toward the camera (radial motion) | speed stays sensible (homography handles radial motion in metres) |
| P6-H5 | Two-wheeler and car runs at the speeds you can reach safely in a permitted area (use a speedometer or phone GPS as reference) | separate confusion matrices; state clearly which classes were reachable and validated |

### Acceptance
- [ ] Synthetic tests pass.
- [ ] Person run sheet and results produced; vehicle results produced or explicitly marked pending.

### Review focus
Time handling, smoothing window, guards, and whether the class thresholds are sensible.

**STOP after the Phase Report.**

---

## Phase 7: Flask + WebSocket dashboard

**Goal:** a browser dashboard that shows live video with overlays, plus distance, proximity alerts and speed class per object.

### Instructions
1. `dashboard/app.py`: Flask + `Flask-SocketIO(async_mode="threading")`.
   - The **pipeline runs in its own background thread** and publishes its latest result to a thread-safe single-slot buffer. The web server reads from it. A slow browser must never slow the pipeline.
   - Emit a `frame_data` Socket.IO event at most `max_send_fps` times per second containing: `jpeg` (base64 of the annotated frame at `jpeg_quality`), `t`, `fps`, and `objects: [{id, cls, distance_m, proximity, speed_kmh, speed_class}]`, plus `stream_ok` (bool).
   - Routes: `/` (page), `/health` (JSON: pipeline running, FPS, camera connected).
2. Page (plain HTML/JS, no build step): live video area, a table of objects (ID, class, distance, proximity state, speed, speed class), colour badges, a visible red banner when any object is `TOO_CLOSE`, FPS and connection status, and a "stream lost" message when `stream_ok` is false or no data arrives for 3 s.
3. A small launcher `tools/run_dashboard.py --source ...`. Accessible from other devices on the LAN. Document opening port 5000 in the Arch firewall only if it is enabled.
4. Add a fake pipeline (`FakePipeline` that emits synthetic objects) for testing the dashboard without a camera.

### Automated tests (`tests/test_dashboard.py`, Flask-SocketIO test client + `FakePipeline`)
| ID | Test | Expected |
|---|---|---|
| P7-A1 | `GET /health` | 200 with JSON |
| P7-A2 | Connect a socket client, wait 3 s | receives `frame_data` events at ≤ `max_send_fps` and the payload schema is valid |
| P7-A3 | Fake pipeline reports a TOO_CLOSE object | the payload carries it |
| P7-A4 | No client connected | pipeline keeps running, memory does not grow |
| P7-A5 | Two clients | both receive data |
| P7-A6 | Pipeline stops | `stream_ok = false` emitted |

### Human tests
| ID | Test | Expected |
|---|---|---|
| P7-H1 | Open the dashboard in a browser on the laptop with the live camera | video + table update live |
| P7-H2 | Open it from the phone browser (same Wi-Fi) | works |
| P7-H3 | Compare the dashboard numbers with the OpenCV overlay | identical |
| P7-H4 | Latency test: film a running stopwatch (on another screen) with the camera and compare it to the dashboard video | glass-to-glass delay recorded, ideally < 0.5-1 s over RTSP |
| P7-H5 | Close the browser tab, reopen it | reconnects, pipeline unaffected |
| P7-H6 | Stop the phone stream | dashboard shows "stream lost" within ~3 s and recovers when it returns |
| P7-H7 | Run for 10 minutes | no growing lag or memory |

### Acceptance
- [ ] Tests pass; dashboard shows live video, distance, proximity alerts and speed class.

### Review focus
Threading and backpressure design, payload size, and the UI information layout.

**STOP after the Phase Report.**

---

## Phase 8: Full evaluation, benchmarks and write-up material

**Goal:** collect all the numbers the paper and the review need, with frozen settings.

### Instructions
1. **Freeze the config:** commit `config/default.yaml` and tag `eval-config-frozen` BEFORE running final evaluations. Every results folder stores the config file and git commit hash used.
2. `tools/bench_fps.py`: measure end-to-end FPS and per-stage latency (capture wait, detection, tracking, merge, geometry+speed, dashboard encode) for `yolo11n` and `yolo11s` at `imgsz` 640 / 480 / 320, with and without the dashboard, using the webcam and the RTSP source. Report the mean, p95, and thread count. Output a table to `results/fps.csv`.
3. `tools/eval_detection.py`: per-class detection accuracy. Method: sample about 100 frames per class from live test sessions (via `record.py`), label them manually (CVAT or labelImg, YOLO txt export), match predictions to labels at IoU 0.5, and report precision, recall and F1 per class (`person`, `two_wheeler`, `car`). For `two_wheeler` the label is the bike box (not the rider).
4. Re-run distance and speed evaluations for all three classes with the frozen config; produce the final tables and plots (`results/`): distance MAE/MRE per class and range bucket, homography vs pinhole, speed confusion matrices, within-one-class accuracy, per-class detection P/R/F1, FPS table.
5. Write `README.md`: setup steps, how to calibrate, how to run live and the dashboard, how to reproduce the results, known limitations (fixed camera, flat ground, range-dependent error, calibrated area only, lighting, rider-without-bike case classed as person, only classes validated listed honestly).
6. Add a limitations list for the paper (moving camera needs pitch compensation and ego-motion, out of scope).

### Human tests
| ID | Test | Expected |
|---|---|---|
| P8-H1 | A complete live demo: person, then two-wheeler, then car, through the calibrated area with the dashboard open | all three work end to end |
| P8-H2 | Re-run calibration on another day/position and repeat a mini distance test | works; documented procedure is sufficient |
| P8-H3 | Different lighting (daylight vs evening) | detection accuracy differences noted |

### Acceptance
- [ ] All the result tables exist in `results/`, and the README is complete.
- [ ] Every number in the report can be traced to a CSV and a commit.

**STOP after the Phase Report. This is the end of the plan unless the human approves extras (object-to-object proximity, lens undistortion, ONNX speed-ups).**

---

## Appendix A: Run-sheet templates

**Distance run sheet** (`data/ground_truth/distance_<class>.csv`)

```
point_label,X_true_m,Y_true_m,d_true_m,notes
```

**Speed run sheet** (`data/ground_truth/speed_<class>.csv`)

```
run_id,cls,lane_length_m,time_s,true_speed_mps,true_speed_kmh,true_class,method,notes
```

`method` is `frame_count`, `stopwatch`, `speedometer` or `gps`.

## Appendix B: Troubleshooting (Arch Linux)

| Problem | Likely fix |
|---|---|
| `cv2.imshow` crashes with "could not load the Qt platform plugin xcb" | Install `xcb-util-cursor`; try `QT_QPA_PLATFORM=xcb` (XWayland) or `QT_QPA_PLATFORM=wayland` depending on your session |
| RTSP works in `ffplay` but not in OpenCV | Set `OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"` before creating `VideoCapture` (and before importing cv2 is safest); confirm the exact URL |
| RTSP video lags more and more | The reader thread must keep only the newest frame; also drop `CAP_PROP_BUFFERSIZE` to 1 if supported |
| Webcam not found | `v4l2-ctl --list-devices`; try indices 0, 1, 2; check permissions |
| ByteTrack fails with a missing `lap` | `uv pip install lapx` (or `lap`) |
| Ultralytics tries to pip-install something and fails | The venv was created without `--seed`; recreate it with `uv venv --python 3.12 --seed` |
| Very low FPS | Use `imgsz` 480/416, `yolo11n`, set `num_threads` to physical cores, close other heavy apps, then consider ONNX export |
| Homography gives nonsense | Check point order in the CSV matches the click order, X/Y not swapped, camera resolution equals the calibration resolution, points not collinear |
| Distances drift between sessions | The camera moved; recalibrate |
