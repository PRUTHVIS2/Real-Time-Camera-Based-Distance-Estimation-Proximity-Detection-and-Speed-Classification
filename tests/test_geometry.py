"""Unit tests for planar geometry, distance estimation, pinhole baseline, and evaluation."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vision.calibration import CalibrationData, world_to_pixel
from vision.config import AppConfig
from vision.geometry import (
    DistanceSmoother,
    distance_from_camera,
    estimate_focal_length_px,
    foot_point,
    is_foot_truncated,
    is_in_range,
    pinhole_distance,
)
from vision.merge import RiderMerger
from vision.pipeline import MeasurementCsvLogger, measure_objects
from vision.types import Detection, Measurement, TrackedObject


def make_synthetic_calibration() -> CalibrationData:
    """Create an exact synthetic camera calibration (1280x720, pitched down 35 deg)."""
    K = np.array([
        [900.0, 0.0, 640.0],
        [0.0, 900.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    theta = np.radians(35.0)
    # Camera +Z_cam looks forward and down (-Z_world); +Y_cam points down in image
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -np.sin(theta), -np.cos(theta)],
        [0.0, np.cos(theta), -np.sin(theta)],
    ], dtype=np.float64)
    camera_xy = [1.0, 0.0]
    camera_h = 1.8
    t = -R @ np.array([camera_xy[0], camera_xy[1], camera_h], dtype=np.float64)
    H_world_to_pix = K @ np.column_stack((R[:, 0], R[:, 1], t))
    H_pix_to_world = np.linalg.inv(H_world_to_pix)
    H_pix_to_world /= H_pix_to_world[2, 2]

    world_pts = [[x, y] for y in [1.0, 2.0, 3.0, 4.0, 5.0] for x in [0.0, 1.0, 2.0]]
    pixel_pts = [list(world_to_pixel(H_pix_to_world, pt)) for pt in world_pts]

    return CalibrationData(
        camera_id="synthetic_cam",
        image_width=1280,
        image_height=720,
        H=H_pix_to_world.tolist(),
        pixel_pts=pixel_pts,
        world_pts=world_pts,
        camera_xy=camera_xy,
        camera_height_m=camera_h,
        focal_px=900.0,
    )


def test_p4_a1_synthetic_camera_exact_distance():
    """P4-A1: Pixels generated from known ground points recover true distance within 1e-6 m."""
    calib = make_synthetic_calibration()
    H = calib.homography
    xc, yc = calib.camera_xy

    test_ground_points = [
        (1.0, 2.0),
        (0.5, 3.5),
        (1.8, 4.2),
        (0.2, 1.8),
    ]

    for X_true, Y_true in test_ground_points:
        u_foot, v_foot = world_to_pixel(H, (X_true, Y_true))
        # Build a bounding box whose bottom-centre is (u_foot, v_foot)
        box = (u_foot - 25.0, v_foot - 120.0, u_foot + 25.0, v_foot)
        obj = TrackedObject(
            track_id=1,
            cls="person",
            box_xyxy=box,
            conf=0.9,
            rider_count=0,
            truncated=False,
            t=1.0,
        )
        fp = foot_point(obj)
        X_est, Y_est, d_est = distance_from_camera(H, fp, (xc, yc))
        d_true = math.hypot(X_true - xc, Y_true - yc)

        assert X_est == pytest.approx(X_true, abs=1e-6)
        assert Y_est == pytest.approx(Y_true, abs=1e-6)
        assert d_est == pytest.approx(d_true, abs=1e-6)


def test_p4_a2_bottom_edge_truncated_returns_none_distance():
    """P4-A2: Object box touching the bottom edge -> distance is None."""
    calib = make_synthetic_calibration()
    cfg = AppConfig()
    smoother = DistanceSmoother(alpha=cfg.geometry.distance_ema_alpha)

    # Box bottom y2 = 718 touches bottom edge of 1280x720 (margin=20 -> >= 700)
    obj_bottom = TrackedObject(
        track_id=7,
        cls="person",
        box_xyxy=(500.0, 400.0, 580.0, 718.0),
        conf=0.9,
        rider_count=0,
        truncated=True,
        t=1.0,
    )
    assert is_foot_truncated(obj_bottom.box_xyxy, (1280, 720), margin_px=cfg.geometry.truncation_margin_px)

    # Also verify YOLO 1080p edge offset (e.g. y2 = 1070 in 1920x1080 -> 10 px from bottom)
    assert is_foot_truncated((500.0, 400.0, 700.0, 1070.0), (1920, 1080), margin_px=cfg.geometry.truncation_margin_px)

    measurements = measure_objects(
        [obj_bottom],
        calib=calib,
        cfg=cfg,
        smoother=smoother,
        frame_size=(1280, 720),
    )
    assert len(measurements) == 1
    assert measurements[0].obj.truncated is True
    assert measurements[0].distance_m is None
    assert measurements[0].world_xy is None
    assert measurements[0].in_range is False

    # Contrast: box touching ONLY top edge (y1=1, feet well inside frame at v=600) still computes distance
    u_foot, v_foot = world_to_pixel(calib.homography, (1.0, 2.0))
    obj_top = TrackedObject(
        track_id=8,
        cls="person",
        box_xyxy=(u_foot - 40.0, 1.0, u_foot + 40.0, v_foot),
        conf=0.9,
        rider_count=0,
        truncated=True,
        t=1.0,
    )
    meas_top = measure_objects(
        [obj_top],
        calib=calib,
        cfg=cfg,
        smoother=smoother,
        frame_size=(1280, 720),
    )
    assert meas_top[0].distance_m is not None
    assert meas_top[0].distance_m == pytest.approx(2.0, abs=1e-4)


def test_p4_a3_foot_point_of_merged_two_wheeler_uses_bike_box():
    """P4-A3: Merged two_wheeler foot point must come from the bike box, not the rider box."""
    bike_box = (300.0, 400.0, 500.0, 620.0)   # bottom-centre: (400.0, 620.0)
    rider_box = (340.0, 250.0, 460.0, 560.0)  # bottom-centre: (400.0, 560.0)

    bike_det = Detection(xyxy=bike_box, conf=0.92, cls="motorcycle", track_id=42)
    rider_det = Detection(xyxy=rider_box, conf=0.88, cls="person", track_id=9)

    merger = RiderMerger(min_containment=0.5)
    objs = merger.merge([bike_det, rider_det], t_capture=1.0, frame_size=(1280, 720))

    assert len(objs) == 1
    merged = objs[0]
    assert merged.cls == "two_wheeler"
    assert merged.rider_count == 1
    assert foot_point(merged) == pytest.approx((400.0, 620.0), abs=1e-9)
    assert foot_point(merged) != pytest.approx((400.0, 560.0), abs=1e-9)


def test_p4_a4_point_outside_calibrated_area_is_out_of_range():
    """P4-A4: Point far outside calibrated hull or beyond max_range_m has in_range = False."""
    calib = make_synthetic_calibration()  # Calibrated hull: X in [0, 2], Y in [1, 5]

    # Point inside calibrated grid -> True
    assert is_in_range(
        (1.0, 3.0),
        camera_xy=calib.camera_xy,
        max_range_m=15.0,
        calibrated_world_pts=calib.world_pts,
        hull_margin_m=1.5,
    ) is True

    # Point slightly outside (0.5 m past border, within 1.5 m tolerance) -> True
    assert is_in_range(
        (2.5, 3.0),
        camera_xy=calib.camera_xy,
        max_range_m=15.0,
        calibrated_world_pts=calib.world_pts,
        hull_margin_m=1.5,
    ) is True

    # Point far outside convex hull (X=6.0 -> 4 m outside hull) -> False
    assert is_in_range(
        (6.0, 3.0),
        camera_xy=calib.camera_xy,
        max_range_m=15.0,
        calibrated_world_pts=calib.world_pts,
        hull_margin_m=1.5,
    ) is False

    # Point inside/along Y but exceeding max_range_m -> False
    assert is_in_range(
        (1.0, 20.0),
        camera_xy=calib.camera_xy,
        max_range_m=15.0,
        calibrated_world_pts=None,
    ) is False


def test_p4_a5_ema_smoother_converges_without_overshoot():
    """P4-A5: Step input to EMA smoother converges monotonically with zero overshoot."""
    smoother = DistanceSmoother(alpha=0.4, track_forget_s=2.0)

    # Initial reading at 10.0 m
    d = smoother.update(1, 10.0, t=0.0)
    assert d == pytest.approx(10.0)

    # Step down to 4.0 m over 20 frames (10 Hz)
    history = [d]
    for i in range(1, 25):
        d = smoother.update(1, 4.0, t=i * 0.1)
        history.append(d)

    # Monotonically decreasing, never dropping below 4.0 m (no overshoot)
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1]
        assert history[i] >= 4.0 - 1e-12

    # Converged close to 4.0 m
    assert history[-1] == pytest.approx(4.0, abs=1e-4)

    # Track expiration resets smoother state
    d_reset = smoother.update(1, 8.0, t=10.0)  # gap > track_forget_s (2.0 s)
    assert d_reset == pytest.approx(8.0)


def test_p4_a6_pinhole_distance_scales_inversely_with_height():
    """P4-A6: Pinhole distance scales inversely with pixel height h_px."""
    focal = estimate_focal_length_px(h_px=340.0, real_height_m=1.7, true_distance_m=5.0)
    assert focal == pytest.approx(1000.0)

    d_near = pinhole_distance(h_px=400.0, real_height_m=1.7, focal_px=focal)
    d_far = pinhole_distance(h_px=200.0, real_height_m=1.7, focal_px=focal)

    assert d_near is not None and d_far is not None
    assert d_far == pytest.approx(2.0 * d_near, rel=1e-9)
    assert d_near == pytest.approx(4.25, rel=1e-9)
    assert pinhole_distance(h_px=0.0, real_height_m=1.7, focal_px=focal) is None


def test_eval_distance_pipeline_and_csv_outputs(tmp_path: Path):
    """Verify MeasurementCsvLogger and tools/eval_distance.py end-to-end."""
    sys.path.insert(0, str(Path("tools").resolve()))
    import eval_distance

    # Build synthetic GT (3 points across buckets 0-4m and 4-8m)
    gt_path = tmp_path / "gt.csv"
    pd.DataFrame([
        {"point_label": "P1", "X_true_m": 1.0, "Y_true_m": 2.0, "d_true_m": 2.0},
        {"point_label": "P2", "X_true_m": 1.0, "Y_true_m": 5.0, "d_true_m": 5.0},
        {"point_label": "P3", "X_true_m": 1.0, "Y_true_m": 7.0, "d_true_m": 7.0},
    ]).to_csv(gt_path, index=False)

    # Create logs via MeasurementCsvLogger
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    for label, d_homo, d_pin in [("P1", 2.05, 2.30), ("P2", 4.90, 5.60), ("P3", 7.14, 8.20)]:
        log_path = logs_dir / f"{label}_t1.csv"
        with MeasurementCsvLogger(log_path) as logger_csv:
            for k in range(10):
                obj = TrackedObject(
                    track_id=1,
                    cls="person",
                    box_xyxy=(500.0, 200.0, 600.0, 500.0),
                    conf=0.9,
                    rider_count=0,
                    truncated=False,
                    t=float(k) * 0.1,
                )
                m = Measurement(
                    obj=obj,
                    foot_px=(550.0, 500.0),
                    world_xy=(1.0, d_homo),
                    distance_m=d_homo,
                    in_range=True,
                    proximity="UNKNOWN",
                    speed_mps=None,
                    speed_kmh=None,
                    speed_class=None,
                    pinhole_distance_m=d_pin,
                )
                logger_csv.log([m])

    out_dir = tmp_path / "results"
    rc = eval_distance.main([
        "--gt", str(gt_path),
        "--logs-dir", str(logs_dir),
        "--cls", "person",
        "--output-dir", str(out_dir),
        "--prefix", "test_person",
    ])
    assert rc == 0
    assert (out_dir / "test_person_per_position.csv").is_file()
    assert (out_dir / "test_person_summary.csv").is_file()
    assert (out_dir / "test_person_buckets.csv").is_file()
    assert (out_dir / "test_person_plot.png").is_file()

    summary = pd.read_csv(out_dir / "test_person_summary.csv")
    homo_row = summary[summary["method"] == "homography"].iloc[0]
    pin_row = summary[summary["method"] == "pinhole"].iloc[0]
    assert homo_row["mae_m"] < pin_row["mae_m"]
    assert homo_row["accuracy"] > 0.95


def test_vision_pipeline_resolution_guard_and_csv_schema(tmp_path: Path):
    """VisionPipeline enforces calibration resolution and writes expected CSV columns."""
    from vision.pipeline import VisionPipeline

    calib = make_synthetic_calibration()  # expects 1280x720
    cfg = AppConfig()

    class _FakeDetector:
        def track(self, frame):
            u, v = world_to_pixel(calib.homography, (1.0, 3.0))
            return [Detection(xyxy=(u - 30.0, v - 180.0, u + 30.0, v), conf=0.9, cls="person", track_id=5)]

    pipe = VisionPipeline(cfg=cfg, calibration=calib, detector=_FakeDetector())

    # Wrong resolution (640x480) must raise ValueError
    wrong_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Calibration frame size mismatch"):
        pipe.process_frame(wrong_frame, t_capture=0.1)

    # Matching resolution (1280x720) succeeds
    ok_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    meas = pipe.process_frame(ok_frame, t_capture=0.2)
    assert len(meas) == 1
    assert meas[0].distance_m == pytest.approx(3.0, abs=1e-4)
    assert meas[0].in_range is True
    assert meas[0].pinhole_distance_m == pytest.approx(900.0 * 1.7 / 180.0, rel=1e-6)

    csv_path = tmp_path / "log.csv"
    with MeasurementCsvLogger(csv_path) as log:
        log.log(meas)
    df = pd.read_csv(csv_path)
    for col in ("t", "id", "cls", "X", "Y", "distance", "in_range", "truncated"):
        assert col in df.columns
