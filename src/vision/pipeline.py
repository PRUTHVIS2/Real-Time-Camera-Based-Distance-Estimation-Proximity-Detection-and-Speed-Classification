"""End-to-end vision processing pipeline wiring capture, detection, tracking, merging, and geometry."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import IO, Sequence, Tuple

import numpy as np

from vision.calibration import CalibrationData
from vision.config import AppConfig
from vision.detector import YoloDetector
from vision.geometry import (
    DistanceSmoother,
    assumed_height_for_class,
    box_height_px,
    distance_from_camera,
    foot_point,
    is_foot_truncated,
    is_in_range,
    pinhole_distance,
)
from vision.merge import RiderMerger
from vision.types import Measurement, TrackedObject

logger = logging.getLogger(__name__)


def measure_objects(
    tracked_objects: Sequence[TrackedObject],
    calib: CalibrationData,
    cfg: AppConfig,
    smoother: DistanceSmoother,
    frame_size: Tuple[int, int] | None = None,
    now_t: float | None = None,
) -> list[Measurement]:
    """Compute ground coordinates, distance, range status, and pinhole baseline for tracked objects.

    Pure evaluation helper (no camera/file I/O).

    Args:
        tracked_objects: Merged TrackedObject instances for one frame.
        calib: Loaded CalibrationData with H, camera_xy, world_pts, and optional focal_px.
        cfg: Application configuration.
        smoother: DistanceSmoother instance for EMA filtering.
        frame_size: Frame (width, height) in pixels used to detect bottom-edge foot truncation.
        now_t: Current timestamp for pruning stale EMA tracks.

    Returns:
        List of Measurement objects (proximity and speed fields left as empty/UNKNOWN in Phase 4).
    """
    H = calib.homography
    camera_xy = (float(calib.camera_xy[0]), float(calib.camera_xy[1]))
    cal_world_pts = calib.world_pts if len(calib.world_pts) >= 3 else None

    if now_t is not None:
        smoother.prune(now_t)

    measurements: list[Measurement] = []

    for obj in tracked_objects:
        fp = foot_point(obj)
        h_px = box_height_px(obj)

        # Bottom-edge truncation hides the foot point -> distance cannot be estimated
        if frame_size is not None:
            foot_cut = is_foot_truncated(
                obj.box_xyxy,
                frame_size=frame_size,
                margin_px=cfg.geometry.truncation_margin_px,
            )
        else:
            foot_cut = obj.truncated

        pinhole_d: float | None = None
        if calib.focal_px is not None and calib.focal_px > 0.0:
            real_h = assumed_height_for_class(obj.cls, cfg.pinhole_baseline)
            if real_h is not None:
                pinhole_d = pinhole_distance(h_px, real_h, calib.focal_px)

        if foot_cut:
            measurements.append(
                Measurement(
                    obj=obj,
                    foot_px=fp,
                    world_xy=None,
                    distance_m=None,
                    in_range=False,
                    proximity="UNKNOWN",
                    speed_mps=None,
                    speed_kmh=None,
                    speed_class=None,
                    pinhole_distance_m=pinhole_d,
                )
            )
            continue

        try:
            wx, wy, raw_d = distance_from_camera(H, fp, camera_xy)
        except ValueError as exc:
            logger.debug(f"Homography projection failed for track {obj.track_id}: {exc}")
            measurements.append(
                Measurement(
                    obj=obj,
                    foot_px=fp,
                    world_xy=None,
                    distance_m=None,
                    in_range=False,
                    proximity="UNKNOWN",
                    speed_mps=None,
                    speed_kmh=None,
                    speed_class=None,
                    pinhole_distance_m=pinhole_d,
                )
            )
            continue

        in_rng = is_in_range(
            world_xy=(wx, wy),
            camera_xy=camera_xy,
            max_range_m=cfg.geometry.max_range_m,
            calibrated_world_pts=cal_world_pts,
            hull_margin_m=cfg.geometry.hull_margin_m,
            distance_m=raw_d,
        )

        smooth_d = smoother.update(obj.track_id, raw_d, obj.t)

        measurements.append(
            Measurement(
                obj=obj,
                foot_px=fp,
                world_xy=(wx, wy),
                distance_m=smooth_d,
                in_range=in_rng,
                proximity="UNKNOWN",
                speed_mps=None,
                speed_kmh=None,
                speed_class=None,
                pinhole_distance_m=pinhole_d,
            )
        )

    return measurements


class VisionPipeline:
    """Stateful pipeline processing BGR frames into per-object Measurements."""

    def __init__(
        self,
        cfg: AppConfig,
        calibration: CalibrationData | str | Path | None = None,
        detector: YoloDetector | None = None,
    ) -> None:
        self.cfg = cfg
        self.detector = detector if detector is not None else YoloDetector(cfg.detector)
        self.merger = RiderMerger(
            min_containment=cfg.merge.min_containment,
            rider_memory_s=cfg.merge.rider_memory_s,
            truncation_margin_px=cfg.geometry.truncation_margin_px,
            default_frame_size=(cfg.camera.width, cfg.camera.height),
        )
        self.smoother = DistanceSmoother(
            alpha=cfg.geometry.distance_ema_alpha,
            track_forget_s=cfg.speed.track_forget_s,
        )

        self._calibration: CalibrationData | None = None
        self._calibration_path: Path | None = None
        self._resolution_verified = False

        if isinstance(calibration, CalibrationData):
            self._calibration = calibration
        elif calibration is not None:
            self._calibration_path = Path(calibration)
            self._calibration = CalibrationData.load_json(self._calibration_path)
        else:
            default_cal_path = Path(f"data/calibration/{cfg.camera.camera_id}.json")
            if default_cal_path.is_file():
                self._calibration_path = default_cal_path
                self._calibration = CalibrationData.load_json(default_cal_path)
            else:
                raise FileNotFoundError(
                    f"Calibration file not found at {default_cal_path}. "
                    "Run tools/calibrate.py first or pass an explicit calibration path."
                )

    @property
    def calibration(self) -> CalibrationData:
        assert self._calibration is not None
        return self._calibration

    def _verify_resolution(self, frame_size: Tuple[int, int]) -> None:
        if self._resolution_verified:
            return
        calib = self.calibration
        expected = (calib.image_width, calib.image_height)
        if frame_size != expected:
            raise ValueError(
                f"Calibration frame size mismatch for {calib.camera_id}: "
                f"calibration was computed for {expected[0]}x{expected[1]}, "
                f"but live frame is {frame_size[0]}x{frame_size[1]}"
            )
        self._resolution_verified = True

    def process_frame(self, frame: np.ndarray, t_capture: float) -> list[Measurement]:
        """Run detection, ByteTrack tracking, rider merging, and distance estimation on one frame."""
        h, w = frame.shape[:2]
        frame_size = (w, h)
        self._verify_resolution(frame_size)

        detections = self.detector.track(frame)
        tracked_objects = self.merger.merge(detections, t_capture=t_capture, frame_size=frame_size)
        return measure_objects(
            tracked_objects=tracked_objects,
            calib=self.calibration,
            cfg=self.cfg,
            smoother=self.smoother,
            frame_size=frame_size,
            now_t=t_capture,
        )


class MeasurementCsvLogger:
    """Writes per-frame object Measurements to a CSV log file."""

    HEADER = [
        "t",
        "id",
        "cls",
        "X",
        "Y",
        "distance",
        "in_range",
        "truncated",
        "box_h_px",
        "pinhole_distance",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def log(self, measurements: Sequence[Measurement]) -> None:
        """Append all Measurement rows for one frame."""
        for m in measurements:
            obj = m.obj
            wx = f"{m.world_xy[0]:.4f}" if m.world_xy is not None else ""
            wy = f"{m.world_xy[1]:.4f}" if m.world_xy is not None else ""
            dist_str = f"{m.distance_m:.4f}" if m.distance_m is not None else ""
            h_px = box_height_px(obj)
            pin_str = f"{m.pinhole_distance_m:.4f}" if m.pinhole_distance_m is not None else ""
            self._writer.writerow([
                f"{obj.t:.6f}",
                obj.track_id,
                obj.cls,
                wx,
                wy,
                dist_str,
                m.in_range,
                obj.truncated,
                f"{h_px:.2f}",
                pin_str,
            ])

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> MeasurementCsvLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
