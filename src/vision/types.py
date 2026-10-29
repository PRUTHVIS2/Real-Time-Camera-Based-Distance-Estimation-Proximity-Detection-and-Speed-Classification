"""Shared data types and dataclasses for the vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """Raw YOLO detection output after mapping to target classes."""
    xyxy: tuple[float, float, float, float]
    conf: float
    cls: str  # 'person' | 'bicycle' | 'motorcycle' | 'car'
    track_id: int | None = None


@dataclass
class TrackedObject:
    """Tracked object representation after rider-two-wheeler merging."""
    track_id: int
    cls: str  # 'person' | 'two_wheeler' | 'car'
    box_xyxy: tuple[float, float, float, float]  # box used for the foot point
    conf: float
    rider_count: int  # 0 for non-two-wheelers or empty bikes
    truncated: bool
    t: float  # time.monotonic() at frame capture


@dataclass
class Measurement:
    """Complete measurement result for a tracked object in one frame."""
    obj: TrackedObject
    foot_px: tuple[float, float]
    world_xy: tuple[float, float] | None  # metres in the calibration frame
    distance_m: float | None
    in_range: bool
    proximity: str  # 'SAFE' | 'TOO_CLOSE' | 'OUT_OF_RANGE' | 'UNKNOWN'
    speed_mps: float | None
    speed_kmh: float | None
    speed_class: str | None  # 'STATIONARY' | 'SLOW' | 'MODERATE' | 'FAST' | 'VERY_FAST' | 'ESTIMATING'

