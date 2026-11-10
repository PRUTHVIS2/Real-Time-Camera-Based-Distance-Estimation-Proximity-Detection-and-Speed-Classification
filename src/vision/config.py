"""Configuration loading, schema validation, and typed dataclasses.

Handles loading default.yaml, deep-merging local.yaml if present,
validating against unknown keys, and producing a typed AppConfig object.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints, Union

import yaml


@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    camera_id: str = "laptop_webcam"
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "MJPG"
    rtsp_transport: str = "tcp"


@dataclass
class DetectorConfig:
    """YOLO detector configuration parameters."""
    model: str = "models/yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    device: str = "cpu"
    num_threads: int = 8


@dataclass
class MergeConfig:
    """Rider and bike merge configuration parameters."""
    min_containment: float = 0.5
    rider_memory_s: float = 1.0


@dataclass
class GeometryConfig:
    """Planar homography and distance geometry configuration."""
    max_range_m: float = 15.0
    hull_margin_m: float = 1.5
    distance_ema_alpha: float = 0.4
    truncation_margin_px: int = 3


@dataclass
class PinholeBaselineConfig:
    """Assumed object heights for baseline pinhole comparison."""
    person_height_m: float = 1.7
    two_wheeler_height_m: float = 1.1
    car_height_m: float = 1.5


@dataclass
class ProximityConfig:
    """Proximity alerting thresholds and hysteresis."""
    threshold_m: dict[str, float] = field(
        default_factory=lambda: {"person": 3.0, "two_wheeler": 5.0, "car": 8.0}
    )
    hysteresis_m: float = 0.3


@dataclass
class SpeedConfig:
    """Speed estimation and classification parameters."""
    window_s: float = 0.7
    min_points: int = 5
    class_vote_len: int = 5
    track_forget_s: float = 2.0
    dead_zone_mps: dict[str, float] = field(
        default_factory=lambda: {"person": 0.3, "two_wheeler": 0.5, "car": 0.5}
    )
    max_plausible_mps: dict[str, float] = field(
        default_factory=lambda: {"person": 12.0, "two_wheeler": 40.0, "car": 45.0}
    )
    person_upper_mps: list[float] = field(
        default_factory=lambda: [0.3, 1.0, 1.8, 3.0]
    )
    vehicle_upper_kmh: list[float] = field(
        default_factory=lambda: [2.0, 30.0, 60.0, 90.0]
    )


@dataclass
class DashboardConfig:
    """Web dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 5000
    max_send_fps: int = 15
    jpeg_quality: int = 70


@dataclass
class AppConfig:
    """Top-level application configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    pinhole_baseline: PinholeBaselineConfig = field(default_factory=PinholeBaselineConfig)
    proximity: ProximityConfig = field(default_factory=ProximityConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into base dictionary.

    Returns a new merged dictionary without mutating arguments.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_keys(data: dict[str, Any], dataclass_type: type, path: str = "") -> None:
    """Validate that data only contains fields defined in dataclass_type.

    Raises:
        ValueError: If an unknown key is encountered.
    """
    valid_hints = get_type_hints(dataclass_type)
    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key
        if key not in valid_hints:
            raise ValueError(
                f"Unknown configuration key: '{current_path}' (not allowed in {dataclass_type.__name__})"
            )
        field_type = valid_hints[key]
        if is_dataclass(field_type) and isinstance(value, dict):
            _validate_keys(value, field_type, current_path)


def _dict_to_dataclass(data: dict[str, Any], dataclass_type: type) -> Any:
    """Convert a dictionary to a dataclass instance recursively."""
    valid_hints = get_type_hints(dataclass_type)
    kwargs = {}
    for key, value in data.items():
        if key in valid_hints:
            field_type = valid_hints[key]
            if is_dataclass(field_type) and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[key] = value
    return dataclass_type(**kwargs)


def load_config(
    default_path: str | Path = "config/default.yaml",
    local_path: str | Path = "config/local.yaml",
) -> AppConfig:
    """Load default config and optional local overrides, returning a validated AppConfig.

    Args:
        default_path: Path to default YAML config file.
        local_path: Path to local override YAML file (optional).

    Returns:
        AppConfig typed configuration object.

    Raises:
        FileNotFoundError: If default_path does not exist.
        ValueError: If unknown keys exist in default or local config files.
    """
    default_p = Path(default_path)
    if not default_p.is_file():
        raise FileNotFoundError(f"Default config not found at: {default_p}")

    with open(default_p, "r", encoding="utf-8") as f:
        raw_default = yaml.safe_load(f) or {}

    merged = raw_default
    local_p = Path(local_path)
    if local_p.is_file():
        with open(local_p, "r", encoding="utf-8") as f:
            raw_local = yaml.safe_load(f) or {}
        merged = deep_merge(raw_default, raw_local)

    # Validate against AppConfig schema
    _validate_keys(merged, AppConfig)

    return _dict_to_dataclass(merged, AppConfig)

