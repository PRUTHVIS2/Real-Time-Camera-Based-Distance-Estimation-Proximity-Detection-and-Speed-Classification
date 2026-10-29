"""Unit tests for configuration loading and validation."""

import pytest
import yaml

from vision.config import AppConfig, deep_merge, load_config


def test_default_config_loads(tmp_path):
    """Test that the default config loads cleanly into an AppConfig instance."""
    cfg = load_config(default_path="config/default.yaml", local_path=tmp_path / "nonexistent.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.camera.camera_id == "phone_main"
    assert cfg.camera.source == 0
    assert cfg.camera.width == 1280
    assert cfg.camera.height == 720
    assert cfg.camera.rtsp_transport == "tcp"
    assert cfg.detector.model == "models/yolo11n.pt"
    assert cfg.detector.imgsz == 640
    assert cfg.detector.conf == 0.25
    assert cfg.detector.device == "cpu"
    assert cfg.detector.num_threads == 8
    assert cfg.merge.min_containment == 0.5
    assert cfg.geometry.max_range_m == 15.0
    assert cfg.proximity.threshold_m["person"] == 3.0
    assert cfg.speed.window_s == 0.7
    assert cfg.speed.person_upper_mps == [0.3, 1.0, 1.8, 3.0]
    assert cfg.dashboard.port == 5000


def test_local_yaml_overrides_value(tmp_path):
    """Test that values in local.yaml override default.yaml without losing other keys."""
    local_yaml = tmp_path / "local.yaml"
    local_data = {
        "camera": {
            "source": "rtsp://192.168.1.100:8554/live",
            "width": 1920,
        },
        "detector": {
            "conf": 0.15,
        },
    }
    with open(local_yaml, "w", encoding="utf-8") as f:
        yaml.dump(local_data, f)

    cfg = load_config(default_path="config/default.yaml", local_path=local_yaml)
    # Overridden values
    assert cfg.camera.source == "rtsp://192.168.1.100:8554/live"
    assert cfg.camera.width == 1920
    assert cfg.detector.conf == 0.15

    # Non-overridden sibling values preserved
    assert cfg.camera.height == 720
    assert cfg.camera.camera_id == "phone_main"
    assert cfg.detector.model == "models/yolo11n.pt"
    assert cfg.detector.iou == 0.5


def test_unknown_top_level_key_raises_error(tmp_path):
    """Test that an unknown top-level key raises ValueError."""
    bad_config = tmp_path / "bad_default.yaml"
    with open("config/default.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["bogus_section"] = {"foo": "bar"}
    with open(bad_config, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    with pytest.raises(ValueError, match="Unknown configuration key: 'bogus_section'"):
        load_config(default_path=bad_config, local_path=tmp_path / "none.yaml")


def test_unknown_nested_key_raises_error(tmp_path):
    """Test that an unknown nested key raises ValueError."""
    local_yaml = tmp_path / "local.yaml"
    local_data = {
        "detector": {
            "unknown_tuning_param": 123,
        }
    }
    with open(local_yaml, "w", encoding="utf-8") as f:
        yaml.dump(local_data, f)

    with pytest.raises(ValueError, match="Unknown configuration key: 'detector.unknown_tuning_param'"):
        load_config(default_path="config/default.yaml", local_path=local_yaml)


def test_deep_merge_utility():
    """Test deep_merge function behavior."""
    base = {"a": 1, "nested": {"x": 10, "y": 20}}
    override = {"b": 2, "nested": {"y": 30, "z": 40}}
    merged = deep_merge(base, override)
    assert merged == {
        "a": 1,
        "b": 2,
        "nested": {"x": 10, "y": 30, "z": 40},
    }
    # Check original dicts were not mutated
    assert base["nested"]["y"] == 20

