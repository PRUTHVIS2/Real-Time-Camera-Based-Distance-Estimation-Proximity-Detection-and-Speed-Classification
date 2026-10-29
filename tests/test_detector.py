"""Unit tests for YOLO detector wrapper and class mapping."""

import pytest

from vision.detector import COCO_CLASS_MAP, map_raw_detections
from vision.types import Detection


def test_class_mapping_filters_non_target_classes():
    """P1-A2: Only person/bicycle/car/motorcycle survive; bus/truck/etc. are dropped."""
    # Synthetic inputs: person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7), dog(16)
    boxes = [
        [10.0, 10.0, 50.0, 100.0],   # person
        [60.0, 20.0, 120.0, 80.0],   # bicycle
        [150.0, 50.0, 300.0, 180.0], # car
        [320.0, 40.0, 400.0, 120.0], # motorcycle
        [410.0, 30.0, 600.0, 250.0], # bus (COCO 5) - should be dropped
        [10.0, 200.0, 200.0, 350.0], # truck (COCO 7) - should be dropped
        [220.0, 250.0, 280.0, 310.0],# dog (COCO 16) - should be dropped
    ]
    confs = [0.92, 0.85, 0.88, 0.75, 0.90, 0.82, 0.70]
    class_ids = [0, 1, 2, 3, 5, 7, 16]

    dets = map_raw_detections(boxes, confs, class_ids)

    # Only 4 target classes should survive
    assert len(dets) == 4

    classes = [d.cls for d in dets]
    assert classes == ["person", "bicycle", "car", "motorcycle"]

    # Verify attributes of mapped objects
    assert dets[0].xyxy == (10.0, 10.0, 50.0, 100.0)
    assert pytest.approx(dets[0].conf) == 0.92
    assert dets[0].cls == "person"
    assert dets[0].track_id is None

    assert dets[1].cls == "bicycle"
    assert dets[2].cls == "car"
    assert dets[3].cls == "motorcycle"


def test_empty_detections():
    """Verify empty input produces empty output without error."""
    dets = map_raw_detections([], [], [])
    assert dets == []
