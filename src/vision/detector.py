"""YOLO detector wrapper and class mapping.

Maps COCO classes 0, 1, 2, 3 to target classes:
  0 -> 'person'
  1 -> 'bicycle'
  2 -> 'car'
  3 -> 'motorcycle'
All other classes are dropped.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch
from ultralytics import YOLO

from vision.config import DetectorConfig
from vision.types import Detection

logger = logging.getLogger(__name__)

# COCO target mapping
COCO_CLASS_MAP: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
}

TARGET_CLASSES = set(COCO_CLASS_MAP.values())


def map_raw_detections(
    boxes_xyxy: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    track_ids: Sequence[int | None] | None = None,
) -> list[Detection]:
    """Pure function to map raw bounding boxes, scores, and class IDs to target Detections.

    Filters out any class ID not in {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'}.

    Args:
        boxes_xyxy: Iterable of [x1, y1, x2, y2] coordinates.
        confidences: Detection confidence scores.
        class_ids: Integer COCO class IDs.
        track_ids: Optional tracker IDs.

    Returns:
        List of Detection objects for matching target classes.
    """
    results: list[Detection] = []
    num_boxes = len(boxes_xyxy)
    if track_ids is None:
        track_ids = [None] * num_boxes

    for box, conf, cls_id, trk_id in zip(boxes_xyxy, confidences, class_ids, track_ids):
        if cls_id in COCO_CLASS_MAP:
            target_cls = COCO_CLASS_MAP[cls_id]
            xyxy_tuple = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            results.append(
                Detection(
                    xyxy=xyxy_tuple,
                    conf=float(conf),
                    cls=target_cls,
                    track_id=int(trk_id) if trk_id is not None else None,
                )
            )
    return results


class YoloDetector:
    """Wrapper around Ultralytics YOLO model with targeted class filtering."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        if self.config.num_threads > 0:
            torch.set_num_threads(self.config.num_threads)
            logger.info(f"PyTorch CPU threads set to {self.config.num_threads}")

        logger.info(f"Loading YOLO detector model from: {self.config.model}")
        self.model = YOLO(self.config.model)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on an input BGR frame.

        Args:
            frame: Input image array in BGR format.

        Returns:
            List of filtered Detection objects.
        """
        # Run prediction on target COCO classes (0: person, 1: bicycle, 2: car, 3: motorcycle)
        results = self.model.predict(
            source=frame,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            classes=[0, 1, 2, 3],
            verbose=False,
        )

        if not results or len(results) == 0:
            return []

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().tolist()
        confs = boxes.conf.cpu().numpy().tolist()
        clss = boxes.cls.int().cpu().numpy().tolist()

        return map_raw_detections(
            boxes_xyxy=xyxy,
            confidences=confs,
            class_ids=clss,
            track_ids=None,
        )
