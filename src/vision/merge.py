"""Rider and two-wheeler bounding box merging and rider memory suppression.

Implements pure geometric containment computation, rider-to-bike assignment,
rider memory suppression across flicker frames, and frame-edge truncation detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, Tuple

from vision.config import MergeConfig
from vision.types import Detection, TrackedObject

logger = logging.getLogger(__name__)


def box_area(box: Tuple[float, float, float, float]) -> float:
    """Compute area of an [x1, y1, x2, y2] bounding box."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h


def box_intersection_area(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute intersection area between box_a and box_b."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    w = max(0.0, ix2 - ix1)
    h = max(0.0, iy2 - iy1)
    return w * h


def compute_containment(
    person_box: Tuple[float, float, float, float],
    bike_box: Tuple[float, float, float, float],
) -> float:
    """Compute share of person_box area that lies inside bike_box.

    Returns:
        float in [0.0, 1.0] representing area(person ∩ bike) / area(person).
    """
    area_p = box_area(person_box)
    if area_p <= 0.0:
        return 0.0
    return box_intersection_area(person_box, bike_box) / area_p


def is_rider_of_bike(
    person_box: Tuple[float, float, float, float],
    bike_box: Tuple[float, float, float, float],
    min_containment: float = 0.5,
    bottom_margin_ratio: float = 0.15,
) -> Tuple[bool, float]:
    """Determine whether a person box qualifies as a rider of a bike box.

    Conditions:
      1. containment >= min_containment
      2. Person's horizontal centre lies within the bike box's x-range
      3. Person's bottom edge is not far below the bike's bottom edge

    Returns:
        (is_rider, containment_value)
    """
    containment = compute_containment(person_box, bike_box)
    if containment < min_containment:
        return False, containment

    # Check horizontal centre
    person_cx = (person_box[0] + person_box[2]) / 2.0
    if not (bike_box[0] <= person_cx <= bike_box[2]):
        return False, containment

    # Check bottom edge not far below bike bottom
    bike_h = max(1.0, bike_box[3] - bike_box[1])
    max_allowed_y2 = bike_box[3] + bottom_margin_ratio * bike_h
    if person_box[3] > max_allowed_y2:
        return False, containment

    return True, containment


def is_box_truncated(
    box: Tuple[float, float, float, float],
    frame_size: Tuple[int, int] = (1280, 720),
    margin_px: int = 3,
) -> bool:
    """Check if a bounding box touches any frame edge within margin_px.

    Args:
        box: (x1, y1, x2, y2)
        frame_size: (width, height) in pixels
        margin_px: Edge proximity threshold in pixels

    Returns:
        True if box touches any boundary.
    """
    w, h = frame_size
    x1, y1, x2, y2 = box
    if x1 <= margin_px or y1 <= margin_px or x2 >= (w - margin_px) or y2 >= (h - margin_px):
        return True
    return False


class RiderMerger:
    """Stateful rider-to-bike merger with rider-memory suppression across flicker frames."""

    def __init__(
        self,
        min_containment: float = 0.5,
        rider_memory_frames: int = 15,
        truncation_margin_px: int = 3,
        default_frame_size: Tuple[int, int] = (1280, 720),
    ) -> None:
        self.min_containment = min_containment
        self.rider_memory_frames = rider_memory_frames
        self.truncation_margin_px = truncation_margin_px
        self.default_frame_size = default_frame_size

        # Maps person_track_id -> consecutive frames seen without a bike
        self._rider_memory: dict[int, int] = {}
        self._fallback_id_counter = 10000

    def reset(self) -> None:
        """Clear tracking memory state."""
        self._rider_memory.clear()
        self._fallback_id_counter = 10000

    def _get_id(self, track_id: int | None) -> int:
        if track_id is not None:
            return track_id
        self._fallback_id_counter += 1
        return self._fallback_id_counter

    def merge(
        self,
        detections: Sequence[Detection],
        t_capture: float,
        frame_size: Tuple[int, int] | None = None,
    ) -> list[TrackedObject]:
        """Merge riders with two-wheelers and produce final TrackedObjects.

        Args:
            detections: List of Detection objects from tracker/detector.
            t_capture: Frame capture timestamp (seconds).
            frame_size: Optional (width, height) to assess truncation.

        Returns:
            List of merged TrackedObject instances.
        """
        f_size = frame_size or self.default_frame_size

        if not detections:
            return []

        # Split detections into persons, bikes (bicycle/motorcycle), and cars
        persons: list[Detection] = []
        bikes: list[Detection] = []
        cars: list[Detection] = []

        for det in detections:
            if det.cls == "person":
                persons.append(det)
            elif det.cls in ("bicycle", "motorcycle"):
                bikes.append(det)
            elif det.cls == "car":
                cars.append(det)

        # For each person, find all qualifying bikes and pick the one with max containment
        # bike_index -> list of person detections assigned as riders
        bike_riders: dict[int, list[Detection]] = {i: [] for i in range(len(bikes))}
        merged_person_indices: set[int] = set()

        for p_idx, p in enumerate(persons):
            best_bike_idx: int | None = None
            best_containment = -1.0

            for b_idx, b in enumerate(bikes):
                qualifies, cont = is_rider_of_bike(
                    person_box=p.xyxy,
                    bike_box=b.xyxy,
                    min_containment=self.min_containment,
                )
                if qualifies and cont > best_containment:
                    best_containment = cont
                    best_bike_idx = b_idx

            if best_bike_idx is not None:
                bike_riders[best_bike_idx].append(p)
                merged_person_indices.add(p_idx)
                # If person has a valid track ID, remember that this person was merged
                p_id = p.track_id
                if p_id is not None:
                    self._rider_memory[p_id] = 0

        output: list[TrackedObject] = []

        # 1. Output two-wheelers (merged or unmerged)
        for b_idx, b in enumerate(bikes):
            riders = bike_riders[b_idx]
            rider_count = len(riders)
            bike_id = self._get_id(b.track_id)
            truncated = is_box_truncated(b.xyxy, f_size, self.truncation_margin_px)

            output.append(
                TrackedObject(
                    track_id=bike_id,
                    cls="two_wheeler",
                    box_xyxy=b.xyxy,
                    conf=b.conf,
                    rider_count=rider_count,
                    truncated=truncated,
                    t=t_capture,
                )
            )

        # 2. Output cars
        for c in cars:
            car_id = self._get_id(c.track_id)
            truncated = is_box_truncated(c.xyxy, f_size, self.truncation_margin_px)
            output.append(
                TrackedObject(
                    track_id=car_id,
                    cls="car",
                    box_xyxy=c.xyxy,
                    conf=c.conf,
                    rider_count=0,
                    truncated=truncated,
                    t=t_capture,
                )
            )

        # 3. Output unmerged persons, respecting rider memory suppression
        for p_idx, p in enumerate(persons):
            if p_idx in merged_person_indices:
                continue

            p_id = p.track_id
            if p_id is not None and p_id in self._rider_memory:
                # Increment unseen bike frame counter
                self._rider_memory[p_id] += 1
                if self._rider_memory[p_id] <= self.rider_memory_frames:
                    # Suppress person from appearing as a pedestrian
                    logger.debug(
                        f"Suppressing rider person ID {p_id} (flicker frame {self._rider_memory[p_id]}/{self.rider_memory_frames})"
                    )
                    continue
                else:
                    # Memory expired; remove from suppression memory and let person appear
                    del self._rider_memory[p_id]

            person_id = self._get_id(p.track_id)
            truncated = is_box_truncated(p.xyxy, f_size, self.truncation_margin_px)
            output.append(
                TrackedObject(
                    track_id=person_id,
                    cls="person",
                    box_xyxy=p.xyxy,
                    conf=p.conf,
                    rider_count=0,
                    truncated=truncated,
                    t=t_capture,
                )
            )

        return output


def merge_riders(
    dets: Sequence[Detection],
    min_containment: float = 0.5,
    t_capture: float = 0.0,
    frame_size: Tuple[int, int] = (1280, 720),
    truncation_margin_px: int = 3,
) -> list[TrackedObject]:
    """Stateless functional wrapper for single-frame rider merging (without history memory)."""
    merger = RiderMerger(
        min_containment=min_containment,
        rider_memory_frames=0,
        truncation_margin_px=truncation_margin_px,
        default_frame_size=frame_size,
    )
    return merger.merge(dets, t_capture=t_capture, frame_size=frame_size)
