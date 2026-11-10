"""Planar homography projection, foot point extraction, distance, and pinhole baseline (pure).

All coordinates in metres (world frame: +X right, +Y forward away from camera, Z=0 ground),
pixel coordinates in (u, v) with top-left origin. Zero camera, network, or file I/O.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from vision.calibration import pixel_to_world
from vision.config import PinholeBaselineConfig
from vision.types import TrackedObject


def foot_point(obj: TrackedObject | Sequence[float]) -> Tuple[float, float]:
    """Return the bottom-centre pixel coordinate (u, v) of an object's bounding box.

    For merged two-wheelers, TrackedObject.box_xyxy is always the bike box
    (wheel-ground contact), never the rider box.

    Args:
        obj: TrackedObject or (x1, y1, x2, y2) bounding box in pixels.

    Returns:
        (u, v) pixel coordinates of bottom-centre foot point.
    """
    box = obj.box_xyxy if isinstance(obj, TrackedObject) else obj
    x1, _, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return ((x1 + x2) * 0.5, y2)


def box_height_px(obj: TrackedObject | Sequence[float]) -> float:
    """Return bounding box vertical height in pixels (y2 - y1)."""
    box = obj.box_xyxy if isinstance(obj, TrackedObject) else obj
    return max(0.0, float(box[3]) - float(box[1]))


def is_foot_truncated(
    box_xyxy: Sequence[float],
    frame_size: Tuple[int, int],
    margin_px: int = 3,
) -> bool:
    """Return True if the bottom edge of the box touches the bottom frame edge.

    When the bottom edge touches the frame boundary, the ground contact foot point
    is outside the image and distance cannot be reliably estimated.

    Args:
        box_xyxy: Bounding box (x1, y1, x2, y2) in pixels.
        frame_size: Frame dimensions (width, height) in pixels.
        margin_px: Bottom-edge margin threshold in pixels.
    """
    _, frame_height = frame_size
    y2 = float(box_xyxy[3])
    return y2 >= (frame_height - margin_px)


def distance_from_camera(
    H: np.ndarray,
    foot_px: Tuple[float, float],
    camera_xy: Tuple[float, float] | Sequence[float],
) -> Tuple[float, float, float]:
    """Project foot pixel (u, v) onto ground plane via homography H and compute distance.

    Args:
        H: 3x3 pixel-to-world homography matrix.
        foot_px: Ground contact pixel coordinate (u, v).
        camera_xy: Camera plumb-line ground coordinate (Xc, Yc) in metres.

    Returns:
        (X, Y, d_m) where (X, Y) is ground position in metres and
        d_m = hypot(X - Xc, Y - Yc) is horizontal ground distance in metres.
    """
    world_x, world_y = pixel_to_world(H, foot_px)
    xc, yc = float(camera_xy[0]), float(camera_xy[1])
    d_m = math.hypot(world_x - xc, world_y - yc)
    return world_x, world_y, d_m


def _point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Euclidean distance from point (px, py) to line segment (ax, ay)-(bx, by)."""
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def distance_to_convex_hull(
    world_xy: Tuple[float, float],
    calibrated_world_pts: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Compute Euclidean distance in metres from world_xy to convex hull of calibration points.

    Returns 0.0 when world_xy lies inside or on the boundary of the convex hull.
    """
    pts = np.asarray(calibrated_world_pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("At least 3 2D points are required to form a calibrated convex hull")

    px, py = float(world_xy[0]), float(world_xy[1])
    try:
        hull = ConvexHull(pts)
    except QhullError as exc:
        raise ValueError(f"Calibration world points are degenerate for convex hull: {exc}") from exc

    # ConvexHull.equations has shape (n_facets, 3): a*x + b*y + c <= 0 for interior points
    p_vec = np.array([px, py], dtype=np.float64)
    signed_dists = hull.equations[:, :2] @ p_vec + hull.equations[:, 2]
    if float(np.max(signed_dists)) <= 1e-9:
        return 0.0

    # Outside the hull: minimum distance to any boundary edge
    vertices = pts[hull.vertices]
    n_verts = len(vertices)
    min_dist = math.inf
    for i in range(n_verts):
        ax, ay = float(vertices[i, 0]), float(vertices[i, 1])
        bx, by = float(vertices[(i + 1) % n_verts, 0]), float(vertices[(i + 1) % n_verts, 1])
        dist = _point_to_segment_distance(px, py, ax, ay, bx, by)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def is_in_range(
    world_xy: Tuple[float, float],
    camera_xy: Tuple[float, float] | Sequence[float],
    max_range_m: float,
    calibrated_world_pts: Sequence[Sequence[float]] | np.ndarray | None = None,
    hull_margin_m: float = 1.5,
    distance_m: float | None = None,
) -> bool:
    """Determine if ground point world_xy is within max_range_m and calibrated hull tolerance.

    Args:
        world_xy: Ground coordinate (X, Y) in metres.
        camera_xy: Camera ground coordinate (Xc, Yc) in metres.
        max_range_m: Maximum valid ground range from camera in metres.
        calibrated_world_pts: Optional Nx2 calibration ground points defining valid region hull.
        hull_margin_m: Maximum distance outside the calibrated convex hull still considered in-range.
        distance_m: Optional precomputed camera-to-object ground distance in metres.
    """
    d_m = (
        distance_m
        if distance_m is not None
        else math.hypot(float(world_xy[0]) - float(camera_xy[0]), float(world_xy[1]) - float(camera_xy[1]))
    )
    if not math.isfinite(d_m) or d_m > max_range_m:
        return False

    if calibrated_world_pts is not None and len(calibrated_world_pts) >= 3:
        hull_dist = distance_to_convex_hull(world_xy, calibrated_world_pts)
        if hull_dist > hull_margin_m:
            return False

    return True


def pinhole_distance(h_px: float, real_height_m: float, focal_px: float) -> float | None:
    """Estimate distance via pinhole height baseline: d = focal_px * real_height_m / h_px.

    Args:
        h_px: Observed bounding box height in pixels.
        real_height_m: Assumed physical object height in metres.
        focal_px: Camera focal length in pixels.

    Returns:
        Estimated distance in metres, or None if h_px <= 0.
    """
    if focal_px <= 0.0:
        raise ValueError(f"focal_px must be positive, got {focal_px}")
    if real_height_m <= 0.0:
        raise ValueError(f"real_height_m must be positive, got {real_height_m}")
    if h_px <= 0.0:
        return None
    return float(focal_px * real_height_m / h_px)


def estimate_focal_length_px(h_px: float, real_height_m: float, true_distance_m: float) -> float:
    """Compute focal length in pixels from one tape-measured reference observation.

    Formula: f = h_px * d_true / H_real.
    """
    if h_px <= 0.0 or real_height_m <= 0.0 or true_distance_m <= 0.0:
        raise ValueError("h_px, real_height_m, and true_distance_m must all be positive")
    return float(h_px * true_distance_m / real_height_m)


def assumed_height_for_class(cls: str, cfg: PinholeBaselineConfig) -> float | None:
    """Lookup assumed physical object height (metres) for pinhole comparison."""
    mapping = {
        "person": cfg.person_height_m,
        "two_wheeler": cfg.two_wheeler_height_m,
        "car": cfg.car_height_m,
    }
    return mapping.get(cls)


class DistanceSmoother:
    """Per-track Exponential Moving Average (EMA) smoother for distance in metres."""

    def __init__(self, alpha: float = 0.4, track_forget_s: float = 2.0) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"distance_ema_alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.track_forget_s = float(track_forget_s)
        self._state: dict[int, Tuple[float, float]] = {}  # track_id -> (smoothed_d, last_t)

    def reset(self) -> None:
        """Clear all track EMA history."""
        self._state.clear()

    def prune(self, now_t: float) -> None:
        """Remove tracks unseen for longer than track_forget_s."""
        expired = [
            tid for tid, (_, last_t) in self._state.items()
            if (now_t - last_t) > self.track_forget_s
        ]
        for tid in expired:
            del self._state[tid]

    def update(self, track_id: int, raw_distance_m: float, t: float) -> float:
        """Update EMA state for track_id and return smoothed distance in metres."""
        if track_id not in self._state:
            smoothed = float(raw_distance_m)
        else:
            prev_d, last_t = self._state[track_id]
            if (t - last_t) > self.track_forget_s:
                smoothed = float(raw_distance_m)
            else:
                smoothed = self.alpha * float(raw_distance_m) + (1.0 - self.alpha) * prev_d
        self._state[track_id] = (smoothed, float(t))
        return smoothed
