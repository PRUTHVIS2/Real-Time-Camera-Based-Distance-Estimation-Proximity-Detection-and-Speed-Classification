"""Camera calibration and planar homography mathematics.

Provides pure functions for:
- Computing homography H mapping pixel coordinates (u, v) to ground world coordinates (X, Y) in metres.
- Forward projection (pixel_to_world) and inverse projection (world_to_pixel).
- Reprojection error computation in metres and in pixels.
- Leave-one-out cross-validation error analysis.
- Collinearity and minimum point guards.
- Serialization and loading of calibration JSON with strict resolution verification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def check_points_validity(pixel_pts: np.ndarray, world_pts: np.ndarray) -> None:
    """Validate that points are sufficient and not degenerate.

    Raises:
        ValueError: If fewer than 4 points or if points are nearly collinear.
    """
    if len(pixel_pts) < 4 or len(world_pts) < 4:
        raise ValueError(
            f"At least 4 correspondence points are required to compute homography, got {len(pixel_pts)}"
        )
    if len(pixel_pts) != len(world_pts):
        raise ValueError(
            f"Mismatched points: {len(pixel_pts)} pixel points vs {len(world_pts)} world points"
        )

    # Check collinearity of pixel points using SVD on centered coordinates
    pix_centered = pixel_pts - np.mean(pixel_pts, axis=0)
    _, s_pix, _ = np.linalg.svd(pix_centered)
    if s_pix[1] < 1e-3 or (s_pix[1] / s_pix[0]) < 1e-4:
        raise ValueError("Pixel points are collinear or nearly collinear; cannot fit a 2D plane homography")

    # Check collinearity of world points
    world_centered = world_pts - np.mean(world_pts, axis=0)
    _, s_world, _ = np.linalg.svd(world_centered)
    if s_world[1] < 1e-3 or (s_world[1] / s_world[0]) < 1e-4:
        raise ValueError("World points are collinear or nearly collinear; cannot fit a 2D plane homography")


def compute_homography(
    pixel_pts: Sequence[Sequence[float]] | np.ndarray,
    world_pts: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Compute homography H mapping pixel coordinates (u, v) to ground plane (X, Y) in metres.

    Uses least-squares findHomography (method=0).

    Args:
        pixel_pts: (N, 2) array of (u, v) coordinates in pixels.
        world_pts: (N, 2) array of (X, Y) ground coordinates in metres.

    Returns:
        3x3 homography matrix H where [X, Y, 1]^T ~ H * [u, v, 1]^T.
    """
    p_pts = np.asarray(pixel_pts, dtype=np.float64)
    w_pts = np.asarray(world_pts, dtype=np.float64)

    check_points_validity(p_pts, w_pts)

    # findHomography with method=0 uses standard least squares
    H, status = cv2.findHomography(p_pts, w_pts, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed to find a valid transformation matrix")

    return H


def pixel_to_world(H: np.ndarray, uv: Tuple[float, float] | Sequence[float]) -> Tuple[float, float]:
    """Map a single pixel coordinate (u, v) to ground coordinate (X, Y) in metres.

    Args:
        H: 3x3 homography matrix.
        uv: (u, v) pixel coordinate.

    Returns:
        (X, Y) ground coordinates in metres.
    """
    u, v = float(uv[0]), float(uv[1])
    vec = np.array([u, v, 1.0], dtype=np.float64)
    mapped = H @ vec
    w = mapped[2]
    if abs(w) < 1e-9:
        raise ValueError(f"Projective coordinate w' is zero ({w}); point lies on horizon or at infinity")
    return float(mapped[0] / w), float(mapped[1] / w)


def world_to_pixel(H: np.ndarray, xy: Tuple[float, float] | Sequence[float]) -> Tuple[float, float]:
    """Map a ground coordinate (X, Y) in metres back to image pixel coordinate (u, v).

    Args:
        H: 3x3 homography matrix mapping pixels to world.
        xy: (X, Y) world coordinates in metres.

    Returns:
        (u, v) pixel coordinate.
    """
    H_inv = np.linalg.inv(H)
    x, y = float(xy[0]), float(xy[1])
    vec = np.array([x, y, 1.0], dtype=np.float64)
    mapped = H_inv @ vec
    w = mapped[2]
    if abs(w) < 1e-9:
        raise ValueError(f"Projective coordinate w' is zero ({w}); point cannot be projected to image")
    return float(mapped[0] / w), float(mapped[1] / w)


def reprojection_errors(
    H: np.ndarray,
    pixel_pts: Sequence[Sequence[float]] | np.ndarray,
    world_pts: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Compute per-point reprojection errors in metres (forward) and pixels (backward).

    Returns:
        dict with:
            errors_m: list of euclidean errors in metres per point
            mean_error_m: float
            max_error_m: float
            errors_px: list of reprojection errors in pixels per point
            mean_error_px: float
            max_error_px: float
    """
    p_pts = np.asarray(pixel_pts, dtype=np.float64)
    w_pts = np.asarray(world_pts, dtype=np.float64)

    errors_m = []
    errors_px = []

    for (u, v), (X_true, Y_true) in zip(p_pts, w_pts):
        # Pixel -> World error in metres
        X_est, Y_est = pixel_to_world(H, (u, v))
        err_m = float(np.hypot(X_est - X_true, Y_est - Y_true))
        errors_m.append(err_m)

        # World -> Pixel error in pixels
        u_est, v_est = world_to_pixel(H, (X_true, Y_true))
        err_px = float(np.hypot(u_est - u, v_est - v))
        errors_px.append(err_px)

    return {
        "errors_m": errors_m,
        "mean_error_m": float(np.mean(errors_m)),
        "max_error_m": float(np.max(errors_m)),
        "errors_px": errors_px,
        "mean_error_px": float(np.mean(errors_px)),
        "max_error_px": float(np.max(errors_px)),
    }


def leave_one_out_errors(
    pixel_pts: Sequence[Sequence[float]] | np.ndarray,
    world_pts: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Compute leave-one-out cross-validation errors in metres.

    Recomputes H without point k and measures prediction error on point k.

    Returns:
        dict with 'loo_errors_m', 'mean_loo_error_m', and 'max_loo_error_m'.
    """
    p_pts = np.asarray(pixel_pts, dtype=np.float64)
    w_pts = np.asarray(world_pts, dtype=np.float64)
    n = len(p_pts)

    if n < 5:
        raise ValueError("Leave-one-out analysis requires at least 5 points")

    loo_errors_m = []

    for i in range(n):
        # Sub-sample all points except index i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        p_sub = p_pts[mask]
        w_sub = w_pts[mask]

        H_sub = compute_homography(p_sub, w_sub)
        X_pred, Y_pred = pixel_to_world(H_sub, p_pts[i])
        err = float(np.hypot(X_pred - w_pts[i, 0], Y_pred - w_pts[i, 1]))
        loo_errors_m.append(err)

    return {
        "loo_errors_m": loo_errors_m,
        "mean_loo_error_m": float(np.mean(loo_errors_m)),
        "max_loo_error_m": float(np.max(loo_errors_m)),
    }


@dataclass
class CalibrationData:
    """Complete calibration parameters and metadata."""
    camera_id: str
    image_width: int
    image_height: int
    H: list[list[float]]  # 3x3 matrix
    pixel_pts: list[list[float]]
    world_pts: list[list[float]]
    camera_xy: list[float]  # [Xc, Yc] ground coordinate of camera plumb line in metres
    camera_height_m: float  # lens height above ground in metres
    focal_px: float | None = None  # estimated focal length for pinhole baseline comparison
    date: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.date:
            self.date = datetime.now().isoformat()

    @property
    def homography(self) -> np.ndarray:
        return np.array(self.H, dtype=np.float64)

    def save_json(self, path: str | Path) -> None:
        """Save calibration data to JSON file."""
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Saved calibration JSON to: {out_p}")

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        verify_frame_size: Tuple[int, int] | None = None,
    ) -> CalibrationData:
        """Load calibration data from JSON, validating resolution if requested.

        Args:
            path: Path to JSON calibration file.
            verify_frame_size: Optional (width, height) tuple to assert frame resolution matches.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If live frame size does not match calibration resolution.
        """
        in_p = Path(path)
        if not in_p.is_file():
            raise FileNotFoundError(f"Calibration file not found at: {in_p}")

        with open(in_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        calib = cls(**data)

        if verify_frame_size is not None:
            expected = (calib.image_width, calib.image_height)
            if verify_frame_size != expected:
                raise ValueError(
                    f"Calibration frame size mismatch for {calib.camera_id}: "
                    f"calibration was computed for {expected[0]}x{expected[1]}, "
                    f"but current frame is {verify_frame_size[0]}x{verify_frame_size[1]}"
                )

        return calib
