"""Interactive camera calibration and ground plane homography tool.

Features:
- Live camera preview with freeze frame ('Space' / 'f').
- Interactive point clicking guided by calibration CSV (label, X, Y).
- Real-time magnifying loupe (4x zoom inset) for sub-pixel clicking precision.
- Undo ('u'), reset ('r'), compute H ('Enter'), validate on held-out points ('v'), save ('s'), quit ('q').
- Computes homography H, reprojection error in METRES and pixels, and leave-one-out errors.
- Overlays 1 m ground metric grid for visual alignment check against floor tape.
- Saves complete calibration parameters and metadata to JSON.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from vision.calibration import (
    CalibrationData,
    compute_homography,
    leave_one_out_errors,
    pixel_to_world,
    reprojection_errors,
    world_to_pixel,
)
from vision.capture import create_source
from vision.config import AppConfig, load_config
from vision.detector import YoloDetector
from vision.geometry import estimate_focal_length_px

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("calibrate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ground plane homography calibration tool.")
    parser.add_argument("--source", type=str, default=None, help="Camera index or URL. Defaults to config.")
    parser.add_argument("--image", type=str, default=None, help="Path to static image file instead of live camera.")
    parser.add_argument(
        "--camera-id",
        type=str,
        default=None,
        help="Camera identifier (defaults to config camera.camera_id).",
    )
    parser.add_argument(
        "--points-csv",
        type=str,
        default="data/calibration/laptop_webcam_points.csv",
        help="CSV file containing calibration grid world points (label,X,Y).",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/calibration/laptop_webcam_val_points.csv",
        help="CSV file containing held-out validation points (label,X,Y).",
    )
    parser.add_argument(
        "--camera-x",
        type=float,
        default=1.0,
        help="Ground X coordinate of camera plumb line in metres.",
    )
    parser.add_argument(
        "--camera-y",
        type=float,
        default=0.0,
        help="Ground Y coordinate of camera plumb line in metres.",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.5,
        help="Camera lens height above ground in metres.",
    )
    parser.add_argument(
        "--calibrate-focal",
        action="store_true",
        help="Estimate pinhole focal_px from one person standing at --ref-distance and save into existing JSON.",
    )
    parser.add_argument(
        "--ref-distance",
        type=float,
        default=None,
        help="True tape-measured camera distance (metres) to the reference person for --calibrate-focal.",
    )
    parser.add_argument(
        "--person-height",
        type=float,
        default=None,
        help="True height (metres) of the reference person (defaults to pinhole_baseline.person_height_m).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to data/calibration/<camera_id>.json.",
    )
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to default config.")
    parser.add_argument("--local-config", type=str, default="config/local.yaml", help="Path to local config.")
    return parser.parse_args()


def load_points_csv(csv_path: str | Path) -> List[Tuple[str, float, float]]:
    """Load (label, X, Y) world points from CSV."""
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Points CSV not found: {p}")

    points = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            x = float(row["X"])
            y = float(row["Y"])
            points.append((label, x, y))
    return points


class CalibrationSession:
    """State management and UI interaction for the calibration tool."""

    def __init__(
        self,
        frame: np.ndarray,
        calib_targets: List[Tuple[str, float, float]],
        val_targets: List[Tuple[str, float, float]],
        camera_id: str,
        camera_xy: List[float],
        camera_height_m: float,
        output_path: Path,
    ) -> None:
        self.orig_frame = frame.copy()
        self.calib_targets = calib_targets
        self.val_targets = val_targets
        self.camera_id = camera_id
        self.camera_xy = camera_xy
        self.camera_height_m = camera_height_m
        self.output_path = output_path

        self.clicked_calib_pts: List[Tuple[float, float]] = []
        self.clicked_val_pts: List[Tuple[float, float]] = []

        self.H: np.ndarray | None = None
        self.validation_mode = False
        self.cursor_pos: Tuple[int, int] = (0, 0)
        self.status_msg = "Click marks in order. Space: unfreeze | Enter: compute | u: undo | r: reset | s: save"

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        self.cursor_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.validation_mode:
                if len(self.clicked_calib_pts) < len(self.calib_targets):
                    self.clicked_calib_pts.append((float(x), float(y)))
                    curr = len(self.clicked_calib_pts)
                    total = len(self.calib_targets)
                    lbl = self.calib_targets[curr - 1][0]
                    self.status_msg = f"Recorded point {curr}/{total}: {lbl} at ({x}, {y})"
                    logger.info(f"Recorded calibration point {curr}/{total}: {lbl} at pixel ({x}, {y})")
                    if len(self.clicked_calib_pts) == len(self.calib_targets):
                        self.status_msg = "All calibration points clicked! Press ENTER to compute H."
            else:
                if len(self.clicked_val_pts) < len(self.val_targets):
                    self.clicked_val_pts.append((float(x), float(y)))
                    curr = len(self.clicked_val_pts)
                    total = len(self.val_targets)
                    lbl = self.val_targets[curr - 1][0]
                    self.status_msg = f"Recorded validation point {curr}/{total}: {lbl} at ({x}, {y})"
                    logger.info(f"Recorded validation point {curr}/{total}: {lbl} at pixel ({x}, {y})")
                    if curr == total:
                        self.validate_held_out()

    def undo(self) -> None:
        if self.validation_mode:
            if self.clicked_val_pts:
                popped = self.clicked_val_pts.pop()
                self.status_msg = f"Undid validation point at {popped}"
        else:
            if self.clicked_calib_pts:
                popped = self.clicked_calib_pts.pop()
                self.H = None
                self.status_msg = f"Undid calibration point at {popped}"

    def reset(self) -> None:
        if self.validation_mode:
            self.clicked_val_pts.clear()
            self.status_msg = "Validation points reset."
        else:
            self.clicked_calib_pts.clear()
            self.clicked_val_pts.clear()
            self.H = None
            self.status_msg = "Calibration points reset."

    def compute(self) -> None:
        if len(self.clicked_calib_pts) < 4:
            self.status_msg = f"Need at least 4 points (currently have {len(self.clicked_calib_pts)})"
            logger.warning(self.status_msg)
            return

        w_pts = [[pt[1], pt[2]] for pt in self.calib_targets[: len(self.clicked_calib_pts)]]
        p_pts = self.clicked_calib_pts

        try:
            self.H = compute_homography(p_pts, w_pts)
            stats = reprojection_errors(self.H, p_pts, w_pts)

            print("\n" + "=" * 60)
            print("HOMOGRAPHY COMPUTATION RESULTS")
            print("=" * 60)
            print(f"Points used: {len(p_pts)}")
            print(f"Mean Reprojection Error: {stats['mean_error_m'] * 100.0:.2f} cm ({stats['mean_error_m']:.4f} m)")
            print(f"Max Reprojection Error:  {stats['max_error_m'] * 100.0:.2f} cm ({stats['max_error_m']:.4f} m)")
            print(f"Mean Pixel Error:        {stats['mean_error_px']:.2f} px")

            print("\nPer-Point Errors (Metres):")
            for i, (tgt, err_m, err_px) in enumerate(zip(self.calib_targets, stats["errors_m"], stats["errors_px"])):
                print(f"  {tgt[0]:>5} (X={tgt[1]:.1f}m, Y={tgt[2]:.1f}m): {err_m * 100.0:6.2f} cm | {err_px:4.1f} px")

            if len(p_pts) >= 5:
                loo = leave_one_out_errors(p_pts, w_pts)
                print(f"\nMean Leave-One-Out Error: {loo['mean_loo_error_m'] * 100.0:.2f} cm")
                print(f"Max Leave-One-Out Error:  {loo['max_loo_error_m'] * 100.0:.2f} cm")

            print("=" * 60 + "\n")
            self.status_msg = f"H computed! Mean err: {stats['mean_error_m']*100:.1f} cm. Press 's' to save, 'v' for validation."

        except Exception as e:
            self.status_msg = f"Error computing H: {e}"
            logger.error(self.status_msg)

    def validate_held_out(self) -> None:
        if self.H is None:
            self.status_msg = "Compute H first before validating held-out points!"
            return

        if not self.clicked_val_pts:
            self.status_msg = f"Click the {len(self.val_targets)} validation points first."
            return

        w_val = [[pt[1], pt[2]] for pt in self.val_targets[: len(self.clicked_val_pts)]]
        p_val = self.clicked_val_pts
        stats = reprojection_errors(self.H, p_val, w_val)

        print("\n" + "=" * 60)
        print("HELD-OUT VALIDATION RESULTS")
        print("=" * 60)
        print(f"Validation Points Tested: {len(p_val)}")
        print(f"Mean Validation Error:    {stats['mean_error_m'] * 100.0:.2f} cm ({stats['mean_error_m']:.4f} m)")
        print(f"Max Validation Error:     {stats['max_error_m'] * 100.0:.2f} cm ({stats['max_error_m']:.4f} m)")
        for tgt, err_m in zip(self.val_targets, stats["errors_m"]):
            print(f"  {tgt[0]:>5} (X={tgt[1]:.1f}m, Y={tgt[2]:.1f}m): {err_m * 100.0:6.2f} cm")
        print("=" * 60 + "\n")
        self.status_msg = f"Validation Mean Err: {stats['mean_error_m']*100:.1f} cm, Max: {stats['max_error_m']*100:.1f} cm"

    def save(self) -> None:
        if self.H is None:
            self.status_msg = "Compute H first before saving!"
            return

        h, w = self.orig_frame.shape[:2]
        w_pts = [[pt[1], pt[2]] for pt in self.calib_targets[: len(self.clicked_calib_pts)]]
        p_pts = [[pt[0], pt[1]] for pt in self.clicked_calib_pts]

        calib_data = CalibrationData(
            camera_id=self.camera_id,
            image_width=w,
            image_height=h,
            H=self.H.tolist(),
            pixel_pts=p_pts,
            world_pts=w_pts,
            camera_xy=self.camera_xy,
            camera_height_m=self.camera_height_m,
            notes=f"Calibrated via tools/calibrate.py with {len(p_pts)} grid points",
        )
        calib_data.save_json(self.output_path)
        self.status_msg = f"Saved calibration to {self.output_path}"
        print(f"[SUCCESS] Saved calibration JSON to {self.output_path}")

    def render(self) -> np.ndarray:
        display = self.orig_frame.copy()
        h, w = display.shape[:2]

        # 1. Overlay 1m metric grid if H is computed
        if self.H is not None:
            self._draw_metric_grid(display)

        # 2. Draw calibration points
        for i, (u, v) in enumerate(self.clicked_calib_pts):
            pt_int = (int(round(u)), int(round(v)))
            cv2.circle(display, pt_int, 4, (0, 0, 255), -1)
            cv2.circle(display, pt_int, 7, (0, 255, 255), 1)
            lbl = self.calib_targets[i][0]
            cv2.putText(
                display,
                f"{i+1}:{lbl}",
                (pt_int[0] + 8, pt_int[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw validation points
        for i, (u, v) in enumerate(self.clicked_val_pts):
            pt_int = (int(round(u)), int(round(v)))
            cv2.drawMarker(display, pt_int, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
            lbl = self.val_targets[i][0]
            cv2.putText(
                display,
                f"VAL:{lbl}",
                (pt_int[0] + 8, pt_int[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # 4. Next target indicator
        if not self.validation_mode and len(self.clicked_calib_pts) < len(self.calib_targets):
            next_idx = len(self.clicked_calib_pts)
            lbl, tx, ty = self.calib_targets[next_idx]
            info = f"Next mark to click [{next_idx+1}/{len(self.calib_targets)}]: {lbl} (X={tx:.1f}m, Y={ty:.1f}m)"
        elif self.validation_mode and len(self.clicked_val_pts) < len(self.val_targets):
            next_idx = len(self.clicked_val_pts)
            lbl, tx, ty = self.val_targets[next_idx]
            info = f"[VAL MODE] Next target [{next_idx+1}/{len(self.val_targets)}]: {lbl} (X={tx:.1f}m, Y={ty:.1f}m)"
        else:
            info = "All points clicked. Press Enter to compute H."

        # 5. Draw status bar at bottom
        cv2.rectangle(display, (0, h - 50), (w, h), (30, 30, 30), -1)
        cv2.putText(display, info, (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, self.status_msg, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 6. Draw 4x Magnifying Loupe in top-right corner
        self._draw_loupe(display)

        return display

    def _draw_metric_grid(self, display: np.ndarray) -> None:
        """Project and draw 1-metre metric grid lines onto the floor plane."""
        assert self.H is not None
        h, w = display.shape[:2]

        # Draw grid lines: X in [-1.0, 3.0] step 1.0; Y in [0.0, 7.0] step 1.0
        x_min, x_max = -1.0, 3.0
        y_min, y_max = 0.5, 6.0

        # Lines along Y (constant X)
        for x in np.arange(x_min, x_max + 0.1, 1.0):
            pts = []
            for y in np.linspace(y_min, y_max, 30):
                try:
                    u, v = world_to_pixel(self.H, (x, y))
                    if 0 <= u < w and 0 <= v < h:
                        pts.append((int(round(u)), int(round(v))))
                except Exception:
                    pass
            for j in range(1, len(pts)):
                cv2.line(display, pts[j - 1], pts[j], (0, 220, 220), 1, cv2.LINE_AA)

        # Lines along X (constant Y)
        for y in np.arange(1.0, y_max + 0.1, 1.0):
            pts = []
            for x in np.linspace(x_min, x_max, 30):
                try:
                    u, v = world_to_pixel(self.H, (x, y))
                    if 0 <= u < w and 0 <= v < h:
                        pts.append((int(round(u)), int(round(v))))
                except Exception:
                    pass
            for j in range(1, len(pts)):
                cv2.line(display, pts[j - 1], pts[j], (0, 220, 220), 1, cv2.LINE_AA)

    def _draw_loupe(self, display: np.ndarray) -> None:
        """Render a 4x magnified loupe in the upper-right corner around current mouse cursor."""
        cx, cy = self.cursor_pos
        h, w = display.shape[:2]

        loupe_w, loupe_h = 180, 180
        src_half = 25  # 50x50 crop magnified ~3.6x to 180x180

        x1 = max(0, cx - src_half)
        y1 = max(0, cy - src_half)
        x2 = min(w, cx + src_half)
        y2 = min(h, cy + src_half)

        crop = self.orig_frame[y1:y2, x1:x2]
        if crop.size > 0:
            zoom = cv2.resize(crop, (loupe_w, loupe_h), interpolation=cv2.INTER_NEAREST)
            # Draw crosshair in loupe
            mid_x, mid_y = loupe_w // 2, loupe_h // 2
            cv2.line(zoom, (mid_x - 15, mid_y), (mid_x + 15, mid_y), (0, 0, 255), 1)
            cv2.line(zoom, (mid_x, mid_y - 15), (mid_x, mid_y + 15), (0, 0, 255), 1)
            cv2.rectangle(zoom, (0, 0), (loupe_w - 1, loupe_h - 1), (0, 255, 255), 2)

            # Paste loupe into top-right corner
            pos_x = w - loupe_w - 10
            pos_y = 10
            display[pos_y : pos_y + loupe_h, pos_x : pos_x + loupe_w] = zoom
            cv2.putText(
                display,
                f"Loupe ({cx}, {cy})",
                (pos_x, pos_y + loupe_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )


def _acquire_frame(
    args: argparse.Namespace,
    cfg: AppConfig,
    window_name: str,
    prompt_text: str,
) -> np.ndarray | None:
    """Acquire a frozen frame either from a static image file or live camera preview."""
    if args.image:
        frozen_frame = cv2.imread(args.image)
        if frozen_frame is None:
            logger.error(f"Failed to read image from {args.image}")
            return None
        return frozen_frame

    source_val = args.source if args.source is not None else cfg.camera.source
    logger.info(f"Opening live camera source: {source_val}")
    source = create_source(
        source_val,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        fourcc=cfg.camera.fourcc,
        rtsp_transport=cfg.camera.rtsp_transport,
    )

    frozen_frame: np.ndarray | None = None
    logger.info("Live preview active. Press SPACE or 'f' to freeze a frame, 'q' to quit.")
    try:
        while True:
            frame, _ = source.read()
            if frame is None:
                cv2.waitKey(10)
                continue

            preview = frame.copy()
            cv2.putText(
                preview,
                prompt_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord(" "), ord("f")):
                frozen_frame = frame.copy()
                logger.info("Frame frozen.")
                break
            if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        source.close()

    return frozen_frame


def _run_focal_calibration(
    args: argparse.Namespace,
    cfg: AppConfig,
    out_json: Path,
    window_name: str,
) -> int:
    """Estimate focal_px from one tape-measured reference person and update calibration JSON."""
    if args.ref_distance is None or args.ref_distance <= 0:
        logger.error("--calibrate-focal requires --ref-distance <metres> (> 0).")
        return 1

    person_h = (
        args.person_height
        if args.person_height is not None
        else cfg.pinhole_baseline.person_height_m
    )

    calib = CalibrationData.load_json(out_json)
    frozen_frame = _acquire_frame(
        args,
        cfg,
        window_name,
        f"Stand person ({person_h:.2f}m tall) at {args.ref_distance:.2f}m, press SPACE",
    )
    if frozen_frame is None:
        cv2.destroyAllWindows()
        return 0

    h, w = frozen_frame.shape[:2]
    if (w, h) != (calib.image_width, calib.image_height):
        raise ValueError(
            f"Calibration frame size mismatch for {calib.camera_id}: "
            f"expected {calib.image_width}x{calib.image_height}, got {w}x{h}"
        )

    detector = YoloDetector(cfg.detector)
    person_dets = [d for d in detector.detect(frozen_frame) if d.cls == "person"]

    selected_box: Tuple[float, float, float, float] | None = None
    if person_dets:
        # Pick largest / highest-confidence person detection
        best = max(person_dets, key=lambda d: (d.xyxy[3] - d.xyxy[1]) * d.conf)
        selected_box = best.xyxy

    clicked_y: list[int] = []

    def _on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_y) < 2:
            clicked_y.append(y)

    cv2.setMouseCallback(window_name, _on_mouse)

    while True:
        vis = frozen_frame.copy()
        h_px: float | None = None
        src_note = ""

        if len(clicked_y) == 2:
            y_top, y_bot = min(clicked_y), max(clicked_y)
            h_px = float(y_bot - y_top)
            cv2.line(vis, (0, y_top), (w, y_top), (255, 0, 255), 2)
            cv2.line(vis, (0, y_bot), (w, y_bot), (255, 0, 255), 2)
            src_note = "manual clicks"
        elif selected_box is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in selected_box)
            h_px = float(selected_box[3] - selected_box[1])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            src_note = "YOLO person box"

        if h_px is not None and h_px > 0:
            focal_px = estimate_focal_length_px(h_px, person_h, args.ref_distance)
            info = (
                f"h_px={h_px:.1f}px ({src_note}) -> focal_px={focal_px:.1f}px "
                f"| Press 's' to save, 'r' click manually, 'q' quit"
            )
        else:
            focal_px = None
            info = "Click HEAD top then FEET bottom (2 clicks). Press 'r' reset, 'q' quit."

        cv2.rectangle(vis, (0, h - 38), (w, h), (30, 30, 30), -1)
        cv2.putText(vis, info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord("r"):
            clicked_y.clear()
            selected_box = None
        elif key == ord("s") and focal_px is not None:
            calib.focal_px = round(focal_px, 2)
            calib.save_json(out_json)
            print(
                f"[SUCCESS] Saved focal_px={calib.focal_px:.2f} px "
                f"(H_person={person_h:.2f}m, d_ref={args.ref_distance:.2f}m, h_px={h_px:.1f}px) to {out_json}"
            )
            break

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(default_path=args.config, local_path=args.local_config)
    camera_id = args.camera_id if args.camera_id is not None else cfg.camera.camera_id

    out_json = (
        Path(args.output)
        if args.output is not None
        else Path(f"data/calibration/{camera_id}.json")
    )

    window_name = "Minor Project - Camera Calibration (Freeze: Space/f | Click points)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if args.calibrate_focal:
        return _run_focal_calibration(args, cfg, out_json, window_name)

    calib_targets = load_points_csv(args.points_csv)
    val_targets = load_points_csv(args.val_csv) if Path(args.val_csv).is_file() else []

    frozen_frame = _acquire_frame(
        args,
        cfg,
        window_name,
        "LIVE: Position floor marks, then press SPACE to freeze frame",
    )
    if frozen_frame is None:
        cv2.destroyAllWindows()
        return 0

    session = CalibrationSession(
        frame=frozen_frame,
        calib_targets=calib_targets,
        val_targets=val_targets,
        camera_id=camera_id,
        camera_xy=[args.camera_x, args.camera_y],
        camera_height_m=args.camera_height,
        output_path=out_json,
    )

    cv2.setMouseCallback(window_name, session.mouse_callback)

    while True:
        rendered = session.render()
        cv2.imshow(window_name, rendered)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            logger.info("Quit requested.")
            break
        elif key == ord("u"):
            session.undo()
        elif key == ord("r"):
            session.reset()
        elif key in (10, 13):  # Enter key
            session.compute()
        elif key == ord("s"):
            session.save()
        elif key == ord("v"):
            session.validation_mode = not session.validation_mode
            mode_str = "Validation (Held-out points)" if session.validation_mode else "Calibration points"
            session.status_msg = f"Switched mode to: {mode_str}"
            logger.info(f"Switched mode to: {mode_str}")
            if session.validation_mode and len(session.clicked_val_pts) == len(session.val_targets):
                session.validate_held_out()

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
