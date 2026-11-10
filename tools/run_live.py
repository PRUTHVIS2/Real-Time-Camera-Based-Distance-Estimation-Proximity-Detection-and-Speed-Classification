"""Live camera tracking, rider-merge, and ground distance visualization tool.

Runs ByteTrack tracking, rider-to-bike merging, and planar homography distance
estimation on a live camera stream or video file. Displays bounding boxes,
persistent IDs, ground foot points, distance (d = X.XX m), an optional top-down
mini-map, and supports CSV measurement logging (--log-csv) and annotated MP4 export.
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from vision.calibration import CalibrationData
from vision.capture import create_source
from vision.config import load_config
from vision.detector import YoloDetector
from vision.merge import RiderMerger
from vision.pipeline import MeasurementCsvLogger, VisionPipeline
from vision.types import Measurement, TrackedObject

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_live")

# Class colors in BGR
CLASS_COLORS = {
    "person": (0, 255, 0),         # Green
    "two_wheeler": (255, 255, 0),  # Cyan
    "car": (0, 200, 255),          # Yellow-Orange
}
DEFAULT_COLOR = (200, 200, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live tracking, merge, and distance viewer.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Camera index (e.g. 0), RTSP URL (rtsp://...), or video file path. Defaults to config.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold override (e.g. 0.25 vs 0.10).",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration JSON. Defaults to data/calibration/<camera_id>.json if present.",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="Run tracking & merge only, without loading homography calibration.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Disable top-down bird's-eye mini-map overlay.",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Write per-frame measurements to CSV (t, id, cls, X, Y, distance, in_range, truncated, ...).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Display raw unmerged detections without rider merging or distance.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Path to output .mp4 file to record annotated video.",
    )
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML.")
    parser.add_argument("--local-config", type=str, default="config/local.yaml", help="Path to local config YAML.")
    parser.add_argument("--no-display", action="store_true", help="Run without opening an OpenCV GUI window.")
    parser.add_argument("--max-frames", type=int, default=None, help="Exit after processing this many frames.")
    return parser.parse_args()


def _draw_measurements(frame: np.ndarray, measurements: Sequence[Measurement]) -> None:
    """Draw bounding boxes, foot points, track IDs, and ground distances onto frame."""
    for m in measurements:
        obj = m.obj
        x1, y1, x2, y2 = (int(v) for v in obj.box_xyxy)
        color = CLASS_COLORS.get(obj.cls, DEFAULT_COLOR)
        if m.distance_m is not None and not m.in_range:
            color = (0, 200, 200)  # Muted yellow for out-of-range detections

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Foot point dot when distance is valid
        if m.distance_m is not None:
            fx, fy = int(round(m.foot_px[0])), int(round(m.foot_px[1]))
            cv2.circle(frame, (fx, fy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (fx, fy), 7, (255, 255, 255), 1)

        label_parts = [f"#{obj.track_id} {obj.cls}"]
        if obj.cls == "two_wheeler" and obj.rider_count > 0:
            label_parts.append(f"riders:{obj.rider_count}")
        if m.distance_m is not None:
            label_parts.append(f"d={m.distance_m:.2f}m")
            if not m.in_range:
                label_parts.append("[OOR]")
        else:
            label_parts.append("[EDGE]")
        label = " ".join(label_parts)

        (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - txt_h - 6)), (x1 + txt_w + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, max(txt_h + 2, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def _draw_minimap(
    frame: np.ndarray,
    measurements: Sequence[Measurement],
    calib: CalibrationData,
    max_range_m: float,
) -> None:
    """Draw a compact top-down ground map in the top-right corner."""
    fh, fw = frame.shape[:2]
    map_size = min(220, max(140, fh // 3))
    margin = 10
    if fw < map_size + 2 * margin or fh < map_size + 2 * margin:
        return

    panel = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    panel[:, :] = (25, 25, 25)

    # Determine bounding box in metres covering camera and calibrated world points
    all_x = [float(calib.camera_xy[0])] + [float(p[0]) for p in calib.world_pts]
    all_y = [float(calib.camera_xy[1])] + [float(p[1]) for p in calib.world_pts]
    for m in measurements:
        if m.world_xy is not None and m.in_range:
            all_x.append(m.world_xy[0])
            all_y.append(m.world_xy[1])

    x_min, x_max = min(all_x) - 1.0, max(all_x) + 1.0
    y_min, y_max = min(all_y) - 0.5, max(max(all_y) + 1.0, float(calib.camera_xy[1]) + 4.0)
    span = max(x_max - x_min, y_max - y_min, 2.0)
    cx_mid = 0.5 * (x_min + x_max)
    cy_mid = 0.5 * (y_min + y_max)

    def to_px(wx: float, wy: float) -> tuple[int, int]:
        u = int(round((wx - (cx_mid - 0.5 * span)) / span * (map_size - 20) + 10))
        v = int(round(map_size - 10 - (wy - (cy_mid - 0.5 * span)) / span * (map_size - 20)))
        return (max(2, min(map_size - 3, u)), max(2, min(map_size - 3, v)))

    # Calibrated area bounding outline
    if calib.world_pts:
        cal_xs = [p[0] for p in calib.world_pts]
        cal_ys = [p[1] for p in calib.world_pts]
        p_bl = to_px(min(cal_xs), min(cal_ys))
        p_tr = to_px(max(cal_xs), max(cal_ys))
        cv2.rectangle(panel, (p_bl[0], p_tr[1]), (p_tr[0], p_bl[1]), (80, 120, 120), 1)

    # Camera position marker
    cam_pt = to_px(float(calib.camera_xy[0]), float(calib.camera_xy[1]))
    cv2.drawMarker(panel, cam_pt, (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 10, 2)
    cv2.putText(panel, "CAM", (cam_pt[0] + 6, cam_pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 120, 255), 1)

    # Tracked objects
    for m in measurements:
        if m.world_xy is None or m.distance_m is None:
            continue
        pt = to_px(m.world_xy[0], m.world_xy[1])
        color = CLASS_COLORS.get(m.obj.cls, DEFAULT_COLOR) if m.in_range else (120, 120, 120)
        cv2.circle(panel, pt, 4, color, -1)
        cv2.putText(
            panel,
            f"#{m.obj.track_id} {m.distance_m:.1f}m",
            (min(map_size - 60, pt[0] + 5), max(12, pt[1] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )

    cv2.rectangle(panel, (0, 0), (map_size - 1, map_size - 1), (100, 100, 100), 1)
    cv2.putText(panel, f"Top-Down (max {max_range_m:.0f}m)", (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

    x_off = fw - map_size - margin
    y_off = margin
    frame[y_off : y_off + map_size, x_off : x_off + map_size] = panel


def _resolve_calibration(args: argparse.Namespace, camera_id: str) -> CalibrationData | None:
    if args.no_calib or args.raw:
        return None
    if args.calibration is not None:
        return CalibrationData.load_json(args.calibration)
    default_path = Path(f"data/calibration/{camera_id}.json")
    if default_path.is_file():
        logger.info(f"Using calibration file: {default_path}")
        return CalibrationData.load_json(default_path)
    logger.warning(f"No calibration found at {default_path}; running without distance estimation.")
    return None


def main() -> int:
    args = parse_args()
    cfg = load_config(default_path=args.config, local_path=args.local_config)

    source_val = args.source if args.source is not None else cfg.camera.source
    if args.conf is not None:
        cfg.detector.conf = args.conf
        logger.info(f"Overriding detector confidence threshold to: {cfg.detector.conf}")

    calib = _resolve_calibration(args, cfg.camera.camera_id)

    logger.info(f"Opening frame source: {source_val}")
    source = create_source(
        source_val,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        fourcc=cfg.camera.fourcc,
        rtsp_transport=cfg.camera.rtsp_transport,
    )
    detector = YoloDetector(cfg.detector)
    pipeline = (
        VisionPipeline(cfg=cfg, calibration=calib, detector=detector)
        if calib is not None
        else None
    )
    merger = RiderMerger(
        min_containment=cfg.merge.min_containment,
        rider_memory_s=cfg.merge.rider_memory_s,
        truncation_margin_px=cfg.geometry.truncation_margin_px,
        default_frame_size=(cfg.camera.width, cfg.camera.height),
    )

    csv_logger = MeasurementCsvLogger(args.log_csv) if args.log_csv else None
    if csv_logger is not None:
        logger.info(f"Logging per-frame measurements to: {args.log_csv}")

    window_name = "Minor Project - Live Pipeline"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps_history = collections.deque(maxlen=30)
    last_loop_t = time.perf_counter()
    frame_count = 0
    video_writer: cv2.VideoWriter | None = None

    logger.info("Starting live processing loop. Press 'q' to quit.")

    try:
        while True:
            frame, t_capture = source.read()
            if frame is None:
                if isinstance(source_val, str) and not source_val.startswith("rtsp"):
                    logger.info("End of file reached.")
                    break
                time.sleep(0.01)
                if not args.no_display:
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                continue

            frame_count += 1
            h, w = frame.shape[:2]

            if args.save_video and video_writer is None:
                save_p = Path(args.save_video)
                save_p.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_fps = getattr(source, "fps", 30.0) if getattr(source, "fps", 0) > 0 else 30.0
                video_writer = cv2.VideoWriter(str(save_p), fourcc, writer_fps, (w, h))
                logger.info(f"Recording annotated video to: {save_p} ({w}x{h} @ {writer_fps:.1f} FPS)")

            if pipeline is not None:
                measurements = pipeline.process_frame(frame, t_capture)
            else:
                detections = detector.track(frame)
                if args.raw:
                    tracked = [
                        TrackedObject(
                            track_id=det.track_id if det.track_id is not None else -1,
                            cls=det.cls,
                            box_xyxy=det.xyxy,
                            conf=det.conf,
                            rider_count=0,
                            truncated=False,
                            t=t_capture,
                        )
                        for det in detections
                    ]
                else:
                    tracked = merger.merge(detections, t_capture=t_capture, frame_size=(w, h))
                measurements = [
                    Measurement(
                        obj=obj,
                        foot_px=((obj.box_xyxy[0] + obj.box_xyxy[2]) * 0.5, obj.box_xyxy[3]),
                        world_xy=None,
                        distance_m=None,
                        in_range=False,
                        proximity="UNKNOWN",
                        speed_mps=None,
                        speed_kmh=None,
                        speed_class=None,
                    )
                    for obj in tracked
                ]

            if csv_logger is not None:
                csv_logger.log(measurements)

            now = time.perf_counter()
            dt = now - last_loop_t
            last_loop_t = now
            if dt > 0:
                fps_history.append(1.0 / dt)
            rolling_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            should_draw = (not args.no_display) or (video_writer is not None)
            if should_draw:
                _draw_measurements(frame, measurements)
                if calib is not None and not args.no_map:
                    _draw_minimap(frame, measurements, calib, cfg.geometry.max_range_m)

                mode_tag = f"cal={calib.camera_id}" if calib is not None else "no-calib"
                status_text = f"FPS: {rolling_fps:.1f} | Objs: {len(measurements)} | {mode_tag} | conf={cfg.detector.conf}"
                cv2.putText(
                    frame,
                    status_text,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                if video_writer is not None:
                    video_writer.write(frame)

            if not args.no_display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed.")
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed.")
                    break
            elif frame_count % 30 == 0:
                logger.info(
                    f"Processed {frame_count} frames | Rolling FPS: {rolling_fps:.1f} | Objects: {len(measurements)}"
                )

            if args.max_frames is not None and frame_count >= args.max_frames:
                logger.info(f"Reached max frames limit ({args.max_frames}).")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        source.close()
        if csv_logger is not None:
            csv_logger.close()
        if video_writer is not None:
            video_writer.release()
            logger.info(f"Annotated video saved successfully to: {args.save_video}")
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
