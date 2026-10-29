"""Live camera detection tool.

Displays live video stream from webcam, RTSP stream, or test video file,
running YOLO11 detection with overlaid bounding boxes, class labels,
confidence scores, and rolling average FPS.
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
import time

import cv2

from vision.capture import create_source
from vision.config import load_config
from vision.detector import YoloDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_live")

# Colors for bounding boxes (BGR)
CLASS_COLORS = {
    "person": (0, 255, 0),        # Green
    "bicycle": (255, 255, 0),     # Cyan
    "car": (0, 200, 255),         # Yellow-Orange
    "motorcycle": (0, 140, 255),  # Orange
}
DEFAULT_COLOR = (200, 200, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live camera detection viewer.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Camera index (e.g. 0), RTSP URL (rtsp://...), or video file path. Defaults to config.",
    )
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold override.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML.")
    parser.add_argument("--local-config", type=str, default="config/local.yaml", help="Path to local config YAML.")
    parser.add_argument("--no-display", action="store_true", help="Run without opening an OpenCV GUI window.")
    parser.add_argument("--max-frames", type=int, default=None, help="Exit after processing this many frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(default_path=args.config, local_path=args.local_config)

    source_val = args.source if args.source is not None else cfg.camera.source
    if args.conf is not None:
        cfg.detector.conf = args.conf

    logger.info(f"Opening frame source: {source_val}")
    source = create_source(
        source_val,
        width=cfg.camera.width,
        height=cfg.camera.height,
        rtsp_transport=cfg.camera.rtsp_transport,
    )
    detector = YoloDetector(cfg.detector)

    window_name = "Minor Project - Live Detection"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps_history = collections.deque(maxlen=30)
    last_loop_t = time.perf_counter()
    frame_count = 0

    logger.info("Starting live processing loop. Press 'q' to quit.")

    try:
        while True:
            frame, t_capture = source.read()
            if frame is None:
                # If reading from file and reached EOF, terminate
                if isinstance(source_val, str) and not source_val.startswith("rtsp"):
                    logger.info("End of file reached.")
                    break
                # Idle briefly if stream is waiting/reconnecting
                time.sleep(0.01)
                if not args.no_display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                continue

            frame_count += 1

            # Run detection
            detections = detector.detect(frame)

            # Compute rolling FPS
            now = time.perf_counter()
            dt = now - last_loop_t
            last_loop_t = now
            if dt > 0:
                fps_history.append(1.0 / dt)
            rolling_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            if not args.no_display:
                # Draw detections
                for det in detections:
                    x1, y1, x2, y2 = (int(v) for v in det.xyxy)
                    color = CLASS_COLORS.get(det.cls, DEFAULT_COLOR)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{det.cls} {det.conf:.2f}"
                    (txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        frame,
                        (x1, max(0, y1 - txt_h - 4)),
                        (x1 + txt_w, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(txt_h, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                # Draw FPS overlay
                fps_text = f"FPS: {rolling_fps:.1f} ({len(detections)} objs)"
                cv2.putText(
                    frame,
                    fps_text,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed.")
                    break

                # Handle window close (X button)
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed.")
                    break
            else:
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames | Rolling FPS: {rolling_fps:.1f} | Detections: {len(detections)}")

            if args.max_frames is not None and frame_count >= args.max_frames:
                logger.info(f"Reached max frames limit ({args.max_frames}).")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        source.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
