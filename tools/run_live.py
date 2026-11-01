"""Live camera tracking and rider-merge visualization tool.

Runs ByteTrack tracking and rider-to-bike merging on live camera stream or video file,
displaying persistent IDs, merged two-wheeler boxes with rider counts, edge truncation tags,
and rolling FPS.
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
from vision.merge import RiderMerger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_live")

# Class colors in BGR
CLASS_COLORS = {
    "person": (0, 255, 0),        # Green
    "two_wheeler": (255, 255, 0),  # Cyan
    "car": (0, 200, 255),         # Yellow-Orange
}
DEFAULT_COLOR = (200, 200, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live camera tracking and rider-merge viewer.")
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
        help="Detection confidence threshold override (e.g. 0.25 vs 0.10 for ID stability tests).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Display raw unmerged detections without rider merging.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Path to output .mp4 file to record the video with bounding boxes and track IDs burned in.",
    )
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
        logger.info(f"Overriding detector confidence threshold to: {cfg.detector.conf}")

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
    merger = RiderMerger(
        min_containment=cfg.merge.min_containment,
        rider_memory_s=cfg.merge.rider_memory_s,
        truncation_margin_px=cfg.geometry.truncation_margin_px,
        default_frame_size=(cfg.camera.width, cfg.camera.height),
    )

    window_name = "Minor Project - Tracking & Rider Merge"
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
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                continue

            frame_count += 1
            h, w = frame.shape[:2]

            # Initialize video writer if --save-video is requested
            if args.save_video and video_writer is None:
                from pathlib import Path
                save_p = Path(args.save_video)
                save_p.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_fps = getattr(source, "fps", 30.0) if getattr(source, "fps", 0) > 0 else 30.0
                video_writer = cv2.VideoWriter(str(save_p), fourcc, writer_fps, (w, h))
                logger.info(f"Recording annotated video to: {save_p} ({w}x{h} @ {writer_fps:.1f} FPS)")

            # 1. Run detection with ByteTrack tracking
            detections = detector.track(frame)

            # 2. Run rider-two-wheeler merging
            if args.raw:
                # Raw unmerged visualization
                tracked_objects = []
                for det in detections:
                    from vision.types import TrackedObject
                    t_id = det.track_id if det.track_id is not None else -1
                    tracked_objects.append(
                        TrackedObject(
                            track_id=t_id,
                            cls=det.cls,
                            box_xyxy=det.xyxy,
                            conf=det.conf,
                            rider_count=0,
                            truncated=False,
                            t=t_capture,
                        )
                    )
            else:
                tracked_objects = merger.merge(detections, t_capture=t_capture, frame_size=(w, h))

            # Compute rolling FPS
            now = time.perf_counter()
            dt = now - last_loop_t
            last_loop_t = now
            if dt > 0:
                fps_history.append(1.0 / dt)
            rolling_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            should_draw = (not args.no_display) or (video_writer is not None)

            if should_draw:
                for obj in tracked_objects:
                    x1, y1, x2, y2 = (int(v) for v in obj.box_xyxy)
                    color = CLASS_COLORS.get(obj.cls, DEFAULT_COLOR)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label: #ID class conf [rider(s): n] [EDGE]
                    label_parts = [f"#{obj.track_id} {obj.cls} {obj.conf:.2f}"]
                    if obj.cls == "two_wheeler" and obj.rider_count > 0:
                        label_parts.append(f"riders:{obj.rider_count}")
                    if obj.truncated:
                        label_parts.append("[EDGE]")
                    label = " ".join(label_parts)

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

                # Draw top overlay banner
                status_text = f"FPS: {rolling_fps:.1f} | Objs: {len(tracked_objects)} | conf={cfg.detector.conf}"
                cv2.putText(
                    frame,
                    status_text,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Write annotated frame to video file if requested
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
            else:
                if frame_count % 30 == 0:
                    logger.info(
                        f"Processed {frame_count} frames | Rolling FPS: {rolling_fps:.1f} | Objects: {len(tracked_objects)}"
                    )

            if args.max_frames is not None and frame_count >= args.max_frames:
                logger.info(f"Reached max frames limit ({args.max_frames}).")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        source.close()
        if video_writer is not None:
            video_writer.release()
            logger.info(f"Annotated video saved successfully to: {args.save_video}")
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
