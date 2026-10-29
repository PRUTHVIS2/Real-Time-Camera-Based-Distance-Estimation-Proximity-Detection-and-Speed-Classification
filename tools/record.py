"""Video recorder with per-frame capture timestamps.

Records raw video stream from webcam or RTSP feed and logs a timestamp CSV
with exact time.monotonic() values for every frame captured.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import cv2

from vision.capture import create_source
from vision.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("record")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record video feed with per-frame timestamps.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Camera index or stream URL. Defaults to config camera.source.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/recordings",
        help="Directory where video and CSV files will be saved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="clip",
        help="Prefix for output file names.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Recording duration in seconds. Omit to record until 'q' is pressed.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV preview window during recording.",
    )
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to default config.")
    parser.add_argument("--local-config", type=str, default="config/local.yaml", help="Path to local config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(default_path=args.config, local_path=args.local_config)
    source_val = args.source if args.source is not None else cfg.camera.source

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.prefix}_{timestamp_str}"
    video_path = output_dir / f"{base_name}.mp4"
    csv_path = output_dir / f"{base_name}_timestamps.csv"

    source = create_source(
        source_val,
        width=cfg.camera.width,
        height=cfg.camera.height,
        rtsp_transport=cfg.camera.rtsp_transport,
    )

    logger.info(f"Opening source: {source_val}")
    logger.info("Waiting for first frame to initialize video writer...")

    # Wait for first valid frame
    frame = None
    t_start = time.monotonic()
    while frame is None and (time.monotonic() - t_start < 10.0):
        frame, _ = source.read()
        if frame is None:
            time.sleep(0.05)

    if frame is None:
        logger.error("Failed to read any frames from source. Aborting recording.")
        source.close()
        return 1

    h, w = frame.shape[:2]
    fps = 30.0  # nominal recording FPS container rate
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        logger.error(f"Failed to create video writer for: {video_path}")
        source.close()
        return 1

    window_name = "Minor Project - Recording (Press 'q' to stop)"
    if not args.no_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx", "t_capture"])

    frame_idx = 0
    record_start_t = time.monotonic()

    logger.info(f"Recording started. Saving to {video_path} and {csv_path}")

    try:
        while True:
            frame, t_capture = source.read()
            if frame is None:
                time.sleep(0.01)
                continue

            writer.write(frame)
            csv_writer.writerow([frame_idx, f"{t_capture:.6f}"])
            frame_idx += 1

            if not args.no_preview:
                preview = frame.copy()
                elapsed = time.monotonic() - record_start_t
                cv2.putText(
                    preview,
                    f"REC [{frame_idx} frames, {elapsed:.1f}s]",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Stop requested by user.")
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Preview window closed.")
                    break

            if args.duration is not None and (time.monotonic() - record_start_t) >= args.duration:
                logger.info(f"Target duration of {args.duration}s reached.")
                break

    except KeyboardInterrupt:
        logger.info("Recording interrupted by Ctrl+C.")
    finally:
        source.close()
        writer.release()
        csv_file.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

    total_time = time.monotonic() - record_start_t
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    logger.info(
        f"Recording finished: {frame_idx} frames in {total_time:.2f}s ({avg_fps:.1f} FPS)\n"
        f"  Video: {video_path}\n"
        f"  Timestamps: {csv_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
