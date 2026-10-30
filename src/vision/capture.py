"""Video frame capture sources with precise timestamps.

Provides unified FrameSource interface and implementations for:
- WebcamSource: direct OpenCV capture from local webcam.
- RtspSource: threaded capture maintaining only the latest frame, with auto-reconnect.
- FileSource: sequential video file reader with synthetic monotonically increasing timestamps.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """Abstract base class for all frame capture sources."""

    @abstractmethod
    def read(self) -> Tuple[np.ndarray | None, float]:
        """Read the next available frame.

        Returns:
            Tuple of (frame, t_capture) where frame is a BGR ndarray (or None if unavailable/EOF),
            and t_capture is the timestamp (seconds) when the frame was read from the device.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release underlying camera or stream resources."""
        raise NotImplementedError

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class WebcamSource(FrameSource):
    """Local USB or integrated webcam capture source."""

    def __init__(
        self,
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        fourcc: str | None = "MJPG",
    ) -> None:
        self.index = index
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            logger.warning(f"Could not open webcam index {index}")

        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width is not None and width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None and height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None and fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))

        # Read back what driver actually granted
        if self.cap.isOpened():
            rw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            rh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rfps = self.cap.get(cv2.CAP_PROP_FPS)
            rfcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            rfcc = "".join([chr((rfcc_int >> (8 * i)) & 0xFF) for i in range(4)])
            logger.info(
                f"Webcam {index} opened: requested ({width}x{height} @ {fps}fps, {fourcc}), "
                f"driver granted ({rw}x{rh} @ {rfps:.1f}fps, {rfcc})"
            )

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def read(self) -> Tuple[np.ndarray | None, float]:
        if not self.cap.isOpened():
            return None, time.monotonic()
        ret, frame = self.cap.read()
        t_capture = time.monotonic()
        if not ret or frame is None:
            return None, t_capture
        return frame, t_capture

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
            logger.info(f"Webcam {self.index} released")


class RtspSource(FrameSource):
    """Threaded RTSP stream reader that maintains only the freshest frame and auto-reconnects."""

    def __init__(self, url: str, retry_interval_s: float = 2.0, transport: str = "tcp") -> None:
        self.url = url
        self.retry_interval_s = retry_interval_s
        self.transport = transport

        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_t: float = 0.0
        self._is_connected = False
        self._stop_event = threading.Event()

        # Set transport option in environment
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.transport}"

        self._thread = threading.Thread(target=self._capture_worker, name="RtspSourceReader", daemon=True)
        self._thread.start()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def _open_capture(self) -> cv2.VideoCapture | None:
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                # Attempt to set buffer size to 1 to reduce stream latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        except Exception as e:
            logger.debug(f"Exception opening RTSP stream: {e}")
        return None

    def _capture_worker(self) -> None:
        logger.info(f"Starting RTSP capture worker for: {self.url}")
        cap: cv2.VideoCapture | None = None

        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                self._is_connected = False
                cap = self._open_capture()
                if cap is None or not cap.isOpened():
                    self._stop_event.wait(self.retry_interval_s)
                    continue
                self._is_connected = True
                logger.info("RTSP stream connected")

            ret, frame = cap.read()
            t_capture = time.monotonic()

            if not ret or frame is None:
                logger.warning("RTSP read failed, reconnecting...")
                self._is_connected = False
                if cap is not None:
                    cap.release()
                    cap = None
                self._stop_event.wait(self.retry_interval_s)
                continue

            with self._lock:
                self._latest_frame = frame
                self._latest_t = t_capture

        if cap is not None and cap.isOpened():
            cap.release()
        self._is_connected = False
        logger.info("RTSP capture worker stopped")

    def read(self) -> Tuple[np.ndarray | None, float]:
        with self._lock:
            if self._latest_frame is None:
                return None, time.monotonic()
            return self._latest_frame.copy(), self._latest_t

    def close(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        with self._lock:
            self._latest_frame = None
        self._is_connected = False
        logger.info("RtspSource closed")


class FileSource(FrameSource):
    """Sequential frame reader for recorded video files, using video frame timing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps > 0 else 30.0
        self.frame_idx = 0

    def read(self) -> Tuple[np.ndarray | None, float]:
        if not self.cap.isOpened():
            return None, self.frame_idx / self.fps

        ret, frame = self.cap.read()
        t_capture = self.frame_idx / self.fps
        if not ret or frame is None:
            return None, t_capture

        self.frame_idx += 1
        return frame, t_capture

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
            logger.info(f"FileSource closed: {self.path}")


def create_source(
    source: int | str,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    fourcc: str | None = "MJPG",
    rtsp_transport: str = "tcp",
) -> FrameSource:
    """Factory function to instantiate the appropriate FrameSource.

    Args:
        source: Integer device index, RTSP URL ('rtsp://...'), or path to video file.
        width: Desired width for webcam capture.
        height: Desired height for webcam capture.
        fps: Desired FPS for webcam capture.
        fourcc: Desired fourcc codec (e.g. 'MJPG') for webcam capture.
        rtsp_transport: Transport protocol for RTSP ('tcp' or 'udp').

    Returns:
        Configured FrameSource instance.
    """
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return WebcamSource(index=int(source), width=width, height=height, fps=fps, fourcc=fourcc)
    if isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("rtsps://") or source.startswith("http://")):
        return RtspSource(url=source, transport=rtsp_transport)
    return FileSource(path=source)
