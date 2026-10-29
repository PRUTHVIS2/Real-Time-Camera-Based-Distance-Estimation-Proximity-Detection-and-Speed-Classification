"""Unit tests for frame capture sources (Webcam, Rtsp, File)."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.capture import FileSource, RtspSource, create_source


@pytest.fixture
def sample_video_file(tmp_path: Path) -> Path:
    """Create a temporary 10-frame synthetic test video file."""
    video_path = tmp_path / "synthetic_test.mp4"
    w, h = 320, 240
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for i in range(10):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (i * 20, 100, 150)
        writer.write(frame)
    writer.release()
    return video_path


def test_file_source_order_and_strictly_increasing_timestamps(sample_video_file: Path):
    """P1-A1: FileSource returns frames in order with strictly increasing timestamps."""
    source = FileSource(sample_video_file)
    assert source.fps == 30.0

    frames_read = 0
    timestamps = []

    while True:
        frame, t = source.read()
        if frame is None:
            break
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)
        timestamps.append(t)
        frames_read += 1

    source.close()

    assert frames_read == 10
    assert len(timestamps) == 10

    # Assert strictly increasing timestamps
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1], f"Timestamps must increase: {timestamps[i]} <= {timestamps[i-1]}"


def test_rtsp_source_bad_url_does_not_crash():
    """P1-A3: RtspSource with an invalid URL does not crash, reports is_connected=False, and retries."""
    bad_url = "rtsp://127.0.0.1:65432/bad_stream"
    source = RtspSource(url=bad_url, retry_interval_s=0.2)

    try:
        # Give reader thread time to attempt initial connection
        time.sleep(0.3)
        assert not source.is_connected

        # read() should return (None, t) safely
        frame, t = source.read()
        assert frame is None
        assert isinstance(t, float)

        # Allow another retry cycle
        time.sleep(0.3)
        assert not source.is_connected
    finally:
        source.close()
        assert not source.is_connected


def test_create_source_factory(sample_video_file: Path):
    """Test factory function selection for webcam, rtsp, and file."""
    s_file = create_source(str(sample_video_file))
    assert isinstance(s_file, FileSource)
    s_file.close()

    s_rtsp = create_source("rtsp://localhost:8554/live")
    assert isinstance(s_rtsp, RtspSource)
    s_rtsp.close()
