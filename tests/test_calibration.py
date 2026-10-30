"""Unit tests for camera calibration, homography computation, and reprojection errors."""

import numpy as np
import pytest

from vision.calibration import (
    CalibrationData,
    compute_homography,
    leave_one_out_errors,
    pixel_to_world,
    reprojection_errors,
    world_to_pixel,
)


def make_synthetic_camera_homography():
    """Construct an exact ground-to-pixel homography for a synthetic camera.

    Camera mounted at (Xc=1.0, Yc=0.0, Zc=1.8m), pitched down by 35 degrees.
    Sensor resolution: 1280x720, focal length fx=fy=900px, cx=640, cy=360.
    """
    # Camera intrinsics
    K = np.array([
        [900.0, 0.0, 640.0],
        [0.0, 900.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Pitch rotation (around X-axis): camera tilts down towards +Y
    theta = np.radians(35.0)
    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ], dtype=np.float64)

    # Coordinate frame convention: camera looks along +Z_cam, +X_cam right, +Y_cam down.
    # World frame: ground plane Z_world = 0, +X_world right, +Y_world away from camera.
    # Rotation from world to camera
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.sin(theta), -np.cos(theta)],
        [0.0, np.cos(theta), np.sin(theta)],
    ], dtype=np.float64)

    t = -R @ np.array([1.0, 0.0, 1.8], dtype=np.float64)  # camera position (1.0, 0.0, 1.8)

    # Ground plane Z = 0: column 1 (X) and column 2 (Y) of R, plus t
    H_world_to_pix = K @ np.column_stack((R[:, 0], R[:, 1], t))
    # Invert to map pixels to world
    H_pix_to_world = np.linalg.inv(H_world_to_pix)
    # Normalize so H[2, 2] = 1
    H_pix_to_world /= H_pix_to_world[2, 2]
    return H_pix_to_world


def test_p3_a1_synthetic_homography_recovery():
    """P3-A1: Recover known H from 12 ground points; mapping error < 1e-6 m."""
    H_true = make_synthetic_camera_homography()

    # 12 known world points on the ground (X from 0 to 2m, Y from 1 to 4m)
    xs = [0.0, 1.0, 2.0]
    ys = [1.0, 2.0, 3.0, 4.0]
    world_pts = [[x, y] for y in ys for x in xs]

    # Generate exact pixel coordinates from H_true
    pixel_pts = [world_to_pixel(H_true, pt) for pt in world_pts]

    # Recover H from points
    H_recovered = compute_homography(pixel_pts, world_pts)

    # Check mapping error for all points
    for p_pt, w_pt in zip(pixel_pts, world_pts):
        X_est, Y_est = pixel_to_world(H_recovered, p_pt)
        err = np.hypot(X_est - w_pt[0], Y_est - w_pt[1])
        assert err < 1e-6, f"Point {w_pt} error {err} exceeds 1e-6 m"


def test_p3_a2_noise_and_range_dependent_error():
    """P3-A2: Add Gaussian noise (sigma=1.5px); error is reported and grows with range."""
    H_true = make_synthetic_camera_homography()

    # 15 world points spanning Y = 1m to 5m
    xs = [0.0, 1.0, 2.0]
    ys = [1.0, 2.0, 3.0, 4.0, 5.0]
    world_pts = [[x, y] for y in ys for x in xs]

    np.random.seed(42)
    # Generate exact pixels, then add Gaussian noise with sigma = 1.5 px
    pixel_pts = np.array([world_to_pixel(H_true, pt) for pt in world_pts])
    noise = np.random.normal(loc=0.0, scale=1.5, size=pixel_pts.shape)
    noisy_pixel_pts = pixel_pts + noise

    H_noisy = compute_homography(noisy_pixel_pts, world_pts)
    stats = reprojection_errors(H_noisy, noisy_pixel_pts, world_pts)

    mean_err_m = stats["mean_error_m"]
    # Mean error should be small (order of few centimeters)
    assert mean_err_m < 0.10, f"Mean error {mean_err_m} m is too large"

    # Verify that error grows with distance: near points (Y <= 2m) vs far points (Y >= 4m)
    near_errors = [stats["errors_m"][i] for i, pt in enumerate(world_pts) if pt[1] <= 2.0]
    far_errors = [stats["errors_m"][i] for i, pt in enumerate(world_pts) if pt[1] >= 4.0]

    mean_near = np.mean(near_errors)
    mean_far = np.mean(far_errors)
    assert mean_far > mean_near, (
        f"Expected error to grow with distance: near ({mean_near:.4f}m) vs far ({mean_far:.4f}m)"
    )


def test_p3_a3_pixel_to_world_and_world_to_pixel_round_trip():
    """P3-A3: Round-trip pixel -> world -> pixel reproduces original values within 1e-6."""
    H = make_synthetic_camera_homography()

    test_pixels = [
        (640.0, 480.0),
        (300.0, 550.0),
        (900.0, 400.0),
        (500.0, 600.0),
    ]

    for (u, v) in test_pixels:
        X, Y = pixel_to_world(H, (u, v))
        u_back, v_back = world_to_pixel(H, (X, Y))
        assert pytest.approx(u_back, abs=1e-6) == u
        assert pytest.approx(v_back, abs=1e-6) == v


def test_p3_a4_fewer_than_4_points_raises_clear_error():
    """P3-A4: Fewer than 4 points raises ValueError."""
    pts_3 = [[100.0, 200.0], [200.0, 300.0], [300.0, 400.0]]
    w_3 = [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]

    with pytest.raises(ValueError, match="At least 4 correspondence points are required"):
        compute_homography(pts_3, w_3)


def test_p3_a5_collinear_points_raises_error():
    """P3-A5: Collinear points raise ValueError."""
    # 5 points all in a straight vertical line in both pixels and world
    pix_collinear = [[100.0, y] for y in [100.0, 200.0, 300.0, 400.0, 500.0]]
    world_collinear = [[0.0, y] for y in [1.0, 2.0, 3.0, 4.0, 5.0]]

    with pytest.raises(ValueError, match="collinear"):
        compute_homography(pix_collinear, world_collinear)


def test_p3_a6_json_save_load_and_resolution_guard(tmp_path):
    """P3-A6: Save -> Load preserves H exactly; loading with different frame size is refused."""
    H = make_synthetic_camera_homography()
    calib = CalibrationData(
        camera_id="laptop_webcam",
        image_width=1280,
        image_height=720,
        H=H.tolist(),
        pixel_pts=[[640.0, 480.0], [320.0, 400.0], [960.0, 400.0], [640.0, 600.0]],
        world_pts=[[1.0, 2.0], [0.0, 3.0], [2.0, 3.0], [1.0, 1.5]],
        camera_xy=[1.0, 0.0],
        camera_height_m=1.8,
        notes="Synthetic test calibration",
    )

    json_file = tmp_path / "laptop_webcam.json"
    calib.save_json(json_file)

    # 1. Load without check
    loaded = CalibrationData.load_json(json_file)
    assert loaded.camera_id == "laptop_webcam"
    assert loaded.image_width == 1280
    assert loaded.image_height == 720
    np.testing.assert_allclose(loaded.homography, H, rtol=1e-12, atol=1e-12)

    # 2. Load with matching resolution -> succeeds
    loaded_match = CalibrationData.load_json(json_file, verify_frame_size=(1280, 720))
    assert loaded_match.camera_id == "laptop_webcam"

    # 3. Load with mismatched resolution -> raises ValueError
    with pytest.raises(ValueError, match="Calibration frame size mismatch"):
        CalibrationData.load_json(json_file, verify_frame_size=(640, 480))


def test_leave_one_out_errors():
    """Verify leave-one-out cross validation function."""
    H_true = make_synthetic_camera_homography()
    xs = [0.0, 1.0, 2.0]
    ys = [1.0, 2.0, 3.0, 4.0]
    world_pts = [[x, y] for y in ys for x in xs]
    pixel_pts = [world_to_pixel(H_true, pt) for pt in world_pts]

    res = leave_one_out_errors(pixel_pts, world_pts)
    assert len(res["loo_errors_m"]) == 12
    # In noise-free case, leave-one-out error should be virtually zero
    assert res["mean_loo_error_m"] < 1e-6
