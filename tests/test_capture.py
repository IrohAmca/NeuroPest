import numpy as np
import pytest

from neuropest.capture import (
    CROP_H,
    CROP_W,
    DIFF_THRESHOLD,
    CaptureProcess,
    bgra_to_gray,
    get_capturer,
)
from neuropest.vision import image_scene


def test_bgra_to_gray():
    # Test known color conversion
    # Pure red: (0, 0, 255, 255) -> 255 * 77 >> 8 = 76
    # Pure green: (0, 255, 0, 255) -> 255 * 150 >> 8 = 149
    # Pure blue: (255, 0, 0, 255) -> 255 * 29 >> 8 = 28
    bgra = np.zeros((2, 2, 4), dtype=np.uint8)
    bgra[0, 0] = [0, 0, 255, 255]    # Red
    bgra[0, 1] = [0, 255, 0, 255]    # Green
    bgra[1, 0] = [255, 0, 0, 255]    # Blue
    bgra[1, 1] = [255, 255, 255, 255] # White

    out = np.empty((2, 2), dtype=np.uint8)
    bgra_to_gray(bgra, out)

    assert abs(int(out[0, 0]) - 76) <= 1
    assert abs(int(out[0, 1]) - 149) <= 1
    assert abs(int(out[1, 0]) - 28) <= 1
    assert abs(int(out[1, 1]) - 254) <= 1


def test_capturer_grab():
    cap = get_capturer(crop_w=100, crop_h=100)
    gray, ox, oy, is_static, diff = cap.grab(500, 500)
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8
    assert ox == 500 - 50
    assert oy == 500 - 50
    # First frame has no previous, so is_static should be False
    assert is_static is False

    # Second grab in the same place without change
    gray2, _, _, is_static2, diff2 = cap.grab(500, 500)
    assert gray2.shape == (100, 100)
    cap.close()


def test_image_scene_with_origin():
    # 10x10 test image centered at origin (100, 200)
    gray = np.full((10, 10), 200, dtype=np.uint8)
    gray[5, 5] = 50 # dark spot in center

    scene = image_scene(gray, origin=(100.0, 200.0), outside=0.5)

    # Point outside image
    val_outside = scene(np.array([50.0]), np.array([50.0]))
    assert val_outside[0] == 0.5

    # Point at center of image (100 + 5, 200 + 5) = (105, 205)
    val_center = scene(np.array([105.0]), np.array([205.0]))
    assert np.isclose(val_center[0], 50.0 / 255.0, atol=0.05)


def test_capture_process_lifecycle():
    import time
    cp = CaptureProcess(crop_w=100, crop_h=100)
    try:
        cp.start()
        cp.update_target(300, 300, enabled=True)

        # Wait for frame (spawn + numba JIT on Windows takes up to ~10-15s)
        seq = 0
        for _ in range(200):
            time.sleep(0.1)
            frame, stamp, ox, oy, is_static, seq = cp.get_latest_frame()
            if seq > 0:
                break

        assert seq > 0
        assert frame.shape == (100, 100)
        assert ox == 300 - 50
        assert oy == 300 - 50
    finally:
        cp.stop()
