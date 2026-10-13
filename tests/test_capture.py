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


def test_part_of_the_crop_outside_the_desktop_is_black_not_a_stale_capture():
    """BitBlt returns TRUE for a source rectangle that sticks out of the desktop and leaves those destination pixels
    alone: the reused DIB would show an older capture there."""
    import ctypes
    import sys

    cap = get_capturer(crop_w=100, crop_h=100)
    if not hasattr(cap, "_bgra_view") or sys.platform != "win32":
        pytest.skip("no GDI capture here")
    try:
        u = ctypes.windll.user32
        vx, vy = u.GetSystemMetrics(76), u.GetSystemMetrics(77)
        cap._bgra_view[:] = 200                                  # what an earlier capture left in the DIB
        gray, ox, oy, _, _ = cap.grab(vx + 20, vy + 300)         # the left 30 px of the crop are outside the desktop
        if not cap.ok:
            pytest.skip("the screen cannot be read in this session")
        assert (ox, oy) == (vx - 30, vy + 250)
        assert not gray[:, :30].any()
    finally:
        cap.close()


class _FakeGdi:
    """BitBlt double: answers from `results`; on success it copies the visible part of a fake desktop into the DIB and,
    like GDI, leaves the destination pixels of a source area outside the desktop untouched."""

    def __init__(self, results, cap, desktop=(0, 0, 10_000, 10_000), value=100):
        self.results = list(results)
        self.calls = []
        self.cap, self.desktop, self.value = cap, desktop, value

    def BitBlt(self, dst, dx, dy, w, h, src_dc, ox, oy, rop):
        self.calls.append((dst, dx, dy, w, h, src_dc, ox, oy, rop))
        ok = self.results.pop(0)
        if ok:
            vx, vy, vw, vh = self.desktop
            for row in range(h):
                for col in range(w):
                    if vx <= ox + col < vx + vw and vy <= oy + row < vy + vh:
                        self.cap._bgra_view[row, col, :3] = self.value
        return ok


class _FakeUser:
    def __init__(self, desktop=(0, 0, 10_000, 10_000)):
        self.released = []
        self.taken = 0
        self.desktop = desktop

    def ReleaseDC(self, hwnd, dc):
        self.released.append(dc)
        return 1

    def GetDC(self, hwnd):
        self.taken += 1
        return 99

    def GetSystemMetrics(self, index):
        vx, vy, vw, vh = self.desktop
        return {76: vx, 77: vy, 78: vw, 79: vh}[index]


def _fake_capturer(results, desktop=(0, 0, 10_000, 10_000), size=8):
    from neuropest.capture import GDIScreenCapturer

    cap = GDIScreenCapturer.__new__(GDIScreenCapturer)
    cap.crop_w = cap.crop_h = size
    cap._gdi32, cap._user32 = _FakeGdi(results, cap, desktop), _FakeUser(desktop)
    cap.hwnd_desktop, cap.hdc_screen, cap.hdc_mem = 1, 2, 3
    cap._bgra_view = np.zeros((size, size, 4), np.uint8)
    cap.gray = np.empty((size, size), np.uint8)
    cap.prev_gray = np.zeros((size, size), np.uint8)
    cap.has_prev = False
    cap.ok = True
    cap._good_origin = None
    return cap


def test_failed_blit_retries_once_on_a_fresh_dc_and_then_keeps_the_last_good_pixels_with_their_origin():
    cap = _fake_capturer([True, False, False, False, True])
    gdi, user = cap._gdi32, cap._user32
    _, ox, oy, _, _ = cap.grab(100, 100)
    assert (ox, oy) == (96, 96) and cap.ok
    good = cap.gray.copy()
    assert good.any()                                            # the fake desktop really was copied

    gdi.value = 30                                               # the screen changes, but it cannot be read now
    gray, ox, oy, is_static, diff = cap.grab(200, 200)           # BitBlt fails, the retry on a fresh DC fails too
    assert not cap.ok
    assert (ox, oy) == (96, 96)                                  # the origin of the pixels the DIB still holds
    assert np.array_equal(gray, good)
    assert is_static and diff == 0.0                             # the consumer sees an unchanged scene
    assert user.released == [2] and user.taken == 1              # the stale DC was released, a new one taken ...
    assert gdi.calls[1][5] == 2 and gdi.calls[2][5] == 99        # ... and the retry blits from the NEW one
    assert gdi.calls[1][6:8] == gdi.calls[2][6:8] == (196, 196)  # at the same origin
    assert len(gdi.calls) == 3

    _, ox, oy, _, _ = cap.grab(300, 300)                         # fails once, the retry succeeds
    assert cap.ok and (ox, oy) == (296, 296)
    assert gdi.calls[3][5] == 99 and gdi.calls[4][5] == 99
    assert user.released == [2, 99]
    assert (cap.gray == cap.gray.flat[0]).all() and cap.gray.flat[0] != good.flat[0]    # the new screen content


def test_first_grab_that_fails_reports_its_own_origin_and_a_black_frame():
    cap = _fake_capturer([False, False])
    gray, ox, oy, _, _ = cap.grab(100, 100)
    assert not cap.ok
    assert (ox, oy) == (96, 96)
    assert not gray.any()


@pytest.mark.parametrize("vx, vy", [(0, 0), (-1920, -360)])      # a monitor left of / above the primary
def test_blanking_outside_the_desktop_handles_every_edge_and_a_crop_fully_outside(vx, vy):
    cap = _fake_capturer([], desktop=(vx, vy, 20, 20))

    def blank_case(dx, dy):                                      # crop origin relative to the desktop's top-left
        cap._bgra_view[:] = 7
        cap._blank_outside_desktop(vx + dx, vy + dy)
        return cap._bgra_view[:, :, 0] == 0

    left = blank_case(-3, 5)                # 3 columns outside on the left
    assert left[:, :3].all() and not left[:, 3:].any()
    top = blank_case(5, -2)                 # 2 rows outside at the top
    assert top[:2, :].all() and not top[2:, :].any()
    right = blank_case(16, 5)               # columns 4.. are outside (20 - 16 = 4 inside)
    assert right[:, 4:].all() and not right[:, :4].any()
    bottom = blank_case(5, 15)              # rows 5.. are outside
    assert bottom[5:, :].all() and not bottom[:5, :].any()
    corner = blank_case(-4, -4)             # top-left corner: a 4-px frame on two sides
    assert corner[:4, :].all() and corner[:, :4].all() and not corner[4:, 4:].any()
    assert blank_case(-50, 0).all()         # fully outside: everything black
    assert not blank_case(5, 5).any()       # fully inside: untouched


def test_blanking_reads_width_and_height_the_right_way_round():
    cap = _fake_capturer([], desktop=(0, 0, 30, 10))                # wide and short: swapping the two would show
    cap._bgra_view[:] = 7
    cap._blank_outside_desktop(5, 5)                                # rows 5..12 of a 10 px tall desktop
    blank = cap._bgra_view[:, :, 0] == 0
    assert blank[5:, :].all() and not blank[:5, :].any()


def test_grab_at_the_desktop_edge_blanks_the_stale_strip_and_keeps_the_screen():
    cap = _fake_capturer([True], desktop=(-1920, 0, 1920, 1080))
    cap._bgra_view[:] = 200                                      # an older capture left in the DIB
    gray, ox, oy, _, _ = cap.grab(-1920 + 1, 500)                # origin x = -1923: 3 columns left of the desktop
    assert (ox, oy) == (-1923, 496)
    assert not gray[:, :3].any()                                 # the stale strip is black
    assert (gray[:, 3:] == 100).all()                            # the rest is the (fake) screen, not the old 200


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
