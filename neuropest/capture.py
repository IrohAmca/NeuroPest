"""Windows GDI-based high-performance screen capture for NeuroPest visual input.

Captures a region (default 480x480 px) around the fly's position on the desktop,
converts it to uint8 grayscale, detects static scenes, and runs in a separate
process to avoid stalling the GUI or simulation engine.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import multiprocessing as mp
import sys
import time
from typing import Tuple

import numba as nb
import numpy as np

CROP_W = 480
CROP_H = 480
FPS_ACTIVE = 8.0
FPS_STATIC = 2.0
DIFF_THRESHOLD = 1.0     # mean absolute pixel difference in [0, 255] below which scene is static

# Slots in the shared metadata double array
(
    META_STAMP,          # time.perf_counter() when frame was captured
    META_ORIGIN_X,       # crop top-left X in screen coordinates
    META_ORIGIN_Y,       # crop top-left Y in screen coordinates
    META_IS_STATIC,      # 1.0 if screen is static, 0.0 if scene changed
    META_SEQ,            # monotonically increasing sequence counter
    META_TARGET_X,       # input: fly X center in screen coordinates
    META_TARGET_Y,       # input: fly Y center in screen coordinates
    META_ENABLED,        # input: 1.0 if capture should run, 0.0 to pause
) = range(8)
META_SIZE = 8


@nb.njit(fastmath=True)
def bgra_to_gray(bgra: np.ndarray, out: np.ndarray) -> None:
    """Fast conversion from BGRA uint8 to Grayscale uint8 using integer weights."""
    h, w, _ = bgra.shape
    for y in range(h):
        for x in range(w):
            b = bgra[y, x, 0]
            g = bgra[y, x, 1]
            r = bgra[y, x, 2]
            out[y, x] = (r * 77 + g * 150 + b * 29) >> 8


@nb.njit(fastmath=True)
def calc_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-allocation mean absolute pixel difference between two uint8 images."""
    h, w = a.shape
    total = 0.0
    for y in range(h):
        for x in range(w):
            d = int(a[y, x]) - int(b[y, x])
            total += -d if d < 0 else d
    return total / (h * w)


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class GDIScreenCapturer:
    """Direct Windows GDI BitBlt capturer with pre-allocated memory DIBSection."""

    def __init__(self, crop_w: int = CROP_W, crop_h: int = CROP_H):
        if sys.platform != "win32":
            raise RuntimeError("GDIScreenCapturer is only available on Windows")

        self.crop_w = crop_w
        self.crop_h = crop_h
        self._user32 = ctypes.windll.user32
        self._gdi32 = ctypes.windll.gdi32

        # Set per-monitor DPI awareness if not already set
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                self._user32.SetProcessDPIAware()
            except Exception:
                pass

        # Switch to interactive desktop if running in an isolated desktop
        try:
            hdesk = self._user32.OpenDesktopW("Default", 0, False, 0x01FF)
            if hdesk:
                self._user32.SetThreadDesktop(hdesk)
        except Exception:
            pass

        # Explicit 64-bit handle types
        HANDLE = ctypes.c_void_p
        self._user32.GetDesktopWindow.restype = HANDLE
        self._user32.GetDC.restype = HANDLE
        self._user32.GetDC.argtypes = [HANDLE]
        self._user32.ReleaseDC.restype = ctypes.c_int
        self._user32.ReleaseDC.argtypes = [HANDLE, HANDLE]

        self._gdi32.CreateCompatibleDC.restype = HANDLE
        self._gdi32.CreateCompatibleDC.argtypes = [HANDLE]
        self._gdi32.DeleteDC.restype = wintypes.BOOL
        self._gdi32.DeleteDC.argtypes = [HANDLE]
        self._gdi32.CreateDIBSection.restype = HANDLE
        self._gdi32.CreateDIBSection.argtypes = [
            HANDLE, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p), HANDLE, wintypes.DWORD
        ]
        self._gdi32.SelectObject.restype = HANDLE
        self._gdi32.SelectObject.argtypes = [HANDLE, HANDLE]
        self._gdi32.DeleteObject.restype = wintypes.BOOL
        self._gdi32.DeleteObject.argtypes = [HANDLE]
        self._gdi32.BitBlt.restype = wintypes.BOOL
        self._gdi32.BitBlt.argtypes = [
            HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            HANDLE, ctypes.c_int, ctypes.c_int, wintypes.DWORD
        ]

        self.hwnd_desktop = self._user32.GetDesktopWindow()
        self.hdc_screen = self._user32.GetDC(self.hwnd_desktop)
        self.hdc_mem = self._gdi32.CreateCompatibleDC(self.hdc_screen)

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = self.crop_w
        bmi.biHeight = -self.crop_h       # top-down DIB
        bmi.biPlanes = 1
        bmi.biBitCount = 32

        self.p_bits = ctypes.c_void_p()
        self.hbm = self._gdi32.CreateDIBSection(
            self.hdc_screen, ctypes.byref(bmi), 0, ctypes.byref(self.p_bits), None, 0
        )
        if not self.hdc_screen or not self.hdc_mem or not self.hbm or not self.p_bits.value:
            raise RuntimeError("Failed to initialize GDI screen capture context")
        self.old_bm = self._gdi32.SelectObject(self.hdc_mem, self.hbm)

        self._bgra_view = np.ctypeslib.as_array(
            ctypes.cast(self.p_bits, ctypes.POINTER(ctypes.c_uint8)),
            shape=(self.crop_h, self.crop_w, 4)
        )
        self.gray = np.empty((self.crop_h, self.crop_w), dtype=np.uint8)
        self.prev_gray = np.zeros((self.crop_h, self.crop_w), dtype=np.uint8)
        self.has_prev = False

    def grab(self, cx: float, cy: float) -> Tuple[np.ndarray, int, int, bool, float]:
        """Grabs the screen centered at (cx, cy).

        Returns:
            (gray_image, origin_x, origin_y, is_static, diff)
        """
        origin_x = int(cx - self.crop_w // 2)
        origin_y = int(cy - self.crop_h // 2)

        # BitBlt 0x00CC0020 = SRCCOPY
        res = self._gdi32.BitBlt(
            self.hdc_mem, 0, 0, self.crop_w, self.crop_h,
            self.hdc_screen, origin_x, origin_y, 0x00CC0020
        )
        if not res:
            # Refresh screen DC if invalid handle
            self._user32.ReleaseDC(self.hwnd_desktop, self.hdc_screen)
            self.hdc_screen = self._user32.GetDC(self.hwnd_desktop)

        bgra_to_gray(self._bgra_view, self.gray)

        if not self.has_prev:
            diff = 100.0
            is_static = False
            self.prev_gray[:] = self.gray
            self.has_prev = True
        else:
            diff = float(calc_diff(self.gray, self.prev_gray))
            is_static = diff < DIFF_THRESHOLD
            if not is_static:
                self.prev_gray[:] = self.gray

        return self.gray, origin_x, origin_y, is_static, diff

    def close(self) -> None:
        """Frees GDI allocations."""
        if hasattr(self, "_gdi32") and self.old_bm:
            self._gdi32.SelectObject(self.hdc_mem, self.old_bm)
            self._gdi32.DeleteObject(self.hbm)
            self._gdi32.DeleteDC(self.hdc_mem)
            self._user32.ReleaseDC(self.hwnd_desktop, self.hdc_screen)
            self.old_bm = None


class FallbackScreenCapturer:
    """Mock/Fallback capturer for non-Windows platforms or test suites."""

    def __init__(self, crop_w: int = CROP_W, crop_h: int = CROP_H):
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.gray = np.full((crop_h, crop_w), 128, dtype=np.uint8)

    def grab(self, cx: float, cy: float) -> Tuple[np.ndarray, int, int, bool, float]:
        origin_x = int(cx - self.crop_w // 2)
        origin_y = int(cy - self.crop_h // 2)
        return self.gray, origin_x, origin_y, True, 0.0

    def close(self) -> None:
        pass


def get_capturer(crop_w: int = CROP_W, crop_h: int = CROP_H):
    """Returns GDIScreenCapturer on Windows or FallbackScreenCapturer elsewhere."""
    if sys.platform == "win32":
        try:
            return GDIScreenCapturer(crop_w, crop_h)
        except Exception:
            return FallbackScreenCapturer(crop_w, crop_h)
    return FallbackScreenCapturer(crop_w, crop_h)


def capture_worker_loop(frame_raw, meta_raw, stop_event: mp.Event, crop_w: int = CROP_W, crop_h: int = CROP_H) -> None:
    """Main function executed in the separate capture worker process."""
    try:
        capturer = get_capturer(crop_w, crop_h)
        frame_np = np.frombuffer(frame_raw, dtype=np.uint8).reshape((crop_h, crop_w))

        while not stop_event.is_set():
            enabled = meta_raw[META_ENABLED] > 0.5
            if not enabled:
                # Sleep briefly when visual input is disabled
                if stop_event.wait(0.1):
                    break
                continue

            cx = meta_raw[META_TARGET_X]
            cy = meta_raw[META_TARGET_Y]

            t0 = time.perf_counter()
            gray, ox, oy, is_static, _ = capturer.grab(cx, cy)
            t_captured = time.perf_counter()

            # Copy frame into shared array
            frame_np[:] = gray

            # Update shared metadata
            meta_raw[META_STAMP] = t_captured
            meta_raw[META_ORIGIN_X] = float(ox)
            meta_raw[META_ORIGIN_Y] = float(oy)
            meta_raw[META_IS_STATIC] = 1.0 if is_static else 0.0
            meta_raw[META_SEQ] += 1.0

            # Dynamic pacing: 8 fps when active/moving, 2 fps when static
            target_fps = FPS_STATIC if is_static else FPS_ACTIVE
            target_period = 1.0 / target_fps
            spent = time.perf_counter() - t0
            sleep_time = max(0.001, target_period - spent)

            if stop_event.wait(sleep_time):
                break
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if "capturer" in locals():
            capturer.close()


class CaptureProcess:
    """Manages the lifecycle of the standalone capture subprocess and its shared memory."""

    def __init__(self, ctx=None, crop_w: int = CROP_W, crop_h: int = CROP_H):
        self._ctx = ctx or mp.get_context("spawn")
        self.crop_w = crop_w
        self.crop_h = crop_h

        # Shared memory buffers
        self.frame_raw = self._ctx.Array("B", crop_w * crop_h, lock=False)
        self.meta_raw = self._ctx.Array("d", META_SIZE, lock=False)
        self.stop_event = self._ctx.Event()
        self._proc = None

    def start(self) -> None:
        """Starts the capture subprocess."""
        if self._proc is not None and self._proc.is_alive():
            return
        self.stop_event.clear()
        self._proc = self._ctx.Process(
            target=capture_worker_loop,
            args=(self.frame_raw, self.meta_raw, self.stop_event, self.crop_w, self.crop_h),
            daemon=True,
            name="neuropest-capture",
        )
        self._proc.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Signals the subprocess to stop and joins."""
        if self._proc is None:
            return
        self.stop_event.set()
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None

    def update_target(self, x: float, y: float, enabled: bool) -> None:
        """GUI/Runner calls this to update the center of interest and enable switch."""
        self.meta_raw[META_TARGET_X] = float(x)
        self.meta_raw[META_TARGET_Y] = float(y)
        self.meta_raw[META_ENABLED] = 1.0 if enabled else 0.0

    def get_latest_frame(self) -> Tuple[np.ndarray, float, float, float, bool, int]:
        """Returns (frame_copy, stamp, origin_x, origin_y, is_static, seq)."""
        seq = int(self.meta_raw[META_SEQ])
        stamp = self.meta_raw[META_STAMP]
        ox = self.meta_raw[META_ORIGIN_X]
        oy = self.meta_raw[META_ORIGIN_Y]
        is_static = self.meta_raw[META_IS_STATIC] > 0.5
        frame = np.frombuffer(self.frame_raw, dtype=np.uint8).reshape((self.crop_h, self.crop_w)).copy()
        return frame, stamp, ox, oy, is_static, seq
