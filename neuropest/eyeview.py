"""Fly's-eye preview: what the visual front end hands to the circuit, drawn as pictures.

Light module (numpy only; the GUI process imports it without numba). The engine worker publishes one snapshot per
vision frame into a shared float32 array while the preview is switched on (`write_snapshot`); the GUI copies it out
(`read_snapshot`) and turns it into RGB images (`render_*`). Three pictures:

  eye      the luminance every medulla column sees (the funnel view of the screen plus the cursor disc), both eyes
           on one azimuth / elevation panorama
  motion   the retinotopic detectors: expansion (looming) and small moving object, per column
  neurons  the drive (Hz) of the LPLC2, LC4, LPC1 and LC10 neurons at their receptive-field columns

It shows the model's INPUT, not a recording of a real optic lobe: the detectors in vision.py are hand-built.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paths import FIELD

MAX_COLS = 2048                      # medulla columns the snapshot can hold (both eyes: 1,584 in the dataset)
MAX_NEURONS = 2048                   # receptive-field neurons of the four groups together (1,001 in the dataset)
HEADER = 6                           # seq, columns, LPLC2, LC4, LPC1, LC10 counts
PREVIEW_LEN = HEADER + 3 * MAX_COLS + 2 * MAX_NEURONS

GROUPS = ("LPLC2", "LC4", "LPC1", "LC10")
GROUP_ROLES = ("looming, size", "looming, speed", "retreat", "small object, steering")
GROUP_COLOURS = ((255, 122, 144), (255, 180, 84), (138, 180, 255), (110, 231, 168))
GROUP_MAX_HZ = (150.0, 150.0, 20.0, 50.0)        # VisionParams: max_loom, max_loom, max_retreat, max_object
EXPANSION_FULL = 3.0                              # expansion value drawn at full orange
ACTIVE_HZ = 0.5                                   # a neuron counts as active above this drive

TABS = ("Eye view", "Motion", "Neurons")
EL_TOP, EL_BOTTOM = 80.0, -90.0                   # elevation range of the panorama, degrees (columns span -87..+75)
BACKGROUND = (14, 14, 17)
GRID_HORIZON = (60, 150, 175)
GRID_AHEAD = (150, 105, 30)


# ----------------------------------------------------------------------------------------------------- snapshot
@dataclass
class Snapshot:
    seq: int
    lum: np.ndarray          # float32 [m] luminance 0..1 seen by each column
    expansion: np.ndarray    # float32 [m] expansion detector (1/s-like, ~0..9)
    obj: np.ndarray          # float32 [m] small-object detector 0..1
    cols: np.ndarray         # int32 [n] column of each neuron's receptive field, groups concatenated
    rates: np.ndarray        # float32 [n] drive in Hz, same order
    sizes: tuple             # (LPLC2, LC4, LPC1, LC10) counts

    def group(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(columns, rates) of group i."""
        lo = int(sum(self.sizes[:i]))
        hi = lo + int(self.sizes[i])
        return self.cols[lo:hi], self.rates[lo:hi]


def write_snapshot(buf: np.ndarray, seq: int, lum, expansion, obj, cols, rates, sizes) -> None:
    """Worker side. `buf` is the shared float32 array of PREVIEW_LEN; the sequence number goes in last, so a reader
    that sees the same number before and after copying has a consistent frame."""
    m, n = min(len(lum), MAX_COLS), min(len(cols), MAX_NEURONS)
    o = HEADER
    buf[o:o + m] = lum[:m]
    o += MAX_COLS
    buf[o:o + m] = expansion[:m]
    o += MAX_COLS
    buf[o:o + m] = obj[:m]
    o += MAX_COLS
    buf[o:o + n] = cols[:n]
    o += MAX_NEURONS
    buf[o:o + n] = rates[:n]
    buf[1] = m
    for i, s in enumerate(sizes):
        buf[2 + i] = s
    buf[0] = float(seq)


def read_snapshot(buf: np.ndarray) -> Snapshot | None:
    """GUI side: a copy of the latest frame, or None when none was written yet or the read was torn by a write."""
    seq = int(buf[0])
    if seq <= 0:
        return None
    m = int(buf[1])
    sizes = tuple(int(buf[2 + i]) for i in range(4))
    n = sum(sizes)
    if not (0 < m <= MAX_COLS and 0 <= n <= MAX_NEURONS):
        return None
    o = HEADER
    lum = buf[o:o + m].copy()
    o += MAX_COLS
    exp = buf[o:o + m].copy()
    o += MAX_COLS
    obj = buf[o:o + m].copy()
    o += MAX_COLS
    cols = buf[o:o + n].astype(np.int32)
    o += MAX_NEURONS
    rates = buf[o:o + n].copy()
    if int(buf[0]) != seq:
        return None
    return Snapshot(seq, lum, exp, obj, np.clip(cols, 0, m - 1), rates, sizes)


# ----------------------------------------------------------------------------------------------------- geometry
class EyeLayout:
    """Azimuth / elevation of the medulla columns (from field.npz), and their pixels on a panorama."""

    def __init__(self, col_dir: np.ndarray):
        d = np.asarray(col_dir, np.float64)
        self.az = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        self.el = np.degrees(np.arcsin(np.clip(d[:, 2], -1.0, 1.0)))
        self._cache: dict = {}

    @classmethod
    def load(cls, path=FIELD) -> "EyeLayout | None":
        try:
            with np.load(path) as z:
                return cls(z["col_dir"])
        except (OSError, KeyError, ValueError):
            return None

    def __len__(self) -> int:
        return len(self.az)

    def pixels(self, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        key = (w, h)
        if key not in self._cache:
            x = np.clip(np.round((self.az + 180.0) / 360.0 * (w - 1)), 0, w - 1).astype(np.int32)
            y = np.clip(np.round((EL_TOP - self.el) / (EL_TOP - EL_BOTTOM) * (h - 1)), 0, h - 1).astype(np.int32)
            self._cache[key] = (x, y)
        return self._cache[key]


def _splat(img: np.ndarray, xs, ys, colours, size: int, x0: int = 0, y0: int = 0, w: int | None = None,
           h: int | None = None) -> None:
    """Write a size x size block of `colours` (n x 3, or one RGB) at every (xs, ys), clipped to the tile."""
    hi_x = (img.shape[1] if w is None else x0 + w) - 1
    hi_y = (img.shape[0] if h is None else y0 + h) - 1
    for dy in range(size):
        for dx in range(size):
            img[np.minimum(ys + y0 + dy, hi_y), np.minimum(xs + x0 + dx, hi_x)] = colours


def _guides(img: np.ndarray, layout: EyeLayout) -> None:
    """Dashed horizon (elevation 0) and the straight-ahead meridian (azimuth 0)."""
    h, w = img.shape[:2]
    y = int(round(EL_TOP / (EL_TOP - EL_BOTTOM) * (h - 1)))
    x = int(round(180.0 / 360.0 * (w - 1)))
    img[y, ::4] = GRID_HORIZON
    img[::4, x] = GRID_AHEAD


# ----------------------------------------------------------------------------------------------------- rendering
def render_eye(snap: Snapshot, layout: EyeLayout, w: int = 440, h: int = 200) -> np.ndarray:
    """RGB uint8 [h, w, 3]: column luminance as grey squares on the az/el panorama."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BACKGROUND
    _guides(img, layout)
    m = min(len(snap.lum), len(layout))
    xs, ys = layout.pixels(w, h)
    g = (np.clip(snap.lum[:m], 0.0, 1.0) * 255.0).astype(np.uint8)
    _splat(img, xs[:m], ys[:m], g[:, None], 3)
    return img


def render_motion(snap: Snapshot, layout: EyeLayout, w: int = 440, h: int = 200) -> np.ndarray:
    """RGB uint8: orange = expansion (looming) detector, cyan = small moving object detector, over the dim lattice."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BACKGROUND
    _guides(img, layout)
    m = min(len(snap.expansion), len(layout))
    xs, ys = layout.pixels(w, h)
    xs, ys = xs[:m], ys[:m]
    _splat(img, xs, ys, (34, 34, 42), 2)
    o = np.clip(snap.obj[:m], 0.0, 1.0)
    on = o > 0.05
    if on.any():
        c = (np.array([76.0, 201.0, 240.0]) * (0.25 + 0.75 * o[on])[:, None]).astype(np.uint8)
        _splat(img, xs[on], ys[on], c, 3)
    e = np.clip(snap.expansion[:m] / EXPANSION_FULL, 0.0, 1.0)
    on = e > 0.02
    if on.any():
        c = (np.array([255.0, 140.0, 40.0]) * (0.35 + 0.65 * e[on])[:, None]).astype(np.uint8)
        _splat(img, xs[on], ys[on], c, 3)
    return img


def tile_rect(i: int, w: int, h: int) -> tuple[int, int, int, int]:
    """(x0, y0, width, height) of the tile of group i in the 2 x 2 neuron picture."""
    tw, th = w // 2, h // 2
    return (i % 2) * tw, (i // 2) * th, tw, th


def render_neurons(snap: Snapshot, layout: EyeLayout, w: int = 440, h: int = 200) -> np.ndarray:
    """RGB uint8: one tile per group, each neuron a dot at its receptive-field column, brighter with more Hz."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BACKGROUND
    for i in range(4):
        x0, y0, tw, th = tile_rect(i, w, h)
        img[y0 + 1:y0 + th - 1, x0 + 1:x0 + tw - 1] = (23, 23, 27)
        cols, rates = snap.group(i)
        if len(cols) == 0:
            continue
        cols = np.minimum(cols, len(layout) - 1)
        # a tile is a panorama of its own, below a one-line header drawn by the widget
        top = 14
        inner_w, inner_h = tw - 8, th - top - 4
        xs, ys = layout.pixels(inner_w, inner_h)
        xs, ys = xs[cols] + 4, ys[cols] + top
        act = rates > ACTIVE_HZ
        if (~act).any():
            _splat(img, xs[~act], ys[~act], (62, 62, 72), 1, x0, y0, tw, th)
        if act.any():
            f = np.clip(rates[act] / GROUP_MAX_HZ[i], 0.0, 1.0)[:, None]
            c = (np.array(GROUP_COLOURS[i], np.float64) * (0.4 + 0.6 * f)).astype(np.uint8)
            _splat(img, xs[act], ys[act], c, 3, x0, y0, tw, th)
    return img


RENDERERS = (render_eye, render_motion, render_neurons)


def group_summary(snap: Snapshot, i: int) -> str:
    """'24/210 active · 58 Hz' for the tile header."""
    _, rates = snap.group(i)
    if len(rates) == 0:
        return "none in this circuit"
    return f"{int((rates > ACTIVE_HZ).sum())}/{len(rates)} active · {float(rates.max()):.0f} Hz"
