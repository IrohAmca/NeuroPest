"""Seeing the screen: the funnel view, retinal features and the receptive fields of projection neurons.

The fly stands on the screen plane. Every medulla column looks along a direction (azimuth, elevation) in the
body frame; a direction that points below the horizon meets the plane at distance h / tan(-elevation) from
the fly (h = eye height), so near ground is seen with the fine, bottom ommatidia and the far screen is
squeezed into the thin band just below the horizon: a funnel with a very low slope.

Directions and receptive fields come from the connectome (`eyebuild.py`, needs the raw data once; results
live in data/circuits/eye.npz and field.npz). A spiking optic lobe driven by the image did not produce
looming or motion selectivity (tools/vision_experiment.py), so the optic lobe is replaced by two simple
retinotopic detectors computed here on the column lattice, whose outputs drive the projection neurons
(LPLC2, LC4, LPC1, LC10) through their own receptive fields:

  expansion  - dark or bright edges growing outward all around a column (LPLC2-like); scene change is
               measured at the CURRENT pose, so the fly's own walking does not look like looming
  object     - a small dark or bright spot against its surround (LC10-like)

Body axes in the dataset form a left-handed frame relative to a true fly (the volume is mirrored with respect
to its left/right labels); the eyes are mirror symmetric, so the model uses the labelled sides.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np

from .paths import EYE, FIELD

EYE_FILE = EYE
KINDS = {"R1-6": 0, "R7": 1, "R8": 2, "L1": 3, "L2": 4, "L3": 5}
VOXEL_UM = np.array([4.0, 4.0, 40.0]) / 1000.0


@dataclass
class Retina:
    idx: np.ndarray          # int32 [m] neuron index of each photoreceptor in the full network
    az: np.ndarray           # float32 [m] azimuth, radians, 0 = ahead, positive = fly's right (clockwise on screen)
    el: np.ndarray           # float32 [m] elevation, radians, positive = up
    eye: np.ndarray          # uint8 [m] 0 = left, 1 = right
    kind: np.ndarray         # uint8 [m] 0 = R1-6, 1 = R7, 2 = R8, 3 = L1, 4 = L2, 5 = L3 (see KINDS)

    def save(self, path: Path = EYE_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, idx=self.idx, az=self.az, el=self.el, eye=self.eye, kind=self.kind)

    @classmethod
    def load(cls, path: Path = EYE_FILE) -> "Retina":
        z = np.load(path)
        return cls(z["idx"], z["az"], z["el"], z["eye"], z["kind"])

    def __len__(self) -> int:
        return len(self.idx)

    def subset(self, mask: np.ndarray) -> "Retina":
        return Retina(self.idx[mask], self.az[mask], self.el[mask], self.eye[mask], self.kind[mask])


# --------------------------------------------------------------------------- the funnel
def plane_hits(retina: Retina, x: float, y: float, heading: float, eye_height: float, max_dist: float):
    """Where each photoreceptor looks on the screen plane.

    (x, y): fly position in px, heading: radians clockwise from +x on screen (y points down),
    eye_height: px above the plane. Returns hit x, hit y (px) and a boolean `hits` (looks below the
    horizon and within `max_dist`).
    """
    below = retina.el < -1e-3
    dist = eye_height / np.tan(np.where(below, -retina.el, 1.0))
    hits = below & (dist <= max_dist)
    dist = np.where(hits, dist, 0.0)                            # columns that miss the plane get a dummy spot
    psi = heading + retina.az                                   # world azimuth on screen
    return x + dist * np.cos(psi), y + dist * np.sin(psi), hits


def sample_scene(retina: Retina, scene, x: float, y: float, heading: float, eye_height: float = 14.0,
                 max_dist: float = 2200.0, sky: float = 0.5) -> np.ndarray:
    """Luminance in [0, 1] seen by every photoreceptor.

    `scene(px, py)` returns luminance for arrays of screen coordinates (a function, or use `image_scene`).
    Photoreceptors that do not meet the screen see `sky`.
    """
    hx, hy, hits = plane_hits(retina, x, y, heading, eye_height, max_dist)
    lum = np.full(len(retina), sky, np.float32)
    if hits.any():
        lum[hits] = scene(hx[hits], hy[hits])
    return lum


def sample_scenes(retina: Retina, scenes, x: float, y: float, heading: float, eye_height: float = 14.0,
                  max_dist: float = 2200.0, sky: float = 0.5) -> list[np.ndarray]:
    """`sample_scene` for several scenes seen from the same pose (the plane hits are computed once)."""
    hx, hy, hits = plane_hits(retina, x, y, heading, eye_height, max_dist)
    hx, hy, any_hit = hx[hits], hy[hits], hits.any()
    out = []
    for scene in scenes:
        lum = np.full(len(retina), sky, np.float32)
        if any_hit:
            lum[hits] = scene(hx, hy)
        out.append(lum)
    return out


def sphere_luminance(col_dir: np.ndarray, cursor: tuple[float, float], x: float, y: float, heading: float,
                     radius_px: float, background: float = 0.5, dark: float = 0.0, elevation_deg: float = 0.0,
                     edge_deg: float = 2.0) -> np.ndarray:
    """Luminance of every column when the cursor is a dark disc facing the fly at eye level.

    The disc sits on the horizon at the cursor's azimuth and subtends atan(radius / distance), so an approach
    expands it at v / d whatever the eye height is (a disc lying on the screen plane shrinks into the thin band
    under the horizon as the eye rises, and the expansion detectors fall silent: it looks like something passing
    underneath rather than a collision). `edge_deg` softens the rim over about one ommatidial acceptance angle, so
    a distant disc smaller than a column still dims the nearest columns by its coverage.
    """
    dx, dy = cursor[0] - x, cursor[1] - y
    az = np.arctan2(dy, dx) - heading
    el = np.radians(elevation_deg)
    centre = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    half = np.arctan2(radius_px, max(np.hypot(dx, dy), 1.0))
    ang = np.arccos(np.clip(col_dir.astype(np.float64) @ centre, -1.0, 1.0))
    cover = np.clip((half - ang) / np.radians(edge_deg) + 0.5, 0.0, 1.0)
    return (background + (dark - background) * cover).astype(np.float32)


def image_scene(gray: np.ndarray, outside: float = 0.5):
    """Scene function over a grayscale image (rows = y, columns = x), bilinear, `outside` beyond its edges."""
    from scipy.ndimage import map_coordinates

    h, w = gray.shape

    def scene(px, py):
        inside = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
        out = np.full(px.shape, outside, np.float32)
        if inside.any():
            out[inside] = map_coordinates(gray, [py[inside], px[inside]], order=1, mode="nearest")
        return out

    return scene


# --------------------------------------------------------------------------- columns, receptive fields
FIELD_FILE = FIELD


@dataclass
class VisualField:
    col_dir: np.ndarray      # float32 [M, 3] unit vectors (ahead, right, up) of the medulla columns, both eyes
    col_eye: np.ndarray      # uint8 [M] 0 = left, 1 = right
    rf_id: np.ndarray        # int64 [K] FlyWire root ids of projection neurons with a receptive field
    rf_dir: np.ndarray       # float32 [K, 3] receptive field centre
    rf_conc: np.ndarray      # float32 [K] 0..1, near 1 = small receptive field
    rf_type: np.ndarray      # int16 [K] index into `types`
    rf_col: np.ndarray       # int32 [K] column nearest to the centre
    rf_side: np.ndarray      # uint8 [K] 0 = left, 1 = right
    types: list

    @classmethod
    def load(cls, path: Path = FIELD_FILE) -> "VisualField":
        z = np.load(path)
        return cls(z["col_dir"], z["col_eye"], z["rf_id"], z["rf_dir"], z["rf_conc"], z["rf_type"], z["rf_col"],
                   z["rf_side"], [str(t) for t in z["types"]])

    @property
    def col_az(self) -> np.ndarray:
        return np.arctan2(self.col_dir[:, 1], self.col_dir[:, 0]).astype(np.float32)

    @property
    def col_el(self) -> np.ndarray:
        return np.arcsin(np.clip(self.col_dir[:, 2], -1, 1)).astype(np.float32)

    def neurons(self, net, type_names) -> tuple[np.ndarray, np.ndarray]:
        """(network index, column index) of the receptive fields of `type_names` that exist in `net`."""
        if net.ids is None:                                # the toy circuit has no FlyWire ids, nothing to match
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        row_of = {int(r): i for i, r in enumerate(net.ids)}
        want = np.isin(self.rf_type, [self.types.index(t) for t in type_names if t in self.types])
        pairs = [(row_of[int(i)], c) for i, c in zip(self.rf_id[want], self.rf_col[want]) if int(i) in row_of]
        if not pairs:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        a = np.array(pairs, np.int64)
        return a[:, 0].astype(np.int32), a[:, 1].astype(np.int32)

    def as_retina(self) -> Retina:
        """The columns as a `Retina` (one 'photoreceptor' per column) for `sample_scene`."""
        m = len(self.col_dir)
        return Retina(np.arange(m, dtype=np.int32), self.col_az, self.col_el, self.col_eye.copy(),
                      np.zeros(m, np.uint8))


@nb.njit(cache=True)
def _weakest_quadrant(grow, ptr, ind, val, out):
    """out[i, c] = min over the four quadrants k of sum_j val[j] * grow[ind[j], c], c = 0, 1; quadrant k of
    column i is row k * m + i of the stacked CSR matrix (ptr, ind, val)."""
    m = out.shape[0]
    for i in range(m):
        best0 = np.inf
        best1 = np.inf
        for k in range(4):
            r = k * m + i
            acc0 = 0.0
            acc1 = 0.0
            for j in range(ptr[r], ptr[r + 1]):
                g = ind[j]
                acc0 += val[j] * grow[g, 0]
                acc1 += val[j] * grow[g, 1]
            best0 = min(best0, acc0)
            best1 = min(best1, acc1)
        out[i, 0] = best0
        out[i, 1] = best1


@nb.njit(cache=True)
def _pool_wmax(values, ptr, ind, w, rows, out):
    """out[i] = max over the neighbour list of w[j] * values[ind[j]], for i in rows (w = receptive-field weight)."""
    for r in range(rows.shape[0]):
        i = rows[r]
        best = 0.0
        for j in range(ptr[i], ptr[i + 1]):
            v = w[j] * values[ind[j]]
            if v > best:
                best = v
        out[i] = best


@nb.njit(cache=True)
def _pool_mean(values, ptr, ind, rows, out):
    """out[i] = mean of values over the neighbour list of i, for i in rows."""
    for r in range(rows.shape[0]):
        i = rows[r]
        acc = 0.0
        for j in range(ptr[i], ptr[i + 1]):
            acc += values[ind[j]]
        out[i] = acc / max(ptr[i + 1] - ptr[i], 1)


@nb.njit(cache=True)
def _pool_max(values, ptr, ind, rows, out):
    """out[i] = max of values over the neighbour list ind[ptr[i]:ptr[i + 1]], for i in rows."""
    for r in range(rows.shape[0]):
        i = rows[r]
        best = values[ind[ptr[i]]]
        for j in range(ptr[i] + 1, ptr[i + 1]):
            v = values[ind[j]]
            if v > best:
                best = v
        out[i] = best


class Features:
    """Retinotopic expansion and small-object detectors on the column lattice.

    update(lum_now, lum_prev_same_pose, dt) takes the luminance every column sees now and what the same
    columns would have seen one frame ago from the SAME pose, so only changes in the scene count.

    expansion: for each polarity (dark, bright) relative to a slow baseline: the column lies inside the
    object (centre value) and the object's extent grows in all four quadrants of a ring around it (the
    weakest quadrant counts, so a translating edge or a one-sided change scores zero). A shrinking dark
    object does not grow the dark channel and its centre is not bright, so contraction scores zero too.
    A change that is the same everywhere in an eye is subtracted (wide-field change is not looming).

    The slow baseline (tau_adapt) is a gain control. With 2 s, a disk that had sat near the fly for seconds left
    a bright footprint when it moved away, which read as a growing bright object (a false retreat); 8 s does not
    (tools/vision_calibrate.py, scenario "recede after 3 s near")."""

    def __init__(self, field: VisualField, ring_deg=(5.0, 18.0), centre_deg=3.5, surround_deg=(5.0, 11.0),
                 tau_adapt=8.0, tau_smooth=0.03, wide_field=1.5, pool_deg=30.0, object_pool_deg=12.0, hold_s=0.15,
                 object_hold_s=0.1, rf_sigma_deg=0.0):
        from scipy.sparse import csr_matrix, vstack

        d, eye = field.col_dir.astype(np.float64), field.col_eye
        m = len(d)
        ang = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
        same = eye[:, None] == eye[None, :]
        np.fill_diagonal(ang, 1e9)
        up = np.array([0.0, 0.0, 1.0])
        e2 = up - (d @ up)[:, None] * d                      # tangent "up" at every column
        e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12
        e1 = np.cross(e2, d)
        sectors = [np.zeros((m, m), np.float32) for _ in range(4)]
        in_ring = same & (ang >= ring_deg[0]) & (ang <= ring_deg[1])
        for i in range(m):
            js = np.flatnonzero(in_ring[i])
            if len(js) == 0:
                continue
            o = d[js] - (d[js] @ d[i])[:, None] * d[i]
            phi = np.degrees(np.arctan2(o @ e2[i], o @ e1[i])) % 360.0
            sec = ((phi + 45.0) // 90.0).astype(int) % 4
            for k in range(4):
                sel = js[sec == k]
                if len(sel):
                    sectors[k][i, sel] = 1.0 / len(sel)
        q = vstack([csr_matrix(x) for x in sectors], format="csr")                 # (4m, m): quadrant means
        self.sectors = (q.indptr.astype(np.int32), q.indices.astype(np.int32), q.data.astype(np.float32))
        cen = same & (ang <= centre_deg)
        np.fill_diagonal(cen, True)
        sur = same & (ang >= surround_deg[0]) & (ang <= surround_deg[1])
        self.centre = csr_matrix((cen / np.maximum(cen.sum(1, keepdims=True), 1)).astype(np.float32))
        surround = csr_matrix((sur / np.maximum(sur.sum(1, keepdims=True), 1)).astype(np.float32))
        self.centre_surround = vstack([self.centre, surround], format="csr")      # (2m, m)
        self.eye, self.m = eye, m
        self.pool_deg = pool_deg
        self.eye_idx = [np.flatnonzero(eye == 0), np.flatnonzero(eye == 1)]
        self.eye_mean = np.zeros((2, m), np.float32)                     # row e averages over the columns of eye e
        for e_, idx in enumerate(self.eye_idx):
            self.eye_mean[e_, idx] = 1.0 / max(len(idx), 1)
        self.all_rows = np.arange(m, dtype=np.int32)

        def neighbours(radius):                                          # neighbour lists, self included
            near = same & (ang <= radius)
            np.fill_diagonal(near, True)
            ptr = np.concatenate([[0], np.cumsum(near.sum(1))]).astype(np.int32)
            return ptr, np.nonzero(near)[1].astype(np.int32)

        self.pool = neighbours(pool_deg)                                # wide pooling (expansion)
        # receptive-field weights of the wide pool: a neuron whose field centre is `a` degrees from the object sees
        # it with weight exp(-a^2 / 2 sigma^2) (sigma 0: flat, the pool is a plain top hat)
        rows_i, cols_j = np.nonzero(same & (ang <= pool_deg) | np.eye(m, dtype=bool))
        a = np.where(rows_i == cols_j, 0.0, ang[rows_i, cols_j])
        self.pool_w = (np.exp(-0.5 * (a / rf_sigma_deg) ** 2) if rf_sigma_deg > 0 else np.ones(len(a))).astype(np.float32)
        self.rf_sigma_deg = rf_sigma_deg
        self.pool_small = neighbours(object_pool_deg)                   # narrow pooling (small objects)
        self.tau_adapt, self.tau_smooth, self.wide = tau_adapt, tau_smooth, wide_field
        self.hold, self.obj_hold = hold_s, object_hold_s
        self.exp_hold = np.zeros(m, np.float32)             # leaky peak hold: a brief expansion drives for ~hold_s
        self.obj_hold_v = np.zeros(m, np.float32)
        self.A = np.full(m, 0.5, np.float32)
        self.dd = np.zeros((m, 2), np.float32)               # smoothed growth of the dark and bright channels
        self._weakest = np.zeros((m, 2), np.float32)
        self._pooled = np.zeros(m, np.float32)
        self._size = np.zeros(m, np.float32)
        self._pooled_small = np.zeros(m, np.float32)

    def reset(self, lum: np.ndarray | None = None):
        """Forget everything. The slow baseline restarts at neutral grey, or at the median of `lum` (a scene-wide
        level, so a dark screen is not all 'dark'); never per column, which would leave an after-image where
        something dark sits when the eye opens (it reads as a growing bright object when it leaves)."""
        self.A[:] = 0.5 if lum is None else float(np.median(lum))
        self.dd[:] = 0
        self.exp_hold[:] = 0
        self.obj_hold_v[:] = 0

    def _channels(self, lum, base):
        """[m, 2]: how far below (dark) and above (bright) the slow baseline each column is."""
        x = 2.0 * (lum - base)
        out = np.empty((len(x), 2), np.float32)
        np.clip(-x, 0.0, 1.0, out=out[:, 0])
        np.clip(x, 0.0, 1.0, out=out[:, 1])
        return out

    def update(self, lum_now: np.ndarray, lum_prev: np.ndarray, dt: float, pool_rows=None, object_rows=None) -> dict:
        """One frame. The pooled outputs are computed for the columns in `pool_rows` / `object_rows` (all by
        default; asking only for the columns the caller reads is cheaper, the rest stay zero)."""
        m = self.m
        now, prev = self._channels(lum_now, self.A), self._channels(lum_prev, self.A)
        self.A += (1.0 - np.exp(-dt / self.tau_adapt)) * (lum_now - self.A)
        self.dd += (1.0 - np.exp(-dt / self.tau_smooth)) * ((now - prev) / dt - self.dd)
        grow = np.maximum(self.dd, 0.0)                                              # [m, 2]
        _weakest_quadrant(grow, *self.sectors, self._weakest)
        wide = self.eye_mean @ grow                                                  # [eye, polarity]
        e = np.maximum(self._weakest * (self.centre @ now) - self.wide * wide[self.eye], 0.0).max(axis=1)
        # small object: centre darker (or brighter) than the surround
        cs = self.centre_surround @ lum_now
        obj = np.clip(np.abs(cs[m:] - cs[:m]) * 2.0, 0.0, 1.0)
        self.exp_hold = np.maximum(e, self.exp_hold * np.exp(-dt / self.hold)).astype(np.float32)
        self.obj_hold_v = np.maximum(obj, self.obj_hold_v * np.exp(-dt / self.obj_hold)).astype(np.float32)
        eye_max = np.array([self.exp_hold[i].max() for i in self.eye_idx], np.float32)
        rows = self.all_rows if pool_rows is None else pool_rows
        _pool_wmax(self.exp_hold, *self.pool, self.pool_w, rows, self._pooled)
        _pool_mean(np.maximum(now[:, 0], now[:, 1]), *self.pool, rows, self._size)   # share of the field the object covers
        _pool_max(self.obj_hold_v, *self.pool_small, self.all_rows if object_rows is None else object_rows,
                  self._pooled_small)
        return {"expansion": self.exp_hold, "expansion_pooled": self._pooled.copy(), "size_pooled": self._size.copy(),
                "expansion_eye": eye_max[self.eye], "object": self.obj_hold_v,
                "object_pooled": self._pooled_small.copy()}
