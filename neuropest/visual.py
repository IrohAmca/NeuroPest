"""From a scene to projection-neuron rates, for the engine worker.

Two ways to show the fly the cursor:
  step_cursor  a dark disc at eye level facing the fly, subtending atan(r / d) (the default, `cursor_model="sphere"`):
               an approach expands it at v / d whatever the eye height, so the detectors answer the same from
               any height
  step         a scene lying on the screen plane (the cursor drawn as a dark disk, or a captured screen image) sampled
               through the funnel view at the fly's pose; this is the geometry for real screen content, where the
               eye height matters, and the first cursor model
Either way the retinotopic detectors of `vision.Features` run on the column luminances and their output becomes
forced-spike rates of LPLC2, LC4 (looming), LPC1 (retreat) and LC10 (small object) neurons through the receptive
fields found in the connectome.

The mushroom body reads the image the same way. The visual projection neurons that reach the Kenyon cells
(`mushroom.CUE_PN_TYPES`: aMe12, MTe32, MTe30, LTe25, MTe40 for the cursor, aMe26, LTe72, MTe37 for looming) get their
drive from the same detectors through their own receptive fields, and `mb_cue` holds each group's strongest drive as a
share of its maximum rate. ASSUMPTIONS: what these cells encode is poorly known; here the cursor group is driven like
LC10 (the small-object detector) and the looming group like LC4 (the expansion detector). Most of them are not in the
smaller simulated tiers (none of the cursor ones at 15,000), so their spikes are not forced in the engine: only the cue
level is computed.
"""
from __future__ import annotations

import numpy as np

from .mushroom import CUE_PN_TYPES
from .vision import Features, VisualField, sample_scenes, sphere_luminance
from .visionparams import DISK_PLANE, VisionParams  # noqa: F401  (re-exported)


def cursor_scene(cx: float, cy: float, radius: float, background: float = 0.5):
    """Dark disk (the cursor) on a neutral screen."""
    def scene(px, py):
        return np.where((px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2, 0.0, background).astype(np.float32)
    return scene


class VisionDrive:
    def __init__(self, net, params: VisionParams = VisionParams(), field: VisualField | None = None):
        self.params = params
        self.expansion = 0.0
        self.field = field or VisualField.load()
        self.col_dir_f64 = self.field.col_dir.astype(np.float64)
        self.retina = self.field.as_retina()
        self.features = Features(self.field, rf_sigma_deg=params.rf_sigma_deg)
        lplc2_idx, lplc2_col = self.field.neurons(net, ["LPLC2"])
        lc4_idx, lc4_col = self.field.neurons(net, ["LC4"])
        self.n_lplc2 = len(lplc2_idx)                    # loom_idx = the LPLC2 neurons, then the LC4 ones
        self.loom_idx, self.loom_col = np.concatenate([lplc2_idx, lc4_idx]), np.concatenate([lplc2_col, lc4_col])
        self.ret_idx, self.ret_col = self.field.neurons(net, ["LPC1"])
        self.obj_idx, self.obj_col = self.field.neurons(net, ["LC10a", "LC10c-2", "LC10d"])
        self.idx = np.concatenate([self.loom_idx, self.ret_idx, self.obj_idx]).astype(np.int32)
        # mushroom-body visual projection neurons: receptive-field columns of the cursor and the looming group
        self.mb_near_col = self.field.cell_columns(CUE_PN_TYPES["cursor_near"])
        self.mb_loom_col = self.field.cell_columns(CUE_PN_TYPES["looming"])
        self.mb_cue = {"cursor_near": 0.0, "looming": 0.0}              # 0..1, from the last frame
        # the columns whose pooled values are read
        self.loom_rows = np.unique(np.concatenate([self.loom_col, self.mb_loom_col])).astype(np.int32)
        self.obj_rows = np.unique(np.concatenate([self.obj_col, self.mb_near_col])).astype(np.int32)
        self.eye_height = params.eye_height          # px above the screen; the GUI can change both while running
        self.skittish = 1.0                          # multiplier on the looming and retreat gains
        self._fresh = True

    @property
    def usable(self) -> bool:
        return len(self.loom_idx) > 0

    @property
    def halo_px(self) -> float:
        """Radius of the cursor disk. It grows with the eye height so it keeps its apparent size: from 250 px up
        a 30 px disk spans less than a column and the expansion detector never saw it (the fly did not flee).
        The eye-level disc ("sphere") does not depend on the height."""
        if self.params.cursor_model == "sphere":
            return self.params.halo_px
        return self.params.halo_px * max(1.0, self.eye_height / self.params.eye_height)

    @property
    def height_gain(self) -> float:
        """Looming is weaker from higher up even for a disk of the same apparent size (expansion for an approach at
        800 px/s: 5.4 at 100 px, 5.2 at 150, 3.1 at 250, 2.9 at 350); the detector outputs are scaled back up."""
        if self.params.cursor_model == "sphere":
            return 1.0
        return max(1.0, (self.eye_height / 150.0) ** 0.8)

    def reset(self):
        self._fresh = True                  # the next frame sets the slow baseline to what the eye sees then

    def step(self, scene_now, scene_prev, x: float, y: float, heading: float, dt: float,
             cursor: tuple[float, float] | None = None, prev_cursor: tuple[float, float] | None = None,
             screen_scale: float = 1.0):
        """One frame: returns (neuron indices, rates in Hz). `scene_prev` is the previous frame's scene, sampled
        here at the CURRENT pose so that the fly's own movement does not count as change."""
        p = self.params
        lum_now, lum_prev = sample_scenes(self.retina, (scene_now, scene_prev), x, y, heading, self.eye_height,
                                          sky=p.background)
        if screen_scale < 1.0:
            lum_prev = lum_now - (lum_now - lum_prev) * np.float32(screen_scale)
        if cursor is not None and p.cursor_model == "sphere":
            cd = self.col_dir_f64
            cur_now = sphere_luminance(cd, cursor, x, y, heading, p.halo_px, p.background)
            cur_prev = sphere_luminance(cd, prev_cursor or cursor, x, y, heading, p.halo_px, p.background)
            cov_now = np.clip((p.background - cur_now) / max(p.background, 1e-4), 0.0, 1.0)
            cov_prev = np.clip((p.background - cur_prev) / max(p.background, 1e-4), 0.0, 1.0)
            lum_now = lum_now * (1.0 - cov_now)
            lum_prev = lum_prev * (1.0 - cov_prev)
        return self._detect(lum_now, lum_prev, dt, self.height_gain, min(1.0, p.eye_height / self.eye_height))

    def step_cursor(self, now: tuple[float, float], prev: tuple[float, float], x: float, y: float, heading: float,
                    dt: float):
        """One frame with the cursor drawn as a disc at eye level (`sphere_luminance`). `now` and `prev` are the
        cursor positions at this and the previous frame (screen px); both are seen from the CURRENT pose, so the
        fly's own walking does not count as the cursor moving."""
        p = self.params
        cd = self.col_dir_f64
        lum_now = sphere_luminance(cd, now, x, y, heading, p.halo_px, p.background)
        lum_prev = sphere_luminance(cd, prev, x, y, heading, p.halo_px, p.background)
        return self._detect(lum_now, lum_prev, dt, 1.0, 1.0)

    def _detect(self, lum_now, lum_prev, dt: float, g: float, go: float):
        """Luminance of the columns now and one frame ago (same pose) -> (neuron indices, rates in Hz).
        `g` scales the looming and retreat detectors, `go` the small-object (steering) one."""
        p = self.params
        if self._fresh:
            self.features.reset(lum_now)
            self._fresh = False
        f = self.features.update(lum_now, lum_prev, dt, self.loom_rows, self.obj_rows)
        e, o, eye = g * f["expansion_pooled"], f["object_pooled"], g * f["expansion_eye"]
        self.expansion = float(e[self.loom_col].max()) if len(self.loom_col) else 0.0   # for the brain's freeze rule
        el = e[self.loom_col]
        if p.loom_split:                                    # LPLC2 reads size (times looming), LC4 the expansion speed
            size = np.clip((f["size_pooled"][self.loom_col[:self.n_lplc2]] - p.size_lo) / (p.size_hi - p.size_lo), 0.0, 1.0)
            el = np.concatenate([el[:self.n_lplc2] * size, el[self.n_lplc2:]])
        ee = eye[self.ret_col]
        retreat = np.minimum(self.skittish * p.gain_retreat * ee, p.max_retreat) * np.clip(
            (p.flee_hi - ee) / (p.flee_hi - p.flee_lo), 0.0, 1.0)
        # mushroom-body cues: the rate each cell would get (as LC10 / LC4 above) as a share of its maximum rate
        near = float(o[self.mb_near_col].max()) * p.gain_object / p.max_object if len(self.mb_near_col) else 0.0
        loom = float(e[self.mb_loom_col].max()) * self.skittish * p.gain_loom / p.max_loom if len(self.mb_loom_col) else 0.0
        self.mb_cue["cursor_near"] = min(1.0, near)
        self.mb_cue["looming"] = min(1.0, loom)
        rates = np.concatenate([np.minimum(self.skittish * p.gain_loom * el, p.max_loom), retreat,
                                np.minimum(go * p.gain_object * o[self.obj_col], go * p.max_object)])
        return self.idx, rates.astype(np.float32)
