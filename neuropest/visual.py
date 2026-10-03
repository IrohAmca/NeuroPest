"""From a scene to projection-neuron rates, for the engine worker.

The scene is what lies on the screen plane around the fly (today: the cursor drawn as a dark disk on a
neutral background; a captured screen image can be plugged in as `scene`). `VisionDrive.step` samples it
through the funnel view at the fly's pose, runs the retinotopic detectors of `vision.Features` and turns
their output into forced-spike rates of LPLC2, LC4 (looming), LPC1 (retreat) and LC10 (small object)
neurons through the receptive fields found in the connectome.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vision import Features, VisualField, sample_scenes


@dataclass(frozen=True)
class VisionParams:
    eye_height: float = 100.0        # px above the screen plane; larger = more top-down view
    halo_px: float = 30.0            # the cursor is drawn as a dark disk this big
    background: float = 0.5
    # gains found by grid search on approach / slide / recede scenes (tools/vision_calibrate.py --grid)
    gain_loom: float = 25.0          # Hz of LPLC2/LC4 drive per unit of expansion
    max_loom: float = 150.0
    gain_retreat: float = 18.0       # Hz of LPC1 drive per unit of eye-wide expansion
    max_retreat: float = 20.0
    flee_lo: float = 3.0             # retreat drive fades out between these expansion values: strong looming flees
    flee_hi: float = 4.5
    gain_object: float = 50.0        # LC10 (steering): at 100 Hz a near disk drove DNa02 past 100 Hz, twice what
    max_object: float = 50.0         # the cursor drive gives at its maximum (79 Hz)


def cursor_scene(cx: float, cy: float, radius: float, background: float = 0.5):
    """Dark disk (the cursor) on a neutral screen."""
    def scene(px, py):
        return np.where((px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2, 0.0, background).astype(np.float32)
    return scene


class VisionDrive:
    def __init__(self, net, params: VisionParams = VisionParams(), field: VisualField | None = None):
        self.params = params
        self.field = field or VisualField.load()
        self.retina = self.field.as_retina()
        self.features = Features(self.field)
        self.loom_idx, self.loom_col = self.field.neurons(net, ["LPLC2", "LC4"])
        self.ret_idx, self.ret_col = self.field.neurons(net, ["LPC1"])
        self.obj_idx, self.obj_col = self.field.neurons(net, ["LC10a", "LC10c-2", "LC10d"])
        self.idx = np.concatenate([self.loom_idx, self.ret_idx, self.obj_idx]).astype(np.int32)
        self.loom_rows = np.unique(self.loom_col).astype(np.int32)       # the columns whose pooled values are read
        self.obj_rows = np.unique(self.obj_col).astype(np.int32)
        self.eye_height = params.eye_height          # px above the screen; the GUI can change both while running
        self.skittish = 1.0                          # multiplier on the looming and retreat gains
        self._fresh = True

    @property
    def usable(self) -> bool:
        return len(self.loom_idx) > 0

    def reset(self):
        self._fresh = True                  # the next frame sets the slow baseline to what the eye sees then

    def step(self, scene_now, scene_prev, x: float, y: float, heading: float, dt: float):
        """One frame: returns (neuron indices, rates in Hz). `scene_prev` is the previous frame's scene, sampled
        here at the CURRENT pose so that the fly's own movement does not count as change."""
        p = self.params
        lum_now, lum_prev = sample_scenes(self.retina, (scene_now, scene_prev), x, y, heading, self.eye_height,
                                          sky=p.background)
        if self._fresh:
            self.features.reset(lum_now)
            self._fresh = False
        f = self.features.update(lum_now, lum_prev, dt, self.loom_rows, self.obj_rows)
        e, o, eye = f["expansion_pooled"], f["object_pooled"], f["expansion_eye"]
        ee = eye[self.ret_col]
        retreat = np.minimum(self.skittish * p.gain_retreat * ee, p.max_retreat) * np.clip(
            (p.flee_hi - ee) / (p.flee_hi - p.flee_lo), 0.0, 1.0)
        rates = np.concatenate([np.minimum(self.skittish * p.gain_loom * e[self.loom_col], p.max_loom), retreat,
                                np.minimum(p.gain_object * o[self.obj_col], p.max_object)])
        return self.idx, rates.astype(np.float32)
