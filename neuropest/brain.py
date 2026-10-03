"""Glue: cursor stimulus -> circuit input, circuit output -> behavior state."""
from __future__ import annotations

import math

import numpy as np

from .toy_circuit import build

STAND, WALK, FLY = "stand", "walk", "fly"


class Brain:
    def __init__(self, circuit=None, groups=None):
        if circuit is None:
            circuit, groups = build()
        self.c, self.g = circuit, groups
        self.state = STAND
        self.walk_bias = 0.0       # Hz tonic drive, a UI knob
        self.last_dist = None

    def stimulus(self, dist: float, closing_speed: float) -> np.ndarray:
        """dist px to cursor, closing_speed px/s (positive = approaching)."""
        ext = np.zeros(self.c.n, dtype=np.float32)
        loom = max(0.0, closing_speed) / max(dist, 40.0) * 400.0   # ~ tau-style looming
        ext[self.g["LOOM"]] = min(loom, 1500.0)
        near = 1.0 / (1.0 + dist / 200.0)
        ext[self.g["VIS"]] = 150.0 * near
        ext[self.g["REST"]] = 120.0
        ext[self.g["WALK"]] = self.walk_bias
        return ext

    def step(self, dt_ms: float, dist: float, closing_speed: float) -> str:
        n_sub = max(1, int(dt_ms / 0.5))
        h = dt_ms / n_sub
        ext = self.stimulus(dist, closing_speed)
        for _ in range(n_sub):
            self.c.step(h, ext)
        r = self.c.rate
        gf, walk, rest = (r[self.g[k]].mean() for k in ("GF", "WALK", "REST"))
        if gf > 15:
            self.state = FLY
        elif self.state == FLY and gf < 3:
            self.state = WALK if walk > rest else STAND
        elif self.state != FLY:
            self.state = WALK if walk > rest * 1.2 else STAND
        return self.state
