"""Glue between the outside world and the engine.

stimulus (cursor) -> Poisson drive on named neuron groups
spikes of output groups -> smoothed rates -> behavior state (stand / walk / fly)
"""
from __future__ import annotations

import math

import numpy as np

from .engine import LIFEngine, Network
from .states import FLY, STAND, STATES, WALK  # noqa: F401  (re-exported)

# stimulus -> rate maps (toy circuit calibration, see tools/calibrate_toy.py)
LOOM_GAIN = 10.0        # Hz per (1/s) of relative expansion rate (closing speed / distance)
LOOM_MAX_HZ = 250.0
VIS_MAX_HZ = 60.0
VIS_FALLOFF_PX = 250.0
REST_DRIVE = (500.0, 3.0)     # Poisson event rate (Hz) and size (mV on g)
WALK_DRIVE = (650.0, 3.0)     # rate scaled by the walk bias in [0, 1]
BULK_HZ = 150.0

GF_ON_HZ, GF_OFF_HZ = 15.0, 3.0
MIN_DWELL_MS = 600.0          # a stand/walk state lasts at least this long
MIN_FLY_MS = 300.0


class Brain:
    def __init__(self, net: Network, dt: float = 0.5, seed: int = 0, ema_ms: float = 80.0):
        self.net = net
        self.engine = LIFEngine(net, dt=dt, seed=seed)
        self.g = net.groups
        self.ema_ms = ema_ms
        self.rates = {"GF": 0.0, "WALK": 0.0, "REST": 0.0}
        self.state = STAND
        self._dwell = 0.0
        self.walk_bias = 0.0
        self._stim = (1e6, 0.0)
        self._drive()

    # ----------------------------------------------------------------- input
    def set_stimulus(self, dist: float, closing_speed: float, walk_bias: float | None = None):
        """dist: px to the cursor; closing_speed: px/s, positive when the cursor approaches."""
        self._stim = (dist, closing_speed)
        if walk_bias is not None:
            self.walk_bias = walk_bias
        self._drive()

    def _drive(self):
        dist, closing = self._stim
        loom = min(LOOM_MAX_HZ, max(0.0, closing) / max(dist, 30.0) * LOOM_GAIN)
        vis = VIS_MAX_HZ / (1.0 + dist / VIS_FALLOFF_PX)
        e, g = self.engine, self.g
        fi = [g["LOOM"], g["VIS"], g["BULK_DRIVE"]]
        fr = [np.full(len(g["LOOM"]), loom), np.full(len(g["VIS"]), vis),
              np.full(len(g["BULK_DRIVE"]), BULK_HZ)]
        e.set_drive(np.concatenate(fi), np.concatenate(fr))
        e.set_current_drive(
            np.concatenate([g["REST"], g["WALK"]]),
            np.concatenate([np.full(len(g["REST"]), REST_DRIVE[0]),
                            np.full(len(g["WALK"]), WALK_DRIVE[0] * self.walk_bias)]),
            np.concatenate([np.full(len(g["REST"]), REST_DRIVE[1]),
                            np.full(len(g["WALK"]), WALK_DRIVE[1])]))

    # ---------------------------------------------------------------- run
    def advance(self, ms: float) -> str:
        e = self.engine
        e.advance(ms)
        a = 1.0 - math.exp(-ms / self.ema_ms)
        for k in self.rates:
            idx = self.g[k]
            inst = e.pop_counts(idx).sum() / (len(idx) * ms / 1000.0)
            self.rates[k] += a * (inst - self.rates[k])
        self._decode(ms)
        return self.state

    def _decode(self, ms: float):
        gf, walk, rest = self.rates["GF"], self.rates["WALK"], self.rates["REST"]
        self._dwell += ms
        new = self.state
        if gf > GF_ON_HZ:
            new = FLY                                   # escape: immediate
        elif self.state == FLY:
            if gf < GF_OFF_HZ and self._dwell >= MIN_FLY_MS:
                new = WALK if walk > rest else STAND
        elif self._dwell >= MIN_DWELL_MS:               # debounce stand <-> walk
            new = WALK if walk > rest * 1.2 else STAND
        if new != self.state:
            self.state, self._dwell = new, 0.0
