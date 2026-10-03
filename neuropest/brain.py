"""Glue between the outside world and the engine.

stimulus (cursor) -> Poisson drive on named neuron groups
spikes of output groups -> smoothed rates -> behavior state (stand / walk / fly)

Stimulus-to-rate maps live in a `BrainSpec`, one per circuit kind, because each circuit has its
own units: the toy circuit's weights are arbitrary, the FlyWire one is calibrated against the
measured response of the Giant Fiber (tools/probe_circuit.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import LIFEngine, Network
from .states import FLY, STAND, STATES, WALK  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class BrainSpec:
    loom_gain: float                  # Hz of looming-input spikes per (1/s) of relative expansion rate
    loom_max_hz: float
    vis_max_hz: float = 0.0           # drive of the VIS group by cursor proximity (toy only)
    vis_falloff_px: float = 250.0
    walk_drive: tuple = (650.0, 3.0)  # Poisson events (Hz, mV on g) on WALK at full bias; rate scales with bias
    rest_drive: tuple | None = None   # tonic events on REST, which competes with WALK (toy only)
    bulk_hz: float = 0.0              # Poisson drive of BULK_DRIVE neurons (toy only)
    gf_on_hz: float = 15.0
    gf_off_hz: float = 3.0
    walk_min_hz: float = 3.0
    min_dwell_ms: float = 600.0       # a stand/walk state lasts at least this long
    min_fly_ms: float = 300.0


# Toy circuit: calibrated with tools/calibrate.py
TOY = BrainSpec(loom_gain=10.0, loom_max_hz=250.0, vis_max_hz=60.0, rest_drive=(500.0, 3.0), bulk_hz=150.0)
# FlyWire circuit: GF reaches 15 Hz at ~6 Hz of LPLC2/LC4 input, 33 Hz at 10 Hz, 217 Hz at 100 Hz
FLYWIRE = BrainSpec(loom_gain=2.0, loom_max_hz=150.0)


def spec_for(net: Network) -> BrainSpec:
    return FLYWIRE if net.meta.get("kind") == "flywire" else TOY


class Brain:
    def __init__(self, net: Network, dt: float = 0.5, seed: int = 0, ema_ms: float = 80.0,
                 spec: BrainSpec | None = None):
        self.net = net
        self.spec = spec or spec_for(net)
        self.engine = LIFEngine(net, dt=dt, seed=seed)
        self.g = net.groups
        self.ema_ms = ema_ms
        self.rates = {"GF": 0.0, "WALK": 0.0, "REST": 0.0}
        self.state = STAND
        self._dwell = 0.0
        self.walk_bias = 0.0
        self.skittish = 1.0
        self._stim = (1e6, 0.0)
        self._drive()

    def group(self, name: str) -> np.ndarray:
        return self.g.get(name, np.zeros(0, np.int32))

    # ----------------------------------------------------------------- input
    def set_stimulus(self, dist: float, closing_speed: float, walk_bias: float | None = None,
                     skittish: float | None = None):
        """dist: px to the cursor; closing_speed: px/s, positive when the cursor approaches.

        walk_bias in [0, 1]: tonic drive of the walking command neurons.
        skittish: multiplier on the looming sensitivity (1 = calibrated)."""
        self._stim = (dist, closing_speed)
        if walk_bias is not None:
            self.walk_bias = walk_bias
        if skittish is not None:
            self.skittish = skittish
        self._drive()

    def _drive(self):
        s = self.spec
        dist, closing = self._stim
        loom = min(s.loom_max_hz, max(0.0, closing) / max(dist, 30.0) * s.loom_gain * self.skittish)
        vis = s.vis_max_hz / (1.0 + dist / s.vis_falloff_px)
        e = self.engine
        fi, fr = [], []
        for name, hz in (("LOOM", loom), ("VIS", vis), ("BULK_DRIVE", s.bulk_hz)):
            idx = self.group(name)
            fi.append(idx)
            fr.append(np.full(len(idx), hz))
        e.set_drive(np.concatenate(fi), np.concatenate(fr))
        ci, cr, cw = [], [], []
        walk, rest = self.group("WALK"), self.group("REST")
        ci += [walk]
        cr += [np.full(len(walk), s.walk_drive[0] * self.walk_bias)]
        cw += [np.full(len(walk), s.walk_drive[1])]
        if s.rest_drive is not None:
            ci += [rest]
            cr += [np.full(len(rest), s.rest_drive[0])]
            cw += [np.full(len(rest), s.rest_drive[1])]
        e.set_current_drive(np.concatenate(ci), np.concatenate(cr), np.concatenate(cw))

    # ---------------------------------------------------------------- run
    def advance(self, ms: float) -> str:
        e = self.engine
        e.advance(ms)
        a = 1.0 - math.exp(-ms / self.ema_ms)
        for k in self.rates:
            idx = self.group(k)
            if len(idx):
                inst = e.pop_counts(idx).sum() / (len(idx) * ms / 1000.0)
                self.rates[k] += a * (inst - self.rates[k])
        self._decode(ms)
        return self.state

    def _decode(self, ms: float):
        s = self.spec
        gf, walk, rest = self.rates["GF"], self.rates["WALK"], self.rates["REST"]
        self._dwell += ms
        new = self.state
        if gf > s.gf_on_hz:
            new = FLY                                   # escape: immediate
        elif self.state == FLY:
            if gf < s.gf_off_hz and self._dwell >= s.min_fly_ms:
                new = WALK if walk > max(rest, s.walk_min_hz) else STAND
        elif self._dwell >= s.min_dwell_ms:             # debounce stand <-> walk
            new = WALK if walk > max(s.walk_min_hz, rest * 1.2) else STAND
        if new != self.state:
            self.state, self._dwell = new, 0.0
