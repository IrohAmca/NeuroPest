"""Glue between the outside world and the engine.

stimulus (cursor) -> Poisson drive on named neuron groups
spikes of output groups -> smoothed rates -> behavior state (stand / walk / fly / retreat) and a steering signal

Stimulus-to-rate maps live in a `BrainSpec`, one per circuit kind, because each circuit has its
own units: the toy circuit's weights are arbitrary, the FlyWire one is calibrated against the
measured response of the anchor neurons (tools/probe_circuit.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import Network, create_engine
from .states import FLY, GROOM, RETREAT, STAND, STATES, WALK  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class BrainSpec:
    loom_gain: float                  # Hz of looming-input spikes per (1/s) of relative expansion rate
    loom_max_hz: float
    retreat_gain: float = 0.0         # same, for the RETREAT_IN group (drives MDN, backward walking)
    retreat_max_hz: float = 0.0
    steer_max_hz: float = 0.0         # LC10 drive at zero distance; split left/right by the cursor bearing
    touch_hz: float = 0.0             # TOUCH drive (head mechanosensory neurons) while the cursor is on the fly
    steer_range_px: float = 700.0     # beyond this the cursor is ignored, so a far cursor costs no CPU
    vis_max_hz: float = 0.0           # drive of the VIS group by cursor proximity (toy only)
    vis_falloff_px: float = 250.0
    walk_drive: tuple = (650.0, 3.0)  # Poisson events (Hz, mV on g) on WALK at full bias; rate scales with bias
    rest_drive: tuple | None = None   # tonic events on REST, which competes with WALK (toy only)
    bulk_hz: float = 0.0              # Poisson drive of BULK_DRIVE neurons (toy only)
    gf_on_hz: float = 15.0
    gf_off_hz: float = 3.0
    mdn_on_hz: float = 5.0
    mdn_off_hz: float = 2.0
    groom_on_hz: float = 8.0          # mean rate of the grooming DNs (aDN1, aDN2)
    groom_off_hz: float = 3.0
    walk_min_hz: float = 3.0
    min_dwell_ms: float = 600.0       # a stand/walk state lasts at least this long
    min_fly_ms: float = 300.0
    min_retreat_ms: float = 400.0
    min_groom_ms: float = 800.0


# Toy circuit: calibrated with tools/calibrate.py
TOY = BrainSpec(loom_gain=10.0, loom_max_hz=250.0, vis_max_hz=60.0, rest_drive=(500.0, 3.0), bulk_hz=150.0)
# FlyWire circuit (15,000-neuron tier, within ~2% of the full brain; tools/probe_circuit.py, probe_combo.py):
#   LPLC2 + LC4 -> GF: 15 Hz at ~6 Hz of input, 63 Hz at 20 Hz, 200 Hz at 80 Hz
#   LPC1 -> MDN: ~0 up to 12 Hz of input, 15 Hz at 20 Hz, 26 Hz at 30 Hz
#   LC10 on one side -> DNa02 on that side: 24 Hz at 10 Hz of input, 79 Hz at 20 Hz
#   LPC1 input suppresses GF: at 20 Hz of LPC1 the take-off needs ~20+ Hz of LPLC2/LC4 input.
#   TOUCH (head bristles + Johnston's organ C/E) on one side -> aDN1/aDN2 (GROOM): ~0 at 25 Hz, ~5 Hz
#   at 50 Hz, 25 to 40 Hz at 100 Hz (mean of the four neurons); it also suppresses GF (tools/probe_touch.py)
# So retreat input saturates at 20 Hz (MDN ~15 Hz), and take-off wins only when the looming input is
# strong: retreat from about 2 /s of expansion, take-off from about 7 /s.
FLYWIRE = BrainSpec(loom_gain=3.0, loom_max_hz=150.0, retreat_gain=8.0, retreat_max_hz=20.0,
                    steer_max_hz=20.0, touch_hz=120.0)


def spec_for(net: Network) -> BrainSpec:
    return FLYWIRE if net.meta.get("kind") == "flywire" else TOY


class Brain:
    def __init__(self, net: Network, dt: float = 0.5, seed: int = 0, ema_ms: float = 80.0,
                 spec: BrainSpec | None = None, backend: str = "cpu", adapter: int | None = None):
        self.net = net
        self.spec = spec or spec_for(net)
        self.engine = create_engine(net, dt=dt, seed=seed, backend=backend, adapter=adapter)
        self.g = net.groups
        self.ema_ms = ema_ms
        self.rates = {k: 0.0 for k in ("GF", "WALK", "REST", "MDN", "DNa02_L", "DNa02_R", "GROOM")}
        # all output groups are read with one call per chunk (on a GPU every read waits for the device)
        parts = [self.group(k) for k in self.rates]
        self._mon_idx = np.concatenate(parts).astype(np.int32)
        self._mon_edges = np.cumsum([0] + [len(p) for p in parts])
        self.state = STAND
        self._dwell = 0.0
        self.walk_bias = 0.0
        self.skittish = 1.0
        self._stim = (1e6, 0.0, 0.0, 0.0)
        self._drive()

    def group(self, name: str) -> np.ndarray:
        return self.g.get(name, np.zeros(0, np.int32))

    @property
    def steer(self) -> float:
        """Right minus left DNa02 rate (Hz); positive turns the fly to its right (clockwise on screen)."""
        return self.rates["DNa02_R"] - self.rates["DNa02_L"]

    # ----------------------------------------------------------------- input
    def set_stimulus(self, dist: float, closing_speed: float, walk_bias: float | None = None,
                     skittish: float | None = None, bearing: float | None = None, touch: float = 0.0):
        """dist: px to the cursor; closing_speed: px/s, positive when the cursor approaches;
        bearing: angle of the cursor relative to the fly's heading, radians, positive to the right.
        touch: 0..1, the cursor is on the fly (hover); it touches the side given by the bearing.

        walk_bias in [0, 1]: tonic drive of the walking command neurons.
        skittish: multiplier on the looming and retreat sensitivity (1 = calibrated)."""
        self._stim = (dist, closing_speed, self._stim[2] if bearing is None else bearing, touch)
        if walk_bias is not None:
            self.walk_bias = walk_bias
        if skittish is not None:
            self.skittish = skittish
        self._drive()

    def _drive(self):
        s = self.spec
        dist, closing, bearing, touch = self._stim
        expansion = max(0.0, closing) / max(dist, 30.0) * self.skittish
        loom = min(s.loom_max_hz, expansion * s.loom_gain)
        retreat = min(s.retreat_max_hz, expansion * s.retreat_gain)
        vis = s.vis_max_hz / (1.0 + dist / s.vis_falloff_px)
        steer = s.steer_max_hz * max(0.0, 1.0 - dist / s.steer_range_px)
        side = math.sin(bearing)                         # >0: cursor to the right
        e = self.engine
        fi, fr = [], []
        for name, hz in (("LOOM", loom), ("RETREAT_IN", retreat), ("VIS", vis), ("BULK_DRIVE", s.bulk_hz),
                         ("LC10_R", steer * max(0.0, side)), ("LC10_L", steer * max(0.0, -side)),
                         ("TOUCH_R", s.touch_hz * touch * (side >= 0)), ("TOUCH_L", s.touch_hz * touch * (side < 0))):
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
        cs = np.concatenate([[0], np.cumsum(e.pop_counts(self._mon_idx), dtype=np.int64)])
        for j, k in enumerate(self.rates):
            lo, hi = self._mon_edges[j], self._mon_edges[j + 1]
            if hi > lo:
                inst = (cs[hi] - cs[lo]) / ((hi - lo) * ms / 1000.0)
                self.rates[k] += a * (inst - self.rates[k])
        self._decode(ms)
        return self.state

    def _decode(self, ms: float):
        """Priority: take-off, then grooming, then retreat, then walk / stand."""
        s = self.spec
        gf, mdn = self.rates["GF"], self.rates["MDN"]
        walk, rest = self.rates["WALK"], self.rates["REST"]
        self._dwell += ms
        cur = self.state
        new = cur
        groom = self.rates["GROOM"]
        if gf > s.gf_on_hz:
            new = FLY                                   # escape: immediate
        elif cur == FLY:
            if gf < s.gf_off_hz and self._dwell >= s.min_fly_ms:
                new = (GROOM if groom > s.groom_on_hz else RETREAT if mdn > s.mdn_on_hz
                       else WALK if walk > max(rest, s.walk_min_hz) else STAND)
        elif cur == GROOM:
            if groom < s.groom_off_hz and self._dwell >= s.min_groom_ms:
                new = RETREAT if mdn > s.mdn_on_hz else (WALK if walk > max(s.walk_min_hz, rest * 1.2) else STAND)
        elif s.groom_on_hz > 0 and groom > s.groom_on_hz:
            new = GROOM
        elif cur == RETREAT:
            if mdn < s.mdn_off_hz and self._dwell >= s.min_retreat_ms:
                new = WALK if walk > max(s.walk_min_hz, rest * 1.2) else STAND
        elif mdn > s.mdn_on_hz:
            new = RETREAT
        elif self._dwell >= s.min_dwell_ms:             # debounce stand <-> walk
            new = WALK if walk > max(s.walk_min_hz, rest * 1.2) else STAND
        if new != cur:
            self.state, self._dwell = new, 0.0
