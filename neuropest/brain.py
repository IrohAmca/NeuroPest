"""Glue between the outside world and the engine.

stimulus (cursor) -> Poisson drive on named neuron groups
spikes of output groups -> smoothed rates -> behavior state (stand / walk / fly / retreat / groom / freeze) and a steering signal

Stimulus-to-rate maps live in a `BrainSpec`, one per circuit kind, because each circuit has its
own units: the toy circuit's weights are arbitrary, the FlyWire one is calibrated against the
measured response of the anchor neurons (tools/probe_circuit.py).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .engine import Network, create_engine
from .mushroom import MushroomBody
from .states import FLY, FREEZE, GROOM, RETREAT, STAND, STATES, WALK  # noqa: F401  (re-exported)


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
    gf_on_hz: float = 15.0            # rate path to take-off (smoothed GF rate); the spike event below is the fast path
    gf_off_hz: float = 3.0
    gf_event_spikes: int = 2          # take-off as soon as this many GF spikes (both neurons together) fall in one
    gf_event_ms: float = 10.0         # window of at least this long (whole engine chunks); 0 spikes = rate path only
    mdn_on_hz: float = 5.0
    mdn_off_hz: float = 2.0
    groom_on_hz: float = 8.0          # mean rate of the grooming DNs (aDN1, aDN2)
    groom_off_hz: float = 3.0
    walk_min_hz: float = 3.0
    walk_on_hz: float = 3.5           # C1: hysteresis threshold to initiate walk
    walk_off_hz: float = 2.0          # C1: hysteresis threshold to stop walk
    min_dwell_ms: float = 600.0       # a stand/walk state lasts at least this long
    min_fly_ms: float = 300.0
    min_retreat_ms: float = 400.0
    min_groom_ms: float = 800.0
    # freeze: a DESIGNED rule on the looming input, not a connectome readout (the model has no neuron that stops the
    # fly). A looming that is too weak for retreat or take-off stops a walking fly for a while (most walking flies freeze
    # to a loom and few jump: Zacarias et al. 2018). `freeze_on` is the smoothed expansion rate of the nearest object (1/s,
    # the same number the looming drive is made of, about 2 for the retreat and 7 for the take-off threshold); 0 = off.
    freeze_on: float = 0.0
    freeze_off: float = 0.3
    min_freeze_ms: float = 1200.0


FREEZE_ON = 0.7         # see BrainSpec.freeze_on; tools/vision_calibrate.py (image) and the cursor numbers agree on it

# Toy circuit: calibrated with tools/calibrate.py
TOY = BrainSpec(loom_gain=10.0, loom_max_hz=250.0, vis_max_hz=60.0, rest_drive=(500.0, 3.0), bulk_hz=150.0,
               gf_event_spikes=0)
# FlyWire circuit (15,000-neuron tier, within ~2% of the full brain; tools/probe_circuit.py, probe_combo.py):
#   LPLC2 + LC4 -> GF: 15 Hz at ~6 Hz of input, 63 Hz at 20 Hz, 200 Hz at 80 Hz
#   LPC1 -> MDN: ~0 up to 12 Hz of input, 15 Hz at 20 Hz, 26 Hz at 30 Hz  (LPC1 is a functional placeholder for
#   the retreat input: see flywire.make_groups and tools/probe_retreat.py)
#   LC10 on one side -> DNa02 on that side: 24 Hz at 10 Hz of input, 79 Hz at 20 Hz
#   LPC1 input suppresses GF: at 20 Hz of LPC1 the take-off needs ~20+ Hz of LPLC2/LC4 input.
#   TOUCH (head bristles + Johnston's organ C/E) on one side -> aDN1/aDN2 (GROOM): ~0 at 25 Hz, ~5 Hz
#   at 50 Hz, 25 to 40 Hz at 100 Hz (mean of the four neurons); it also suppresses GF (tools/probe_touch.py)
# So retreat input saturates at 20 Hz (MDN ~15 Hz), and take-off wins only when the looming input is
# strong: retreat from about 2 /s of expansion, take-off from about 7 /s.
FLYWIRE = BrainSpec(loom_gain=3.0, loom_max_hz=150.0, retreat_gain=8.0, retreat_max_hz=20.0,
                    steer_max_hz=20.0, touch_hz=120.0, freeze_on=FREEZE_ON)


def spec_for(net: Network) -> BrainSpec:
    return FLYWIRE if net.meta.get("kind") == "flywire" else TOY


class Brain:
    def __init__(self, net: Network, dt: float = 0.5, seed: int = 0, ema_ms: float = 80.0,
                 spec: BrainSpec | None = None, backend: str = "cpu", adapter: int | None = None,
                 learning: bool = True):
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
        self._gf_slot = list(self.rates).index("GF")
        self._gf_hist: deque = deque()      # (chunk ms, GF spikes in it), newest last: the window of the spike event
        self.gf_window = 0                  # GF spikes in that window now
        self.state = STAND
        self._dwell = 0.0
        self.walk_bias = 0.0
        self.skittish = 1.0
        self.arousal = 0.0                  # B3: defensive arousal state (0..1)
        self.tau_arousal_s = 30.0           # slow arousal decay constant (Gibson et al. 2015)
        self._last_arousal_drive = 0.0
        self.walk_smoothed = 0.0            # C1: 400 ms smoothed DNp09 rate for robust walking decision
        self.walk_smooth_ms = 400.0
        self._stim = (1e6, 0.0, 0.0, 0.0)
        self._vision_expansion = 0.0
        self._vision = None        # (neuron indices, rates in Hz) from the image, replaces the cursor numbers
        self._loom_in = 0.0        # expansion rate (1/s) of the nearest object now, from the cursor numbers or the image
        self.loom = 0.0            # the same, smoothed like the rates: what the freeze rule reads
        # learned valence (mushroom body, mushroom.py): a designed layer, 0 for a naive fly
        self.mb = MushroomBody() if learning else None
        self.valence = 0.0                  # -1 fear .. +1 desire, for the cues present now
        self._mb_ms = 0.0
        self._reward = 0.0                  # PAM drive: feeding, held ~0.8 s after the contact
        self._punish = 0.0                  # PPL1 drive: threat read from the spiking escape / retreat outputs
        self._valence_driven = 0.0          # the valence the drive was last built with
        self.phero_steer = 0.0     # bilateral tropotaxis steering bias (Hz)
        self.phero_drive = 0.0     # attractive pheromone intensity (0..1)
        self.phero_repel = 0.0     # repulsive warning/border intensity (0..1)
        self.at_target = 0.0       # 1.0 if arrived at attractive source
        self.flight_urge = 0.0     # accumulated motivation to initiate long-mode pursuit flight
        # central motor pool (diagram): V_motor = threat + desire + unreachability tension. A READOUT of what drives the
        # state machine below, not its input: the state still comes from the spiking descending neurons.
        self.v_threat = self.v_desire = 0.0
        self.v_motor = 0.0
        self.flight_mode = ""      # "short" (escape, Giant Fiber) or "long" (pursuit, non-GF) while in FLY
        self._last_fi: np.ndarray | None = None
        self._last_fr: np.ndarray | None = None
        self._last_ci: np.ndarray | None = None
        self._last_cr: np.ndarray | None = None
        self._last_cw: np.ndarray | None = None
        self._drive()

    def group(self, name: str) -> np.ndarray:
        return self.g.get(name, np.zeros(0, np.int32))

    def set_vision(self, idx: np.ndarray, rates: np.ndarray, expansion: float = 0.0):
        """Forced spike rates of individual projection neurons computed from the screen image (see vision.py).

        While set, they replace the looming / retreat / bearing drive derived from the cursor position.
        `expansion` is the strongest looming expansion rate in the image (1/s), the input of the freeze rule."""
        self._vision = (np.asarray(idx, np.int32), np.asarray(rates, np.float32))
        self._vision_expansion = float(expansion)
        self._drive()

    def clear_vision(self):
        self._vision = None
        self._drive()

    @property
    def gear(self) -> str:
        """The diagram's gear: stand, walk, fly_short (escape), fly_long (chase / search); other states count as stand."""
        if self.state == FLY:
            return "fly_long" if self.flight_mode == "long" else "fly_short"
        return WALK if self.state == WALK else STAND

    @property
    def steer(self) -> float:
        """Right minus left DNa02 rate (Hz); positive turns the fly to its right (clockwise on screen)."""
        return self.rates["DNa02_R"] - self.rates["DNa02_L"]

    # ----------------------------------------------------------------- input
    def set_stimulus(self, dist: float, closing_speed: float, walk_bias: float | None = None,
                     skittish: float | None = None, bearing: float | None = None, touch: float = 0.0,
                     phero_steer: float = 0.0, phero_drive: float = 0.0, phero_repel: float = 0.0,
                     at_target: float = 0.0):
        """dist: px to the cursor; closing_speed: px/s, positive when the cursor approaches;
        bearing: angle of the cursor relative to the fly's heading, radians, positive to the right.
        touch: 0..1, the cursor is on the fly (hover); it touches the side given by the bearing.

        walk_bias in [0, 1]: tonic drive of the walking command neurons.
        skittish: multiplier on the looming and retreat sensitivity (1 = calibrated).
        phero_steer: right minus left tropotaxis bias (Hz), drives DNa02 steering.
        phero_drive: attractive pheromone intensity (0..1), boosts walking/foraging.
        phero_repel: repulsive warning/border intensity (0..1).
        at_target: 1.0 when arrived at target/nectar source."""
        self._stim = (dist, closing_speed, self._stim[2] if bearing is None else bearing, touch)
        if walk_bias is not None:
            self.walk_bias = walk_bias
        if skittish is not None:
            self.skittish = skittish
        self.phero_steer = phero_steer
        self.phero_drive = phero_drive
        self.phero_repel = phero_repel
        self.at_target = at_target
        self._drive()

    def _drive(self):
        s = self.spec
        dist, closing, bearing, touch = self._stim
        v = self.valence
        self._valence_driven = v
        desire, fear = max(0.0, v), max(0.0, -v)
        effective_skittish = self.skittish * (1.0 + 0.5 * self.arousal) * (1.0 + 0.8 * fear)
        effective_walk_bias = min(1.0, (self.walk_bias + 0.65 * self.phero_drive + 0.4 * desire)
                                  * (1.0 + 0.3 * self.arousal))
        # Food odor / attractant actively suppresses predator looming escape (GF) and backward retreat (MDN)
        if self.phero_drive > 0.05:
            expansion = 0.0
            loom = 0.0
            retreat = min(s.retreat_max_hz, self.phero_repel * 8.0)
        else:
            expansion = max(0.0, closing) / max(dist, 30.0) * effective_skittish
            loom = min(s.loom_max_hz, expansion * s.loom_gain)
            retreat = min(s.retreat_max_hz, expansion * s.retreat_gain + self.phero_repel * 8.0)

        # High chemical alarm repulsion excites the Giant Fiber escape pathway
        if self.phero_repel > 0.6:
            alarm_takeoff_hz = min(s.loom_max_hz, (self.phero_repel - 0.6) * 200.0)
            loom = max(loom, alarm_takeoff_hz)

        # High accumulated voluntary pursuit urge drives takeoff through the motor pathway
        if self.flight_urge >= 1.0:
            loom = max(loom, 80.0)

        vis = s.vis_max_hz / (1.0 + dist / s.vis_falloff_px)
        steer = s.steer_max_hz * max(0.0, 1.0 - dist / s.steer_range_px)
        side = math.sin(bearing)                         # >0: cursor to the right

        # Bilateral steering drive combining cursor angle and pheromone tropotaxis
        # learned valence scales it: toward what the fly learned to want, away from what it learned to fear
        gain = min(2.5, max(-1.0, 1.0 + 1.5 * v))
        cur_r, cur_l = steer * max(0.0, side), steer * max(0.0, -side)
        ph_r, ph_l = max(0.0, self.phero_steer), max(0.0, -self.phero_steer)
        if gain < 0:                                     # aversion reverses the turn: the opposite side is driven
            cur_r, cur_l, ph_r, ph_l = cur_l, cur_r, ph_l, ph_r
        cur_r, cur_l, ph_r, ph_l = (abs(gain) * x for x in (cur_r, cur_l, ph_r, ph_l))

        e = self.engine
        fi, fr = [], []
        # an object in contact or food being approached is not an attacking predator
        self._loom_in = 0.0 if (touch > 0 or self.phero_drive > 0.05) else (
            self._vision_expansion if self._vision is not None
            else max(0.0, closing) / max(dist, 30.0)) * effective_skittish
        if self._vision is not None:                     # the image replaces the visual part of the cursor drive
            fi.append(self._vision[0])
            fr.append(self._vision[1])
            cursor_drive = (("VIS", vis), ("BULK_DRIVE", s.bulk_hz),
                            ("LC10_R", ph_r), ("LC10_L", ph_l))
        else:
            cursor_drive = (("LOOM", loom), ("RETREAT_IN", retreat), ("VIS", vis), ("BULK_DRIVE", s.bulk_hz),
                            ("LC10_R", cur_r + ph_r), ("LC10_L", cur_l + ph_l))
        touch_drive = (("TOUCH_R", s.touch_hz * touch * (side >= 0)), ("TOUCH_L", s.touch_hz * touch * (side < 0)))
        for name, hz in cursor_drive + touch_drive:      # touch is mechanical, not visual: it always comes from the cursor
            idx = self.group(name)
            fi.append(idx)
            fr.append(np.full(len(idx), hz))

        new_fi = np.concatenate(fi)
        new_fr = np.concatenate(fr)
        drive_changed = True
        if self._last_fi is not None and self._last_fr is not None:
            if len(self._last_fi) == len(new_fi) and np.array_equal(self._last_fi, new_fi):
                if np.max(np.abs(self._last_fr - new_fr)) < 0.2:
                    drive_changed = False
        if drive_changed:
            self._last_fi, self._last_fr = new_fi, new_fr
            e.set_drive(new_fi, new_fr)

        ci, cr, cw = [], [], []
        walk, rest = self.group("WALK"), self.group("REST")
        ci += [walk]
        cr += [np.full(len(walk), s.walk_drive[0] * effective_walk_bias)]
        cw += [np.full(len(walk), s.walk_drive[1])]
        if s.rest_drive is not None:
            ci += [rest]
            cr += [np.full(len(rest), s.rest_drive[0])]
            cw += [np.full(len(rest), s.rest_drive[1])]

        new_ci = np.concatenate(ci)
        new_cr = np.concatenate(cr)
        new_cw = np.concatenate(cw)
        curr_changed = True
        if self._last_ci is not None and self._last_cr is not None and self._last_cw is not None:
            if len(self._last_ci) == len(new_ci) and np.array_equal(self._last_ci, new_ci):
                if np.max(np.abs(self._last_cr - new_cr)) < 0.2 and np.max(np.abs(self._last_cw - new_cw)) < 1e-4:
                    curr_changed = False
        if curr_changed:
            self._last_ci, self._last_cr, self._last_cw = new_ci, new_cr, new_cw
            e.set_current_drive(new_ci, new_cr, new_cw)

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
        self.loom += a * (self._loom_in - self.loom)
        self._track_gf_spikes(ms, int(cs[self._mon_edges[self._gf_slot + 1]] - cs[self._mon_edges[self._gf_slot]]))
        # B3: decay defensive arousal
        if self.arousal > 1e-4:
            decay = math.exp(-ms / (self.tau_arousal_s * 1000.0))
            self.arousal = max(0.0, self.arousal * decay)
            if abs(self.arousal - self._last_arousal_drive) > 0.05:
                self._last_arousal_drive = self.arousal
                self._drive()
        self._decode(ms)
        if self.mb is not None:
            self._learn(ms)
        return self.state

    MB_STEP_MS = 50.0

    def _learn(self, ms: float):
        """Mushroom-body step every 50 ms: cues in, dopamine from the unconditioned stimuli, valence out.

        Reward (PAM) = feeding at a pheromone source. Punishment (PPL1) = threat read from the spiking outputs: Giant
        Fiber (escape), MDN (backward retreat), touch. A chase (food odor / pursuit urge) is not a threat even though it
        drives the Giant Fiber, so punishment is held off then."""
        s = self.spec
        self._reward = 1.0 if self.at_target > 0.5 else self._reward * math.exp(-ms / 800.0)
        threat = max(self.rates["GF"] / (2.0 * s.gf_on_hz), self.rates["MDN"] / (3.0 * s.mdn_on_hz))
        if self.gf_window >= max(1, s.gf_event_spikes):
            threat = 1.0
        if self.phero_drive > 0.05 or self.flight_urge >= 0.5:
            threat = 0.0
        self._punish = min(1.0, threat)
        self._mb_ms += ms
        if self._mb_ms < self.MB_STEP_MS:
            return
        dt, self._mb_ms = self._mb_ms / 1000.0, 0.0
        dist, _, _, touch = self._stim
        cues = (self.phero_drive, self.phero_repel, 1.0 / (1.0 + dist / 250.0) if dist < 1e5 else 0.0,
                min(1.0, self.loom / 3.0), touch)
        self.valence = self.mb.step(cues, self._reward, self._punish, dt)
        if abs(self.valence - self._valence_driven) > 0.05:
            self._drive()

    def _track_gf_spikes(self, ms: float, spikes: int):
        """Keep the GF spike counts of the last `gf_event_ms` (rounded up to whole chunks)."""
        h = self._gf_hist
        h.append((ms, spikes))
        self.gf_window += spikes
        while len(h) > 1 and sum(m for m, _ in h) - h[0][0] >= self.spec.gf_event_ms:
            self.gf_window -= h.popleft()[1]

    def _decode(self, ms: float):
        """Priority: take-off, retreat, freeze, grooming, then walk / stand (a higher one interrupts a lower one at
        once; leaving a state needs its hysteresis and minimum duration)."""
        s = self.spec
        gf, mdn = self.rates["GF"], self.rates["MDN"]
        walk, rest = self.rates["WALK"], self.rates["REST"]
        self._dwell += ms
        cur = self.state
        new = cur
        groom = self.rates["GROOM"]
        retreat_on = mdn > s.mdn_on_hz
        freeze_on = s.freeze_on > 0 and self.loom > s.freeze_on
        groom_on = s.groom_on_hz > 0 and groom > s.groom_on_hz

        # C1: 400 ms decision reader smoothing and hysteresis
        a_w = 1.0 - math.exp(-ms / self.walk_smooth_ms)
        self.walk_smoothed += a_w * (walk - self.walk_smoothed)
        if cur == WALK:
            quiet = STAND if self.walk_smoothed < max(s.walk_off_hz, rest * 1.2) else WALK
        else:
            quiet = WALK if self.walk_smoothed > max(s.walk_on_hz, rest * 1.2) else STAND

        def settle(skip=()):
            return next((st for st, on in ((RETREAT, retreat_on), (FREEZE, freeze_on), (GROOM, groom_on))
                         if on and st not in skip), quiet)

        # V_motor terms. threat: how hard the escape / retreat outputs and the alarm odor push; desire: the odor plus
        # what the fly learned to want; tension: the urge to fly that builds while a wanted goal stays out of reach.
        self.v_threat = min(1.5, max(gf / (2.0 * s.gf_on_hz), mdn / (2.0 * s.mdn_on_hz), self.phero_repel))
        self.v_desire = min(1.5, self.phero_drive + max(0.0, self.valence))
        self.v_motor = self.v_threat + self.v_desire + self.flight_urge

        # Long-mode voluntary takeoff / goal-directed pursuit accumulation; learned desire speeds it up, a threat
        # stops it (a chase does not start under attack)
        if cur == WALK and self.phero_drive > 0.25 and self.at_target < 0.5 and self.v_threat < 0.5:
            pull = self.phero_drive * (1.0 + max(0.0, self.valence))
            self.flight_urge = min(1.2, self.flight_urge + (ms / 1000.0) * (pull * 0.85))
        else:
            self.flight_urge = max(0.0, self.flight_urge - (ms / 1000.0) * 0.35)

        # take-off is an event, not a rate: a few GF spikes in ~10 ms (the animal takes off on one or two);
        event = s.gf_event_spikes > 0 and self.gf_window >= s.gf_event_spikes
        if event or gf > s.gf_on_hz:
            new = FLY                                   # take-off: driven directly by Giant Fiber neural output
            if cur != FLY:
                self.flight_mode = "long" if self.flight_urge >= 0.5 and self.v_threat < 0.5 else "short"
        elif cur == FLY:
            landing_ready = False
            # Dynamic landing decision: not hardcoded time
            if self._dwell >= 180.0:                    # min takeoff completion dwell
                if self.at_target > 0.5:                # arrived at target/nectar source -> land on target
                    landing_ready = True
                elif gf < s.gf_off_hz and self.phero_repel < 0.25 and self.loom < 0.3 and self.flight_urge <= 0.2:
                    landing_ready = True                # threat has dissipated -> land on safe surface
                elif self._dwell >= 4500.0:             # flight fatigue safeguard
                    landing_ready = True
            if landing_ready:
                new = settle()
                self.flight_urge = 0.0
        elif cur == RETREAT:
            if mdn < s.mdn_off_hz and self._dwell >= s.min_retreat_ms:
                new = settle(skip=(RETREAT,))
        elif retreat_on:
            new = RETREAT
        elif cur == FREEZE:
            if self.loom < s.freeze_off and self._dwell >= s.min_freeze_ms:
                new = GROOM if groom_on else quiet
        elif freeze_on:
            new = FREEZE
        elif cur == GROOM:
            if groom < s.groom_off_hz and self._dwell >= s.min_groom_ms:
                new = quiet
        elif groom_on:
            new = GROOM
        elif self._dwell >= s.min_dwell_ms:             # debounce stand <-> walk
            new = quiet
        if new != cur:
            self.state, self._dwell = new, 0.0
            if new != FLY:
                self.flight_mode = ""
            # B3: defensive arousal surge on threat
            if new == FLY:
                self.arousal = min(1.0, self.arousal + 0.5)
                self._last_arousal_drive = self.arousal
                self._drive()
            elif new == RETREAT:
                self.arousal = min(1.0, self.arousal + 0.05)
                self._last_arousal_drive = self.arousal
                self._drive()
            elif new == FREEZE:
                self.arousal = min(1.0, self.arousal + 0.02)
                self._last_arousal_drive = self.arousal
                self._drive()
