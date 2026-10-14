"""Glue between the outside world and the engine.

stimulus (cursor) -> Poisson drive on named neuron groups
spikes of output groups -> smoothed rates -> behavior state (stand / walk / fly / retreat / groom / freeze) and a steering signal

Stimulus-to-rate maps live in a `BrainSpec`, one per circuit kind, because each circuit has its
own units: the toy circuit's weights are arbitrary, the FlyWire one is calibrated against the
measured response of the anchor neurons (tools/probe_circuit.py).
"""
from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from .engine import Network, create_engine
from .mushroom import MushroomBody
from .states import FLY, FREEZE, GROOM, LAB_GROUPS, RETREAT, STAND, STATES, WALK  # noqa: F401  (re-exported)


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

# Food odor and the startle reflexes. A hungry fly that smells food is less ready to treat what approaches as a predator:
# the looming, retreat and freeze reflexes are scaled by `odor_gate`, which falls from 1 (no odor) to 0 at an odor
# strength (phero_drive x hunger) of ODOR_GATE_FULL. A faint plume edge costs a little sensitivity, the middle of a plume
# or the cursor's own lure silences them (the cursor is food there, not a predator). One function for every place that
# applies it, so they can not drift apart: the cursor-number expansion and the freeze input here (the worker passes the
# raw closing speed), and the image path in the worker (runner._Eye scales the detector gain, VisionDrive.skittish, so
# the gate acts before the per-cell rate caps and the freeze -> retreat -> take-off order is kept). 0.2 keeps the lure working down to hunger 0.2 (the default start is 0.35) while a fast
# approach from outside the plume still startles: calibrated against the old hard 0.05 gate in tools/probe_odor_gate.py
# (0.3 and 0.5 let the pet retreat from or take off at its own attractive cursor when it is only a little hungry).
ODOR_GATE_FULL = 0.2
ODOR_LEARN_GATE = 0.05  # below this odor strength an escape still teaches fear; above it a chase is not punished


def odor_gate(strength: float) -> float:
    """Sensitivity of the startle reflexes under a food odor of `strength` (phero_drive x hunger): 1 .. 0."""
    return max(0.0, 1.0 - float(strength) / ODOR_GATE_FULL)


# The neuroscience lab (Brain.set_lab, the 3D view's tools) [DESIGNED]: silencing clamps a group's neurons in the engine
# (they cannot spike, so nothing downstream hears them; an acute GtACR-like silencing), stimulation forces their spikes
# (Poisson, like a light-gated channel). Whatever the fly does then comes from the circuit's own response.
LAB_MAX_HZ = 300.0          # cap on a forced lab rate
LAB_PULSE_MS = 300.0        # one click's light pulse
LAB_HOLD_MS = 10_000.0      # every stimulation ends by itself after at most this (a held button included)
LAB_TAIL_MS = 600.0         # plasticity stays paused this long after the last manipulation ends (7.5 rate EMAs)
LAB_RATE_GROUPS = ("LC4", "LPLC2", "LPC1", "LC10_L", "LC10_R")   # read every chunk like the outputs (lab readouts)

# Foraging flight toward a food source [DESIGNED]: a hungry fly that smells a source it is not getting closer to on foot
# builds an urge to fly there and, at 1, takes off in long mode WITHOUT the Giant Fiber (von Reyn et al. 2014: long-mode
# take-offs do not need it) and lands at the food (fly.py steers it there: the GUI tells where the source is, the same kind
# of shortcut as the cursor bearing into LC10). Escape (a GF event) stays the only short-mode take-off.
GOAL_REACH_PX = 180.0       # nearer than this the fly walks
GOAL_MAX_PX = 650.0
GOAL_ODOR_MIN = 0.015       # the goal's own odor at the fly x hunger
GOAL_HUNGER_MIN = 0.45
GOAL_URGE_RATE = 0.8        # urge per second at full pull (it takes off at 1)
GOAL_WALK_PX_S = 70.0       # walking speed (fly.py): closing in this fast on foot means walking works (no urge)
GOAL_LOST_MS = 400.0        # the goal gone this long: land
GOAL_MAX_FLIGHT_MS = 5000.0
GOAL_REFRACTORY_MS = 3000.0
GOAL_MEAL_REFRACTORY_MS = 8000.0
CUE_PULSE_MS = 800.0        # a learning-sandbox cue is shown to the mushroom body this long

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
                 learning: bool = True, mb_wiring: str = "random"):
        self.net = net
        self.spec = spec or spec_for(net)
        self.engine = create_engine(net, dt=dt, seed=seed, backend=backend, adapter=adapter)
        self.g = dict(net.groups)                       # own dict: the lab names LPC1, the retreat input, as well
        if "RETREAT_IN" in self.g:
            self.g.setdefault("LPC1", self.g["RETREAT_IN"])
        self.ema_ms = ema_ms
        self.rates = {k: 0.0 for k in ("GF", "WALK", "REST", "MDN", "DNa02_L", "DNa02_R", "GROOM")}
        self.lab_rates = {k: 0.0 for k in LAB_RATE_GROUPS if len(self.group(k))}   # visual inputs the lab reads
        # all output groups are read with one call per chunk (on a GPU every read waits for the device)
        self._mon_keys = list(self.rates) + list(self.lab_rates)
        parts = [self.group(k) for k in self._mon_keys]
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
        self._vision_cues = None   # (cursor, looming) drive of the mushroom-body visual PNs from the image, or None
        self._loom_in = 0.0        # expansion rate (1/s) of the nearest object now, from the cursor numbers or the image
        self.loom = 0.0            # the same, smoothed like the rates: what the freeze rule reads
        # learned valence (mushroom body, mushroom.py): a designed layer, 0 for a naive fly
        self.mb = None
        if learning:
            self.mb = MushroomBody()
            if mb_wiring == "flywire":                  # opt-in: the real FlyWire KC / MBON / DAN / PN wiring
                try:
                    self.mb = MushroomBody.from_flywire()
                except Exception as e:                  # not built, empty, damaged or of another schema
                    # (OSError, EOFError, BadZipFile, zlib.error, KeyError ...): a saved preference for the real cells must
                    # not keep the engine from starting; the warning names what went wrong
                    warnings.warn(f"real mushroom-body wiring not available ({type(e).__name__}: {e}); "
                                  "using the random model")
        self.valence = 0.0                  # -1 fear .. +1 desire, for the cues present now
        self._last_cues = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._mb_ms = 0.0
        self._reward = 0.0                  # PAM drive: feeding, held ~0.8 s after the contact
        self._punish = 0.0                  # PPL1 drive: threat read from the spiking escape / retreat outputs
        self._valence_driven = 0.0          # the valence the drive was last built with
        self.phero_steer = 0.0     # bilateral tropotaxis steering bias (Hz)
        self.phero_drive = 0.0     # attractive pheromone intensity (0..1)
        self.phero_repel = 0.0     # repulsive warning/border intensity (0..1)
        self.at_target = 0.0       # 1.0 if arrived at attractive source
        self.hunger = 1.0          # internal hunger drive in [0, 1]
        self.flight_urge = 0.0     # accumulated motivation to initiate long-mode pursuit flight
        self.wall_bump = 0.0       # mechanical wall collision impact [0, 1]
        # central motor pool (diagram): V_motor = threat + desire + unreachability tension. A READOUT of what drives the
        # state machine below, not its input: the state still comes from the spiking descending neurons.
        self.v_threat = self.v_desire = 0.0
        self.v_motor = 0.0
        self.flight_mode = ""      # "short" (escape, Giant Fiber) or "long" (foraging flight, no GF) while in FLY
        # the food the GUI says the fly would fly to (set_goal): gap to it (px, <= 0 at it), bearing, its odor here
        self.goal_gap, self.goal_bearing, self.goal_odor = 1e6, 0.0, 0.0
        self._gap_prev: float | None = None
        self._gap_rate = 0.0                    # px/s, smoothed; negative while closing in
        self._fed_events = 0                    # meals so far (note_fed)
        self._goal_fed0 = 0                     # ... when the foraging flight took off
        self._goal_lost_ms = 0.0
        self._goal_refractory_ms = 0.0
        # learning sandbox: a cue shown to the mushroom body for a moment (present_cue)
        self._cue_food = self._cue_food_ms = 0.0
        self._cue_near = self._cue_near_ms = 0.0
        # the neuroscience lab (set_lab)
        self._lab_sil: tuple[str, ...] = ()
        self._lab_stim: dict[str, float] = {}
        self._lab_fi = np.zeros(0, np.int32)
        self._lab_fr = np.zeros(0, np.float32)
        self._lab_left_ms = 0.0
        self._lab_tail_ms = 0.0
        self._last_fi: np.ndarray | None = None
        self._last_fr: np.ndarray | None = None
        self._last_ci: np.ndarray | None = None
        self._last_cr: np.ndarray | None = None
        self._last_cw: np.ndarray | None = None

        # Pre-allocate static drive buffers for cursor and current inputs
        cursor_groups = ("LOOM", "RETREAT_IN", "VIS", "BULK_DRIVE", "LC10_R", "LC10_L", "TOUCH_R", "TOUCH_L")
        c_parts = [self.group(k) for k in cursor_groups]
        self._cursor_fi = np.concatenate(c_parts).astype(np.int32) if c_parts else np.zeros(0, np.int32)
        self._cursor_slices = []
        cur_offset = 0
        for p in c_parts:
            lp = len(p)
            self._cursor_slices.append(slice(cur_offset, cur_offset + lp))
            cur_offset += lp
        self._cursor_fr = np.zeros(len(self._cursor_fi), dtype=np.float32)

        walk = self.group("WALK")
        rest = self.group("REST") if self.spec.rest_drive is not None else np.zeros(0, np.int32)
        self._curr_ci = np.concatenate([walk, rest]).astype(np.int32)
        self._walk_slice = slice(0, len(walk))
        self._rest_slice = slice(len(walk), len(walk) + len(rest))
        self._curr_cr = np.zeros(len(self._curr_ci), dtype=np.float32)
        self._curr_cw = np.zeros(len(self._curr_ci), dtype=np.float32)
        self._curr_cw[self._walk_slice] = self.spec.walk_drive[1]
        if self.spec.rest_drive is not None:
            self._curr_cw[self._rest_slice] = self.spec.rest_drive[1]

        self._drive()

    def group(self, name: str) -> np.ndarray:
        return self.g.get(name, np.zeros(0, np.int32))

    def set_vision(self, idx: np.ndarray, rates: np.ndarray, expansion: float = 0.0,
                   mb_cues: tuple[float, float] | None = None):
        """Forced spike rates of individual projection neurons computed from the screen image (see vision.py).

        While set, they replace the looming / retreat / bearing drive derived from the cursor position.
        `expansion` is the strongest looming expansion rate in the image (1/s), the input of the freeze rule.
        `mb_cues` (cursor, looming), each 0..1, is the drive of the visual projection neurons that reach the Kenyon
        cells (visual.VisionDrive.mb_cue): the mushroom body reads these instead of the cursor numbers.
        The rates and cues already carry the food-odor gate: the worker scales the detector gain by odor_gate."""
        self._vision = (np.asarray(idx, np.int32), np.asarray(rates, np.float32))
        self._vision_expansion = float(expansion)
        self._vision_cues = mb_cues
        self._drive()

    def clear_vision(self):
        self._vision = None
        self._vision_cues = None
        self._drive()

    # ----------------------------------------------------------------- the lab
    def set_lab(self, silenced=None, stimulated: dict | None = None, stim_ms: float = LAB_PULSE_MS):
        """Silence groups and / or stimulate groups (names from states.LAB_GROUPS); None leaves that part as it is, an
        empty one ends it.

        silenced: the groups whose neurons are clamped (engine.set_silenced): they cannot spike, whatever drives them.
        stimulated: {group: Hz}, forced Poisson spikes on every neuron of the group for stim_ms of simulated time (each
            stimulation ends by itself, after LAB_HOLD_MS at most; a new one replaces it). Silence wins over it.
        Groups this circuit does not have are dropped. Returns (silenced, stimulated) as applied."""
        was = self.lab_on
        if silenced is not None:
            want = set(silenced)
            sil = tuple(n for n in LAB_GROUPS if n in want and len(self.group(n)))
            if sil != self._lab_sil:
                idx = np.unique(np.concatenate([self.group(n) for n in sil])) if sil else np.zeros(0, np.int32)
                self.engine.set_silenced(idx)
                self._lab_sil = sil
        if stimulated is not None:
            stim = {n: min(float(hz), LAB_MAX_HZ) for n, hz in dict(stimulated).items()
                    if n in LAB_GROUPS and float(hz) > 0.0 and len(self.group(n))}
            if stim:
                fi = np.concatenate([self.group(n) for n in stim]).astype(np.int32)
                fr = np.concatenate([np.full(len(self.group(n)), hz, np.float32) for n, hz in stim.items()])
                li = np.unique(fi)
                lr = np.zeros(len(li), np.float32)
                np.maximum.at(lr, np.searchsorted(li, fi), fr)
                self._lab_fi, self._lab_fr = li.astype(np.int32), lr
                self._lab_left_ms = min(stim_ms if stim_ms > 0 else LAB_HOLD_MS, LAB_HOLD_MS)
            else:
                self._lab_fi, self._lab_fr = np.zeros(0, np.int32), np.zeros(0, np.float32)
                self._lab_left_ms = 0.0
            self._lab_stim = stim
            self._last_fi = None                        # the forced drive is rebuilt with (or without) it now
            self._drive()
        if was and not self.lab_on:
            self._lab_tail_ms = LAB_TAIL_MS
        return self._lab_sil, tuple(self._lab_stim)

    @property
    def lab_on(self) -> bool:
        """A lab manipulation is applied now."""
        return bool(self._lab_sil or self._lab_stim)

    @property
    def lab_active(self) -> bool:
        """... or ended less than LAB_TAIL_MS ago: the mushroom body does not learn, nothing is punished."""
        return self.lab_on or self._lab_tail_ms > 0.0

    @property
    def lab_left_ms(self) -> float:
        return self._lab_left_ms

    def lab_available(self) -> tuple[str, ...]:
        """The lab groups this circuit has."""
        return tuple(n for n in LAB_GROUPS if len(self.group(n)))

    def _merge_lab(self, fi: np.ndarray, fr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The sensory drive with the lab's stimulation on top: one entry per stimulated neuron, at the larger of the
        two rates (the CPU engines fire a neuron listed twice twice per step, the GPU once; one entry keeps them equal)."""
        li, lr = self._lab_fi, self._lab_fr
        fi = np.asarray(fi, np.int32)
        fr = np.asarray(fr, np.float32)
        hit = np.isin(fi, li)
        if hit.any():
            lr = lr.copy()
            np.maximum.at(lr, np.searchsorted(li, fi[hit]), fr[hit])
        return np.concatenate([fi[~hit], li]), np.concatenate([fr[~hit], lr])

    # ----------------------------------------------------------------- goal, meals, sandbox
    def set_goal(self, gap: float, bearing: float, odor: float) -> None:
        """The food source the fly would fly to: gap = px to it minus its eating radius (<= 0: at it), bearing (rad, +
        to the right), odor = that source's own odor at the fly (0: no goal). Stored only; the decision reads it."""
        self.goal_gap, self.goal_bearing, self.goal_odor = float(gap), float(bearing), max(0.0, float(odor))

    def note_fed(self, n: int = 1) -> None:
        """n meals since the last call (a counter in the shared input, so none is lost): ends a foraging flight and is
        the feeding reward even when the one-frame at_target was missed."""
        if n > 0:
            self._fed_events += int(n)
            self._reward = max(self._reward, self.hunger)

    def present_cue(self, kind: str, amount: float = 1.0, ms: float = CUE_PULSE_MS) -> None:
        """Learning sandbox: show the mushroom body a cue for `ms` ("food": the food odor, "near": the cursor close) so a
        reward or punishment given meanwhile is learned for it. Only the mushroom body sees it; the motor drive and the
        reflexes do not change."""
        amount = max(0.0, min(1.0, float(amount)))
        if kind == "food":
            self._cue_food, self._cue_food_ms = max(self._cue_food, amount), float(ms)
        elif kind == "near":
            self._cue_near, self._cue_near_ms = max(self._cue_near, amount), float(ms)

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
                     at_target: float = 0.0, hunger: float = 1.0, wall_bump: float = 0.0):
        """dist: px to the cursor; closing_speed: px/s, positive when the cursor approaches;
        bearing: angle of the cursor relative to the fly's heading, radians, positive to the right.
        touch: 0..1, the cursor is on the fly (hover); it touches the side given by the bearing.

        walk_bias in [0, 1]: tonic drive of the walking command neurons.
        skittish: multiplier on the looming and retreat sensitivity (1 = calibrated).
        phero_steer: right minus left tropotaxis bias (Hz), drives DNa02 steering.
        phero_drive: attractive pheromone intensity (an unclamped sum of plumes, ~0..1), boosts walking/foraging;
            phero_drive x hunger also weakens the looming / retreat / freeze reflexes (odor_gate).
        phero_repel: repulsive warning/border intensity (0..1).
        at_target: 1.0 when arrived at target/nectar source.
        hunger: 0..1 internal metabolic hunger drive.
        wall_bump: 1.0 when colliding with screen boundaries (mechanosensory pain)."""
        self._stim = (dist, closing_speed, self._stim[2] if bearing is None else bearing, touch)
        if walk_bias is not None:
            self.walk_bias = walk_bias
        if skittish is not None:
            self.skittish = skittish
        self.phero_steer = phero_steer
        self.phero_drive = phero_drive
        self.phero_repel = phero_repel
        self.at_target = at_target
        self.hunger = float(hunger)
        self.wall_bump = float(wall_bump)
        self._drive()

    def _drive(self):
        s = self.spec
        dist, closing, bearing, touch = self._stim
        v = self.valence
        h = self.hunger
        self._valence_driven = v
        desire, fear = max(0.0, v), max(0.0, -v)
        effective_skittish = self.skittish * (1.0 + 0.5 * self.arousal) * (1.0 + 0.8 * fear)

        # Hunger gates olfactory sensitivity
        eff_phero_drive = self.phero_drive * h
        effective_walk_bias = min(1.0, (self.walk_bias + 0.65 * eff_phero_drive + 0.4 * desire)
                                  * (1.0 + 0.3 * self.arousal))

        # Food odor / attractant suppresses predator looming escape (GF), backward retreat (MDN) and freezing in
        # proportion to its strength (odor_gate); a satiated fly (eff_phero_drive == 0) keeps normal predator avoidance
        gate = odor_gate(eff_phero_drive)
        expansion = max(0.0, closing) / max(dist, 30.0) * effective_skittish * gate
        loom = min(s.loom_max_hz, expansion * s.loom_gain)
        retreat = min(s.retreat_max_hz, expansion * s.retreat_gain + self.phero_repel * 8.0)

        # Mechanical collision with solid boundaries triggers acute retreat reflex (MDN)
        if self.wall_bump > 0.0:
            retreat = max(retreat, min(s.retreat_max_hz, 16.0 * self.wall_bump))

        # High chemical alarm repulsion excites the Giant Fiber escape pathway
        if self.phero_repel > 0.6:
            alarm_takeoff_hz = min(s.loom_max_hz, (self.phero_repel - 0.6) * 200.0)
            loom = max(loom, alarm_takeoff_hz)

        vis = s.vis_max_hz / (1.0 + dist / s.vis_falloff_px)
        steer = s.steer_max_hz * max(0.0, 1.0 - dist / s.steer_range_px)
        side = math.sin(bearing)                         # >0: cursor to the right

        # Bilateral steering drive combining cursor angle and pheromone tropotaxis
        # learned valence scales it: toward what the fly learned to want, away from what it learned to fear
        gain = min(2.5, max(-1.0, 1.0 + 1.5 * v))
        cur_r, cur_l = steer * max(0.0, side), steer * max(0.0, -side)
        eff_phero_steer = self.phero_steer * h
        ph_r, ph_l = max(0.0, eff_phero_steer), max(0.0, -eff_phero_steer)
        if gain < 0:                                     # aversion reverses the turn: the opposite side is driven
            cur_r, cur_l, ph_r, ph_l = cur_l, cur_r, ph_l, ph_r
        cur_r, cur_l, ph_r, ph_l = (abs(gain) * x for x in (cur_r, cur_l, ph_r, ph_l))

        e = self.engine
        fi, fr = [], []
        # an object in contact or food being approached is not an attacking predator
        self._loom_in = 0.0 if touch > 0 else (
            self._vision_expansion if self._vision is not None
            else max(0.0, closing) / max(dist, 30.0)) * effective_skittish * gate
        if self._vision is not None:                     # the image replaces the visual part of the cursor drive
            fi = [self._vision[0]]
            fr = [self._vision[1]]                       # already odor-gated through the detector gain (runner._Eye)
            cursor_drive = (("VIS", vis), ("BULK_DRIVE", s.bulk_hz),
                            ("LC10_R", ph_r), ("LC10_L", ph_l))
            touch_drive = (("TOUCH_R", s.touch_hz * touch * (side >= 0)), ("TOUCH_L", s.touch_hz * touch * (side < 0)))
            for name, hz in cursor_drive + touch_drive:      # touch is mechanical, not visual: it always comes from the cursor
                idx = self.group(name)
                fi.append(idx)
                fr.append(np.full(len(idx), hz, dtype=np.float32))
            new_fi = np.concatenate(fi)
            new_fr = np.concatenate(fr)
        else:
            rates_tuple = (
                loom, retreat, vis, s.bulk_hz,
                cur_r + ph_r, cur_l + ph_l,
                s.touch_hz * touch * (side >= 0),
                s.touch_hz * touch * (side < 0)
            )
            for sl, r in zip(self._cursor_slices, rates_tuple):
                self._cursor_fr[sl] = r
            new_fi = self._cursor_fi
            new_fr = self._cursor_fr
        if len(self._lab_fi):                            # the lab's stimulation on top
            new_fi, new_fr = self._merge_lab(new_fi, new_fr)

        drive_changed = True
        if self._last_fi is not None and self._last_fr is not None:
            if len(self._last_fi) == len(new_fi) and (self._last_fi is new_fi or np.array_equal(self._last_fi, new_fi)):
                if np.max(np.abs(self._last_fr - new_fr)) < 0.2:
                    drive_changed = False
        if drive_changed:
            self._last_fi = new_fi
            self._last_fr = new_fr.copy()
            e.set_drive(new_fi, new_fr)

        self._curr_cr[self._walk_slice] = s.walk_drive[0] * effective_walk_bias
        if s.rest_drive is not None:
            self._curr_cr[self._rest_slice] = s.rest_drive[0]
        new_ci = self._curr_ci
        new_cr = self._curr_cr
        new_cw = self._curr_cw
        curr_changed = True
        if self._last_ci is not None and self._last_cr is not None and self._last_cw is not None:
            if len(self._last_ci) == len(new_ci) and (self._last_ci is new_ci or np.array_equal(self._last_ci, new_ci)):
                if np.max(np.abs(self._last_cr - new_cr)) < 0.2 and np.max(np.abs(self._last_cw - new_cw)) < 1e-4:
                    curr_changed = False
        if curr_changed:
            self._last_ci = new_ci
            self._last_cr = new_cr.copy()
            self._last_cw = new_cw
            e.set_current_drive(new_ci, new_cr, new_cw)

    # ---------------------------------------------------------------- run
    def advance(self, ms: float) -> str:
        e = self.engine
        e.advance(ms)
        a = 1.0 - math.exp(-ms / self.ema_ms)
        counts = e.pop_counts(self._mon_idx)
        dt_s = ms / 1000.0
        gf_spikes = 0
        for j, k in enumerate(self._mon_keys):
            lo, hi = self._mon_edges[j], self._mon_edges[j + 1]
            if hi > lo:
                grp_sum = int(counts[lo:hi].sum())
                if j == self._gf_slot:
                    gf_spikes = grp_sum
                inst = grp_sum / ((hi - lo) * dt_s)
                rates = self.rates if k in self.rates else self.lab_rates
                rates[k] += a * (inst - rates[k])
        self.loom += a * (self._loom_in - self.loom)
        self._track_gf_spikes(ms, gf_spikes)
        self._tick_timers(ms)
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

    def _tick_timers(self, ms: float) -> None:
        """The lab's pulse (it ends by itself) and its tail, the sandbox cues."""
        if self._lab_left_ms > 0.0:
            self._lab_left_ms -= ms
            if self._lab_left_ms <= 0.0:
                self.set_lab(stimulated={})
        elif self._lab_tail_ms > 0.0 and not self.lab_on:
            self._lab_tail_ms = max(0.0, self._lab_tail_ms - ms)
        if self._cue_food_ms > 0.0:
            self._cue_food_ms -= ms
            if self._cue_food_ms <= 0.0:
                self._cue_food = 0.0
        if self._cue_near_ms > 0.0:
            self._cue_near_ms -= ms
            if self._cue_near_ms <= 0.0:
                self._cue_near = 0.0

    MB_STEP_MS = 50.0

    def _learn(self, ms: float):
        """Mushroom-body step every 50 ms: cues in, dopamine from the unconditioned stimuli, valence out.

        Reward (PAM) = feeding at a pheromone source. Punishment (PPL1) = threat read from the spiking outputs: Giant
        Fiber (escape), MDN (backward retreat), touch. A startle under food odor or during a chase / foraging flight is
        not a threat, so punishment is held off then; while the lab manipulates the circuit nothing is learned at all."""
        s = self.spec
        h = self.hunger
        eff_phero_drive = self.phero_drive * h
        # Reward is strictly gated by hunger (Krashes et al. 2009): satiated feeding does not reinforce
        reward_in = (1.0 if self.at_target > 0.5 else 0.0) * h
        self._reward = reward_in if self.at_target > 0.5 else self._reward * math.exp(-ms / 800.0)
        threat = max(self.rates["GF"] / (2.0 * s.gf_on_hz), self.rates["MDN"] / (3.0 * s.mdn_on_hz))
        if self.gf_window >= max(1, s.gf_event_spikes):
            threat = 1.0
        goal_flight = self.state == FLY and self.flight_mode == "long"
        lab = self.lab_active
        if eff_phero_drive > ODOR_LEARN_GATE or self.flight_urge >= 0.5 or goal_flight or lab:
            threat = 0.0      # a startle under food odor, a chase, or what the lab evoked is not a punishment
        # Physical wall collision delivers acute nociceptive punishment (PPL1 dopamine)
        if self.wall_bump > 0.0 and not goal_flight and not lab:
            threat = max(threat, min(1.0, self.wall_bump))
        punish_in = min(1.0, threat)
        self._punish = punish_in if punish_in > 0.05 else self._punish * math.exp(-ms / 800.0)
        self._mb_ms += ms
        if self._mb_ms < self.MB_STEP_MS:
            return
        dt, self._mb_ms = self._mb_ms / 1000.0, 0.0
        dist, _, _, touch = self._stim
        if self._vision is not None and self._vision_cues is not None:
            near, loom = self._vision_cues                 # seen: the cells' own drive from the image
        else:
            near = 1.0 / (1.0 + dist / 250.0) if dist < 1e5 else 0.0      # numbers standing in for the same stimulus
            loom = min(1.0, self.loom / 3.0)
        cues = (max(eff_phero_drive, self._cue_food), self.phero_repel, max(near, self._cue_near), loom, touch)
        self._last_cues = cues
        if lab:                                            # the lab's pause: the cells fire, no synapse learns
            self.valence = self.mb.step(cues, self._reward, self._punish, dt, plastic=False)
        else:
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
        h = self.hunger
        eff_phero_drive = self.phero_drive * h
        self.v_threat = min(1.5, max(gf / (2.0 * s.gf_on_hz), mdn / (2.0 * s.mdn_on_hz), self.phero_repel))
        self.v_desire = min(1.5, eff_phero_drive + max(0.0, self.valence))
        self.v_motor = self.v_threat + self.v_desire + self.flight_urge

        # Foraging flight (see GOAL_*): the urge builds while a hungry fly smells a food source within flying range that
        # it is not closing in on by walking; learned desire speeds it up, and a threat, an alarm odor or fear stop it
        dt_s = ms / 1000.0
        if self._gap_prev is not None and abs(self.goal_gap - self._gap_prev) < 50.0:
            v_gap = (self.goal_gap - self._gap_prev) / dt_s
        else:
            v_gap = self._gap_rate                       # a new goal (or none): no jump in the rate
        self._gap_prev = self.goal_gap
        self._gap_rate += (1.0 - math.exp(-ms / 300.0)) * (v_gap - self._gap_rate)
        self._goal_refractory_ms = max(0.0, self._goal_refractory_ms - ms)
        goal_ok = (cur in (WALK, STAND) and self.goal_odor > 0.0 and self.goal_odor * h >= GOAL_ODOR_MIN
                   and h >= GOAL_HUNGER_MIN and GOAL_REACH_PX <= self.goal_gap <= GOAL_MAX_PX
                   and self.v_threat < 0.5 and self.phero_repel < 0.25 and self.loom < s.freeze_off
                   and self.valence > -0.2 and self._goal_refractory_ms <= 0.0)
        if goal_ok:
            progress = min(1.0, max(0.0, 1.0 + self._gap_rate / GOAL_WALK_PX_S))   # walking straight at it: no urge
            self.flight_urge = min(1.2, self.flight_urge
                                   + dt_s * GOAL_URGE_RATE * progress * (1.0 + max(0.0, self.valence)))
        else:
            self.flight_urge = max(0.0, self.flight_urge - dt_s * 0.35)

        # take-off is an event, not a rate: a few GF spikes in ~10 ms (the animal takes off on one or two);
        event = s.gf_event_spikes > 0 and self.gf_window >= s.gf_event_spikes
        if event or gf > s.gf_on_hz:
            new = FLY                                   # take-off: driven directly by Giant Fiber neural output
            if cur != FLY or self.flight_mode == "long":
                if cur == FLY:                          # attacked on the way to food: the flight becomes an escape
                    self._surge(0.5)
                self.flight_mode = "short"              # a Giant Fiber take-off is always an escape
        elif cur == FLY:
            landing_ready = False
            if self.flight_mode == "long":              # foraging flight: land at the food, or when the goal is lost
                lost = self.goal_odor <= 0.0
                self._goal_lost_ms = self._goal_lost_ms + ms if lost else 0.0
                fed = self._fed_events != self._goal_fed0 or (not lost and self.goal_gap <= 0.0)
                if self._dwell >= 150.0 and (fed or self._goal_lost_ms >= GOAL_LOST_MS
                                             or self._dwell >= GOAL_MAX_FLIGHT_MS):
                    landing_ready = True
                    self._goal_refractory_ms = GOAL_MEAL_REFRACTORY_MS if fed else GOAL_REFRACTORY_MS
            elif self._dwell >= 180.0:                  # escape: land once the threat has passed
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
        elif goal_ok and self.flight_urge >= 1.0:       # foraging take-off: no Giant Fiber
            new, self.flight_mode, self.flight_urge = FLY, "long", 0.0
            self._goal_fed0, self._goal_lost_ms = self._fed_events, 0.0
        elif self._dwell >= s.min_dwell_ms:             # debounce stand <-> walk
            new = quiet
        if new != cur:
            self.state, self._dwell = new, 0.0
            if new != FLY:
                self.flight_mode = ""
            # B3: defensive arousal surge on threat (an escape; not a foraging flight, not what the lab evoked)
            if new == FLY:
                self._surge(0.5 if self.flight_mode == "short" else 0.0)
            elif new == RETREAT:
                self._surge(0.05)
            elif new == FREEZE:
                self._surge(0.02)

    def _surge(self, amount: float) -> None:
        """Defensive arousal up by `amount` (none while the lab drives the circuit), then the drive is rebuilt."""
        if amount > 0.0 and not self.lab_active:
            self.arousal = min(1.0, self.arousal + amount)
            self._last_arousal_drive = self.arousal
        self._drive()
