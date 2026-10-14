"""Runs the brain in a separate process so the overlay never stalls on a heavy circuit.

GUI -> worker: stimulus in a shared double array; worker -> GUI: behavior state and telemetry in another one. Both are
fresh on every start (both ends always run the same code), and every slot has exactly one writer. Three kinds of slot:
- levels: the latest value wins (the cursor, the pose, hunger ...);
- counters: the GUI only adds to them and the worker acts on the change (forget, meals, the sandbox's injections), so an
  event can not be lost between two reads, as it could when the worker read and then cleared a pulse;
- the lab block: several slots written under a sequence lock (I_LAB_SEQ odd while the GUI writes), so the worker never
  applies half of a request; it acknowledges what it applied in O_LAB_SEQ.
The worker paces itself to wall-clock time; if the circuit is too heavy it falls
behind, and `lag_ms` / `rt` in the telemetry say so (the fly then moves in slow motion).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from .eyeview import PREVIEW_LEN, read_snapshot
from .paths import CACHE
from .states import GEARS, LAB_FULL_HZ, LAB_GROUPS, STAND, STATES, lab_mask, lab_names

# ---- input slots (GUI writes, worker reads) ----
I_DIST, I_CLOSING, I_BIAS, I_SKITTISH, I_BEARING, I_TOUCH = 0, 1, 2, 3, 4, 5
I_X, I_Y, I_HEAD, I_CX, I_CY, I_VISION, I_HEIGHT = 6, 7, 8, 9, 10, 11, 12     # fly pose, cursor, vision switch
I_STAMP = 13                # clock reading (s) of the frame the GUI sampled the cursor and pose at; 0 = not sent
I_PHERO_STEER, I_PHERO_DRIVE, I_PHERO_REPEL, I_AT_TARGET = 14, 15, 16, 17
I_FORGET = 18               # counter: every increase makes the fly forget what it learned
I_HUNGER = 19               # metabolic hunger drive [0, 1]
I_WALL_BUMP = 20            # mechanical wall collision event [0, 1]
I_INJECT_REWARD = 21        # counter (sum of amounts): learning sandbox, reward (PAM)
I_INJECT_PUNISH = 22        # counter: learning sandbox, punishment (PPL1)
I_INJECT_CUE_FOOD = 23      # counter: learning sandbox, the food odor shown to the mushroom body for a moment
I_INJECT_CUE_NEAR = 24      # counter: learning sandbox, the cursor-near cue shown to the mushroom body for a moment
I_PREVIEW = 25              # 1 = the GUI shows the fly's-eye preview: publish a snapshot every vision frame
I_GOAL_GAP = 26             # px from the fly to the food it would fly to, minus the eating radius (<= 0: at it)
I_GOAL_BEARING = 27         # rad, + = to the fly's right
I_GOAL_ODOR = 28            # that source's own odor at the fly; 0 = no goal
I_FED = 29                  # counter: one per meal (the GUI ate a source or fed at the cursor)
# 30, 31: free
I_LAB_SEQ = 32              # the lab block's sequence lock: odd while the GUI writes it, even when it is whole
I_LAB_SILENCE = 33          # bit mask over states.LAB_GROUPS: the groups to keep silenced (level)
I_LAB_STIM_ID = 34          # bumped by every stimulation request (a stop too): the worker (re)arms only on a new id
I_LAB_STIM_MS = 35          # how long the stimulation lasts, simulated ms from when the worker applies it
I_LAB_RATE0 = 36            # Hz per lab group: I_LAB_RATE0 + k for LAB_GROUPS[k]
I_LAB_RATE_END = 52         # (reserved up to here)
INP_LEN = 64
# ---- output slots (worker writes, GUI reads) ----
(O_READY, O_STATE, O_GF, O_WALK, O_REST, O_RT, O_ACTIVE, O_N, O_CPU, O_LAG, O_SIM_S, O_BEAT, O_MDN, O_STEER,
 O_SPIKES, O_GROOM, O_VISION, O_VALENCE, O_VMOTOR, O_GEAR,
 O_PAM, O_PPL1, O_MBON_APP, O_MBON_AV, O_KC_ACTIVE,
 O_CUE_FOOD, O_CUE_ALARM, O_CUE_NEAR, O_CUE_LOOM, O_CUE_TOUCH,
 O_MB_WEIGHTS_DIRTY) = range(31)
O_KC_N = 31                 # Kenyon cells of the running mushroom-body model = valid length of the shared `mb_active` mask
O_URGE = 32                 # the foraging-flight urge (takes off at 1)
O_LAB_SEQ = 33              # the lab request (I_LAB_SEQ) the worker applied last
O_LAB_AVAIL = 34            # bit mask: the lab groups this circuit has
O_LAB_SILENCED = 35         # bit mask: silenced now
O_LAB_STIM = 36             # bit mask: stimulated now (0 again after the pulse)
O_LAB_LEFT_MS = 37          # simulated ms of stimulation left
O_LAB_GATE = 38             # 1 while plasticity is paused by the lab
O_RATE_LC4, O_RATE_LPLC2, O_RATE_LPC1, O_RATE_LC10_L, O_RATE_LC10_R = 39, 40, 41, 42, 43   # spiking rates, Hz
O_RATE_DNA02_L, O_RATE_DNA02_R = 44, 45
O_MOTION = 46               # state index + len(STATES) * gear index, in ONE slot: the GUI reads both with one load
OUT_LEN = 64
LAB_RATE_SLOTS = {"LC4": O_RATE_LC4, "LPLC2": O_RATE_LPLC2, "LPC1": O_RATE_LPC1, "LC10_L": O_RATE_LC10_L,
                  "LC10_R": O_RATE_LC10_R, "DNa02_L": O_RATE_DNA02_L, "DNa02_R": O_RATE_DNA02_R}

MB_WEIGHTS_LEN = 8192       # shared per-Kenyon-cell valence buffer, one slot per KC (random model 2,000; real cells 5,177)
CHUNK_MS = 4.0              # simulated time advanced per loop iteration
CHUNK_MS_GPU = 12.0         # a GPU read-back costs ~1 ms regardless of size; measured x1.8 -> x2.8-4 on a GTX 1650
STATS_EVERY_S = 0.25
MEMORY_SAVE_S = 60.0        # the learned weights are written this often (and on exit) when they changed
VISION_PERIOD_MS = 20.0     # the vision pipeline looks at a new image this often: 5 CPU chunks (60 Hz was rounded up to this anyway)
MAX_FRAME_GAP_S = 0.25      # two cursor samples further apart than this say nothing about its speed (a stalled GUI)
MAX_LAG_S = 0.25            # beyond this the backlog is dropped (slow motion instead of catching up)


@dataclass(frozen=True)
class EngineConfig:
    circuit: str = "toy"    # "toy" (hand-built, plus synthetic load) or "flywire" (real connectome tiers)
    n: int = 146            # neurons in the circuit
    dt: float = 0.5         # integration step, ms
    seed: int = 1
    backend: str = "cpu"    # "cpu" (event-driven numba) or "gpu" (WebGPU)
    adapter: int | None = None   # GPU adapter index (see list_gpus); None = first discrete GPU
    symmetry: str = "individual"  # "individual" (default; FlyWire connectome) or "symmetric" (hemisphere-balanced)
    learning: bool = True   # mushroom-body valence learning (mushroom.py); a naive fly behaves as without it
    memory_path: str | None = None   # where the learned weights are kept between runs; None = forget on exit
    mb_wiring: str = "random"   # mushroom body: "random" (designed) or "flywire" (real KC/MBON/DAN/PN cells, mushroom.py)


GPU_AUTO_MIN_NEURONS = 50_000   # "auto" uses a GPU from this tier on: the CPU holds smaller ones in real time


def list_gpus(timeout: float = 40.0) -> list[dict]:
    """GPUs WebGPU can use, found in a short-lived process (enumerating adapters commits ~100 MB)."""
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_list_gpus_worker, args=(child,), daemon=True)
    proc.start()
    out = []
    if parent.poll(timeout):
        out = parent.recv()
    proc.join(1.0)
    if proc.is_alive():
        proc.terminate()
    return out


def _list_gpus_worker(conn) -> None:
    try:
        from .engine.lif_wgpu import list_adapters

        conn.send([a for a in list_adapters() if a["type"] in ("DiscreteGPU", "IntegratedGPU")])
    except Exception:           # noqa: BLE001  (wgpu not installed or no adapter)
        conn.send([])


def pick_gpu(gpus: list[dict]) -> int | None:
    """Best adapter: a discrete GPU first; among equals Vulkan (it was faster than D3D12 on Intel)."""
    ranked = sorted(gpus, key=lambda g: (g["type"] != "DiscreteGPU", g["backend"] != "Vulkan", g["index"]))
    return ranked[0]["index"] if ranked else None


MB_WIRINGS = ("random", "flywire")


def mb_wiring_from_env() -> str | None:
    """NEUROPEST_MB=flywire|random forces the mushroom-body wiring over the saved preference; None when it is not set."""
    value = os.environ.get("NEUROPEST_MB", "").strip().lower()
    return value if value in MB_WIRINGS else None


def effective_mb_wiring(preferred: str) -> str:
    """The wiring that will really run: NEUROPEST_MB wins over the saved preference, and the real cells need their npz,
    present and loadable (tools/build_mushroom.py; else the worker falls back to the random model with only a console
    warning)."""
    from .mushroom import wiring_usable
    from .paths import MUSHROOM

    wiring = mb_wiring_from_env() or (preferred if preferred in MB_WIRINGS else "random")
    return "flywire" if wiring == "flywire" and wiring_usable(MUSHROOM) else "random"


def default_config() -> EngineConfig:
    """The real connectome at 15,000 neurons (within ~2% of the full brain for every cursor
    stimulus, tools/fidelity.py) when its cache has been built, else the toy circuit."""
    from .paths import MEMORY

    # NEUROPEST_MB=flywire switches the mushroom body to the real cells (needs tools/build_mushroom.py to have been run)
    wiring = mb_wiring_from_env() or "random"
    return EngineConfig("flywire", 15_000, memory_path=str(MEMORY), mb_wiring=wiring) if CACHE.exists() else EngineConfig()


def build_network(cfg: EngineConfig):
    if cfg.circuit == "flywire":
        from . import flywire

        return flywire.load_tier(cfg.n, symmetry=cfg.symmetry)
    from .toy_circuit import build

    return build(cfg.n, cfg.seed)


def _worker_main(cfg: EngineConfig, inp, out, stop, cap_frame=None, cap_meta=None, mb_weights=None,
                 eye_preview=None, mb_active=None) -> None:
    try:
        _run(cfg, inp, out, stop, cap_frame, cap_meta, mb_weights, eye_preview, mb_active)
    except BaseException:
        import traceback

        traceback.print_exc()
        out[O_READY] = -1.0              # the GUI shows "engine error"
        raise


def _run(cfg: EngineConfig, inp, out, stop, cap_frame=None, cap_meta=None, mb_weights=None, eye_preview=None,
         mb_active=None) -> None:
    from .brain import Brain

    net = build_network(cfg)
    brain = Brain(net, dt=cfg.dt, seed=cfg.seed, backend=cfg.backend, adapter=cfg.adapter, learning=cfg.learning,
                  mb_wiring=cfg.mb_wiring)
    memory = brain.mb.memory_file(cfg.memory_path) if brain.mb is not None and cfg.memory_path else None
    if memory is not None:
        brain.mb.load(memory)
    try:
        _loop(cfg, brain, net, inp, out, stop, cap_frame, cap_meta, memory, mb_weights, eye_preview, mb_active)
    finally:
        if memory is not None:
            try:
                brain.mb.save(memory)
            except OSError:
                pass
        close = getattr(brain.engine, "close", None)
        if close:
            close()                                 # free the GPU buffers


def _loop(cfg: EngineConfig, brain, net, inp, out, stop, cap_frame=None, cap_meta=None, memory=None, mb_weights=None,
          eye_preview=None, mb_active=None) -> None:
    brain.advance(cfg.dt * 4)                       # triggers/loads the compiled kernel

    chunk = CHUNK_MS if cfg.backend == "cpu" else CHUNK_MS_GPU
    eye = _Eye(net, cap_frame, cap_meta, eye_preview)
    approach = _Approach()
    mb_view = np.frombuffer(mb_weights, dtype=np.float32) if mb_weights is not None else None   # one memcpy per publish
    act_view = np.frombuffer(mb_active, dtype=np.int8) if mb_active is not None else None
    last_in = None
    forget_seen = 0.0
    counters = _Counters()
    lab = _Lab()
    last_save = time.perf_counter()
    t0 = time.perf_counter()
    sim_ms = 0.0
    spikes0 = brain.engine.total_spikes
    w_wall, w_cpu, w_comp, w_sim, w_active, w_iters = t0, time.process_time(), 0.0, 0.0, 0.0, 0
    out[O_N] = net.n
    out[O_LAB_AVAIL] = float(lab_mask(brain.lab_available()))
    lab.update(brain, inp, out)                    # a lesion carried over from the previous worker is there from the start
    out[O_READY] = 1.0
    while not stop.is_set():
        t = time.perf_counter()
        lab.update(brain, inp, out)
        eye.update(brain, inp, out, sim_ms)
        closing = approach.update(inp)
        raw_closing = inp[I_CLOSING] if closing is None else closing
        hunger = inp[I_HUNGER]
        eff_closing = raw_closing                       # the food-odor gate on the startle reflexes lives in Brain._drive
        eff_bias = inp[I_BIAS]
        wall_bump = inp[I_WALL_BUMP]
        cur = (round(inp[I_DIST], 1), round(eff_closing, 1), round(eff_bias, 3), inp[I_SKITTISH],
               round(inp[I_BEARING], 2), inp[I_TOUCH],
               round(inp[I_PHERO_STEER], 1), round(inp[I_PHERO_DRIVE], 2),
               round(inp[I_PHERO_REPEL], 2), inp[I_AT_TARGET],
               round(hunger, 2), round(wall_bump, 2))
        if eye.on:                                      # the image, not cursor numbers, drives the looming inputs;
            cur = (1e6, 0.0, cur[2], cur[3], cur[4] if cur[5] else 0.0, cur[5], cur[6], cur[7], cur[8], cur[9], cur[10], cur[11])
        if cur != last_in:
            brain.set_stimulus(*cur)
            last_in = cur
        if inp[I_FORGET] != forget_seen and brain.mb is not None:
            forget_seen = inp[I_FORGET]
            brain.mb.reset()
        # the learning sandbox (control panel / 3D view): counters, so a click is never lost between two reads
        if (d := counters.take(inp, I_INJECT_REWARD)) > 0.0:
            brain._reward = max(brain._reward, min(1.0, d))
        if (d := counters.take(inp, I_INJECT_PUNISH)) > 0.0:
            brain._punish = max(brain._punish, min(1.0, d))
        if (d := counters.take(inp, I_INJECT_CUE_FOOD)) > 0.0:
            brain.present_cue("food", d)
        if (d := counters.take(inp, I_INJECT_CUE_NEAR)) > 0.0:
            brain.present_cue("near", d)
        if (d := counters.take(inp, I_FED)) > 0.0:
            brain.note_fed(int(round(d)))
        brain.set_goal(inp[I_GOAL_GAP], inp[I_GOAL_BEARING], inp[I_GOAL_ODOR])
        state = brain.advance(chunk)
        spent = time.perf_counter() - t
        w_comp += spent
        w_sim += chunk
        w_active += brain.engine.n_active if w_active >= 0 else 0
        if brain.engine.n_active < 0:
            w_active = -1.0                         # GPU engine: every neuron is updated every step
        w_iters += 1
        sim_ms += chunk
        gear = GEARS.index(brain.gear)
        out[O_MOTION] = float(STATES.index(state) + len(STATES) * gear)   # what Runner.state_gear reads
        out[O_GEAR] = float(gear)
        out[O_STATE] = float(STATES.index(state))
        out[O_GF], out[O_WALK], out[O_REST] = (brain.rates["GF"], brain.rates["WALK"], brain.rates["REST"])
        out[O_MDN], out[O_STEER], out[O_GROOM] = brain.rates["MDN"], brain.steer, brain.rates["GROOM"]
        out[O_VALENCE] = brain.valence
        out[O_VMOTOR] = brain.v_motor
        out[O_URGE] = brain.flight_urge
        for name, slot in LAB_RATE_SLOTS.items():
            out[slot] = brain.lab_rates.get(name, brain.rates.get(name, 0.0))
        lab.publish(brain, out)
        if brain.mb is not None:
            out[O_PAM] = float(brain.mb.pam)
            out[O_PPL1] = float(brain.mb.ppl1)
            out[O_MBON_APP] = float(brain.mb.mbon_app)
            out[O_MBON_AV] = float(brain.mb.mbon_av)
            if hasattr(brain.mb, "active") and len(brain.mb.active) > 0:
                out[O_KC_ACTIVE] = float(len(brain.mb.active))
            elif hasattr(brain.mb, "kc"):
                out[O_KC_ACTIVE] = float(np.count_nonzero(brain.mb.kc))
            if act_view is not None:
                # which Kenyon cells fired in the last step (the 3D view paints exactly these); one row, one memcpy
                if hasattr(brain.mb, "active"):
                    row = np.zeros(brain.mb.n_kc, np.int8)
                    row[brain.mb.active] = 1
                else:
                    row = (brain.mb.kc > 0).astype(np.int8)
                n_row = min(len(row), len(act_view))
                act_view[:n_row] = row[:n_row]
                out[O_KC_N] = float(n_row)
            cues = getattr(brain, "_last_cues", (0.0, 0.0, 0.0, 0.0, 0.0))
            if len(cues) >= 5:
                out[O_CUE_FOOD], out[O_CUE_ALARM], out[O_CUE_NEAR], out[O_CUE_LOOM], out[O_CUE_TOUCH] = (
                    float(cues[0]), float(cues[1]), float(cues[2]), float(cues[3]), float(cues[4])
                )
            if mb_view is not None and brain.mb.dirty:
                v_vec = getattr(brain.mb, "valence_vector", None)
                if v_vec is not None:
                    n_cp = min(len(mb_view), len(v_vec))
                    mb_view[:n_cp] = v_vec[:n_cp]
                    out[O_MB_WEIGHTS_DIRTY] += 1.0
        if (memory is not None and brain.mb is not None and brain.mb.dirty and not brain.lab_active
                and time.perf_counter() - last_save > MEMORY_SAVE_S):
            last_save = time.perf_counter()
            try:
                brain.mb.save(memory)
            except OSError:
                pass

        ahead = t0 + sim_ms / 1000.0 - time.perf_counter()
        if ahead > 0:
            time.sleep(ahead)
        elif ahead < -MAX_LAG_S:
            t0 -= ahead + MAX_LAG_S                 # drop the backlog
        lag = max(0.0, -ahead)

        now = time.perf_counter()
        if now - w_wall >= STATS_EVERY_S:
            cpu = time.process_time()
            out[O_RT] = (w_sim / 1000.0) / w_comp if w_comp > 0 else 0.0
            out[O_ACTIVE] = w_active / max(w_iters, 1) if w_active >= 0 else -1.0     # -1: GPU, not tracked
            spikes = brain.engine.total_spikes
            out[O_SPIKES] = (spikes - spikes0) / (now - w_wall)
            spikes0 = spikes
            out[O_CPU] = (cpu - w_cpu) / (now - w_wall)
            out[O_LAG] = lag * 1000.0
            out[O_SIM_S] = sim_ms / 1000.0
            out[O_BEAT] += 1
            w_wall, w_cpu, w_comp, w_sim, w_active, w_iters = now, cpu, 0.0, 0.0, 0.0, 0


class _Counters:
    """Worker side of the counter slots: the GUI only adds to them, an event is the change since the last read."""

    def __init__(self):
        self.seen: dict[int, float] = {}

    def take(self, inp, slot: int) -> float:
        v = float(inp[slot])
        d = v - self.seen.get(slot, 0.0)            # the arrays are fresh on every start: nothing is pending at 0
        self.seen[slot] = v
        return d


class _Lab:
    """Worker side of the lab block (I_LAB_*): applies a whole request once, under the sequence lock."""

    def __init__(self):
        self.applied = 0.0                          # I_LAB_SEQ of the request applied last (0: none, a fresh array)
        self.stim_id = 0.0

    def update(self, brain, inp, out) -> None:
        seq = inp[I_LAB_SEQ]
        if seq == self.applied or int(seq) % 2:
            return                                  # nothing new, or the GUI is writing it right now
        mask, stim_id, ms = inp[I_LAB_SILENCE], inp[I_LAB_STIM_ID], inp[I_LAB_STIM_MS]
        rates = [inp[I_LAB_RATE0 + k] for k in range(len(LAB_GROUPS))]
        if inp[I_LAB_SEQ] != seq:
            return                                  # torn: written while read; the next iteration gets it whole
        stim = None
        if stim_id != self.stim_id:                 # a new stimulation (or stop): otherwise only the silencing changed
            stim = {n: r for n, r in zip(LAB_GROUPS, rates) if r > 0.0}
            self.stim_id = stim_id
        brain.set_lab(silenced=lab_names(mask), stimulated=stim, stim_ms=ms)
        self.applied = seq
        self.publish(brain, out)                    # the status first, then the acknowledgement that it is current
        out[O_LAB_SEQ] = seq

    @staticmethod
    def publish(brain, out) -> None:
        out[O_LAB_SILENCED] = float(lab_mask(brain._lab_sil))
        out[O_LAB_STIM] = float(lab_mask(brain._lab_stim))
        out[O_LAB_LEFT_MS] = max(0.0, brain.lab_left_ms)
        out[O_LAB_GATE] = 1.0 if brain.lab_active else 0.0


class _Approach:
    """Closing speed of the cursor, measured here from the cursor and fly positions the GUI sends with a clock stamp.

    The GUI used to divide the change of distance by its frame time clamped to 50 ms, so after a stall of 200 ms
    the speed came out 4 times too high (a false retreat or take-off). Here the divisor is the real time between the
    two samples, and the previous cursor position is measured from the fly's CURRENT position: the fly's own steps
    toward a standing cursor are not the cursor looming."""

    def __init__(self):
        self.stamp = 0.0
        self.cursor = None
        self.closing = 0.0

    def update(self, inp) -> float | None:
        """px/s, positive when the cursor approaches; None when the GUI sent no stamp (use its own number)."""
        stamp = inp[I_STAMP]
        if stamp <= 0.0:
            return None
        if stamp != self.stamp:
            cursor, fly = (inp[I_CX], inp[I_CY]), (inp[I_X], inp[I_Y])
            dt = stamp - self.stamp
            if self.cursor is None or not 0.002 < dt <= MAX_FRAME_GAP_S:
                self.closing = 0.0
            else:
                before = math.hypot(self.cursor[0] - fly[0], self.cursor[1] - fly[1])
                now = math.hypot(cursor[0] - fly[0], cursor[1] - fly[1])
                self.closing = (before - now) / dt
            self.cursor, self.stamp = cursor, stamp
        return self.closing


class _Eye:
    """Worker side of the visual input: while switched on, turns the scene around the fly into projection
    neuron rates about every VISION_PERIOD_MS of simulated time and hands them to the brain."""

    def __init__(self, net, cap_frame=None, cap_meta=None, eye_preview=None):
        import numpy as np
        from .brain import odor_gate
        from .capture import CROP_H, CROP_W, META_ORIGIN_X, META_ORIGIN_Y, META_SEQ, META_STAMP
        from .vision import image_scene
        from .visual import cursor_scene

        self.net = net
        self.cap_frame = cap_frame
        self.cap_meta = cap_meta
        self.preview_buf = None if eye_preview is None else np.frombuffer(eye_preview, dtype=np.float32)
        self.preview_seq = 0
        self.drive = None               # visual.VisionDrive, built on first use (loads eye.npz / field.npz)
        self.missing = False            # the eye data is not there
        self._builder = None
        self.on = False
        self.failed = False             # asked for, but not available here: do not retry until switched off
        self.next_ms = self.last_ms = 0.0
        self.cursor = None              # where the cursor was at the previous frame
        self.stamp = 0.0                # and the GUI clock reading of that sample
        self.last_seq = -1
        self.last_cap_stamp = 0.0
        self.frame_now = None
        self.frame_prev = None
        self.origin_now = (0.0, 0.0)
        self.origin_prev = (0.0, 0.0)
        self._np = np
        self._crop_h = CROP_H
        self._crop_w = CROP_W
        self._meta_ox = META_ORIGIN_X
        self._meta_oy = META_ORIGIN_Y
        self._meta_seq = META_SEQ
        self._meta_stamp = META_STAMP
        self._image_scene = image_scene
        self._cursor_scene = cursor_scene
        self._odor_gate = odor_gate

    def update(self, brain, inp, out, sim_ms: float) -> None:
        want = inp[I_VISION] > 0.5
        if not want:
            self.failed = False
        if want != self.on and not self.failed:
            self._switch(want, brain, out, sim_ms)
        if self.on and sim_ms >= self.next_ms:
            self._frame(brain, inp, sim_ms)

    def _build(self) -> None:
        """Runs in a thread: loading the eye files and building the detectors takes ~0.5 s (scipy import, dense
        column matrices), which would freeze the simulation loop if it happened there."""
        try:
            from .visual import VisionDrive

            self.drive = VisionDrive(self.net)
        except FileNotFoundError:
            self.missing = True                             # eye.npz / field.npz not built (tools/build_eye.py)

    def _switch(self, want: bool, brain, out, sim_ms: float) -> None:
        if not want:
            brain.clear_vision()
            self.on = False
            self.frame_now = self.frame_prev = None
            self.last_seq = -1
            out[O_VISION] = 0.0
            return
        if self.drive is None and not self.missing:
            if self._builder is None:
                self._builder = threading.Thread(target=self._build, daemon=True, name="neuropest-eye")
                self._builder.start()
            if self._builder.is_alive():
                return                                      # still building: stay off, look again next iteration
            if self.drive is None and not self.missing:
                self._builder = None                        # the thread died on something else: report unusable
                self.missing = True
        if self.missing:
            self.failed, out[O_VISION] = True, -1.0
            return
        if not self.drive.usable:
            self.failed, out[O_VISION] = True, -1.0         # this circuit has no LPLC2 / LC4 with receptive fields
            return
        self.drive.reset()
        self.on, self.cursor, self.stamp = True, None, 0.0
        self.frame_now = self.frame_prev = None
        self.last_seq = -1
        self.last_cap_stamp = 0.0
        self.next_ms = self.last_ms = sim_ms
        out[O_VISION] = 1.0

    def _frame(self, brain, inp, sim_ms: float) -> None:
        d = self.drive
        d.eye_height = max(10.0, inp[I_HEIGHT])
        # a food odor weakens the startle here as in the cursor path: through the detector gain, i.e. BEFORE the per-cell
        # rate caps (scaling the capped rates would leave the retreat input under its threshold while take-off survives)
        d.skittish = inp[I_SKITTISH] * self._odor_gate(inp[I_PHERO_DRIVE] * inp[I_HUNGER])
        p = d.params
        cursor = (inp[I_CX], inp[I_CY])
        prev = self.cursor or cursor
        stamp = inp[I_STAMP]
        if stamp > 0.0 and self.stamp > 0.0:
            dt = min(max(stamp - self.stamp, 0.005), MAX_FRAME_GAP_S)   # real time between the two cursor samples
        else:
            dt = max(sim_ms - self.last_ms, 1.0) / 1000.0                 # no stamps (tests): simulated time

        used_screen = False
        if self.cap_frame is not None and self.cap_meta is not None:
            seq = int(self.cap_meta[self._meta_seq])
            if seq > 0:
                is_new = (seq != self.last_seq)
                if is_new:
                    raw = self._np.frombuffer(self.cap_frame, dtype=self._np.uint8).reshape((self._crop_h, self._crop_w)).copy()
                    ox = self.cap_meta[self._meta_ox]
                    oy = self.cap_meta[self._meta_oy]
                    cap_stamp = float(self.cap_meta[self._meta_stamp])
                    dt_screen = max(cap_stamp - self.last_cap_stamp, dt) if self.last_cap_stamp > 0.0 else dt
                    self.last_cap_stamp = cap_stamp
                    self.frame_prev = self.frame_now if self.frame_now is not None else raw
                    self.origin_prev = self.origin_now if self.frame_now is not None else (ox, oy)
                    self.frame_now = raw
                    self.origin_now = (ox, oy)
                    self.last_seq = seq
                    scale_screen = min(1.0, dt / dt_screen)
                else:
                    scale_screen = 0.0

                if self.frame_now is not None and self.frame_prev is not None:
                    # When cursor_model == "sphere" (default), do not draw the flat disk onto the 2D plane image;
                    # composite it at eye level onto the columns inside d.step (B2).
                    plane_cursor = cursor if p.cursor_model != "sphere" else None
                    plane_prev = prev if p.cursor_model != "sphere" else None
                    scene_now = self._image_scene(self.frame_now, origin=self.origin_now, outside=p.background,
                                                  cursor=plane_cursor, cursor_radius=d.halo_px)
                    scene_prev = self._image_scene(self.frame_prev, origin=self.origin_prev, outside=p.background,
                                                   cursor=plane_prev, cursor_radius=d.halo_px)
                    idx, rates = d.step(scene_now, scene_prev, inp[I_X], inp[I_Y], inp[I_HEAD], dt,
                                        cursor=cursor, prev_cursor=prev, screen_scale=scale_screen)
                    # Next ticks before a new capture frame arrives must treat the screen background as unchanged (B1)
                    self.frame_prev = self.frame_now
                    self.origin_prev = self.origin_now
                    used_screen = True

        if not used_screen:
            if p.cursor_model == "sphere":
                idx, rates = d.step_cursor(cursor, prev, inp[I_X], inp[I_Y], inp[I_HEAD], dt)
            else:
                idx, rates = d.step(self._cursor_scene(cursor[0], cursor[1], d.halo_px, p.background),
                                    self._cursor_scene(prev[0], prev[1], d.halo_px, p.background),
                                    inp[I_X], inp[I_Y], inp[I_HEAD], dt)

        brain.set_vision(idx, rates, d.expansion, (d.mb_cue["cursor_near"], d.mb_cue["looming"]))
        if self.preview_buf is not None and inp[I_PREVIEW] > 0.5:
            self._publish(d, rates)
        self.cursor, self.last_ms, self.stamp = cursor, sim_ms, stamp
        self.next_ms = sim_ms + VISION_PERIOD_MS

    def _publish(self, d, rates) -> None:
        """Copy what the eye saw this frame into the shared preview array (only while the GUI shows it)."""
        if d.last is None:
            return
        from .eyeview import write_snapshot

        lum, expansion, obj = d.last
        n_loom = len(d.loom_idx)
        sizes = (d.n_lplc2, n_loom - d.n_lplc2, len(d.ret_idx), len(d.obj_idx))
        cols = self._np.concatenate([d.loom_col, d.ret_col, d.obj_col])
        self.preview_seq += 1
        write_snapshot(self.preview_buf, self.preview_seq, lum, expansion, obj, cols, rates, sizes)


class Runner:
    """GUI-side handle to the worker process.

    `start` returns at once: the old worker is shut down and the new one launched on a helper thread (one swap at a
    time, so a worker is never loading its circuit while another still holds its memory). Each start gets fresh shared
    arrays, so a worker that is still shutting down cannot write into its successor's telemetry."""

    def __init__(self, cfg: EngineConfig | None = None):
        from .capture import CaptureProcess

        self._ctx = mp.get_context("spawn")
        self.capture = CaptureProcess(self._ctx)
        self.capture.start()
        self.inp = self.out = None
        self._proc = None
        self._stop = None
        self._swap_lock = threading.Lock()      # serialises retire-then-launch
        self._gen = 0                           # bumped by every start/stop; a swap that is no longer current gives up
        self._starting = False
        self.bias = 0.65          # walking drive, 0..1
        self.skittish = 1.0       # looming sensitivity multiplier
        self.wall_pain = 1.0      # nociceptive wall collision punishment multiplier
        self.vision = False       # see the screen (funnel view, retinotopic detectors) instead of cursor numbers
        self.eye_height = 100.0
        self.preview = False      # the GUI shows the fly's-eye preview: the worker publishes a snapshot per vision frame
        self._lab_init()
        self.cfg = cfg or default_config()
        self.start(self.cfg)

    def _lab_init(self) -> None:
        """The lab as the GUI asked for it (written into each new worker's arrays: a lesion outlives a restart, a
        stimulation does not)."""
        self._lab_silenced: set[str] = set()
        self._lab_rates = [0.0] * len(LAB_GROUPS)
        self._lab_ms = 0.0
        self._lab_seq = 0.0
        self._lab_stim_id = 0.0

    def start(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._gen += 1
        gen = self._gen
        inp, out = self._ctx.Array("d", INP_LEN, lock=False), self._ctx.Array("d", OUT_LEN, lock=False)
        mb_weights = self._ctx.Array("f", MB_WEIGHTS_LEN, lock=False)
        mb_active = self._ctx.Array("b", MB_WEIGHTS_LEN, lock=False)
        eye_preview = self._ctx.Array("f", PREVIEW_LEN, lock=False)
        old = (self._proc, self._stop)
        self._proc = self._stop = None
        self.inp, self.out, self.mb_weights, self.eye_preview = inp, out, mb_weights, eye_preview
        self.mb_active = mb_active
        self.send(1e6, 0.0)
        self._lab_seq = 0.0                         # a fresh array: the sequence starts again
        self._lab_rates = [0.0] * len(LAB_GROUPS)   # a stimulation does not carry over (it would fire whenever the
        self._lab_ms = 0.0                          # new worker happens to become ready); the silencing does
        if self._lab_silenced:
            self._write_lab()
        self._starting = True
        threading.Thread(target=self._swap, args=(gen, cfg, inp, out, old, mb_weights, eye_preview, mb_active),
                         daemon=True, name="neuropest-swap").start()

    def _swap(self, gen: int, cfg: EngineConfig, inp, out, old, mb_weights=None, eye_preview=None,
              mb_active=None) -> None:
        with self._swap_lock:
            _retire(*old)
            if gen != self._gen:
                return                          # a newer start (or stop) took over while this one waited
            stop = self._ctx.Event()
            proc = self._ctx.Process(target=_worker_main,
                                     args=(cfg, inp, out, stop, self.capture.frame_raw, self.capture.meta_raw, mb_weights,
                                          eye_preview, mb_active),
                                     daemon=True,
                                     name="neuropest-engine")
            proc.start()
            if gen != self._gen:                # ... while the process was being spawned
                _retire(proc, stop)
                return
            self._proc, self._stop, self._starting = proc, stop, False

    def stop(self) -> None:
        """Shut the worker down and wait for it (the application is quitting)."""
        self._gen += 1                          # cancels a swap that has not launched yet
        with self._swap_lock:                   # waits for one that is launching right now
            proc, stop, self._proc, self._stop = self._proc, self._stop, None, None
            self._starting = False
            _retire(proc, stop)
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.stop()

    def send(self, dist: float, closing: float, bearing: float = 0.0, touch: float = 0.0,
             pose=(0.0, 0.0, 0.0), cursor=(0.0, 0.0), stamp: float = 0.0,
             capture_pos=None,
             phero_steer: float = 0.0, phero_drive: float = 0.0,
             phero_repel: float = 0.0, at_target: bool = False,
             hunger: float = 1.0, wall_bump: bool = False,
             goal_gap: float = 1e6, goal_bearing: float = 0.0, goal_odor: float = 0.0) -> None:
        """Per-frame input. pose = fly (x, y, heading) and cursor = (x, y) in screen px feed the visual input.

        stamp: `time.perf_counter()` of the frame. With it the worker measures the cursor's closing speed itself from
        pose and cursor (`closing` is then ignored); without it (0) `closing` is used as given.
        goal_*: the food source the fly would fly to (Brain.set_goal); the defaults say there is none."""
        inp = self.inp
        inp[I_DIST], inp[I_CLOSING], inp[I_BIAS], inp[I_SKITTISH] = dist, closing, self.bias, self.skittish
        inp[I_BEARING], inp[I_TOUCH] = bearing, touch
        inp[I_X], inp[I_Y], inp[I_HEAD] = pose
        inp[I_CX], inp[I_CY] = cursor
        inp[I_STAMP] = stamp
        inp[I_VISION], inp[I_HEIGHT] = float(self.vision), self.eye_height
        inp[I_PREVIEW] = float(self.preview and self.vision)
        inp[I_PHERO_STEER] = phero_steer
        inp[I_PHERO_DRIVE] = phero_drive
        inp[I_PHERO_REPEL] = phero_repel
        inp[I_AT_TARGET] = 1.0 if at_target else 0.0
        inp[I_HUNGER] = float(hunger)
        inp[I_WALL_BUMP] = (1.0 if wall_bump else 0.0) * getattr(self, "wall_pain", 1.0)
        inp[I_GOAL_GAP], inp[I_GOAL_BEARING], inp[I_GOAL_ODOR] = goal_gap, goal_bearing, goal_odor

        if hasattr(self, "capture") and self.capture is not None:
            cx, cy = capture_pos if capture_pos is not None else (pose[0], pose[1])
            self.capture.update_target(cx, cy, enabled=self.vision)

    @property
    def valence(self) -> float:
        """Learned valence of what the fly senses now, -1 (fear) .. +1 (desire); 0 for a naive fly."""
        return self.out[O_VALENCE] if self.ready else 0.0

    @property
    def v_motor(self) -> float:
        """Central motor pool value: threat + desire + tension (see Brain)."""
        return self.out[O_VMOTOR] if self.ready else 0.0

    @property
    def gear(self) -> str:
        return self.state_gear()[1]

    def state_gear(self) -> tuple[str, str]:
        """(state, gear) from one shared slot: both belong to the same worker step (two reads could straddle one)."""
        if not self.ready:
            return STAND, GEARS[0]
        m = int(self.out[O_MOTION])
        return STATES[m % len(STATES)], GEARS[m // len(STATES)]

    def forget(self) -> None:
        """Amnesia: the worker resets the learned weights (and saves the empty memory soon after)."""
        self.inp[I_FORGET] += 1.0

    def note_fed(self) -> None:
        """One meal (a counter: the worker lands a foraging flight on it and takes it as the feeding reward)."""
        self.inp[I_FED] += 1.0

    @property
    def urge(self) -> float:
        """The foraging-flight urge, 0..1.2 (takes off at 1)."""
        return self.out[O_URGE] if self.ready else 0.0

    @property
    def steer(self) -> float:
        """Right minus left DNa02 rate, Hz (positive: turn right)."""
        return self.out[O_STEER] if self.ready else 0.0

    @property
    def ready(self) -> bool:
        return self.out[O_READY] > 0

    @property
    def failed(self) -> bool:
        return self.out[O_READY] < 0

    @property
    def alive(self) -> bool:
        """The worker runs, or is about to (a restart in progress is not a failure)."""
        return self._starting or (self._proc is not None and self._proc.is_alive())

    @property
    def state(self) -> str:
        return self.state_gear()[0]

    @property
    def pam(self) -> float:
        """PAM cluster dopaminergic reinforcement level (0..1)."""
        return self.out[O_PAM] if self.ready else 0.0

    @property
    def ppl1(self) -> float:
        """PPL1 cluster dopaminergic reinforcement level (0..1)."""
        return self.out[O_PPL1] if self.ready else 0.0

    @property
    def mbon_app(self) -> float:
        """Approach MBON pool output rate."""
        return self.out[O_MBON_APP] if self.ready else 1.0

    @property
    def mbon_av(self) -> float:
        """Avoidance MBON pool output rate."""
        return self.out[O_MBON_AV] if self.ready else 1.0

    @property
    def kc_active(self) -> int:
        """Active Kenyon cell count during current frame."""
        return int(self.out[O_KC_ACTIVE]) if self.ready else 0

    # ---------------------------------------------------------------- the learning sandbox (designed signals)
    def inject_reward(self, amount: float = 1.0) -> None:
        """Reward (PAM dopamine, the designed mushroom-body layer) for the cue present now."""
        if self.ready:
            self.inp[I_INJECT_REWARD] += float(amount)

    def inject_punish(self, amount: float = 1.0) -> None:
        """Punishment (PPL1 dopamine, the designed mushroom-body layer) for the cue present now."""
        if self.ready:
            self.inp[I_INJECT_PUNISH] += float(amount)

    def inject_cue_food(self, amount: float = 1.0) -> None:
        """Show the mushroom body the food odor for a moment (Brain.present_cue); the fly does not smell it."""
        if self.ready:
            self.inp[I_INJECT_CUE_FOOD] += float(amount)

    def inject_cue_near(self, amount: float = 1.0) -> None:
        """Show the mushroom body the cursor-near cue for a moment (Brain.present_cue); nothing is seen."""
        if self.ready:
            self.inp[I_INJECT_CUE_NEAR] += float(amount)

    # ---------------------------------------------------------------- the lab (spiking neurons of the circuit)
    def stimulate(self, group: str | dict, intensity: float = 1.0, ms: float = 300.0) -> bool:
        """Forced spiking of a lab group (states.LAB_GROUPS) at intensity x its LAB_FULL_HZ for `ms` simulated ms (it
        ends by itself, after 10 s at most), or of several at once: {group: intensity}. Replaces a stimulation still on.
        False (nothing sent) while the worker is not ready or no group is known."""
        want = group if isinstance(group, dict) else {group: intensity}
        rates = [0.0] * len(LAB_GROUPS)
        for name, level in want.items():
            if name in LAB_GROUPS and level > 0.0:
                rates[LAB_GROUPS.index(name)] = LAB_FULL_HZ[name] * min(2.0, float(level))
        if not self.ready or not any(rates):
            return False
        self._lab_rates, self._lab_ms = rates, float(ms)
        self._lab_stim_id += 1.0
        self._write_lab()
        return True

    def stop_stimulation(self) -> None:
        """End a stimulation now (a held button released)."""
        self._lab_rates, self._lab_ms = [0.0] * len(LAB_GROUPS), 0.0
        self._lab_stim_id += 1.0
        self._write_lab()

    def silence(self, group: str, on: bool = True) -> None:
        """Silence a lab group (its neurons can not spike) or release it. Kept across engine restarts, not saved."""
        if group in LAB_GROUPS:
            (self._lab_silenced.add if on else self._lab_silenced.discard)(group)
            self._write_lab()

    def clear_lab(self) -> None:
        """Release every silenced group and end the stimulation."""
        self._lab_silenced.clear()
        self.stop_stimulation()

    def _write_lab(self) -> None:
        """The lab block under the sequence lock: odd while it is written, even when whole (the worker reads it only
        then, and only once per value)."""
        inp = self.inp
        if inp is None:
            return
        self._lab_seq += 1.0
        inp[I_LAB_SEQ] = self._lab_seq
        inp[I_LAB_SILENCE] = float(lab_mask(self._lab_silenced))
        inp[I_LAB_STIM_ID] = self._lab_stim_id
        inp[I_LAB_STIM_MS] = self._lab_ms
        for k, rate in enumerate(self._lab_rates):
            inp[I_LAB_RATE0 + k] = rate
        self._lab_seq += 1.0
        inp[I_LAB_SEQ] = self._lab_seq

    def lab_status(self) -> dict:
        """What the worker really has applied (not what was asked): groups this circuit has, silenced, stimulated now,
        ms of stimulation left, whether plasticity is paused, and whether the last request has been applied."""
        o = self.out
        if o is None or not self.ready:
            return dict(available=(), silenced=(), stimulated=(), left_ms=0.0, gate=False, acked=False,
                        asked_silenced=tuple(n for n in LAB_GROUPS if n in self._lab_silenced))
        return dict(available=lab_names(o[O_LAB_AVAIL]), silenced=lab_names(o[O_LAB_SILENCED]),
                    stimulated=lab_names(o[O_LAB_STIM]), left_ms=float(o[O_LAB_LEFT_MS]), gate=o[O_LAB_GATE] > 0.5,
                    acked=o[O_LAB_SEQ] == self.inp[I_LAB_SEQ],
                    asked_silenced=tuple(n for n in LAB_GROUPS if n in self._lab_silenced))

    def lab_rates(self) -> dict:
        """Spiking rates (Hz) of the lab's visual input groups and of DNa02 per side, for the 3D view's cards."""
        if self.out is None or not self.ready:
            return {}
        have = set(lab_names(self.out[O_LAB_AVAIL]))
        return {name: float(self.out[slot]) for name, slot in LAB_RATE_SLOTS.items() if name in have}

    # optogenetics names kept for callers of the old API: they now stimulate the real neurons
    _OPTO = {"gf": "GF", "takeoff": "GF", "escape": "GF", "mdn": "MDN", "moonwalk": "MDN", "retreat": "MDN",
             "walk": "WALK", "lc4": "LC4", "lplc2": "LPLC2", "lpc1": "LPC1", "lc10_l": "LC10_L", "lc10_r": "LC10_R",
             "steer_l": "DNa02_L", "steer_r": "DNa02_R", "dna02_l": "DNa02_L", "dna02_r": "DNa02_R", "groom": "GROOM"}

    def inject_gf(self, amount: float = 1.0) -> None:
        """Stimulate the Giant Fiber (DNp01) neurons for a pulse."""
        self.stimulate("GF", amount)

    def inject_mdn(self, amount: float = 1.0) -> None:
        """Stimulate the moonwalker descending neurons (MDN) for a pulse."""
        self.stimulate("MDN", amount)

    def inject_walk(self, amount: float = 1.0) -> None:
        """Stimulate the walking command neurons (DNp09) for a pulse."""
        self.stimulate("WALK", amount)

    def inject_optogenetic(self, target: str, amount: float = 1.0) -> None:
        """A pulse on a lab group by name (see _OPTO), or a sandbox signal ("reward", "punish", "food", "near")."""
        t = target.strip().lower()
        sandbox = {"reward": self.inject_reward, "pam": self.inject_reward, "punish": self.inject_punish,
                   "ppl1": self.inject_punish, "food": self.inject_cue_food, "near": self.inject_cue_near}
        if t in sandbox:
            sandbox[t](amount)
        elif t in self._OPTO:
            self.stimulate(self._OPTO[t], amount)

    def preview_snapshot(self):
        """Latest frame of the fly's-eye preview (eyeview.Snapshot), or None: not switched on, nothing yet, or torn."""
        buf = getattr(self, "eye_preview", None)
        if buf is None:
            return None
        return read_snapshot(np.frombuffer(buf, dtype=np.float32))

    def get_mb_weights(self) -> np.ndarray:
        """Return a copy of the learned net valence per Kenyon cell."""
        if hasattr(self, "mb_weights") and self.mb_weights is not None:
            try:
                buf = self.mb_weights.get_obj() if hasattr(self.mb_weights, "get_obj") else self.mb_weights
                return np.frombuffer(buf, dtype=np.float32).copy()
            except Exception:
                pass
        return np.zeros(MB_WEIGHTS_LEN, dtype=np.float32)

    def get_kc_active(self) -> np.ndarray | None:
        """Mask (bool, one entry per Kenyon cell of the running mushroom-body model) of the cells that fired in its last
        step, or None while the worker has not published a model (starting, or learning is off)."""
        buf = getattr(self, "mb_active", None)
        if buf is None or self.out is None:
            return None
        n = int(self.out[O_KC_N])
        if n <= 0:
            return None
        return np.frombuffer(buf, dtype=np.int8)[:n] > 0

    def stats(self) -> dict:
        o = self.out
        motion = int(o[O_MOTION])                 # state and gear of the same worker step
        return dict(ready=self.ready, n=int(o[O_N]), rt=o[O_RT], active=o[O_ACTIVE], cpu=o[O_CPU],
                    lag_ms=o[O_LAG], gf=o[O_GF], walk=o[O_WALK], rest=o[O_REST], mdn=o[O_MDN],
                    steer=o[O_STEER], groom=o[O_GROOM], sim_s=o[O_SIM_S], spikes=o[O_SPIKES],
                    vision=o[O_VISION], valence=o[O_VALENCE], v_motor=o[O_VMOTOR], gear=GEARS[motion // len(STATES)],
                    pam=o[O_PAM], ppl1=o[O_PPL1], mbon_app=o[O_MBON_APP], mbon_av=o[O_MBON_AV],
                    kc_active=int(o[O_KC_ACTIVE]), cue_food=o[O_CUE_FOOD], cue_near=o[O_CUE_NEAR],
                    weights_dirty=int(o[O_MB_WEIGHTS_DIRTY]), urge=o[O_URGE], state=STATES[motion % len(STATES)],
                    heading=float(self.inp[I_HEAD]))     # the fly's on-screen heading, rad (the 3D compass shows it)


def _retire(proc, stop) -> None:
    """Ask a worker to stop, then terminate it if it does not within 2 s."""
    if proc is None:
        return
    stop.set()
    proc.join(2.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
