"""Runs the brain in a separate process so the overlay never stalls on a heavy circuit.

GUI -> worker: stimulus in a shared double array (latest value wins).
worker -> GUI: behavior state and telemetry in another shared array.
The worker paces itself to wall-clock time; if the circuit is too heavy it falls
behind, and `lag_ms` / `rt` in the telemetry say so (the fly then moves in slow motion).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass

from .paths import CACHE
from .states import STAND, STATES

# input slots
I_DIST, I_CLOSING, I_BIAS, I_SKITTISH, I_BEARING, I_TOUCH = 0, 1, 2, 3, 4, 5
I_X, I_Y, I_HEAD, I_CX, I_CY, I_VISION, I_HEIGHT = 6, 7, 8, 9, 10, 11, 12     # fly pose, cursor, vision switch
I_STAMP = 13                # clock reading (s) of the frame the GUI sampled the cursor and pose at; 0 = not sent
# output slots
(O_READY, O_STATE, O_GF, O_WALK, O_REST, O_RT, O_ACTIVE, O_N, O_CPU, O_LAG, O_SIM_S, O_BEAT, O_MDN, O_STEER,
 O_SPIKES, O_GROOM, O_VISION) = range(17)

CHUNK_MS = 4.0              # simulated time advanced per loop iteration
CHUNK_MS_GPU = 12.0         # a GPU read-back costs ~1 ms regardless of size; measured x1.8 -> x2.8-4 on a GTX 1650
STATS_EVERY_S = 0.25
VISION_PERIOD_MS = 20.0     # the vision pipeline looks at a new image this often: 5 CPU chunks (60 Hz was rounded up to this anyway)
MAX_FRAME_GAP_S = 0.25      # two cursor samples further apart than this say nothing about its speed (a stalled GUI)
MAX_LAG_S = 0.25            # beyond this the backlog is dropped (slow motion instead of catching up)


@dataclass(frozen=True)
class EngineConfig:
    circuit: str = "toy"    # "toy" (hand-built, plus synthetic load) or "flywire" (real connectome tiers)
    n: int = 146            # neurons in the circuit
    dt: float = 0.5         # integration step, ms
    seed: int = 1
    backend: str = "cpu"    # "cpu" (event-driven numba) or "gpu" (WebGPU, needs `uv sync --extra gpu`)
    adapter: int | None = None   # GPU adapter index (see list_gpus); None = first discrete GPU


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


def default_config() -> EngineConfig:
    """The real connectome at 15,000 neurons (within ~2% of the full brain for every cursor
    stimulus, tools/fidelity.py) when its cache has been built, else the toy circuit."""
    return EngineConfig("flywire", 15_000) if CACHE.exists() else EngineConfig()


def build_network(cfg: EngineConfig):
    if cfg.circuit == "flywire":
        from . import flywire

        return flywire.load_tier(cfg.n)
    from .toy_circuit import build

    return build(cfg.n, cfg.seed)


def _worker_main(cfg: EngineConfig, inp, out, stop) -> None:
    try:
        _run(cfg, inp, out, stop)
    except BaseException:
        import traceback

        traceback.print_exc()
        out[O_READY] = -1.0              # the GUI shows "engine error"
        raise


def _run(cfg: EngineConfig, inp, out, stop) -> None:
    from .brain import Brain

    net = build_network(cfg)
    brain = Brain(net, dt=cfg.dt, seed=cfg.seed, backend=cfg.backend, adapter=cfg.adapter)
    try:
        _loop(cfg, brain, net, inp, out, stop)
    finally:
        close = getattr(brain.engine, "close", None)
        if close:
            close()                                 # free the GPU buffers


def _loop(cfg: EngineConfig, brain, net, inp, out, stop) -> None:
    brain.advance(cfg.dt * 4)                       # triggers/loads the compiled kernel
    out[O_N] = net.n
    out[O_READY] = 1.0

    chunk = CHUNK_MS if cfg.backend == "cpu" else CHUNK_MS_GPU
    eye = _Eye(net)
    approach = _Approach()
    last_in = None
    t0 = time.perf_counter()
    sim_ms = 0.0
    spikes0 = brain.engine.total_spikes
    w_wall, w_cpu, w_comp, w_sim, w_active, w_iters = t0, time.process_time(), 0.0, 0.0, 0.0, 0
    while not stop.is_set():
        t = time.perf_counter()
        eye.update(brain, inp, out, sim_ms)
        closing = approach.update(inp)
        cur = (inp[I_DIST], inp[I_CLOSING] if closing is None else closing, inp[I_BIAS], inp[I_SKITTISH],
               round(inp[I_BEARING], 2), inp[I_TOUCH])
        if eye.on:                                      # the image, not cursor numbers, drives the looming inputs;
            cur = (1e6, 0.0, cur[2], cur[3], cur[4] if cur[5] else 0.0, cur[5])    # touch still needs its side
        if cur != last_in:
            brain.set_stimulus(*cur)
            last_in = cur
        state = brain.advance(chunk)
        spent = time.perf_counter() - t
        w_comp += spent
        w_sim += chunk
        w_active += brain.engine.n_active if w_active >= 0 else 0
        if brain.engine.n_active < 0:
            w_active = -1.0                         # GPU engine: every neuron is updated every step
        w_iters += 1
        sim_ms += chunk
        out[O_STATE] = float(STATES.index(state))
        out[O_GF], out[O_WALK], out[O_REST] = (brain.rates["GF"], brain.rates["WALK"], brain.rates["REST"])
        out[O_MDN], out[O_STEER], out[O_GROOM] = brain.rates["MDN"], brain.steer, brain.rates["GROOM"]

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

    def __init__(self, net):
        self.net = net
        self.drive = None               # visual.VisionDrive, built on first use (loads eye.npz / field.npz)
        self.missing = False            # the eye data is not there
        self._builder = None
        self.on = False
        self.failed = False             # asked for, but not available here: do not retry until switched off
        self.next_ms = self.last_ms = 0.0
        self.cursor = None              # where the cursor was at the previous frame
        self.stamp = 0.0                # and the GUI clock reading of that sample

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
        self.next_ms = self.last_ms = sim_ms
        out[O_VISION] = 1.0

    def _frame(self, brain, inp, sim_ms: float) -> None:
        from .visual import cursor_scene

        d = self.drive
        d.eye_height, d.skittish = max(10.0, inp[I_HEIGHT]), inp[I_SKITTISH]
        p = d.params
        cursor = (inp[I_CX], inp[I_CY])
        prev = self.cursor or cursor
        stamp = inp[I_STAMP]
        if stamp > 0.0 and self.stamp > 0.0:
            dt = min(max(stamp - self.stamp, 0.005), MAX_FRAME_GAP_S)   # real time between the two cursor samples
        else:
            dt = max(sim_ms - self.last_ms, 1.0) / 1000.0                 # no stamps (tests): simulated time
        if p.cursor_model == "sphere":
            idx, rates = d.step_cursor(cursor, prev, inp[I_X], inp[I_Y], inp[I_HEAD], dt)
        else:
            idx, rates = d.step(cursor_scene(cursor[0], cursor[1], d.halo_px, p.background),
                                cursor_scene(prev[0], prev[1], d.halo_px, p.background),
                                inp[I_X], inp[I_Y], inp[I_HEAD], dt)
        brain.set_vision(idx, rates, d.expansion)
        self.cursor, self.last_ms, self.stamp = cursor, sim_ms, stamp
        self.next_ms = sim_ms + VISION_PERIOD_MS


class Runner:
    """GUI-side handle to the worker process.

    `start` returns at once: the old worker is shut down and the new one launched on a helper thread (one swap at a
    time, so a worker is never loading its circuit while another still holds its memory). Each start gets fresh shared
    arrays, so a worker that is still shutting down cannot write into its successor's telemetry."""

    def __init__(self, cfg: EngineConfig | None = None):
        self._ctx = mp.get_context("spawn")
        self.inp = self.out = None
        self._proc = None
        self._stop = None
        self._swap_lock = threading.Lock()      # serialises retire-then-launch
        self._gen = 0                           # bumped by every start/stop; a swap that is no longer current gives up
        self._starting = False
        self.bias = 0.65          # walking drive, 0..1
        self.skittish = 1.0       # looming sensitivity multiplier
        self.vision = False       # see the screen (funnel view, retinotopic detectors) instead of cursor numbers
        self.eye_height = 100.0
        self.cfg = cfg or default_config()
        self.start(self.cfg)

    def start(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._gen += 1
        gen = self._gen
        inp, out = self._ctx.Array("d", 16, lock=False), self._ctx.Array("d", 24, lock=False)
        old = (self._proc, self._stop)
        self._proc = self._stop = None
        self.inp, self.out = inp, out
        self.send(1e6, 0.0)
        self._starting = True
        threading.Thread(target=self._swap, args=(gen, cfg, inp, out, old), daemon=True, name="neuropest-swap").start()

    def _swap(self, gen: int, cfg: EngineConfig, inp, out, old) -> None:
        with self._swap_lock:
            _retire(*old)
            if gen != self._gen:
                return                          # a newer start (or stop) took over while this one waited
            stop = self._ctx.Event()
            proc = self._ctx.Process(target=_worker_main, args=(cfg, inp, out, stop), daemon=True,
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

    def send(self, dist: float, closing: float, bearing: float = 0.0, touch: float = 0.0,
             pose=(0.0, 0.0, 0.0), cursor=(0.0, 0.0), stamp: float = 0.0) -> None:
        """Per-frame input. pose = fly (x, y, heading) and cursor = (x, y) in screen px feed the visual input.

        stamp: `time.perf_counter()` of the frame. With it the worker measures the cursor's closing speed itself from
        pose and cursor (`closing` is then ignored); without it (0) `closing` is used as given."""
        inp = self.inp
        inp[I_DIST], inp[I_CLOSING], inp[I_BIAS], inp[I_SKITTISH] = dist, closing, self.bias, self.skittish
        inp[I_BEARING], inp[I_TOUCH] = bearing, touch
        inp[I_X], inp[I_Y], inp[I_HEAD] = pose
        inp[I_CX], inp[I_CY] = cursor
        inp[I_STAMP] = stamp
        inp[I_VISION], inp[I_HEIGHT] = float(self.vision), self.eye_height

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
        return STATES[int(self.out[O_STATE])] if self.ready else STAND

    def stats(self) -> dict:
        o = self.out
        return dict(ready=self.ready, n=int(o[O_N]), rt=o[O_RT], active=o[O_ACTIVE], cpu=o[O_CPU],
                    lag_ms=o[O_LAG], gf=o[O_GF], walk=o[O_WALK], rest=o[O_REST], mdn=o[O_MDN],
                    steer=o[O_STEER], groom=o[O_GROOM], sim_s=o[O_SIM_S], spikes=o[O_SPIKES],
                    vision=o[O_VISION])


def _retire(proc, stop) -> None:
    """Ask a worker to stop, then terminate it if it does not within 2 s."""
    if proc is None:
        return
    stop.set()
    proc.join(2.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
