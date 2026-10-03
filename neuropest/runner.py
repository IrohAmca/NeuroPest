"""Runs the brain in a separate process so the overlay never stalls on a heavy circuit.

GUI -> worker: stimulus in a shared double array (latest value wins).
worker -> GUI: behavior state and telemetry in another shared array.
The worker paces itself to wall-clock time; if the circuit is too heavy it falls
behind, and `lag_ms` / `rt` in the telemetry say so (the fly then moves in slow motion).
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass

from .paths import CACHE
from .states import STAND, STATES

# input slots
I_DIST, I_CLOSING, I_BIAS, I_SKITTISH, I_BEARING = 0, 1, 2, 3, 4
I_X, I_Y, I_HEAD, I_CX, I_CY, I_VISION, I_HEIGHT = 5, 6, 7, 8, 9, 10, 11     # pose, cursor, vision switch
# output slots
(O_READY, O_STATE, O_GF, O_WALK, O_REST, O_RT, O_ACTIVE, O_N, O_CPU, O_LAG, O_SIM_S, O_BEAT, O_MDN, O_STEER,
 O_SPIKES, O_VISION) = range(16)

CHUNK_MS = 4.0              # simulated time advanced per loop iteration
CHUNK_MS_GPU = 12.0         # a GPU read-back costs ~1 ms regardless of size; measured x1.8 -> x2.8-4 on a GTX 1650
STATS_EVERY_S = 0.25
VISION_PERIOD_MS = 1000.0 / 60.0     # the vision pipeline looks at a new image about this often
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

        return flywire.load_cache().prefix(cfg.n)
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
    last_in = None
    t0 = time.perf_counter()
    sim_ms = 0.0
    spikes0 = brain.engine.total_spikes
    w_wall, w_cpu, w_comp, w_sim, w_active, w_iters = t0, time.process_time(), 0.0, 0.0, 0.0, 0
    while not stop.is_set():
        t = time.perf_counter()
        eye.update(brain, inp, out, sim_ms)
        cur = (inp[I_DIST], inp[I_CLOSING], inp[I_BIAS], inp[I_SKITTISH], round(inp[I_BEARING], 2))
        if eye.on:
            cur = (1e6, 0.0, cur[2], cur[3], 0.0)   # the image, not cursor numbers, drives the looming inputs
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
        out[O_MDN], out[O_STEER] = brain.rates["MDN"], brain.steer

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


class _Eye:
    """Worker side of the visual input: while switched on, turns the scene around the fly into projection
    neuron rates about every VISION_PERIOD_MS of simulated time and hands them to the brain."""

    def __init__(self, net):
        self.net = net
        self.drive = None               # visual.VisionDrive, built on first use (loads eye.npz / field.npz)
        self.on = False
        self.failed = False             # asked for, but not available here: do not retry until switched off
        self.next_ms = self.last_ms = 0.0
        self.cursor = None              # where the cursor was at the previous frame

    def update(self, brain, inp, out, sim_ms: float) -> None:
        want = inp[I_VISION] > 0.5
        if not want:
            self.failed = False
        if want != self.on and not self.failed:
            self._switch(want, brain, out, sim_ms)
        if self.on and sim_ms >= self.next_ms:
            self._frame(brain, inp, sim_ms)

    def _switch(self, want: bool, brain, out, sim_ms: float) -> None:
        if not want:
            brain.clear_vision()
            self.on = False
            out[O_VISION] = 0.0
            return
        from .visual import VisionDrive

        if self.drive is None:
            try:
                self.drive = VisionDrive(self.net)
            except FileNotFoundError:
                self.failed, out[O_VISION] = True, -1.0     # eye.npz / field.npz not built (tools/build_eye.py)
                return
        if not self.drive.usable:
            self.failed, out[O_VISION] = True, -1.0         # this circuit has no LPLC2 / LC4 with receptive fields
            return
        self.drive.reset()
        self.on, self.cursor = True, None
        self.next_ms = self.last_ms = sim_ms
        out[O_VISION] = 1.0

    def _frame(self, brain, inp, sim_ms: float) -> None:
        from .visual import cursor_scene

        d = self.drive
        d.eye_height, d.skittish = max(10.0, inp[I_HEIGHT]), inp[I_SKITTISH]
        p = d.params
        cursor = (inp[I_CX], inp[I_CY])
        prev = self.cursor or cursor
        dt = max(sim_ms - self.last_ms, 1.0) / 1000.0
        idx, rates = d.step(cursor_scene(cursor[0], cursor[1], p.halo_px, p.background),
                            cursor_scene(prev[0], prev[1], p.halo_px, p.background),
                            inp[I_X], inp[I_Y], inp[I_HEAD], dt)
        brain.set_vision(idx, rates)
        self.cursor, self.last_ms = cursor, sim_ms
        self.next_ms = sim_ms + VISION_PERIOD_MS


class Runner:
    """GUI-side handle to the worker process."""

    def __init__(self, cfg: EngineConfig | None = None):
        self._ctx = mp.get_context("spawn")
        self.inp = self._ctx.Array("d", 16, lock=False)
        self.out = self._ctx.Array("d", 16, lock=False)
        self._proc = None
        self._stop = None
        self.bias = 0.65          # walking drive, 0..1
        self.skittish = 1.0       # looming sensitivity multiplier
        self.vision = False       # see the screen (funnel view, retinotopic detectors) instead of cursor numbers
        self.eye_height = 100.0
        self.cfg = cfg or default_config()
        self.start(self.cfg)

    def start(self, cfg: EngineConfig) -> None:
        self.stop()
        self.cfg = cfg
        for i in range(len(self.out)):
            self.out[i] = 0.0
        self.send(1e6, 0.0)
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(target=_worker_main, args=(cfg, self.inp, self.out, self._stop),
                                       daemon=True, name="neuropest-engine")
        self._proc.start()

    def stop(self) -> None:
        if self._proc is None:
            return
        self._stop.set()
        self._proc.join(2.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(1.0)
        self._proc = None

    def send(self, dist: float, closing: float, bearing: float = 0.0, pose=(0.0, 0.0, 0.0), cursor=(0.0, 0.0)) -> None:
        """Per-frame input. pose = fly (x, y, heading) and cursor = (x, y) in screen px, used by the vision pipeline."""
        inp = self.inp
        inp[I_DIST], inp[I_CLOSING], inp[I_BIAS], inp[I_SKITTISH] = dist, closing, self.bias, self.skittish
        inp[I_BEARING] = bearing
        inp[I_X], inp[I_Y], inp[I_HEAD] = pose
        inp[I_CX], inp[I_CY] = cursor
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
        return self._proc is not None and self._proc.is_alive()

    @property
    def state(self) -> str:
        return STATES[int(self.out[O_STATE])] if self.ready else STAND

    def stats(self) -> dict:
        o = self.out
        return dict(ready=self.ready, n=int(o[O_N]), rt=o[O_RT], active=o[O_ACTIVE], cpu=o[O_CPU],
                    lag_ms=o[O_LAG], gf=o[O_GF], walk=o[O_WALK], rest=o[O_REST], mdn=o[O_MDN],
                    steer=o[O_STEER], sim_s=o[O_SIM_S], spikes=o[O_SPIKES], vision=o[O_VISION])
