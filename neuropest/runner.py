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
I_DIST, I_CLOSING, I_BIAS, I_SKITTISH = 0, 1, 2, 3
# output slots
O_READY, O_STATE, O_GF, O_WALK, O_REST, O_RT, O_ACTIVE, O_N, O_CPU, O_LAG, O_SIM_S, O_BEAT = range(12)

CHUNK_MS = 4.0              # simulated time advanced per loop iteration
STATS_EVERY_S = 0.25
MAX_LAG_S = 0.25            # beyond this the backlog is dropped (slow motion instead of catching up)


@dataclass(frozen=True)
class EngineConfig:
    circuit: str = "toy"    # "toy" (hand-built, plus synthetic load) or "flywire" (real connectome tiers)
    n: int = 146            # neurons in the circuit
    dt: float = 0.5         # integration step, ms
    seed: int = 1


def default_config() -> EngineConfig:
    """The real connectome at 2,000 neurons when its cache has been built, else the toy circuit."""
    return EngineConfig("flywire", 2_000) if CACHE.exists() else EngineConfig()


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
    brain = Brain(net, dt=cfg.dt, seed=cfg.seed)
    brain.advance(cfg.dt * 4)                       # triggers/loads the compiled kernel
    out[O_N] = net.n
    out[O_READY] = 1.0

    last_in = None
    t0 = time.perf_counter()
    sim_ms = 0.0
    w_wall, w_cpu, w_comp, w_sim, w_active, w_iters = t0, time.process_time(), 0.0, 0.0, 0.0, 0
    while not stop.is_set():
        cur = (inp[I_DIST], inp[I_CLOSING], inp[I_BIAS], inp[I_SKITTISH])
        if cur != last_in:
            brain.set_stimulus(*cur)
            last_in = cur
        t = time.perf_counter()
        state = brain.advance(CHUNK_MS)
        spent = time.perf_counter() - t
        w_comp += spent
        w_sim += CHUNK_MS
        w_active += brain.engine.n_active
        w_iters += 1
        sim_ms += CHUNK_MS
        out[O_STATE] = float(STATES.index(state))
        out[O_GF], out[O_WALK], out[O_REST] = (brain.rates["GF"], brain.rates["WALK"], brain.rates["REST"])

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
            out[O_ACTIVE] = w_active / max(w_iters, 1)
            out[O_CPU] = (cpu - w_cpu) / (now - w_wall)
            out[O_LAG] = lag * 1000.0
            out[O_SIM_S] = sim_ms / 1000.0
            out[O_BEAT] += 1
            w_wall, w_cpu, w_comp, w_sim, w_active, w_iters = now, cpu, 0.0, 0.0, 0.0, 0


class Runner:
    """GUI-side handle to the worker process."""

    def __init__(self, cfg: EngineConfig | None = None):
        self._ctx = mp.get_context("spawn")
        self.inp = self._ctx.Array("d", 8, lock=False)
        self.out = self._ctx.Array("d", 16, lock=False)
        self._proc = None
        self._stop = None
        self.bias = 0.65          # walking drive, 0..1
        self.skittish = 1.0       # looming sensitivity multiplier
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

    def send(self, dist: float, closing: float) -> None:
        inp = self.inp
        inp[I_DIST], inp[I_CLOSING], inp[I_BIAS], inp[I_SKITTISH] = dist, closing, self.bias, self.skittish

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
                    lag_ms=o[O_LAG], gf=o[O_GF], walk=o[O_WALK], rest=o[O_REST], sim_s=o[O_SIM_S])
