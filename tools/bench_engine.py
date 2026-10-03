"""Throughput of the LIF engine on synthetic FlyWire-sized networks.

Answers "how many neurons can we simulate in real time on this machine, at which
activity level and time step?". Not a correctness test (see tests/test_engine.py).
Degree distribution is heavy tailed like the connectome, targets are random;
`gain` and the forced-drive fraction sweep the activity level.

Run: uv run python tools/bench_engine.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

from neuropest.engine import LIFEngine, Network


def bench(n, deg, gain, dt, drive_frac, sim_ms=1000.0):
    net = Network.random(n, mean_deg=deg, gain=gain, seed=0)
    rng = np.random.default_rng(1)
    k = max(5, int(n * drive_frac))
    drive = rng.choice(n, k, replace=False)
    e = LIFEngine(net, dt=dt)
    e.set_drive(drive, 150.0)
    e.advance(100.0)                           # warm up (also compiles on first call)
    s0, u0, st0 = e.total_spikes, e.neuron_updates, e.steps
    t = time.perf_counter()
    e.advance(sim_ms)
    wall = time.perf_counter() - t
    steps = e.steps - st0
    return dict(
        active=(e.neuron_updates - u0) / steps / n,       # mean fraction of neurons updated per step
        rate=(e.total_spikes - s0) / (sim_ms / 1000) / n,  # mean Hz per neuron
        rt=sim_ms / 1000 / wall,                           # simulated s per wall s
        ms_step=wall / steps * 1000,
    )


def main():
    print(f"{'N':>7} {'nnz':>9} {'dt':>4} {'gain':>5} {'drive':>6} | "
          f"{'active%':>8} {'mean Hz':>8} {'ms/step':>8} {'realtime x':>10}")
    for n, deg in [(10_000, 20), (50_000, 20), (139_000, 20), (139_000, 100)]:
        for dt in (0.5, 1.0):
            for gain, drive_frac in [(1.0, 0.001), (3.0, 0.001), (4.5, 0.002), (6.0, 0.005)]:
                nnz = n * deg
                r = bench(n, deg, gain, dt, drive_frac)
                print(f"{n:>7} {nnz:>9} {dt:>4} {gain:>5} {drive_frac:>6} | "
                      f"{100 * r['active']:>8.2f} {r['rate']:>8.2f} {r['ms_step']:>8.3f} {r['rt']:>10.2f}",
                      flush=True)


if __name__ == "__main__":
    sys.exit(main())
