"""End-to-end cost of the Brain (toy core + synthetic bulk) at several sizes and time steps.

Run: uv run python tools/bench_brain.py
"""
from __future__ import annotations

import time

from neuropest.brain import Brain
from neuropest.toy_circuit import build

SIZES = [146, 2_000, 10_000, 50_000, 139_000]


def main():
    print(f"{'N':>8} {'dt':>4} | {'build s':>7} {'active%':>8} {'ms/step':>8} {'realtime x':>10} {'calm/loom GF Hz':>16}")
    for n in SIZES:
        t = time.perf_counter()
        net = build(n)
        build_s = time.perf_counter() - t
        for dt in (0.1, 0.5, 1.0):
            b = Brain(net, dt=dt)
            b.set_stimulus(900, 0, 0.5)
            for _ in range(100):                      # 400 ms warm-up
                b.advance(4.0)
            e = b.engine
            u0, s0 = e.neuron_updates, e.steps
            t = time.perf_counter()
            for _ in range(250):                      # 1 s of simulated time
                b.advance(4.0)
            wall = time.perf_counter() - t
            steps = e.steps - s0
            calm_gf = b.rates["GF"]
            b.set_stimulus(150, 3000, 0.5)
            for _ in range(100):
                b.advance(4.0)
            print(f"{net.n:>8} {dt:>4} | {build_s:>7.2f} {100 * (e.neuron_updates - u0) / steps / net.n:>8.2f} "
                  f"{wall / steps * 1000:>8.3f} {1.0 / wall:>10.2f} {calm_gf:>8.1f}/{b.rates['GF']:<7.1f}", flush=True)


if __name__ == "__main__":
    main()
