"""What does the full brain cost on the CPU when photoreceptors (R1-6, R7, R8) are driven by an image?

Drives a random fraction of the 11,391 photoreceptors with Poisson spikes. A stand-in for a screen image:
how many neurons become active, how many spikes and synaptic events per second, how fast is the CPU engine.

Run: uv run python tools/probe_retina.py
"""
from __future__ import annotations

import time

import numpy as np

from neuropest import flywire
from neuropest.engine import LIFEngine


def main():
    net = flywire.load_cache()
    ann = flywire.load_annotations(net.ids)
    r = np.flatnonzero(ann.cell_type.isin(["R1-6", "R7", "R8"]).to_numpy()).astype(np.int32)
    g = net.groups
    outdeg = np.diff(net.indptr)
    every = np.arange(net.n)
    rng = np.random.default_rng(0)
    print(f"photoreceptors: {len(r)}  | brain: {net.n:,} neurons, {net.nnz:,} edges")
    print(f"{'fraction':>8} {'rate Hz':>8} | {'active %':>8} {'spikes/s':>10} {'events/s':>11} {'GF Hz':>6} | "
          f"{'ms/step':>8} {'realtime x':>10}")
    for frac, rate in [(0.1, 20), (0.5, 20), (0.5, 50), (1.0, 50), (1.0, 100)]:
        sub = rng.choice(r, int(len(r) * frac), replace=False)
        e = LIFEngine(net, dt=0.5, seed=1)
        e.set_drive(sub, float(rate))
        e.advance(300)
        e.pop_counts(every)
        u0, s0, st0 = e.neuron_updates, e.total_spikes, e.steps
        t = time.perf_counter()
        e.advance(1000)
        wall = time.perf_counter() - t
        steps = e.steps - st0
        c = e.pop_counts(every).astype(np.float64)
        print(f"{frac:>8.1f} {rate:>8} | {100 * (e.neuron_updates - u0) / steps / net.n:>8.1f} "
              f"{e.total_spikes - s0:>10,} {int((c * outdeg).sum()):>11,} {c[g['GF']].mean():>6.1f} | "
              f"{wall / steps * 1000:>8.3f} {1.0 / wall:>10.2f}", flush=True)


if __name__ == "__main__":
    main()
