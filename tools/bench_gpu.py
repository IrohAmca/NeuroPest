"""CPU engine vs GPU engine (WebGPU) on retina-like and looming loads of the full FlyWire brain.

Run: uv run python tools/bench_gpu.py [adapter_index ...]     (no args: list adapters and exit)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from neuropest import flywire
from neuropest.engine import LIFEngine
from neuropest.engine.lif_wgpu import WGPUEngine, list_adapters


def photoreceptors(net):
    ann = flywire.load_annotations(net.ids)
    return np.flatnonzero(ann.cell_type.isin(["R1-6", "R7", "R8"]).to_numpy()).astype(np.int32)


def workloads(net):
    r = photoreceptors(net)
    rng = np.random.default_rng(0)
    half = rng.choice(r, len(r) // 2, replace=False)
    g = net.groups
    return [
        ("retina 10% @20Hz", r[rng.choice(len(r), len(r) // 10, replace=False)], 20.0),
        ("retina 50% @20Hz", half, 20.0),
        ("retina 50% @50Hz", half, 50.0),
        ("retina 100% @50Hz", r, 50.0),
        ("retina 100% @100Hz", r, 100.0),
        ("looming 50Hz", g["LOOM"], 50.0),
    ]


def measure(make, idx, rate, sim_ms=1000.0, warm_ms=300.0):
    e = make()
    e.set_drive(idx, rate)
    e.advance(warm_ms)
    e.pop_counts(np.arange(5))
    if hasattr(e, "sync"):
        e.sync()
    s0 = e.total_spikes
    t = time.perf_counter()
    e.advance(sim_ms)
    e.pop_counts(np.arange(5))                     # also waits for the GPU
    wall = time.perf_counter() - t
    spikes = e.total_spikes - s0
    if hasattr(e, "close"):
        e.close()
    return sim_ms / 1000.0 / wall, spikes


def main():
    ads = list_adapters()
    if len(sys.argv) == 1:
        for a in ads:
            print(a)
        return
    net = flywire.load_cache()
    wl = workloads(net)
    print(f"full brain: {net.n:,} neurons, {net.nnz:,} edges, dt 0.5 ms")
    print(f"{'workload':20s} | {'CPU x':>7} {'spikes/s':>10} |" + "".join(
        f" {ads[int(i)]['name'][:12]}/{ads[int(i)]['backend'][:6]:>6} x" for i in sys.argv[1:]))
    for name, idx, rate in wl:
        cpu_x, cpu_s = measure(lambda: LIFEngine(net, dt=0.5, seed=1), idx, rate)
        cols = []
        for i in sys.argv[1:]:
            gx, gs = measure(lambda: WGPUEngine(net, dt=0.5, seed=1, adapter=int(i)), idx, rate)
            cols.append(f" {gx:>16.2f} ({gs / 1e3:.0f}k)")
        print(f"{name:20s} | {cpu_x:>7.2f} {cpu_s:>10,} |" + "".join(cols), flush=True)


if __name__ == "__main__":
    main()
