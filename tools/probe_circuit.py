"""How does a FlyWire tier respond when one input group is driven at a range of rates?

Run: uv run python tools/probe_circuit.py [n_neurons ...] [--group LOOM|RETREAT_IN|LC10_L|LC10_R]
                                          [--rates 1,2,5,...]            (default: full network, LOOM)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from neuropest import flywire
from neuropest.engine import LIFEngine

RATES = (0, 10, 25, 50, 100, 150)


def probe(net, group="LOOM", dt=0.5, rates=RATES, warm_ms=300.0, run_ms=1000.0):
    g = net.groups
    out = []
    for r in rates:
        e = LIFEngine(net, dt=dt, seed=3)
        e.set_drive(g[group], float(r))
        e.advance(warm_ms)
        e.pop_counts(np.arange(net.n))
        s0, u0, st0 = e.total_spikes, e.neuron_updates, e.steps
        t = time.perf_counter()
        e.advance(run_ms)
        wall = time.perf_counter() - t
        steps = e.steps - st0
        counts = e.pop_counts(np.arange(net.n)) / (run_ms / 1000.0)        # Hz per neuron
        out.append(dict(rate=r, GF=counts[g["GF"]].mean(), WALK=counts[g["WALK"]].mean(),
                        MDN=counts[g["MDN"]].mean(), DNa02_L=counts[g["DNa02_L"]].mean(),
                        DNa02_R=counts[g["DNa02_R"]].mean(), DN=counts[g["DN"]].mean(),
                        spikes=(e.total_spikes - s0) / (run_ms / 1000.0),
                        active=100 * (e.neuron_updates - u0) / steps / net.n, rt=run_ms / 1000.0 / wall))
    return out


def show(net, tag, group, rates):
    print(f"\n== {tag}: {net.n:,} neurons, {net.nnz:,} edges, driving {group}")
    print(f"{'Hz':>6} | {'GF':>6} {'WALK':>6} {'MDN':>6} {'DNa02L':>7} {'DNa02R':>7} {'DN avg':>7} | "
          f"{'spikes/s':>9} {'active%':>8} {'realtime x':>10}")
    for r in probe(net, group, rates=rates):
        print(f"{r['rate']:>6} | {r['GF']:>6.1f} {r['WALK']:>6.1f} {r['MDN']:>6.1f} {r['DNa02_L']:>7.1f} "
              f"{r['DNa02_R']:>7.1f} {r['DN']:>7.2f} | {r['spikes']:>9.0f} {r['active']:>8.2f} {r['rt']:>10.2f}",
              flush=True)


def main():
    args = sys.argv[1:]
    rates, group = RATES, "LOOM"
    for flag in ("--rates", "--group"):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            if flag == "--rates":
                rates = tuple(float(x) for x in val.split(","))
            else:
                group = val
            args = args[:i] + args[i + 2:]
    net = flywire.load_cache()
    for n in [int(a) for a in args] or [net.n]:
        show(net.prefix(n), f"top {n:,}" if n < net.n else "full brain", group, rates)


if __name__ == "__main__":
    main()
