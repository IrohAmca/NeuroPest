"""How does a FlyWire tier respond when the looming inputs (LPLC2 + LC4) are driven?

Run: uv run python tools/probe_circuit.py [n_neurons ...]     (default: full network)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from neuropest import flywire
from neuropest.engine import LIFEngine

RATES = (0, 10, 25, 50, 100, 150)


def probe(net, dt=0.5, rates=RATES, warm_ms=300.0, run_ms=1000.0):
    g = net.groups
    out = []
    for r in rates:
        e = LIFEngine(net, dt=dt, seed=3)
        e.set_drive(g["LOOM"], float(r))
        e.advance(warm_ms)
        e.pop_counts(np.arange(net.n))
        s0, u0, st0 = e.total_spikes, e.neuron_updates, e.steps
        t = time.perf_counter()
        e.advance(run_ms)
        wall = time.perf_counter() - t
        steps = e.steps - st0
        counts = e.pop_counts(np.arange(net.n)) / (run_ms / 1000.0)        # Hz per neuron
        row = dict(rate=r,
                   GF=counts[g["GF"]].mean(), WALK=counts[g["WALK"]].mean(), MDN=counts[g["MDN"]].mean(),
                   DNa02=counts[np.concatenate([g["DNa02_L"], g["DNa02_R"]])].mean(),
                   DN=counts[g["DN"]].mean(), dn_on=int((counts[g["DN"]] > 1).sum()),
                   spikes=(e.total_spikes - s0) / (run_ms / 1000.0),
                   active=100 * (e.neuron_updates - u0) / steps / net.n,
                   rt=run_ms / 1000.0 / wall, counts=counts)
        out.append(row)
    return out


def show(net, tag, rates=RATES):
    print(f"\n== {tag}: {net.n:,} neurons, {net.nnz:,} edges")
    print(f"{'LOOM Hz':>8} | {'GF':>6} {'WALK':>6} {'MDN':>6} {'DNa02':>6} {'DN avg':>7} {'DN>1Hz':>7} | "
          f"{'spikes/s':>9} {'active%':>8} {'realtime x':>10}")
    for r in probe(net, rates=rates):
        print(f"{r['rate']:>8} | {r['GF']:>6.1f} {r['WALK']:>6.1f} {r['MDN']:>6.1f} {r['DNa02']:>6.1f} "
              f"{r['DN']:>7.2f} {r['dn_on']:>7} | {r['spikes']:>9.0f} {r['active']:>8.2f} {r['rt']:>10.2f}", flush=True)


def main():
    args = sys.argv[1:]
    rates = RATES
    if "--rates" in args:
        i = args.index("--rates")
        rates = tuple(float(x) for x in args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    net = flywire.load_cache()
    sizes = [int(a) for a in args] or [net.n]
    for n in sizes:
        sub = net.prefix(n)
        show(sub, f"top {n:,}" if n < net.n else "full brain", rates)


if __name__ == "__main__":
    main()
