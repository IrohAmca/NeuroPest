"""GF and MDN when looming (LOOM) and retreat (RETREAT_IN) inputs are driven together.

Run: uv run python tools/probe_combo.py [n_neurons]   (default: full network)
"""
from __future__ import annotations

import sys

import numpy as np

from neuropest import flywire
from neuropest.engine import LIFEngine

LOOM = (0, 4, 6, 10, 20, 40, 80)
RETREAT = (0, 3, 5, 8, 12, 20, 30, 60)


def main():
    net = flywire.load_cache()
    if len(sys.argv) > 1:
        net = net.prefix(int(sys.argv[1]))
    g = net.groups
    every = np.arange(net.n)
    print(f"{net.n:,} neurons.  cells: GF Hz / MDN Hz")
    hdr = "LOOM \\ RETREAT_IN"
    print(f"{hdr:>18} " + " ".join(f"{r:>11}" for r in RETREAT))
    for lo in LOOM:
        row = []
        for re in RETREAT:
            e = LIFEngine(net, dt=0.5, seed=9)
            e.add_drive(g["LOOM"], float(lo))
            e.add_drive(g["RETREAT_IN"], float(re))
            e.advance(300)
            e.pop_counts(every)
            e.advance(1000)
            c = e.pop_counts(every).astype(float)
            row.append(f"{c[g['GF']].mean():5.0f}/{c[g['MDN']].mean():<4.0f}")
        print(f"{lo:>18} " + " ".join(f"{x:>11}" for x in row), flush=True)


if __name__ == "__main__":
    main()
