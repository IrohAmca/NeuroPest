"""Does driving one side's visual projection neurons make DNa02 left or right fire?

Run: uv run python tools/probe_side.py [cell_type ...]
"""
from __future__ import annotations

import sys

import numpy as np

from neuropest import flywire
from neuropest.engine import LIFEngine


def main():
    types = sys.argv[1:] or ["LPLC4", "LLPC1", "LC10c-2", "LC10d", "LC11", "LC18", "LC10a"]
    net = flywire.load_cache()
    ann = flywire.load_annotations(net.ids)
    g = net.groups
    every = np.arange(net.n)
    pos = {int(r): i for i, r in enumerate(net.ids)}
    print(f"{'type':10s} {'drive':6s} | DNa02_L DNa02_R | DNa01_L DNa01_R | active")
    for ct in types:
        for side in ("left", "right"):
            sub = ann[(ann.cell_type == ct) & (ann.side == side)]
            idx = np.array([pos[int(r)] for r in sub.index if int(r) in pos], np.int32)
            e = LIFEngine(net, dt=0.5, seed=3)
            e.set_drive(idx, 50.0)
            e.advance(300)
            e.pop_counts(every)
            e.advance(1000)
            c = e.pop_counts(every).astype(float)
            print(f"{ct:10s} {side:6s} | {c[g['DNa02_L']].mean():7.1f} {c[g['DNa02_R']].mean():7.1f} | "
                  f"{c[g['DNa01_L']].mean():7.1f} {c[g['DNa01_R']].mean():7.1f} | {int((c > 0).sum())}", flush=True)


if __name__ == "__main__":
    main()
