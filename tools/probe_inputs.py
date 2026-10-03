"""Which visual projection neuron types drive which anchor descending neurons?

Drives each VPN cell type alone at a fixed Poisson rate in the full network and prints the anchor
responses, strongest first. Run: uv run python tools/probe_inputs.py [rate_hz]
"""
from __future__ import annotations

import sys

import numpy as np

from neuropest import flywire
from neuropest.engine import LIFEngine


def main():
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    net = flywire.load_cache()
    ann = flywire.load_annotations(net.ids)
    vpn = ann[ann.super_class == "visual_projection"]
    g = net.groups
    anchors = {"GF": g["GF"], "WALK": g["WALK"], "MDN": g["MDN"],
               "DNa02": np.concatenate([g["DNa02_L"], g["DNa02_R"]]), "DNa01": np.concatenate([g["DNa01_L"], g["DNa01_R"]])}
    rows = []
    every = np.arange(net.n)
    by_id = np.argsort(net.ids)
    for ct, sub in vpn.groupby("cell_type"):
        idx = by_id[np.searchsorted(net.ids[by_id], sub.index.values)].astype(np.int32)   # root id -> index
        if len(idx) < 4:
            continue
        e = LIFEngine(net, dt=0.5, seed=7)
        e.set_drive(idx, rate)
        e.advance(300)
        e.pop_counts(every)
        e.advance(1000)
        c = e.pop_counts(every).astype(float)
        rows.append((ct, len(idx), *(c[a].mean() for a in anchors.values()), (c > 0).sum()))
        print(f"{ct:10s} n={len(idx):4d} " + " ".join(f"{k} {v:6.1f}" for k, v in zip(anchors, rows[-1][2:7]))
              + f" | active {rows[-1][-1]}", flush=True)


if __name__ == "__main__":
    main()
