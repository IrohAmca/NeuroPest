"""Build data/circuits/flywire_v783.npz from the raw FlyWire files in data/raw.

Run: uv run python tools/build_flywire.py
"""
from __future__ import annotations

import time

import numpy as np

from neuropest import flywire


def main():
    t = time.perf_counter()
    net = flywire.build()
    print(f"built in {time.perf_counter() - t:.1f} s: {net.n:,} neurons, {net.nnz:,} synaptic edges")
    for k, v in net.groups.items():
        print(f"  group {k:8s} {len(v):6d} neurons")
    rank = np.empty(net.n, np.int32)
    rank[net.order] = np.arange(net.n)
    for k in ("GF", "WALK", "MDN", "DNa02_L", "LOOM"):
        r = rank[net.groups[k]]
        print(f"  rank of {k:8s}: min {r.min()} median {int(np.median(r))} max {r.max()}")
    flywire.CACHE.parent.mkdir(parents=True, exist_ok=True)
    net.save(flywire.CACHE)
    print("saved", flywire.CACHE, f"({flywire.CACHE.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
