"""Hand-built stand-in circuit with the same shape as the real extraction.

Populations (placeholders for FlyWire cell types):
  LOOM   looming detectors (LC/LPLC2-like)      <- visual stimulus
  GF     giant fiber / escape command           -> FLY
  VIS    generic visual motion/object           <- stimulus (weaker)
  WALK   walking descending neurons (DNa/DNg)   -> WALK
  REST   rest/idle drive                        -> STAND
Mutual inhibition WALK<->REST, GF suppresses both.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .circuit import Circuit

SIZES = {"LOOM": 40, "VIS": 40, "GF": 6, "WALK": 30, "REST": 30}


def build(seed: int = 1, scale: int = 1) -> tuple[Circuit, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    sizes = {k: v * scale for k, v in SIZES.items()}
    idx, off = {}, 0
    for k, n in sizes.items():
        idx[k] = np.arange(off, off + n)
        off += n
    N = off
    W = sp.lil_matrix((N, N), dtype=np.float32)

    def connect(src, dst, p, w):
        for i in idx[dst]:
            pre = idx[src][rng.random(len(idx[src])) < p]
            W[i, pre] = w * rng.uniform(0.6, 1.4, len(pre))

    connect("LOOM", "GF", 0.6, 4.0)
    connect("VIS", "WALK", 0.4, 3.0)
    connect("WALK", "WALK", 0.15, 1.5)
    connect("REST", "REST", 0.15, 1.5)
    connect("WALK", "REST", 0.3, -3.0)
    connect("REST", "WALK", 0.3, -3.0)
    connect("GF", "WALK", 0.8, -6.0)
    connect("GF", "REST", 0.8, -6.0)
    names = [f"{k}_{i}" for k, ids in idx.items() for i in range(len(ids))]
    return Circuit(W.tocsr(), names, seed=seed), idx
