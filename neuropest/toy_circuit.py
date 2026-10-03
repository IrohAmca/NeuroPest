"""Hand-built stand-in for the real FlyWire subcircuit, with the same shape.

Functional core (placeholders for FlyWire cell types, 146 neurons):
  LOOM   looming detectors (LC4/LPLC2-like)     <- stimulus
  VIS    generic visual object neurons          <- stimulus
  GF     giant fiber / escape command           -> FLY
  WALK   walking descending neurons             -> WALK
  REST   rest/idle drive                        -> STAND
WALK and REST inhibit each other, GF suppresses both.

`n_total > 146` appends a random "bulk" network that does not feed back into the
core. It exists only to exercise the size / performance controls until real
connectome tiers are plugged in; it does not change the behavior.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .engine import Network

CORE_SIZES = {"LOOM": 40, "VIS": 40, "GF": 6, "WALK": 30, "REST": 30}
CORE_N = sum(CORE_SIZES.values())

# (src, dst, connection probability, weight in mV on g)
CORE_EDGES = [
    ("LOOM", "GF", 0.6, 1.5),
    ("VIS", "WALK", 0.4, 0.8),
    ("WALK", "WALK", 0.15, 0.6),
    ("REST", "REST", 0.15, 0.6),
    ("WALK", "REST", 0.3, -1.2),
    ("REST", "WALK", 0.3, -1.2),
    ("GF", "WALK", 0.8, -8.0),
    ("GF", "REST", 0.8, -8.0),
]


def build(n_total: int = CORE_N, seed: int = 1) -> Network:
    rng = np.random.default_rng(seed)
    idx, off = {}, 0
    for k, n in CORE_SIZES.items():
        idx[k] = np.arange(off, off + n, dtype=np.int32)
        off += n
    pre, post, w = [], [], []
    for src, dst, p, wt in CORE_EDGES:
        hit = rng.random((len(idx[dst]), len(idx[src]))) < p
        d, s = np.nonzero(hit)
        pre.append(idx[src][s])
        post.append(idx[dst][d])
        w.append(wt * rng.uniform(0.7, 1.3, len(s)))
    pre, post, w = np.concatenate(pre), np.concatenate(post), np.concatenate(w)

    n_bulk = max(0, n_total - CORE_N)
    groups = dict(idx)
    if n_bulk:
        bulk = Network.random(n_bulk, mean_deg=20, gain=3.5, seed=seed + 1)
        m = sp.block_diag([sp.csr_matrix((w, (pre, post)), shape=(CORE_N, CORE_N)),
                           bulk.to_csr()], format="csr")
        # VIS -> a few bulk targets each, so the bulk also reacts to the cursor
        t = CORE_N + rng.integers(0, n_bulk, (len(idx["VIS"]), 3))
        extra = sp.csr_matrix((np.full(t.size, 3.0, np.float32),
                               (np.repeat(idx["VIS"], 3), t.ravel())), shape=m.shape)
        m = (m + extra).tocsr()
        groups["BULK_DRIVE"] = (CORE_N + rng.choice(n_bulk, max(3, n_bulk // 1000),
                                                    replace=False)).astype(np.int32)
        return Network._from_csr(m, groups=groups, meta={"kind": "toy", "n_bulk": n_bulk})
    m = sp.csr_matrix((w, (pre, post)), shape=(CORE_N, CORE_N))
    groups["BULK_DRIVE"] = np.zeros(0, np.int32)
    return Network._from_csr(m, groups=groups, meta={"kind": "toy", "n_bulk": 0})
