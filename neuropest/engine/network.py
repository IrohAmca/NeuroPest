"""Sparse, signed, weighted directed network in presynaptic-major (CSC-like) layout.

`data[k]` is the voltage kick (mV) a presynaptic spike adds to the postsynaptic
conductance variable: signed synapse count * w_syn, as in Shiu et al. 2024.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np


@dataclass
class Network:
    indptr: np.ndarray                      # int64 [n+1]; out-edges of neuron i: indptr[i]:indptr[i+1]
    indices: np.ndarray                     # int32 [nnz]; postsynaptic neuron
    data: np.ndarray                        # float32 [nnz]; mV
    ids: np.ndarray | None = None           # int64 [n]; FlyWire root ids (None for synthetic nets)
    groups: dict[str, np.ndarray] = field(default_factory=dict)   # name -> int32 neuron indices
    meta: dict = field(default_factory=dict)
    order: np.ndarray | None = None         # int32 [n]; neuron indices, most to least relevant
    extra: dict[str, np.ndarray] = field(default_factory=dict)    # named per-neuron arrays

    @property
    def n(self) -> int:
        return len(self.indptr) - 1

    @property
    def nnz(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_coo(cls, pre, post, w, n: int, **kw) -> "Network":
        import scipy.sparse as sp          # lazy: scipy is slow to import and the worker rarely needs it

        m = sp.csr_matrix((np.asarray(w, np.float32), (np.asarray(pre), np.asarray(post))),
                          shape=(n, n))
        m.sum_duplicates()
        return cls._from_csr(m, **kw)

    @classmethod
    def _from_csr(cls, m, **kw) -> "Network":
        m = m.tocsr()
        m.eliminate_zeros()
        return cls(m.indptr.astype(np.int64), m.indices.astype(np.int32),
                   m.data.astype(np.float32), **kw)

    def to_csr(self):
        import scipy.sparse as sp

        return sp.csr_matrix((self.data, self.indices, self.indptr), shape=(self.n, self.n))

    @classmethod
    def random(cls, n: int, mean_deg: float = 20.0, seed: int = 0, exc_frac: float = 0.7,
               gain: float = 1.0, w_syn: float = 0.275) -> "Network":
        """Heavy-tailed random net with Dale's law; for tests and benchmarks."""
        rng = np.random.default_rng(seed)
        sigma = 1.0
        deg = np.clip(rng.lognormal(np.log(mean_deg) - sigma**2 / 2, sigma, n), 1, n - 1).astype(np.int64)
        indptr = np.zeros(n + 1, np.int64)
        indptr[1:] = np.cumsum(deg)
        nnz = int(indptr[-1])
        indices = rng.integers(0, n, nnz).astype(np.int32)
        counts = (5 + rng.geometric(0.25, nnz)).astype(np.float32)
        sign = np.where(rng.random(n) < exc_frac, 1.0, -1.0).astype(np.float32)
        data = counts * np.repeat(sign, deg) * np.float32(w_syn * gain)
        return cls(indptr, indices, data.astype(np.float32))

    # ------------------------------------------------------------- subnetwork
    def induced(self, keep: np.ndarray) -> "Network":
        """Subnetwork on neurons `keep` (indices into this net), renumbered 0..len(keep)-1.

        The order of `keep` is kept, so a relevance-sorted prefix gives nested tiers.
        """
        keep = np.ascontiguousarray(keep, dtype=np.int32)
        remap = np.full(self.n, -1, np.int32)
        remap[keep] = np.arange(len(keep), dtype=np.int32)
        indptr, indices, data = _induced(self.indptr, self.indices, self.data, keep, remap)
        groups = {}
        for k, idx in self.groups.items():
            r = remap[idx]
            groups[k] = r[r >= 0].astype(np.int32)
        ids = None if self.ids is None else self.ids[keep]
        extra = {k: v[keep] for k, v in self.extra.items()}
        return Network(indptr, indices, data, ids=ids, groups=groups, meta=dict(self.meta), extra=extra)

    def prefix(self, n: int) -> "Network":
        """The `n` most relevant neurons (needs `order`); everything if `n` covers the whole net."""
        if self.order is None:
            raise ValueError("network has no relevance order")
        return self if n >= self.n else self.induced(self.order[:n])

    # --------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        extra = {f"group__{k}": v.astype(np.int32) for k, v in self.groups.items()}
        if self.ids is not None:
            extra["ids"] = self.ids
        if self.order is not None:
            extra["order"] = self.order.astype(np.int32)
        extra.update({f"extra__{k}": v for k, v in self.extra.items()})
        np.savez(path, indptr=self.indptr, indices=self.indices, data=self.data,
                 meta_json=np.array(json.dumps(self.meta)), **extra)

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        z = np.load(path)
        groups = {k[len("group__"):]: z[k] for k in z.files if k.startswith("group__")}
        extra = {k[len("extra__"):]: z[k] for k in z.files if k.startswith("extra__")}
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        return cls(z["indptr"], z["indices"], z["data"],
                   ids=z["ids"] if "ids" in z.files else None, groups=groups, meta=meta,
                   order=z["order"] if "order" in z.files else None, extra=extra)


@nb.njit(cache=True)
def _induced(indptr, indices, data, keep, remap):
    m = keep.shape[0]
    new_ptr = np.zeros(m + 1, np.int64)
    for i in range(m):
        o = keep[i]
        c = 0
        for k in range(indptr[o], indptr[o + 1]):
            if remap[indices[k]] >= 0:
                c += 1
        new_ptr[i + 1] = new_ptr[i] + c
    out_idx = np.empty(new_ptr[m], np.int32)
    out_dat = np.empty(new_ptr[m], np.float32)
    for i in range(m):
        o = keep[i]
        w = new_ptr[i]
        for k in range(indptr[o], indptr[o + 1]):
            t = remap[indices[k]]
            if t >= 0:
                out_idx[w] = t
                out_dat[w] = data[k]
                w += 1
    return new_ptr, out_idx, out_dat
