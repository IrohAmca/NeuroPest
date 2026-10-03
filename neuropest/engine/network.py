"""Sparse, signed, weighted directed network in presynaptic-major (CSC-like) layout.

`data[k]` is the voltage kick (mV) a presynaptic spike adds to the postsynaptic
conductance variable: signed synapse count * w_syn, as in Shiu et al. 2024.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp


@dataclass
class Network:
    indptr: np.ndarray                      # int64 [n+1]; out-edges of neuron i: indptr[i]:indptr[i+1]
    indices: np.ndarray                     # int32 [nnz]; postsynaptic neuron
    data: np.ndarray                        # float32 [nnz]; mV
    ids: np.ndarray | None = None           # int64 [n]; FlyWire root ids (None for synthetic nets)
    groups: dict[str, np.ndarray] = field(default_factory=dict)   # name -> int32 neuron indices
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.indptr) - 1

    @property
    def nnz(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_coo(cls, pre, post, w, n: int, **kw) -> "Network":
        m = sp.csr_matrix((np.asarray(w, np.float32), (np.asarray(pre), np.asarray(post))),
                          shape=(n, n))
        m.sum_duplicates()
        return cls._from_csr(m, **kw)

    @classmethod
    def _from_csr(cls, m: sp.csr_matrix, **kw) -> "Network":
        m = m.tocsr()
        m.eliminate_zeros()
        return cls(m.indptr.astype(np.int64), m.indices.astype(np.int32),
                   m.data.astype(np.float32), **kw)

    def to_csr(self) -> sp.csr_matrix:
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
        net = cls(indptr, indices, data.astype(np.float32))
        return net

    # ------------------------------------------------------------- subnetwork
    def induced(self, keep: np.ndarray) -> "Network":
        """Subnetwork on neurons `keep` (indices into this net), renumbered 0..len(keep)-1.

        Order of `keep` is preserved, so a score-sorted ranking prefix gives nested tiers.
        """
        keep = np.asarray(keep, dtype=np.int64)
        remap = np.full(self.n, -1, np.int64)
        remap[keep] = np.arange(len(keep))
        m = self.to_csr()[keep][:, keep]
        groups = {}
        for k, idx in self.groups.items():
            r = remap[idx]
            groups[k] = r[r >= 0].astype(np.int32)
        ids = None if self.ids is None else self.ids[keep]
        return Network._from_csr(m, ids=ids, groups=groups, meta=dict(self.meta))

    # --------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        extra = {f"group__{k}": v.astype(np.int32) for k, v in self.groups.items()}
        if self.ids is not None:
            extra["ids"] = self.ids
        np.savez_compressed(path, indptr=self.indptr, indices=self.indices,
                            data=self.data, **extra)

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        z = np.load(path)
        groups = {k[len("group__"):]: z[k] for k in z.files if k.startswith("group__")}
        return cls(z["indptr"], z["indices"], z["data"],
                   ids=z["ids"] if "ids" in z.files else None, groups=groups)
