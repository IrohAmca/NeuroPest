"""Leaky integrate-and-fire network on a sparse weight matrix.

Same model family as Shiu et al. 2024 (LIF, synaptic weight = signed synapse
count * w_syn), but a plain NumPy/SciPy Euler step so a small circuit runs in
real time inside the GUI loop without Brian2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class LIFParams:
    v_rest: float = -52.0   # mV
    v_reset: float = -52.0
    v_th: float = -45.0
    tau_m: float = 20.0     # ms
    tau_syn: float = 5.0    # ms
    t_ref: float = 2.2      # ms
    w_syn: float = 0.275    # mV per synapse (Shiu et al. 2024)


class Circuit:
    """W[i, j] = signed synapse count from neuron j onto neuron i."""

    def __init__(self, W: sp.spmatrix, names: list[str] | None = None,
                 params: LIFParams | None = None, seed: int = 0):
        self.W = sp.csr_matrix(W, dtype=np.float32)
        self.n = self.W.shape[0]
        self.names = names or [str(i) for i in range(self.n)]
        self.p = params or LIFParams()
        self.rng = np.random.default_rng(seed)
        self.v = np.full(self.n, self.p.v_rest, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.ref = np.zeros(self.n, dtype=np.float32)
        self.rate = np.zeros(self.n, dtype=np.float32)  # smoothed Hz

    def step(self, dt_ms: float, ext_rate_hz: np.ndarray | None = None,
             ext_w: float = 8.0) -> np.ndarray:
        """Advance dt_ms. ext_rate_hz: Poisson drive per neuron (sensory input)."""
        p = self.p
        if ext_rate_hz is not None:
            lam = ext_rate_hz * dt_ms / 1000.0
            self.g += self.rng.poisson(lam).astype(np.float32) * ext_w
        self.g *= np.exp(-dt_ms / p.tau_syn)
        active = self.ref <= 0
        dv = (p.v_rest - self.v + self.g) * (dt_ms / p.tau_m)
        self.v = np.where(active, self.v + dv, self.v)
        spk = self.v >= p.v_th
        if spk.any():
            self.v[spk] = p.v_reset
            self.ref[spk] = p.t_ref
            self.g += self.W[:, spk].sum(axis=1).A1 * p.w_syn
        self.ref = np.maximum(self.ref - dt_ms, 0)
        a = dt_ms / 100.0  # 100 ms rate smoothing
        self.rate += a * (spk / (dt_ms / 1000.0) - self.rate)
        return spk
