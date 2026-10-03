"""Dense NumPy reference engine: same discretization as `LIFEngine`, no active set.

Slow by design; it is the test oracle for the compiled engine.
"""
from __future__ import annotations

import numpy as np

from .network import Network
from .params import LIFParams


class ReferenceEngine:
    def __init__(self, net: Network, params: LIFParams = LIFParams(), dt: float = 0.5,
                 seed: int = 0, eps: float = 0.0):
        self.net, self.p, self.dt = net, params, float(dt)
        n = self.n = net.n
        f32 = np.float32
        self.v = np.full(n, params.v_rest, f32)
        self.g = np.zeros(n, f32)
        self.ref = np.zeros(n, f32)
        self.counts = np.zeros(n, np.int32)
        self.dly = max(1, int(round(params.t_dly / dt)))
        self.ring = np.zeros((self.dly + 1, n), f32)
        self.pos = 0
        self.a = f32(np.exp(-dt / params.tau_m))
        self.b = f32(np.exp(-dt / params.tau_syn))
        self.c = f32(params.tau_syn / (params.tau_m - params.tau_syn))
        self.rng = np.random.default_rng(seed)
        self.f_idx = np.zeros(0, np.int32)
        self.f_p = np.zeros(0, f32)
        self.c_idx = np.zeros(0, np.int32)
        self.c_p = np.zeros(0, f32)
        self.c_w = np.zeros(0, f32)
        self._rem = 0.0
        self.steps = 0
        self.total_spikes = 0

    def set_drive(self, idx, rate_hz):
        idx = np.asarray(idx, np.int32)
        rate = np.broadcast_to(np.asarray(rate_hz, np.float32), idx.shape)
        keep = rate > 0
        self.f_idx = idx[keep]
        self.f_p = (1.0 - np.exp(-rate[keep] * self.dt / 1000.0)).astype(np.float32)

    def add_drive(self, idx, rate_hz):
        prev_i, prev_p = self.f_idx, self.f_p
        self.set_drive(idx, rate_hz)
        self.f_idx = np.concatenate([prev_i, self.f_idx])
        self.f_p = np.concatenate([prev_p, self.f_p])

    def set_current_drive(self, idx, rate_hz, w_mv):
        idx = np.asarray(idx, np.int32)
        rate = np.broadcast_to(np.asarray(rate_hz, np.float32), idx.shape)
        w = np.broadcast_to(np.asarray(w_mv, np.float32), idx.shape)
        keep = rate > 0
        self.c_idx = idx[keep]
        self.c_p = (1.0 - np.exp(-rate[keep] * self.dt / 1000.0)).astype(np.float32)
        self.c_w = np.ascontiguousarray(w[keep])

    def _emit(self, spk: np.ndarray, slot: int) -> None:
        net = self.net
        starts, ends = net.indptr[spk], net.indptr[spk + 1]
        lens = ends - starts
        tot = int(lens.sum())
        if tot == 0:
            return
        sel = np.arange(tot) + np.repeat(starts - (np.cumsum(lens) - lens), lens)
        np.add.at(self.ring[slot], net.indices[sel], net.data[sel])

    def step(self) -> None:
        p = self.p
        self.g += self.ring[self.pos]
        self.ring[self.pos] = 0
        slot = (self.pos + self.dly) % len(self.ring)
        if len(self.c_idx):
            hit = self.rng.random(len(self.c_idx)) < self.c_p
            np.add.at(self.g, self.c_idx[hit], self.c_w[hit])
        free = self.ref <= 0
        u = (self.v - np.float32(p.v_rest)) * self.a + self.g * self.c * (self.a - self.b)
        g_new = self.g * self.b
        v_new = np.float32(p.v_rest) + u
        self.v = np.where(free, v_new, self.v)
        self.g = np.where(free, g_new, self.g)
        self.ref = np.where(free, self.ref, np.maximum(self.ref - np.float32(self.dt), 0))
        spk = np.flatnonzero(free & (self.v > np.float32(p.v_th)))
        if len(spk):
            self.v[spk] = p.v_rest
            self.g[spk] = 0
            self.ref[spk] = p.t_ref
            self.counts[spk] += 1
            self.total_spikes += len(spk)
            self._emit(spk, slot)
        if len(self.f_idx):
            f = self.f_idx[self.rng.random(len(self.f_idx)) < self.f_p]
            if len(f):
                self.v[f] = p.v_rest
                self.g[f] = 0
                self.ref[f] = 0
                self.counts[f] += 1
                self.total_spikes += len(f)
                self._emit(f, slot)
        self.pos = (self.pos + 1) % len(self.ring)

    def advance(self, ms: float) -> int:
        total = ms + self._rem
        n = int(total / self.dt + 1e-9)
        self._rem = total - n * self.dt
        for _ in range(n):
            self.step()
        self.steps += n
        return n

    def pop_counts(self, idx):
        out = self.counts[idx].copy()
        self.counts[idx] = 0
        return out

    @property
    def n_active(self) -> int:
        return int(((self.v != self.p.v_rest) | (self.g != 0) | (self.ref > 0)).sum())

    @property
    def sim_ms(self) -> float:
        return self.steps * self.dt
