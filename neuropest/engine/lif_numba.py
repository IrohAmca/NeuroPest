"""Event-driven LIF engine (numba).

Same neuron model as Shiu et al. 2024, integrated exactly per time step:

    dv/dt = (v_rest - v + g) / tau_m        dg/dt = -g / tau_syn
    spike when v > v_th: v = v_rest, g = 0, refractory for t_ref (v and g frozen)
    a presynaptic spike adds w (mV) to g of each target after t_dly

Cost per step is proportional to the number of *active* neurons (non-resting
state) plus delivered synaptic events, not to the network size: a silent
neuron costs nothing, because the model has no spontaneous activity.
"""
from __future__ import annotations

import numba as nb
import numpy as np

from .network import Network
from .params import LIFParams


@nb.njit(cache=True)
def _advance(n_steps, dt, a, b, c, dly, v0, v_th, t_ref, eps,
             v, g, ref, counts, ring, touch, ntouch, tflag,
             act, in_act, sc, indptr, indices, data, f_idx, f_p, c_idx, c_p, c_w, seed):
    np.random.seed(seed)
    D = ring.shape[0]
    pos = sc[0]
    nact = sc[1]
    nf = f_idx.shape[0]
    nc = c_idx.shape[0]
    spikes = 0
    work = 0
    for _ in range(n_steps):
        # 1. deliver synaptic input that arrives now
        for q in range(ntouch[pos]):
            i = touch[pos, q]
            g[i] += ring[pos, i]
            ring[pos, i] = 0.0
            tflag[pos, i] = 0
            if in_act[i] == 0:
                in_act[i] = 1
                act[nact] = i
                nact += 1
        ntouch[pos] = 0
        slot = (pos + dly) % D

        # 1b. graded Poisson synaptic drive (events add w to g, like an outside presynaptic cell)
        for q in range(nc):
            if np.random.random() < c_p[q]:
                i = c_idx[q]
                g[i] += c_w[q]
                if in_act[i] == 0:
                    in_act[i] = 1
                    act[nact] = i
                    nact += 1

        # 2. integrate the active neurons, collect spikes, compact the list
        w = 0
        for q in range(nact):
            i = act[q]
            if ref[i] > 0.0:
                ref[i] -= dt
                if ref[i] < 0.0:
                    ref[i] = 0.0
                act[w] = i
                w += 1
                continue
            gi = g[i]
            ui = (v[i] - v0) * a + gi * c * (a - b)
            gi = gi * b
            if ui + v0 > v_th:
                v[i] = v0
                g[i] = 0.0
                ref[i] = t_ref
                counts[i] += 1
                spikes += 1
                for k in range(indptr[i], indptr[i + 1]):
                    j = indices[k]
                    ring[slot, j] += data[k]
                    if tflag[slot, j] == 0:
                        tflag[slot, j] = 1
                        touch[slot, ntouch[slot]] = j
                        ntouch[slot] += 1
                act[w] = i
                w += 1
            elif abs(ui) < eps and abs(gi) < eps:
                v[i] = v0
                g[i] = 0.0
                in_act[i] = 0
            else:
                v[i] = v0 + ui
                g[i] = gi
                act[w] = i
                w += 1
        work += nact
        nact = w

        # 3. externally driven (Poisson) spikes: no refractory period, as in Shiu
        for q in range(nf):
            if np.random.random() < f_p[q]:
                i = f_idx[q]
                v[i] = v0
                g[i] = 0.0
                ref[i] = 0.0
                counts[i] += 1
                spikes += 1
                for k in range(indptr[i], indptr[i + 1]):
                    j = indices[k]
                    ring[slot, j] += data[k]
                    if tflag[slot, j] == 0:
                        tflag[slot, j] = 1
                        touch[slot, ntouch[slot]] = j
                        ntouch[slot] += 1
        pos += 1
        if pos == D:
            pos = 0
    sc[0] = pos
    sc[1] = nact
    sc[2] += spikes
    sc[3] += work


class LIFEngine:
    """Spiking engine over a `Network`. Interface shared with `ReferenceEngine`."""

    def __init__(self, net: Network, params: LIFParams = LIFParams(), dt: float = 0.5,
                 seed: int = 0, eps: float = 0.01):
        self.net, self.p, self.dt, self.eps = net, params, float(dt), float(eps)
        n = net.n
        self.n = n
        self.v = np.full(n, params.v_rest, np.float32)
        self.g = np.zeros(n, np.float32)
        self.ref = np.zeros(n, np.float32)
        self.counts = np.zeros(n, np.int32)
        self.dly = max(1, int(round(params.t_dly / dt)))
        D = self.dly + 1
        self.ring = np.zeros((D, n), np.float32)
        self.touch = np.zeros((D, n), np.int32)
        self.ntouch = np.zeros(D, np.int64)
        self.tflag = np.zeros((D, n), np.uint8)
        self.act = np.zeros(n, np.int32)
        self.in_act = np.zeros(n, np.uint8)
        self.sc = np.zeros(4, np.int64)     # pos, n_active, total_spikes, neuron_updates
        self.a = np.float32(np.exp(-dt / params.tau_m))
        self.b = np.float32(np.exp(-dt / params.tau_syn))
        self.c = np.float32(params.tau_syn / (params.tau_m - params.tau_syn))
        self._seed = seed
        self._rem = 0.0
        self.steps = 0
        self.f_idx = np.zeros(0, np.int32)
        self.f_p = np.zeros(0, np.float32)
        self.c_idx = np.zeros(0, np.int32)
        self.c_p = np.zeros(0, np.float32)
        self.c_w = np.zeros(0, np.float32)

    # -------------------------------------------------------------- input
    def set_drive(self, idx: np.ndarray, rate_hz: np.ndarray | float) -> None:
        """Neurons `idx` emit Poisson spikes at `rate_hz` (forced, like Shiu's PoissonInput)."""
        idx = np.asarray(idx, np.int32)
        rate = np.broadcast_to(np.asarray(rate_hz, np.float32), idx.shape)
        keep = rate > 0
        self.f_idx = np.ascontiguousarray(idx[keep])
        self.f_p = np.ascontiguousarray(1.0 - np.exp(-rate[keep] * self.dt / 1000.0), dtype=np.float32)

    def set_current_drive(self, idx: np.ndarray, rate_hz: np.ndarray | float,
                          w_mv: np.ndarray | float) -> None:
        """Poisson synaptic input: events at `rate_hz`, each adding `w_mv` to g of neurons `idx`."""
        idx = np.asarray(idx, np.int32)
        rate = np.broadcast_to(np.asarray(rate_hz, np.float32), idx.shape)
        w = np.broadcast_to(np.asarray(w_mv, np.float32), idx.shape)
        keep = rate > 0
        self.c_idx = np.ascontiguousarray(idx[keep])
        self.c_p = np.ascontiguousarray(1.0 - np.exp(-rate[keep] * self.dt / 1000.0), dtype=np.float32)
        self.c_w = np.ascontiguousarray(w[keep])

    # ---------------------------------------------------------------- run
    def advance(self, ms: float) -> int:
        total = ms + self._rem
        n_steps = int(total / self.dt + 1e-9)
        self._rem = total - n_steps * self.dt
        if n_steps <= 0:
            return 0
        self._seed += 1
        p, net = self.p, self.net
        _advance(n_steps, np.float32(self.dt), self.a, self.b, self.c, self.dly,
                 np.float32(p.v_rest), np.float32(p.v_th), np.float32(p.t_ref),
                 np.float32(self.eps), self.v, self.g, self.ref, self.counts,
                 self.ring, self.touch, self.ntouch, self.tflag, self.act, self.in_act,
                 self.sc, net.indptr, net.indices, net.data, self.f_idx, self.f_p,
                 self.c_idx, self.c_p, self.c_w, self._seed)
        self.steps += n_steps
        return n_steps

    # ------------------------------------------------------------- output
    def pop_counts(self, idx: np.ndarray) -> np.ndarray:
        """Spikes of neurons `idx` since their last pop; resets those counters."""
        out = self.counts[idx].copy()
        self.counts[idx] = 0
        return out

    @property
    def n_active(self) -> int:
        return int(self.sc[1])

    @property
    def total_spikes(self) -> int:
        return int(self.sc[2])

    @property
    def neuron_updates(self) -> int:
        return int(self.sc[3])

    @property
    def sim_ms(self) -> float:
        return self.steps * self.dt
