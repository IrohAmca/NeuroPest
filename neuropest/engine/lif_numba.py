"""Event-driven LIF engine (numba).

Same neuron model as Shiu et al. 2024, integrated exactly per time step:

    dv/dt = (v_rest - v + g) / tau_m        dg/dt = -g / tau_syn
    spike when v > v_th: v = v_rest, g = 0, refractory for t_ref (v and g frozen)
    a presynaptic spike adds w (mV) to g of each target after t_dly

plus one mechanism the Shiu model lacks, on the neurons the network asks for it (`net.extra["adapt_mv"]`, mV per
spike; FlyWire: the antennal-lobe projection and local neurons): spike-frequency adaptation. Each spike of such a neuron
raises its threshold by adapt_mv, and the raise decays with tau_adapt. It is kept lazily (value and the step it was set
at), so it costs nothing while the neuron is silent.

Cost per step is proportional to the number of *active* neurons (non-resting
state) plus delivered synaptic events, not to the network size: a silent
neuron costs nothing, because the model has no spontaneous activity.

The Poisson drive is drawn as geometric gaps between events (one random number per event, a counter per step) instead of
one Bernoulli trial per driven neuron per step: the same process (the number of steps to the first success of a Bernoulli
trial is geometric; Devroye 1986, inverse transform), ~10-17x cheaper for groups of hundreds of neurons.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

from .hostdrive import HostDrive
from .network import Network
from .params import LIFParams


@nb.njit(cache=True)
def _gap(lp):
    """Steps until the next event of a per-step Bernoulli(p), >= 1; lp = log(1 - p) (-inf for p = 1)."""
    if lp == -np.inf:
        return 1
    k = math.log(1.0 - np.random.random()) / lp          # 1 - random() is in (0, 1]
    if k >= 4.0e18:
        return 4_000_000_000_000_000_000
    return 1 + int(k)


@nb.njit(cache=True)
def _advance(n_steps, s0, dt, a, b, c, dly, v0, v_th, t_ref, eps,
             v, g, ref, counts, ring, touch, ntouch, tflag,
             act, in_act, sc, indptr, indices, data, f_idx, f_lp, f_next, c_idx, c_lp, c_next, c_w, seed,
             has_adapt, adapt_mv, th, th_s, dt_tau_a):
    np.random.seed(seed)
    D = ring.shape[0]
    pos = sc[0]
    nact = sc[1]
    nf = f_idx.shape[0]
    nc = c_idx.shape[0]
    spikes = 0
    work = 0
    for st in range(n_steps):
        s = s0 + st
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
            k = c_next[q]
            if k == 0:
                k = _gap(c_lp[q])
            k -= 1
            if k == 0:
                k = _gap(c_lp[q])
                i = c_idx[q]
                g[i] += c_w[q]
                if in_act[i] == 0:
                    in_act[i] = 1
                    act[nact] = i
                    nact += 1
            c_next[q] = k

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
            thr = v_th
            if has_adapt and th[i] > 0.0:
                thr = v_th + th[i] * math.exp(-(s - th_s[i]) * dt_tau_a)
            if ui + v0 > thr:
                v[i] = v0
                g[i] = 0.0
                ref[i] = t_ref
                counts[i] += 1
                spikes += 1
                if has_adapt and adapt_mv[i] > 0.0:
                    th[i] = th[i] * math.exp(-(s - th_s[i]) * dt_tau_a) + adapt_mv[i]
                    th_s[i] = s
                for k in range(indptr[i], indptr[i + 1]):
                    j = indices[k]
                    ring[slot, j] += data[k]
                    if tflag[slot, j] == 0:
                        tflag[slot, j] = 1
                        touch[slot, ntouch[slot]] = j
                        ntouch[slot] += 1
                act[w] = i
                w += 1
            elif abs(ui) < eps and abs(gi) < eps:      # the adapted threshold decays lazily: it may leave
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
            k = f_next[q]
            if k == 0:
                k = _gap(f_lp[q])
            k -= 1
            if k == 0:
                k = _gap(f_lp[q])
                i = f_idx[q]
                v[i] = v0
                g[i] = 0.0
                ref[i] = 0.0
                counts[i] += 1
                spikes += 1
                for kk in range(indptr[i], indptr[i + 1]):
                    j = indices[kk]
                    ring[slot, j] += data[kk]
                    if tflag[slot, j] == 0:
                        tflag[slot, j] = 1
                        touch[slot, ntouch[slot]] = j
                        ntouch[slot] += 1
            f_next[q] = k
        pos += 1
        if pos == D:
            pos = 0
    sc[0] = pos
    sc[1] = nact
    sc[2] += spikes
    sc[3] += work


class LIFEngine(HostDrive):
    """Spiking engine over a `Network`. Interface shared with `ReferenceEngine` (drive and silencing: HostDrive)."""

    def __init__(self, net: Network, params: LIFParams = LIFParams(), dt: float = 0.5,
                 seed: int = 0, eps: float = 0.1):
        """`eps` (mV): a neuron within eps of rest with no conductance is snapped to rest and
        leaves the active set. 0.1 mV (1.4% of the 7 mV threshold gap) changed the spike count
        of a 139k-neuron run by 1 in 46,000 and cut the work ~1.6x versus never snapping."""
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
        self.adapt_mv = adapt_of(net)
        self.th = np.zeros(n, np.float32)                # threshold raise (mV) as of step th_s
        self.th_s = np.zeros(n, np.int64)
        self._seed = seed
        self._rem = 0.0
        self.steps = 0
        self._init_drive()

    # ---------------------------------------------------------------- run
    def advance(self, ms: float) -> int:
        total = ms + self._rem
        n_steps = int(total / self.dt + 1e-9)
        self._rem = total - n_steps * self.dt
        if n_steps <= 0:
            return 0
        self._seed += 1
        p, net = self.p, self.net
        _advance(n_steps, np.int64(self.steps), np.float32(self.dt), self.a, self.b, self.c, self.dly,
                 np.float32(p.v_rest), np.float32(p.v_th), np.float32(p.t_ref),
                 np.float32(self.eps), self.v, self.g, self.ref, self.counts,
                 self.ring, self.touch, self.ntouch, self.tflag, self.act, self.in_act,
                 self.sc, net.indptr, net.indices, net.data, self.f_idx, self.f_lp, self.f_next,
                 self.c_idx, self.c_lp, self.c_next, self.c_w, self._seed,
                 self.adapt_mv is not None, self.adapt_mv if self.adapt_mv is not None else self.th, self.th,
                 self.th_s, self.dt / p.tau_adapt)
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


def adapt_of(net: Network) -> np.ndarray | None:
    """Per-neuron threshold raise per spike (mV, float32) the network asks for, or None when no neuron adapts."""
    a = net.extra.get("adapt_mv")
    if a is None:
        return None
    a = np.ascontiguousarray(a, np.float32)
    return a if a.shape == (net.n,) and bool(np.any(a > 0)) else None
