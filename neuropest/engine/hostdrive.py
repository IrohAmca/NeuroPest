"""External drive and silencing, host side of the CPU engines (`LIFEngine`, `ReferenceEngine`).

The kernels read the effective drive lists: forced Poisson spikes (`f_idx`, `f_p`) and Poisson synaptic events
(`c_idx`, `c_p`, `c_w`); the compiled engine draws them as geometric gaps (`f_lp`, `f_next`, `c_lp`, `c_next`). The
lists as set are kept raw, and the effective ones are derived from them: entries with a positive rate, minus the
silenced neurons.

Silencing (the lab's lesions) clamps a neuron in the refractory state (`ref = REF_CLAMP`): it integrates nothing and can
not cross the threshold, so nothing downstream hears it. Two things have to go with the clamp, on every engine:
- a forced spike resets the refractory time, so a silenced neuron's drive entries are dropped from the effective lists
  (kept in the raw ones: releasing the neuron brings its drive back without a new set_drive);
- synaptic input keeps adding to g while a neuron is refractory (without decay), so v and g go back to rest on release,
  or the neuron would fire a burst.
"""
from __future__ import annotations

import numpy as np

from .params import REF_CLAMP


class HostDrive:
    """Mixin: needs `n`, `dt`, `p` (LIFParams) and the state arrays `v`, `g`, `ref`; call `_init_drive()` in __init__."""

    def _init_drive(self) -> None:
        self._sil = np.zeros(self.n, np.bool_)
        self._n_sil = 0
        self._f_raw = (np.zeros(0, np.int32), np.zeros(0, np.float32))
        self._c_raw = (np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros(0, np.float32))
        self._apply_f()
        self._apply_c()

    # -------------------------------------------------------------- input
    def set_drive(self, idx: np.ndarray, rate_hz: np.ndarray | float) -> None:
        """Neurons `idx` emit Poisson spikes at `rate_hz` (forced, like Shiu's PoissonInput)."""
        idx = np.array(idx, np.int32)            # copies: callers reuse their arrays in place (Brain._cursor_fr)
        self._f_raw = (idx, np.array(np.broadcast_to(np.asarray(rate_hz, np.float32), idx.shape)))
        self._apply_f()

    def add_drive(self, idx: np.ndarray, rate_hz: np.ndarray | float) -> None:
        """Like `set_drive` but keeps the drive already set (for several stimulated groups)."""
        idx = np.array(idx, np.int32)
        rate = np.array(np.broadcast_to(np.asarray(rate_hz, np.float32), idx.shape))
        self._f_raw = (np.concatenate([self._f_raw[0], idx]), np.concatenate([self._f_raw[1], rate]))
        self._apply_f()

    def set_current_drive(self, idx: np.ndarray, rate_hz: np.ndarray | float,
                          w_mv: np.ndarray | float) -> None:
        """Poisson synaptic input: events at `rate_hz`, each adding `w_mv` to g of neurons `idx`."""
        idx = np.array(idx, np.int32)
        rate = np.array(np.broadcast_to(np.asarray(rate_hz, np.float32), idx.shape))
        w = np.array(np.broadcast_to(np.asarray(w_mv, np.float32), idx.shape))
        self._c_raw = (idx, rate, w)
        self._apply_c()

    def _keep(self, idx: np.ndarray, rate: np.ndarray) -> np.ndarray:
        keep = rate > 0
        if self._n_sil:
            keep &= ~self._sil[idx]
        return keep

    def _apply_f(self) -> None:
        idx, rate = self._f_raw
        keep = self._keep(idx, rate)
        self.f_idx = np.ascontiguousarray(idx[keep])
        self.f_p = np.ascontiguousarray(1.0 - np.exp(-rate[keep] * self.dt / 1000.0), dtype=np.float32)
        self.f_lp, self.f_next = _gaps_of(rate[keep], self.dt)

    def _apply_c(self) -> None:
        idx, rate, w = self._c_raw
        keep = self._keep(idx, rate)
        self.c_idx = np.ascontiguousarray(idx[keep])
        self.c_p = np.ascontiguousarray(1.0 - np.exp(-rate[keep] * self.dt / 1000.0), dtype=np.float32)
        self.c_lp, self.c_next = _gaps_of(rate[keep], self.dt)
        self.c_w = np.ascontiguousarray(w[keep])

    # ---------------------------------------------------------- silencing
    def set_silenced(self, idx: np.ndarray) -> None:
        """Silence exactly the neurons `idx` (replaces the previous set; an empty one releases everything)."""
        new = np.zeros(self.n, np.bool_)
        new[np.asarray(idx, np.int64)] = True
        on = np.flatnonzero(new & ~self._sil)
        off = np.flatnonzero(self._sil & ~new)
        v0 = np.float32(self.p.v_rest)
        self.ref[on], self.v[on], self.g[on] = REF_CLAMP, v0, 0.0
        self.ref[off], self.v[off], self.g[off] = 0.0, v0, 0.0
        self._sil = new
        self._n_sil = int(new.sum())
        self._apply_f()
        self._apply_c()

    @property
    def silenced(self) -> np.ndarray:
        """Indices of the silenced neurons."""
        return np.flatnonzero(self._sil)


def _gaps_of(rate: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """For the compiled engine's geometric draws: log(1 - p) per entry (float64; p = 1 - exp(-rate dt) is the chance of
    an event in one step, so log(1 - p) = -rate dt exactly) and the steps to each entry's next event (0: not drawn yet).
    A new drive starts every entry afresh, which is exact for a memoryless process."""
    lp = np.ascontiguousarray(-np.asarray(rate, np.float64) * dt / 1000.0)
    return lp, np.zeros(len(lp), np.int64)
