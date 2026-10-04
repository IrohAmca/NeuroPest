"""Mushroom-body learning layer: dopamine-gated plasticity that turns experience into a valence.

A DESIGNED layer on top of the spiking connectome, not a readout of it. The FlyWire circuit has no plasticity or
neuromodulation (engine/params.py), so the learning lives here as a small rate model with the structure of the fly
mushroom body, and only its output (the valence) is handed back to the brain's drive. What follows the biology, and
what does not:

  * cues -> projection neurons (PN) -> Kenyon cells (KC): every KC reads a few random PNs (~6, as in Caron et al. 2013)
    and one global inhibitory unit (APL) keeps only the top few percent of KCs active (sparse code, Turner et al. 2008).
    Random wiring is what the connectome shows for the calyx; the KC count is a free (cheap) choice.
  * KC -> MBON in compartments, a dopamine neuron (DAN) per compartment: a reward DAN (PAM) depresses the KC->MBON
    synapses of the avoidance MBON, a punishment DAN (PPL1) those of the approach MBON (Aso et al. 2014; Hige et al.
    2015), only for KCs that were active shortly before (an eligibility trace). Depression recovers slowly (forgetting).
  * valence = approach MBON - avoidance MBON, in -1..+1: >0 desire, <0 fear. A naive fly has valence 0 for every cue,
    so with no experience the behaviour is unchanged.

Not modelled (yet): dopamine as a prediction error, KC->KC and MBON->DAN loops (extinction, second-order conditioning),
several compartments with different time scales, and the real FlyWire KC/MBON/DAN cells (their ids and the PN-KC
wiring could replace the random matrix here without changing the interface).

Cost: one 50 ms step is a gather of ~6 values per KC and two dot products, a few tens of microseconds for 2000 KCs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# cue channels, in the order of the `cues` vector given to MushroomBody.step
CUES = ("food_odor", "alarm_odor", "cursor_near", "looming", "touch")


@dataclass(frozen=True)
class MBParams:
    n_kc: int = 2000
    pn_per_kc: int = 6
    levels: int = 4                   # PNs per cue channel: Gaussian tuning over the cue's level, so 0.2 and 0.9 differ
    sparsity: float = 0.05            # fraction of KCs active (APL)
    eta: float = 0.8                  # depression rate (1/s) per unit of dopamine on a fully eligible KC
    tau_trace_s: float = 2.0          # KC eligibility trace: a cue slightly before the reinforcement still pairs
    tau_forget_s: float = 1800.0      # weights recover toward baseline with this time constant
    w_min: float = 0.0
    seed: int = 7


class MushroomBody:
    def __init__(self, params: MBParams | None = None, n_cues: int = len(CUES)):
        p = self.p = params or MBParams()
        rng = np.random.default_rng(p.seed)
        self.n_cues = n_cues
        self.n_pn = n_cues * p.levels
        self.centers = np.tile(np.linspace(0.0, 1.0, p.levels), n_cues).astype(np.float32)
        self.pn_idx = rng.integers(0, self.n_pn, (p.n_kc, p.pn_per_kc)).astype(np.int32)
        self.pn_w = rng.uniform(0.5, 1.5, (p.n_kc, p.pn_per_kc)).astype(np.float32)
        self.k_active = max(1, int(round(p.n_kc * p.sparsity)))
        self.w_app = np.ones(p.n_kc, np.float32)      # KC -> approach MBON
        self.w_av = np.ones(p.n_kc, np.float32)       # KC -> avoidance MBON
        self.trace = np.zeros(p.n_kc, np.float32)
        self.kc = np.zeros(p.n_kc, np.float32)
        self.valence = 0.0
        self.mbon_app = self.mbon_av = 0.0
        self.pam = self.ppl1 = 0.0                    # dopamine now (0..1), for display and tests

    # ------------------------------------------------------------------ code
    def encode(self, cues: np.ndarray) -> np.ndarray:
        """Cue levels (0..1 each) -> binary KC activity with a fixed number of winners (APL inhibition)."""
        c = np.repeat(np.clip(np.asarray(cues, np.float32), 0.0, 1.0), self.p.levels)
        pn = np.exp(-((c - self.centers) ** 2) / (2 * 0.2 ** 2))
        # a silent channel must not feed the KCs: level 0 is "absent", not a stimulus
        pn *= np.repeat(np.asarray(cues, np.float32) > 0.02, self.p.levels)
        drive = (pn[self.pn_idx] * self.pn_w).sum(axis=1)
        kc = np.zeros(self.p.n_kc, np.float32)
        if drive.max() <= 0.0:
            return kc
        top = np.argpartition(drive, -self.k_active)[-self.k_active:]
        kc[top[drive[top] > 0.0]] = 1.0
        return kc

    # ------------------------------------------------------------------ step
    def step(self, cues: np.ndarray, reward: float, punishment: float, dt: float) -> float:
        """Advance `dt` seconds. reward / punishment: 0..1 drive of PAM / PPL1 (the unconditioned stimuli).
        Returns the valence (-1..+1) read out for the current cues."""
        p = self.p
        self.kc = self.encode(cues)
        self.trace += (self.kc - self.trace) * (1.0 - math.exp(-dt / p.tau_trace_s))
        self.pam, self.ppl1 = float(reward), float(punishment)
        # three-factor rule: eligibility (KC) x dopamine -> depression of that KC's output synapse
        if reward > 0.0:
            self.w_av -= p.eta * reward * dt * self.trace * self.w_av
        if punishment > 0.0:
            self.w_app -= p.eta * punishment * dt * self.trace * self.w_app
        np.maximum(self.w_av, p.w_min, out=self.w_av)
        np.maximum(self.w_app, p.w_min, out=self.w_app)
        # slow forgetting toward the naive state
        r = dt / p.tau_forget_s
        self.w_app += (1.0 - self.w_app) * r
        self.w_av += (1.0 - self.w_av) * r
        n = float(self.k_active)
        self.mbon_app = float(self.kc @ self.w_app) / n
        self.mbon_av = float(self.kc @ self.w_av) / n
        self.valence = self.mbon_app - self.mbon_av if self.kc.any() else 0.0
        return self.valence

    def read(self, cues: np.ndarray) -> float:
        """Valence a cue pattern would give now, without learning (for tests and the UI)."""
        kc = self.encode(cues)
        if not kc.any():
            return 0.0
        return float(kc @ (self.w_app - self.w_av)) / float(self.k_active)

    # ---------------------------------------------------------------- memory
    def reset(self) -> None:
        """Amnesia: back to the naive fly."""
        self.w_app[:] = 1.0
        self.w_av[:] = 1.0
        self.trace[:] = 0.0
        self.valence = 0.0

    def save(self, path: str | Path) -> None:
        np.savez(path, w_app=self.w_app, w_av=self.w_av, n_kc=self.p.n_kc, seed=self.p.seed)

    def load(self, path: str | Path) -> bool:
        """Load a saved memory; False (and nothing changes) if it is missing or made for another KC layout."""
        try:
            z = np.load(path)
            if int(z["n_kc"]) != self.p.n_kc or int(z["seed"]) != self.p.seed:
                return False
            self.w_app[:], self.w_av[:] = z["w_app"], z["w_av"]
            return True
        except (OSError, KeyError, ValueError):
            return False
