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

  * dopamine is a prediction error: the valence a cue already has is subtracted from the reinforcement (the role of the
    MBON -> DAN feedback), so a learned cue stops teaching (blocking, saturation) and a predicted reinforcement that does
    not come (a desired cue with no food, a feared cue with no threat) is an omission that undoes the memory a little
    (extinction, `eta_ext`), on top of the slow forgetting.

Not modelled (yet): KC->KC loops and second-order conditioning, and different time scales per compartment.

Cost: one 50 ms step is a gather of ~6 values per KC and two dot products, a few tens of microseconds for 2000 KCs.

`MushroomBody.from_flywire()` swaps the random matrices for the real FlyWire v783 cells (class FlywireMushroomBody, data
from tools/build_mushroom.py). The default stays the random model above. What is real there and what is still designed
is written at that class.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import MUSHROOM

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
    pred_gain: float = 0.8            # dopamine = reinforcement minus what the learned valence already predicts (0: off)
    eta_ext: float = 0.05             # extinction: recovery rate (1/s) of the memory while the predicted reinforcement is missing
    w_min: float = 0.0
    seed: int = 7


class MushroomBody:
    kind = "random"                                   # which wiring: "random" (this class) or "flywire"

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
        self.dirty = False                            # learned since the last save

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
        n = float(self.k_active)
        # dopamine = reinforcement - prediction (the valence this cue already has); a missing predicted reinforcement
        # is an omission (extinction)
        v0 = float(self.kc @ (self.w_app - self.w_av)) / n if p.pred_gain > 0.0 and self.kc.any() else 0.0
        pe_r = reward - p.pred_gain * max(v0, 0.0)
        pe_p = punishment - p.pred_gain * max(-v0, 0.0)
        dop_r, dop_p, ext_r, ext_p = max(pe_r, 0.0), max(pe_p, 0.0), max(-pe_r, 0.0), max(-pe_p, 0.0)
        self.pam, self.ppl1 = dop_r, dop_p
        # three-factor rule: eligibility (KC) x dopamine -> depression of that KC's output synapse
        self.dirty = self.dirty or dop_r > 0.0 or dop_p > 0.0 or (ext_r > 0.0 or ext_p > 0.0) and bool(self.trace.any())
        if dop_r > 0.0:
            self.w_av -= p.eta * dop_r * dt * self.trace * self.w_av
        if dop_p > 0.0:
            self.w_app -= p.eta * dop_p * dt * self.trace * self.w_app
        if ext_r > 0.0:                                  # reward predicted, none came: the avoidance MBON recovers
            self.w_av += (1.0 - self.w_av) * (p.eta_ext * ext_r * dt) * self.trace
        if ext_p > 0.0:
            self.w_app += (1.0 - self.w_app) * (p.eta_ext * ext_p * dt) * self.trace
        np.maximum(self.w_av, p.w_min, out=self.w_av)
        np.maximum(self.w_app, p.w_min, out=self.w_app)
        # slow forgetting toward the naive state
        r = dt / p.tau_forget_s
        self.w_app += (1.0 - self.w_app) * r
        self.w_av += (1.0 - self.w_av) * r
        self.mbon_app = float(self.kc @ self.w_app) / n
        self.mbon_av = float(self.kc @ self.w_av) / n
        self.valence = self.mbon_app - self.mbon_av if self.kc.any() else 0.0
        return self.valence

    @property
    def valence_vector(self) -> np.ndarray:
        """Net learned valence per Kenyon cell: w_app - w_av in [-1, +1]."""
        return (self.w_app - self.w_av).astype(np.float32)

    def read(self, cues: np.ndarray) -> float:
        """Valence a cue pattern would give now, without learning (for tests and the UI)."""
        kc = self.encode(cues)
        if not kc.any():
            return 0.0
        return float(kc @ (self.w_app - self.w_av)) / float(self.k_active)

    @classmethod
    def from_flywire(cls, path: str | Path | None = None, params: MBParams | None = None, **kw) -> "FlywireMushroomBody":
        """The same layer on the real FlyWire cells (see FlywireMushroomBody); raises FileNotFoundError without the file
        that tools/build_mushroom.py writes."""
        return FlywireMushroomBody(load_wiring(path or MUSHROOM), params, **kw)

    # ---------------------------------------------------------------- memory
    def memory_file(self, path: str | Path) -> Path:
        """Where this wiring keeps its memory, given the configured path (the random model: that path itself)."""
        return Path(path)

    def reset(self) -> None:
        """Amnesia: back to the naive fly."""
        self.w_app[:] = 1.0
        self.w_av[:] = 1.0
        self.trace[:] = 0.0
        self.valence = 0.0
        self.dirty = True

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w_app=self.w_app, w_av=self.w_av, n_kc=self.p.n_kc, seed=self.p.seed)
        self.dirty = False

    def load(self, path: str | Path) -> bool:
        """Load a saved memory; False (and nothing changes) if it is missing or made for another KC layout."""
        try:
            z = np.load(path)
            if "kind" in z.files and str(z["kind"]) != self.kind:      # a memory of the real KCs is not ours
                return False
            if int(z["n_kc"]) != self.p.n_kc or int(z["seed"]) != self.p.seed:
                return False
            self.w_app[:], self.w_av[:] = z["w_app"], z["w_av"]
            return True
        except (OSError, KeyError, ValueError):
            return False



# ====================================================================================================================
# The same layer on the real FlyWire v783 mushroom body
# ====================================================================================================================
#
# REAL (read from the connectome, tools/build_mushroom.py -> data/circuits/mushroom_flywire.npz):
#   * the Kenyon cells themselves (5177 in both hemispheres, all subtypes) and their MBONs (96 neurons, 35 types);
#   * KC -> MBON synapse counts per neuron pair: which KCs reach which MBON, and how strongly (the compartment structure);
#   * PN -> KC synapse counts per pair (the calyx wiring: ~7 PN claws per KC, olfactory ALPN and visual PNs);
#   * how strongly PAM and PPL1 dopamine neurons contact each MBON (DAN -> MBON synapses): the dopamine "coupling" of
#     a compartment. A KC -> MBON synapse is depressed at a rate proportional to its MBON's coupling to the DAN class
#     that is active (reward -> PAM, punishment -> PPL1), as in the three-factor rule of the random model.
#
# DESIGNED / ASSUMED (nothing in the connectome says it):
#   * which PN types stand for which of the app's cues (CUE_PN_TYPES). The desktop fly has no antennae, and the visual
#     PNs that reach the MB are not the cells the eye model drives, so a cue is only a stand-in that fires a pool of
#     real PNs; what is real is the KC code those PNs produce, not the sensory story. Touch has no PN route at all.
#   * the sign (approach / avoidance) of each MBON: from the literature where the table below has it, otherwise
#     DERIVED from the dopamine coupling ("a compartment with a reward DAN holds an avoidance MBON, a punishment DAN an
#     approach MBON", Aso et al. 2014; Hige et al. 2015), which has known exceptions. Every MBON's source is kept in
#     `mbon_sign_src` ("literature" / "derived" / "none"). MBONs without a sign (too little KC input or no dopamine)
#     are left out of the model: they could not change the valence.
#   * the sparse code (APL): still "the top few percent of KCs fire". The APL -> KC synapse counts are stored in the
#     file but not used.
#   * the readout (mean output of the approach pool minus that of the avoidance pool, relative to the naive fly), the
#     rates, the eligibility trace and the forgetting; the DAN -> KC counts and the MBON -> DAN feedback are not used.
#
# Cost per 50 ms step (5177 KCs): see tools/bench_mushroom.py. The code works on the indices of the ~260 active KCs, not
# on 5177-long vectors; learning (only while dopamine is on) touches the KCs whose eligibility trace is up.

# MBON type -> valence when the neuron is ACTIVE: +1 approach, -1 avoidance. FlyWire / hemibrain type names (Li et al. 2020).
# Only entries a source states directly; everything else is derived from the dopamine coupling. Sources: MBON01 (gamma5
# beta'2a), 03 (beta'2mp), 04 (beta'2mp bilateral), 05 (gamma4>gamma1gamma2), 06 (beta1>alpha), all glutamatergic and
# aversive, and MBON12 (gamma2alpha'1, cholinergic) attractive: Aso et al. 2014 (eLife 04580). MBON21 and MBON29 (gamma4
# gamma5, cholinergic) are aversive despite the transmitter: Rubin & Aso 2024. NOT in the table although often assumed:
# MBON02 and MBON07 (reports conflict), MBON11 (conflicting), MBON09 / 13-19 (only inferred from "GABA or ACh = attraction").
MBON_VALENCE_LITERATURE: dict[str, int] = {"MBON01": -1, "MBON03": -1, "MBON04": -1, "MBON05": -1, "MBON06": -1,
                                           "MBON12": +1, "MBON21": -1, "MBON29": -1}

# cue -> PN types (FlyWire cell_type) whose neurons it fires. ASSUMPTION (see above): olfactory ones by what the
# glomerulus is known for (DM1/DM2/DM3/DM4: fruit and vinegar odours; DA2: geosmin; V: CO2; DL5: aversive odours), the
# visual and touch ones arbitrary stand-ins. The visual PNs that reach the MB end on only ~380 KCs, nearly all gamma-d
# (Li et al. 2020 found the same): two visual cues can only be kept apart by sending them to the two visual KC groups
# (gamma-d for the cursor, alpha/beta-posterior for looming), which leaves looming ~70 KCs; any other split shares KCs.
CUE_PN_TYPES: dict[str, tuple[str, ...]] = {
    "food_odor": ("DM1_lPN", "DM2_lPN", "DM3_adPN", "DM4_adPN"),
    "alarm_odor": ("DA2_lPN", "DL5_adPN", "V_ilPN"),
    "cursor_near": ("aMe12", "MTe32", "MTe30", "LTe25", "MTe40"),           # gamma-d Kenyon cells
    "looming": ("aMe26", "LTe72", "MTe37"),                                  # alpha/beta-posterior Kenyon cells
    "touch": ("VM5d_adPN", "VM5v_adPN", "VM3_adPN"),
}

DAN_SYN_FULL = 50.0       # DAN -> MBON synapses at which a compartment counts as fully coupled to its dopamine
SIGN_MIN_DAN_SYN = 30     # derived sign needs at least this much dopamine contact ...
SIGN_MIN_KC_SYN = 100     # ... and this much KC input (the rest are MBONs outside the lobes, or barely sampled)
SIGN_DOMINANCE = 2.0      # ... and one DAN class contributing at least this many times the other
LEARN_TRACE_MIN = 1e-3    # KCs with an eligibility trace below this do not learn
MEMORY_VERSION = 1


def load_wiring(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class FlywireMushroomBody(MushroomBody):
    kind = "flywire"

    def __init__(self, wiring: dict[str, np.ndarray], params: MBParams | None = None, n_cues: int = len(CUES),
                 cue_pn_types: dict[str, tuple[str, ...]] | None = None,
                 valence_table: dict[str, int] | None = None):
        p = self.p = params or MBParams()
        w = wiring
        self.n_cues = n_cues
        self.n_kc, n_mbon = len(w["kc_id"]), len(w["mbon_id"])
        # --- dopamine coupling per MBON from the DAN -> MBON synapse counts
        dtype = w["dan_type"][w["dan_mbon_dan"]]
        dn = w["dan_mbon_n"].astype(np.float64)
        pam = np.bincount(w["dan_mbon_mbon"], weights=dn * np.char.startswith(dtype, "PAM"), minlength=n_mbon)
        ppl1 = np.bincount(w["dan_mbon_mbon"], weights=dn * np.char.startswith(dtype, "PPL1"), minlength=n_mbon)
        tot = pam + ppl1
        full = np.minimum(1.0, tot / DAN_SYN_FULL)
        share = np.divide(pam, tot, out=np.zeros(n_mbon), where=tot > 0)
        c_reward = full * share                                  # PAM coupling of each MBON, 0..1
        c_punish = full * (1.0 - share) * (tot > 0)              # PPL1 coupling
        # --- MBON sign: literature where known, else derived from the coupling
        table = MBON_VALENCE_LITERATURE if valence_table is None else valence_table
        kc_syn = np.bincount(w["kc_mbon_mbon"], weights=w["kc_mbon_n"].astype(np.float64), minlength=n_mbon)
        sign = np.zeros(n_mbon, np.int8)
        src = np.full(n_mbon, "none", dtype="U10")
        for i, t in enumerate(w["mbon_type"]):
            if t in table and kc_syn[i] > 0:
                sign[i], src[i] = table[t], "literature"
            elif tot[i] >= SIGN_MIN_DAN_SYN and kc_syn[i] >= SIGN_MIN_KC_SYN:
                if pam[i] >= SIGN_DOMINANCE * ppl1[i]:
                    sign[i], src[i] = -1, "derived"              # reward DAN here: avoidance MBON
                elif ppl1[i] >= SIGN_DOMINANCE * pam[i]:
                    sign[i], src[i] = +1, "derived"              # punishment DAN here: approach MBON
        self.mbon_type, self.mbon_sign, self.mbon_sign_src = w["mbon_type"], sign, src
        # the model keeps the MBONs that vote, approach pool first: columns 0..n_app are approach, the rest avoidance
        self.vote = np.concatenate([np.flatnonzero(sign > 0), np.flatnonzero(sign < 0)])
        self.n_app = int((sign > 0).sum())
        col = np.full(n_mbon, -1, np.int64)
        col[self.vote] = np.arange(len(self.vote))
        self.c_reward = c_reward[self.vote].astype(np.float32)
        self.c_punish = c_punish[self.vote].astype(np.float32)
        # --- KC -> MBON: naive output weights (synapse counts) and the depression on top of them
        self.W = np.zeros((self.n_kc, len(self.vote)), np.float32)
        c = col[w["kc_mbon_mbon"]]
        self.W[w["kc_mbon_kc"][c >= 0], c[c >= 0]] = w["kc_mbon_n"][c >= 0]
        self.dev = np.zeros_like(self.W)             # W * depressed fraction, in units of 1/self.g (lazy forgetting)
        self.g = 1.0                                 # forgetting factor shared by all synapses: true = g * stored
        # --- PN -> KC, folded into one matrix (PN group x KC): a group is a set of real PNs sharing a level
        types = cue_pn_types or CUE_PN_TYPES
        rng = np.random.default_rng(p.seed)
        self.n_groups = n_cues * p.levels
        group_of = np.full(len(w["pn_id"]), -1, np.int64)
        self.cue_pools = {}                          # cue -> number of real PNs firing for it
        for ci, name in enumerate(list(types)[:n_cues]):
            pool = np.flatnonzero(np.isin(w["pn_type"], types[name]))
            self.cue_pools[name] = len(pool)
            for j, pn in enumerate(rng.permutation(pool)):
                group_of[pn] = ci * p.levels + j % p.levels
        e_n = w["pn_kc_n"].astype(np.float32)
        g = group_of[w["pn_kc_pn"]]
        ok = g >= 0
        self.GT = np.zeros((self.n_groups, self.n_kc), np.float32)
        np.add.at(self.GT, (g[ok], w["pn_kc_kc"][ok]), e_n[ok] / float(e_n.mean()))
        self.centers = np.tile(np.linspace(0.0, 1.0, p.levels), n_cues).astype(np.float32)
        self.k_active = max(1, int(round(self.n_kc * p.sparsity)))
        self.trace = np.zeros(self.n_kc, np.float32)
        self.active = np.zeros(0, np.int64)          # indices of the KCs that fired in the last step
        self.valence = 0.0
        self.mbon_app = self.mbon_av = 1.0           # relative output of the approach / avoidance pool (1 = naive)
        self.pam = self.ppl1 = 0.0
        self.dirty = False
        self.sig = hashlib.sha1(b"".join(w[k].tobytes() for k in ("kc_id", "mbon_id", "kc_mbon_kc", "kc_mbon_mbon",
                                                                    "kc_mbon_n")) + self.vote.tobytes()).hexdigest()

    @property
    def kc(self) -> np.ndarray:
        out = np.zeros(self.n_kc, np.float32)
        out[self.active] = 1.0
        return out

    # ------------------------------------------------------------------ code
    def _fire(self, cues: np.ndarray) -> np.ndarray:
        """Cue levels (0..1) -> indices of the firing KCs: PN groups with Gaussian level tuning drive the real KCs
        through their PN claws; the k most driven KCs fire (APL)."""
        cues = np.asarray(cues, np.float32)
        c = np.repeat(np.clip(cues, 0.0, 1.0), self.p.levels)
        act = np.exp(-((c - self.centers) ** 2) / (2 * 0.2 ** 2)) * np.repeat(cues > 0.02, self.p.levels)
        on = np.flatnonzero(act > 1e-3)
        if on.size == 0:
            return np.zeros(0, np.int64)
        drive = act[on].astype(np.float32) @ self.GT[on]
        cand = np.flatnonzero(drive > 0.0)
        if cand.size > self.k_active:
            cand = cand[np.argpartition(drive[cand], -self.k_active)[-self.k_active:]]
        return cand

    def encode(self, cues: np.ndarray) -> np.ndarray:
        """Binary KC activity for a cue pattern (dense, for tests and display)."""
        out = np.zeros(self.n_kc, np.float32)
        out[self._fire(cues)] = 1.0
        return out

    # --------------------------------------------------------------- readout
    def _readout(self, act: np.ndarray) -> float:
        """Approach pool minus avoidance pool, each as its output relative to the naive fly for this KC pattern."""
        if act.size == 0 or len(self.vote) == 0:
            self.mbon_app = self.mbon_av = 1.0
            return 0.0
        naive = self.W[act].sum(axis=0)
        cur = naive - self.g * self.dev[act].sum(axis=0)
        a = self.n_app
        n_app, n_av = float(naive[:a].sum()), float(naive[a:].sum())
        self.mbon_app = float(cur[:a].sum()) / n_app if n_app > 0.0 else 1.0
        self.mbon_av = float(cur[a:].sum()) / n_av if n_av > 0.0 else 1.0
        return self.mbon_app - self.mbon_av

    @property
    def valence_vector(self) -> np.ndarray:
        """Net learned valence per Kenyon cell: avoidance depression minus approach depression."""
        a = self.n_app
        w_naive_app = self.W[:, :a].sum(axis=1)
        w_naive_av = self.W[:, a:].sum(axis=1)
        dep_app = (self.g * self.dev[:, :a].sum(axis=1)) / np.maximum(1.0, w_naive_app)
        dep_av = (self.g * self.dev[:, a:].sum(axis=1)) / np.maximum(1.0, w_naive_av)
        return (dep_av - dep_app).astype(np.float32)

    # ------------------------------------------------------------------ step
    def step(self, cues: np.ndarray, reward: float, punishment: float, dt: float) -> float:
        p = self.p
        self.active = act = self._fire(cues)
        a = 1.0 - math.exp(-dt / p.tau_trace_s)
        self.trace *= 1.0 - a
        self.trace[act] += a
        # dopamine = reinforcement - prediction (the valence this cue already has); a missing predicted reinforcement
        # is an omission (extinction)
        v0 = self._readout(act) if p.pred_gain > 0.0 else 0.0
        pe_r = reward - p.pred_gain * max(v0, 0.0)
        pe_p = punishment - p.pred_gain * max(-v0, 0.0)
        dop_r, dop_p, ext_r, ext_p = max(pe_r, 0.0), max(pe_p, 0.0), max(-pe_r, 0.0), max(-pe_p, 0.0)
        self.pam, self.ppl1 = dop_r, dop_p
        changed = False
        if dop_r > 0.0 or dop_p > 0.0 or ext_r > 0.0 or ext_p > 0.0:
            rows = np.flatnonzero(self.trace > LEARN_TRACE_MIN)
            if rows.size:
                # three-factor rule, Euler step per synapse: eligibility (KC trace) x dopamine (per MBON) x what is left;
                # extinction takes back a part of the depression at the compartments of the missing reinforcement
                dep = np.outer(self.trace[rows], p.eta * dt * (self.c_reward * dop_r + self.c_punish * dop_p))
                rec = np.outer(self.trace[rows], p.eta_ext * dt * (self.c_reward * ext_r + self.c_punish * ext_p))
                d = self.dev[rows]
                d += (self.W[rows] / self.g - d) * dep - d * rec
                self.dev[rows] = d
                changed = bool(dep.any() or (rec.any() and d.any()))
                self.dirty = self.dirty or changed
        # slow forgetting toward the naive state, for every synapse at once
        self.g *= math.exp(-dt / p.tau_forget_s)
        if self.g < 1e-3:
            self.dev *= self.g
            self.g = 1.0
        if changed or p.pred_gain <= 0.0:
            self.valence = self._readout(act)
        else:
            self.valence = v0
        return self.valence

    def read(self, cues: np.ndarray) -> float:
        keep = (self.mbon_app, self.mbon_av)
        v = self._readout(self._fire(cues))
        self.mbon_app, self.mbon_av = keep
        return v

    # ---------------------------------------------------------------- memory
    def memory_file(self, path: str | Path) -> Path:
        path = Path(path)
        return path.with_name(f"{path.stem}_flywire{path.suffix}")

    def reset(self) -> None:
        self.dev[:] = 0.0
        self.g = 1.0
        self.trace[:] = 0.0
        self.active = np.zeros(0, np.int64)
        self.valence = 0.0
        self.dirty = True

    def save(self, path: str | Path) -> None:
        """Only the KC rows that learned something are written (a few hundred rows instead of 5177)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = np.flatnonzero(self.dev.any(axis=1))
        np.savez(path, kind=self.kind, version=MEMORY_VERSION, sig=self.sig, n_kc=self.n_kc, n_vote=len(self.vote),
                 rows=rows.astype(np.int32), dev=(self.g * self.dev[rows]).astype(np.float32))
        self.dirty = False

    def load(self, path: str | Path) -> bool:
        """Load a saved memory; False (and nothing changes) if it is missing, of the random model, or made for other cells."""
        try:
            z = np.load(path)
            if (str(z["kind"]) != self.kind or int(z["version"]) > MEMORY_VERSION or str(z["sig"]) != self.sig
                    or int(z["n_kc"]) != self.n_kc or int(z["n_vote"]) != len(self.vote)):
                return False
            rows, dev = z["rows"], z["dev"]
            self.dev[:] = 0.0
            self.dev[rows] = dev
            self.g = 1.0
            return True
        except (OSError, KeyError, ValueError):
            return False
