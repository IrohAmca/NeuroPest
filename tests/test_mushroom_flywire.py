"""The mushroom body on real FlyWire cells: the extraction (tools/build_mushroom.py) on a tiny FlyWire-shaped dataset
written to a temp dir, the model built from it, its memory file, and the Brain's opt-in switch."""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_mushroom  # noqa: E402

from neuropest import mushroom  # noqa: E402
from neuropest.brain import Brain  # noqa: E402
from neuropest.mushroom import FlywireMushroomBody, MBParams, MushroomBody  # noqa: E402
from neuropest.paths import MUSHROOM  # noqa: E402
from neuropest.toy_circuit import build  # noqa: E402

N, ROOT0 = 140, 720575940000000000
KC0, N_KC = 0, 120
MBON = {"MBON01": (120, 121), "MBON11": (122, 123), "MBON10": (124, 125)}   # PAM-coupled / PPL1-coupled / sampled by nobody
PAM, PPL1 = (126, 127), (128, 129)
APL = (130, 131)
PNS = {"DM1_lPN": range(132, 134), "DM2_lPN": range(134, 136), "DA2_lPN": range(136, 138), "aMe12": range(138, 140)}
CUE_TYPES = {"food_odor": ("DM1_lPN", "DM2_lPN"), "alarm_odor": ("DA2_lPN",), "cursor_near": ("aMe12",),
             "looming": (), "touch": ()}
FOOD = np.array([0.9, 0, 0, 0, 0])
ALARM = np.array([0, 0.9, 0, 0, 0])
QUIET = np.zeros(5)


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    ids = ROOT0 + np.arange(N) * 7
    cc, ct, hb, sc = ([""] * N for _ in range(4))
    side = ["left", "right"] * (N // 2)
    for i in range(N_KC):
        cc[i], ct[i] = "Kenyon_Cell", "KCg-m" if i < 90 else "KCab"
    for t, (a, b) in MBON.items():
        for i in (a, b):
            cc[i], ct[i] = "MBON", t
    for lo, t in ((PAM[0], "PAM05"), (PPL1[0], "PPL101")):
        for i in (lo, lo + 1):
            cc[i], ct[i], hb[i] = "DAN", t, t
    for i in APL:
        cc[i], ct[i] = "MBIN", "APL"
    for t, rows in PNS.items():
        for i in rows:
            ct[i] = t
            cc[i], sc[i] = ("ALPN", "central") if t != "aMe12" else ("", "visual_projection")
    pd.DataFrame({"Completed": True}, index=pd.Index(ids)).to_csv(d / "Completeness_783.csv")
    pd.DataFrame({"root_id": ids, "super_class": sc, "cell_class": cc, "cell_type": ct, "hemibrain_type": hb,
                  "side": side, "top_nt": "acetylcholine"}).to_csv(d / "Supplemental_file1_neuron_annotations.tsv",
                                                                   sep="\t", index=False)
    pre, post, cnt = [], [], []

    def edge(a, b, c):
        pre.append(a), post.append(b), cnt.append(c)

    pn_all = [i for rows in PNS.values() for i in rows]
    for k in range(N_KC):
        for pn in rng.choice(pn_all, 3, replace=False):                  # three PN claws per KC
            edge(pn, k, int(rng.integers(3, 9)))
        edge(rng.choice(pn_all), k, 1)                                  # a one-synapse contact: below min_syn
        for t in ("MBON01", "MBON11"):                                  # KC -> MBON01 / MBON11, both hemispheres
            for m in MBON[t]:
                edge(k, m, int(rng.integers(3, 9)))
        edge(k, MBON["MBON10"][0], 1)                                    # MBON10 is only touched by single synapses
        edge(APL[0], k, 5)
        edge(PAM[0], k, 2)
        edge(PPL1[0], k, 1)
    for d_ in PAM:
        for m in MBON["MBON01"]:
            edge(d_, m, 40)
    for d_ in PPL1:
        for m in MBON["MBON11"]:
            edge(d_, m, 40)
    pre, post, cnt = (np.asarray(a) for a in (pre, post, cnt))
    o = np.argsort(pre, kind="stable")
    pre, post, cnt = pre[o], post[o], cnt[o]
    pq.write_table(pa.table({
        "Presynaptic_ID": ids[pre], "Postsynaptic_ID": ids[post], "Presynaptic_Index": pre.astype(np.int64),
        "Postsynaptic_Index": post.astype(np.int64), "Connectivity": cnt.astype(np.int64),
        "Excitatory": np.ones(len(pre), np.int64), "Excitatory x Connectivity": cnt.astype(np.int64)}),
        d / "Connectivity_783.parquet", row_group_size=500)
    return d


@pytest.fixture(scope="module")
def wiring(raw):
    return build_mushroom.extract(raw)


def _mb(wiring, **kw):
    return FlywireMushroomBody(wiring, cue_pn_types=CUE_TYPES, **kw)


def _pair(mb, cue, reward=0.0, punish=0.0, trials=4):
    for _ in range(trials):
        for _ in range(40):
            mb.step(cue, reward, punish, 0.05)
        for _ in range(100):
            mb.step(QUIET, 0, 0, 0.05)


# ------------------------------------------------------------------------------------------------ extraction
def test_extract_finds_the_cells_and_drops_single_synapse_edges(wiring):
    w = wiring
    assert len(w["kc_id"]) == N_KC and len(w["mbon_id"]) == 6 and len(w["dan_id"]) == 4 and len(w["apl_id"]) == 2
    assert set(w["pn_type"]) == set(PNS) and set(w["pn_class"]) == {"olfactory", "visual"}
    assert w["kc_mbon_n"].min() >= 3 and w["pn_kc_n"].min() >= 3                 # min_syn = 3
    ok10 = w["mbon_type"][w["kc_mbon_mbon"]] == "MBON10"
    assert not ok10.any()                                                         # only single-synapse contacts
    assert len(w["kc_mbon_n"]) == N_KC * 4 and len(w["pn_kc_n"]) <= N_KC * 3
    dtype = w["dan_type"][w["dan_mbon_dan"]]
    assert set(dtype[w["mbon_type"][w["dan_mbon_mbon"]] == "MBON01"]) == {"PAM05"}
    assert set(dtype[w["mbon_type"][w["dan_mbon_mbon"]] == "MBON11"]) == {"PPL101"}


def test_extract_keeps_per_kc_totals_of_dan_and_apl_contacts(wiring):
    w = wiring
    assert w["dan_kc"].shape == (N_KC, 2) and (w["dan_kc"][:, 0] == 2).all() and (w["dan_kc"][:, 1] == 1).all()
    assert (w["apl_kc"] == 5).all()


def test_extract_roundtrips_through_the_npz(wiring, tmp_path):
    f = tmp_path / "m.npz"
    np.savez_compressed(f, **wiring)
    z = mushroom.load_wiring(f)
    assert set(z) == set(wiring) and all(np.array_equal(z[k], wiring[k]) for k in z)


# ----------------------------------------------------------------------------------------------- the model
def test_signs_come_from_the_dopamine_coupling_unless_the_literature_says_otherwise(wiring):
    mb = _mb(wiring, valence_table={})
    by = {t: (int(s), src) for t, s, src in zip(mb.mbon_type, mb.mbon_sign, mb.mbon_sign_src)}
    assert by["MBON01"] == (-1, "derived") and by["MBON11"] == (1, "derived") and by["MBON10"] == (0, "none")
    assert len(mb.vote) == 4 and mb.n_app == 2                                   # MBON10 has no vote and is dropped
    flipped = _mb(wiring, valence_table={"MBON01": +1})
    sign = dict(zip(flipped.mbon_type, zip(flipped.mbon_sign, flipped.mbon_sign_src)))
    assert sign["MBON01"] == (1, "literature")


def test_each_cue_fires_its_own_real_pns(wiring):
    mb = _mb(wiring)
    assert mb.cue_pools == {"food_odor": 4, "alarm_odor": 2, "cursor_near": 2, "looming": 0, "touch": 0}
    assert mb.encode(FOOD).sum() > 0 and mb.encode(QUIET).sum() == 0
    assert mb.encode(np.array([0, 0, 0, 0.9, 0])).sum() == 0                     # a cue without PNs fires no KC
    assert mb.encode(FOOD).sum() <= mb.k_active


def test_naive_fly_has_no_valence(wiring):
    mb = _mb(wiring)
    assert mb.read(FOOD) == 0.0 and mb.read(ALARM) == 0.0 and mb.step(FOOD, 0, 0, 0.05) == 0.0


def test_reward_makes_the_paired_cue_desired_and_the_other_cue_much_less(wiring):
    mb = _mb(wiring)
    _pair(mb, FOOD, reward=1.0)
    assert mb.read(FOOD) > 0.5 and mb.read(ALARM) < 0.3 * mb.read(FOOD)


def test_punishment_makes_the_paired_cue_feared(wiring):
    mb = _mb(wiring)
    _pair(mb, ALARM, punish=1.0)
    assert mb.read(ALARM) < -0.5 and abs(mb.read(FOOD)) < 0.3


def test_no_dopamine_no_learning(wiring):
    mb = _mb(wiring)
    _pair(mb, FOOD)
    assert mb.read(FOOD) == 0.0 and not mb.dirty


def test_memory_fades_and_reset_clears_it(wiring):
    mb = _mb(wiring, params=MBParams(tau_forget_s=60.0))
    _pair(mb, FOOD, reward=1.0)
    v0 = mb.read(FOOD)
    for _ in range(int(120 / 0.5)):
        mb.step(QUIET, 0, 0, 0.5)
    assert 0 < mb.read(FOOD) < v0 * 0.25
    mb.reset()
    assert mb.read(FOOD) == 0.0


def test_forgetting_survives_a_rescale_of_the_shared_factor(wiring):
    a, b = _mb(wiring, params=MBParams(tau_forget_s=2.0)), _mb(wiring, params=MBParams(tau_forget_s=2.0))
    _pair(a, FOOD, reward=1.0, trials=1)
    b.dev[:], b.g = a.dev.copy(), a.g
    g0 = a.g
    for _ in range(400):                                   # 20 s at tau = 2 s: g falls below 1e-3 and is folded back once
        a.step(QUIET, 0, 0, 0.05)
    assert a.g > g0 * np.exp(-10.0) * 100                  # (it was reset to 1 on the way)
    b.g *= np.exp(-20.0 / 2.0)                             # the same decay done in one go
    assert abs(a.read(FOOD) - b.read(FOOD)) < 1e-4


def test_a_predicted_reward_that_does_not_come_extinguishes_the_memory(wiring):
    mb = _mb(wiring)
    _pair(mb, FOOD, reward=1.0)
    v0 = mb.read(FOOD)
    for _ in range(int(60 / 0.05)):
        mb.step(FOOD, 0, 0, 0.05)
    assert mb.read(FOOD) < 0.5 * v0
    off = _mb(wiring, params=MBParams(pred_gain=0.0))
    _pair(off, FOOD, reward=1.0)
    for _ in range(int(60 / 0.05)):
        off.step(FOOD, 0, 0, 0.05)
    assert off.read(FOOD) > 0.95 * v0


def test_naive_fly_stays_clean_with_the_prediction_error(wiring):
    mb = _mb(wiring)
    for _ in range(200):
        mb.step(FOOD, 0, 0, 0.05)
    assert mb.read(FOOD) == 0.0 and not mb.dirty and not mb.dev.any()


# ---------------------------------------------------------------------------------------------- the memory
def test_memory_roundtrip_and_what_it_refuses(wiring, tmp_path):
    mb = _mb(wiring)
    _pair(mb, FOOD, reward=1.0)
    f = tmp_path / "mem.npz"
    mb.save(f)
    assert not mb.dirty
    other = _mb(wiring)
    assert other.load(f) and abs(other.read(FOOD) - mb.read(FOOD)) < 1e-6
    assert not other.load(tmp_path / "missing.npz")
    assert not MushroomBody().load(f)                                             # the random model refuses it ...
    r = tmp_path / "random.npz"
    MushroomBody().save(r)
    assert not other.load(r)                                                       # ... and this one refuses its file
    changed = dict(wiring, kc_mbon_n=wiring["kc_mbon_n"] + 1)
    assert not _mb(changed).load(f)                                                # other synapse counts: other memory


def test_memory_file_is_next_to_the_random_one_not_on_top_of_it(wiring):
    assert _mb(wiring).memory_file("data/memory/fly_memory.npz") == Path("data/memory/fly_memory_flywire.npz")
    assert MushroomBody().memory_file("data/memory/fly_memory.npz") == Path("data/memory/fly_memory.npz")


def test_old_random_memory_files_without_a_kind_still_load(tmp_path):
    mb = MushroomBody()
    f = tmp_path / "old.npz"
    np.savez(f, w_app=mb.w_app, w_av=mb.w_av, n_kc=mb.p.n_kc, seed=mb.p.seed)    # what v1 of the file looked like
    assert MushroomBody().load(f)


# -------------------------------------------------------------------------------------------- Brain switch
def _toy_brain(**kw):
    net = build(146, 1)
    net.groups["LC10_R"] = np.array([5], np.int32)
    net.groups["LC10_L"] = np.array([6], np.int32)
    return Brain(net, **kw)


def test_brain_uses_the_real_wiring_only_when_asked_and_falls_back_without_the_file(wiring, tmp_path, monkeypatch):
    assert _toy_brain().mb.kind == "random"                                       # the default is unchanged
    monkeypatch.setattr(mushroom, "MUSHROOM", tmp_path / "missing.npz")
    with pytest.warns(UserWarning, match="random model"):
        assert _toy_brain(mb_wiring="flywire").mb.kind == "random"
    f = tmp_path / "mushroom_flywire.npz"
    np.savez_compressed(f, **wiring)
    monkeypatch.setattr(mushroom, "MUSHROOM", f)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = _toy_brain(mb_wiring="flywire")
    assert b.mb.kind == "flywire" and b.mb.n_kc == N_KC
    assert _toy_brain(learning=False, mb_wiring="flywire").mb is None


# ---------------------------------------------------------------------------------------- the real file
needs_real = pytest.mark.skipif(not MUSHROOM.exists(), reason="data/circuits/mushroom_flywire.npz not built here")


@needs_real
def test_real_wiring_has_the_expected_cells_pools_and_learns_selectively():
    mb = MushroomBody.from_flywire()
    assert mb.n_kc == 5177 and len(mb.vote) > 40 and mb.n_app > 10
    assert all(n > 0 for n in mb.cue_pools.values())
    assert (mb.mbon_sign_src == "none").sum() < len(mb.mbon_sign)
    cues = [np.eye(5)[i] * 0.9 for i in range(5)]
    assert all(mb.encode(c).sum() > 0 for c in cues)
    _pair(mb, cues[0], reward=1.0)
    assert mb.read(cues[0]) > 0.5
    assert max(abs(mb.read(c)) for c in cues[1:]) < 0.3 * mb.read(cues[0])


@needs_real
def test_real_wiring_step_cost_stays_far_below_the_50ms_step():
    mb = MushroomBody.from_flywire()
    cues = np.array([0.5, 0.2, 0.7, 0.1, 0.0])
    for _ in range(50):
        mb.step(cues, 0.0, 0.0, 0.05)
    t = time.perf_counter()
    for _ in range(200):
        mb.step(cues, 1.0, 0.0, 0.05)                      # the dearer case: learning
    assert (time.perf_counter() - t) / 200 < 3e-3
