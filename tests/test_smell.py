"""Smell: the engines' spike-frequency adaptation, the geometric Poisson drive, the olfactory pathway of the tiers and the
two antennae's ORN input in the Brain."""
import numpy as np
import pandas as pd
import pytest

from neuropest import flywire
from neuropest.brain import FLYWIRE, FLYWIRE_SMELL, Brain, orn_level
from neuropest.engine import LIFEngine, Network, ReferenceEngine
from neuropest.paths import CACHE, cache_version
from neuropest.pheromone import ATTRACT_STEER_HZ


# ------------------------------------------------------------------ the geometric Poisson drive
def test_geometric_drive_gives_poisson_spike_counts_at_the_set_rate():
    n = 400
    e = LIFEngine(Network(np.zeros(n + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32)), dt=0.5, seed=1)
    e.set_drive(np.arange(n), 40.0)
    windows = []
    for _ in range(100):                                    # 100 windows of 100 ms
        e.advance(100.0)
        windows.append(e.pop_counts(np.arange(n)))
    c = np.array(windows, float)
    rate = c.sum() / n / 10.0
    assert abs(rate - 40.0) < 0.6                           # 160,000 events: within ~1.5%
    p = 1.0 - np.exp(-40.0 * 0.5 / 1000.0)                  # Bernoulli per 0.5 ms step: var/mean = 1 - p
    assert abs(c.var() / c.mean() - (1.0 - p)) < 0.05


def test_geometric_drive_matches_the_reference_engines_bernoulli_drive_through_a_network():
    net = Network.random(300, mean_deg=15, seed=5, gain=1.3)
    rates = {}
    for cls in (LIFEngine, ReferenceEngine):
        tot = 0
        for seed in range(3):
            e = cls(net, dt=0.5, seed=seed) if cls is ReferenceEngine else cls(net, dt=0.5, seed=seed, eps=0.0)
            e.set_drive(np.arange(0, 300, 10), 30.0)
            e.set_current_drive(np.arange(5, 300, 10), 200.0, 4.0)
            e.advance(2000.0)
            tot += e.counts.sum()
        rates[cls.__name__] = tot
    assert rates["ReferenceEngine"] > 3000
    assert abs(rates["LIFEngine"] - rates["ReferenceEngine"]) < 0.05 * rates["ReferenceEngine"]


def test_a_new_drive_and_a_released_lesion_restart_the_draws():
    e = LIFEngine(Network.from_coo([0], [1], [1.0], 3), dt=0.5)
    e.set_drive([0, 2], [50.0, 1e9])                        # 1e9 Hz: p == 1, an event every step
    e.advance(200.0)
    assert e.counts[2] == 400 and 0 < e.counts[0] < 40
    e.set_silenced([2])
    e.advance(10.0)
    assert e.counts[2] == 400
    e.set_silenced([])
    e.advance(10.0)
    assert e.counts[2] == 420


# ------------------------------------------------------------------ spike-frequency adaptation
def _adapting(net, frac=0.3, mv=4.0, seed=0):
    rng = np.random.default_rng(seed)
    net.extra["adapt_mv"] = np.where(rng.random(net.n) < frac, np.float32(mv), np.float32(0.0)).astype(np.float32)
    return net


def test_adaptation_lowers_the_sustained_rate_of_a_driven_neuron_and_decays():
    def run(mv):
        net = Network.from_coo([0], [1], [20.0], 2, extra={"adapt_mv": np.array([0.0, mv], np.float32)})
        e = LIFEngine(net, dt=0.5, eps=0.0)
        e.set_drive([0], 400.0)
        e.advance(100.0)
        early = e.pop_counts([1])[0]
        e.advance(900.0)
        late = e.pop_counts([1])[0] / 9.0
        return early, late, e

    early0, late0, _ = run(0.0)
    early4, late4, e = run(4.0)
    assert early0 > 0 and abs(late0 - early0) <= 0.3 * early0       # no adaptation: steady
    assert late4 < 0.6 * early4 and late4 < 0.6 * late0             # adaptation: a transient, then lower
    e.set_drive([0], 0.0)
    e.advance(3000.0)                                                 # 10 tau: the raise is gone
    raise_now = e.th[1] * np.exp(-(e.steps - e.th_s[1]) * e.dt / e.p.tau_adapt)
    assert raise_now < 0.01


def test_adaptation_matches_the_reference_engine():
    net = _adapting(Network.random(300, mean_deg=15, seed=3, gain=1.4))
    fast, ref = LIFEngine(net, dt=0.5, eps=0.0), ReferenceEngine(net, dt=0.5)
    drive = np.arange(0, 300, 12)
    for e in (fast, ref):
        for t in range(300):
            e.set_drive(drive, 1e9 if t % 12 == 0 else 0.0)
            e.advance(0.5)
    assert ref.total_spikes > 300
    assert abs(fast.total_spikes - ref.total_spikes) <= 0.02 * ref.total_spikes
    assert np.corrcoef(fast.counts, ref.counts)[0, 1] > 0.98
    plain = LIFEngine(Network(net.indptr, net.indices, net.data), dt=0.5, eps=0.0)
    for t in range(300):
        plain.set_drive(drive, 1e9 if t % 12 == 0 else 0.0)
        plain.advance(0.5)
    assert plain.total_spikes > fast.total_spikes                     # it does adapt


def test_the_gpu_engine_adapts_like_the_reference():
    pytest.importorskip("wgpu")
    from neuropest.engine.lif_wgpu import WGPUEngine, gpu_available

    if not gpu_available():
        pytest.skip("no GPU adapter")
    net = _adapting(Network.random(300, mean_deg=15, seed=3, gain=1.4))
    gpu, ref = WGPUEngine(net, dt=0.5), ReferenceEngine(net, dt=0.5)
    drive = np.arange(0, 300, 12)
    for e in (gpu, ref):
        for t in range(300):
            e.set_drive(drive, 1e9 if t % 12 == 0 else 0.0)
            e.advance(0.5)
    g = np.asarray(gpu.pop_counts(np.arange(300)))
    assert abs(g.sum() - ref.counts.sum()) <= 0.05 * ref.counts.sum()
    assert np.corrcoef(g, ref.counts)[0, 1] > 0.97
    with pytest.raises(ValueError):
        WGPUEngine(Network.from_coo([0], [1], [1.0], 2, extra={"adapt_mv": np.array([2.0, 4.0], np.float32)}))


# ------------------------------------------------------------------ the olfactory pathway of the tiers
def _olf_world():
    """ORN L/R (0-3) -> PN (4-7) -> X (8, 9) -> DNa02 L/R (10, 11); Y (12, left only typed twin 13 on the right) above
    DNa02_R; filler 14..39 with weak links; 'base' is the activity order."""
    n = 40
    ct = [""] * n
    side = ["left", "right"] * (n // 2)
    for i, t in enumerate(["ORN_DM1", "ORN_DM1", "ORN_DM2", "ORN_DM2", "PNa", "PNa", "PNb", "PNb", "Xx", "Xx",
                           "DNa02", "DNa02", "Yy", "Yy"]):
        ct[i] = t
    side[0:14] = ["left", "right"] * 7
    edges = [(0, 4, 30), (1, 5, 30), (2, 6, 30), (3, 7, 30), (4, 8, 30), (5, 9, 30), (6, 8, 30), (7, 9, 30),
             (8, 10, 30), (9, 11, 30), (13, 11, 15)]
    edges += [(i, i + 1, 2) for i in range(14, 39)]
    pre, post, w = zip(*edges)
    net = Network.from_coo(pre, post, np.array(w, np.float32) * 0.275, n, meta={"w_syn": 0.275})
    meta = pd.DataFrame({"cell_type": ct, "side": side, "cell_class": [""] * n})
    return net, meta


def test_the_olfactory_pathway_holds_the_orn_to_dn_paths_and_both_sides_twins():
    net, meta = _olf_world()
    base = np.array([10, 11, 13] + list(range(14, 30)))                 # DNa02 and the right Y are in the tier already
    got = set(flywire.olfactory_pathway(net, meta, base).tolist())
    assert set(range(0, 10)) <= got                                    # ORNs, PNs, X on the paths
    assert 12 in got                                                   # the left twin of Y, which is above DNa02_R
    assert not got & set(base.tolist()) and not got & set(range(30, 40))


def test_fit_olfactory_leaves_smaller_tiers_alone_and_ends_the_fitted_tier_with_the_pathway():
    net, meta = _olf_world()
    order = np.array([10, 11, 13] + list(range(14, 40)) + list(range(0, 10)) + [12], np.int32)
    out = flywire.fit_olfactory(order, net, meta, tier=25)
    assert sorted(out.tolist()) == list(range(40))
    olf = set(range(0, 10)) | {12}
    assert olf <= set(out[:25].tolist())
    k = 25 - len(olf)
    assert out[:k].tolist() == order[:k].tolist()                      # every tier up to there is unchanged
    assert flywire.fit_olfactory(order, net, meta, tier=100) is order


def test_adapt_mv_marks_the_antennal_lobe_neurons():
    meta = pd.DataFrame({"cell_class": ["ALPN", "ALLN", "Kenyon_Cell", np.nan]})
    assert flywire.adapt_mv(meta).tolist() == [flywire.AL_ADAPT_MV, flywire.AL_ADAPT_MV, 0.0, 0.0]


# ------------------------------------------------------------------ the Brain's ORN input
def _smell_net():
    """ORNs per side (0-9 left, 10-19 right) -> DNa02 of the same side (20, 21); LC10 left / right (22-25); a walking
    DN (26)."""
    pre, post, w = [], [], []
    for i in range(10):
        pre += [i, 10 + i]
        post += [20, 21]
        w += [3.0, 3.0]
    net = Network.from_coo(pre, post, w, 27, meta={"kind": "flywire"})
    net.groups = {"ODOR_L": np.arange(10, dtype=np.int32), "ODOR_R": np.arange(10, 20, dtype=np.int32),
                  "DNa02_L": np.array([20], np.int32), "DNa02_R": np.array([21], np.int32),
                  "LC10_L": np.array([22, 23], np.int32), "LC10_R": np.array([24, 25], np.int32),
                  "WALK": np.array([26], np.int32)}
    return net


def test_orn_level_saturates():
    assert orn_level(0.0, 0.2) == 0.0 and orn_level(-1.0, 0.2) == 0.0
    assert orn_level(0.2, 0.2) == 0.5 and 0.8 < orn_level(1.0, 0.2) < 1.0


def test_the_odor_drives_each_antennas_orns_and_only_the_repellent_steers_through_lc10():
    assert not Brain(_smell_net(), learning=False).smell              # the default steers through LC10
    b = Brain(_smell_net(), learning=False, spec=FLYWIRE_SMELL)
    assert b.smell
    delta = 0.3
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, phero_drive=0.3, phero_steer=ATTRACT_STEER_HZ * delta, hunger=1.0,
                   odor_delta=delta)
    left, right = b.orn_hz
    assert 0.0 < left < right
    assert np.isclose(right, FLYWIRE_SMELL.orn_max_hz * orn_level(0.3 + delta / 2, FLYWIRE_SMELL.orn_half))
    fi, fp = b.engine.f_idx, b.engine.f_p
    assert set(fi.tolist()) == set(range(20))                          # the ORNs, not LC10
    assert fp[fi >= 10].mean() > fp[fi < 10].mean()
    for _ in range(250):
        b.advance(4.0)
    assert b.rates["DNa02_R"] > b.rates["DNa02_L"]                     # the circuit turns it toward the odor
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, phero_drive=0.3, hunger=0.0, odor_delta=delta)
    assert b.orn_hz == (0.0, 0.0)                                      # a sated fly: no gain
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, phero_steer=-30.0, phero_repel=0.3, hunger=1.0)
    assert set(b.engine.f_idx.tolist()) & {22, 23}                     # a repellent on the right turns it left via LC10


def test_without_an_olfactory_pathway_the_odor_steers_through_lc10_as_before():
    net = _smell_net()
    net.groups.pop("ODOR_R")
    b = Brain(net, learning=False, spec=FLYWIRE_SMELL)
    assert not b.smell
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, phero_drive=0.3, phero_steer=ATTRACT_STEER_HZ * 0.04, hunger=1.0,
                   odor_delta=0.04)
    assert b.orn_hz == (0.0, 0.0) and set(b.engine.f_idx.tolist()) == {24, 25}


# ------------------------------------------------------------------ the real 15,000 tier
needs_cache = pytest.mark.skipif(not CACHE.exists() or cache_version() < 3,
                                 reason="needs the FlyWire cache, version 3 (tools/build_flywire.py)")


@needs_cache
def test_the_15k_tier_smells_without_a_runaway_and_turns_toward_the_odor_side():
    net = flywire.symmetrize(flywire.load_cache().prefix(15_000))
    g = net.groups
    assert len(g["ODOR_L"]) > 50 and len(g["ODOR_R"]) > 50
    assert (net.extra["adapt_mv"] > 0).sum() > 300
    every = np.arange(net.n)
    right_more = 0
    for seed in range(6):
        steer = []
        for side in ("ODOR_L", "ODOR_R"):
            e = LIFEngine(net, dt=0.5, seed=1_000_003 * (seed + 1))
            e.set_current_drive(g["WALK"], 650 * 0.65, 3.0)
            e.set_drive(g[side], 60.0)
            e.advance(300.0)
            e.pop_counts(every)
            e.advance(1000.0)
            c = e.pop_counts(every)
            steer.append(c[g["DNa02_R"]].mean() - c[g["DNa02_L"]].mean())
        right_more += steer[1] > steer[0]
    assert right_more >= 4
    e.set_drive(np.zeros(0, np.int32), 0.0)                            # the odor ends: the circuit falls silent
    e.advance(500.0)
    e.pop_counts(every)
    e.advance(1000.0)
    assert (e.pop_counts(every) > 0).sum() < 50
