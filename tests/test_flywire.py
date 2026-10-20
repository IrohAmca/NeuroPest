"""Loader, ranking, tiers and the Brain on a tiny FlyWire-shaped dataset written to a temp dir."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from neuropest import flywire
from dataclasses import replace

from neuropest.brain import FEED, FLY, FLYWIRE, FREEZE, GROOM, RETREAT, STAND, WALK, Brain, sugar_gain
from neuropest.engine import LIFEngine

N = 400
ROOT0 = 720575940000000000


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    ids = ROOT0 + np.arange(N) * 7
    types = np.array([""] * N, dtype=object)
    sc = np.array(["central"] * N, dtype=object)
    hb = np.array([""] * N, dtype=object)
    side = np.array(["left", "right"] * (N // 2), dtype=object)
    # 0..19 LPLC2, 20..29 LC4, 30..49 other visual projection, 50,51 giant fiber, 52..55 P9, 56..59 other DNs,
    # 60..69 LPC1, 70..73 MDN, 74/75 DNa02 left/right, 76..85 LC10 left, 86..95 LC10 right
    types[0:20], types[20:30] = "LPLC2", "LC4"
    sc[0:50] = "visual_projection"
    types[50:52], hb[50:52], sc[50:52] = "DNp01", "Giant Fiber", "descending"
    types[52:54], hb[52:54], sc[52:54] = "DNp09", "DNp09", "descending"
    types[54:56], hb[54:56], sc[54:56] = "DNp71", "DNp09", "descending"
    types[56:60], sc[56:60] = "DNx", "descending"
    types[60:70], sc[60:70] = "LPC1", "visual_projection"
    types[70:74], sc[70:74] = "MDN", "descending"
    types[74:76], hb[74:76], sc[74:76] = "DNa02", "DNa02", "descending"
    side[74], side[75] = "left", "right"
    types[76:96], sc[76:96] = "LC10c-2", "visual_projection"
    side[76:86], side[86:96] = "left", "right"
    # 100..109 head bristles (BM_InOm) and 110..114 Johnston's organ C/E, left; 120..129 and 130..134 right;
    # 140,141 taste bristles (must not count as touch); 150..153 grooming DNs (aDN1 = DNg62, aDN2 = DNge078)
    cc = np.array([np.nan] * N, dtype=object)
    for lo, s_ in ((100, "left"), (120, "right")):
        types[lo:lo + 10], types[lo + 10:lo + 15], side[lo:lo + 15] = "BM_InOm", "JO-CE", s_
        cc[lo:lo + 15] = "mechanosensory"
    types[140:142], cc[140:142] = "BM_Taste", "mechanosensory"
    types[150:152], types[152:154], sc[150:154] = "DNg62", "DNge078", "descending"
    # 160..165 sugar receptor neurons (LB3c, "sugar"), 166,167 "sugar/low_salt" (LB3b: not in the group), 170,171 MN9
    sub = np.array([np.nan] * N, dtype=object)
    types[160:166], sub[160:166], cc[160:168] = "LB3c", "sugar", "gustatory"
    types[166:168], sub[166:168] = "LB3b", "sugar/low_salt"
    types[170:172], sc[170:172] = "CB0701", "motor"
    pd.DataFrame({"Completed": True}, index=pd.Index(ids)).to_csv(d / "Completeness_783.csv")
    ann = pd.DataFrame({"root_id": ids, "flow": "intrinsic", "super_class": sc, "cell_class": cc,
                        "cell_sub_class": sub, "cell_type": types, "hemibrain_type": hb, "side": side})
    ann.to_csv(d / "Supplemental_file1_neuron_annotations.tsv", sep="\t", index=False)
    pre, post, cnt = [], [], []
    for i in range(N):                                    # sparse random wiring
        tgt = rng.choice(N, 12, replace=False)
        tgt = tgt[tgt != i]
        pre += [i] * len(tgt)
        post += list(tgt)
        cnt += list(rng.integers(1, 12, len(tgt)))
    strong = []                                           # (pre range, post list, synapse count)
    strong += [(range(0, 30), (50, 51), 15)]              # looming -> GF
    strong += [(range(60, 70), (70, 71, 72, 73), 40)]     # LPC1 -> MDN
    strong += [(range(76, 86), (74,), 80), (range(86, 96), (75,), 80)]   # LC10 -> DNa02 of the same side
    strong += [(range(100, 115), (150, 152), 40), (range(120, 135), (151, 153), 40)]   # touch -> grooming DNs
    strong += [(range(160, 166), (170, 171), 12)]         # sugar -> MN9
    for rows, targets, c in strong:
        for i in rows:
            for j in targets:
                pre.append(i)
                post.append(j)
                cnt.append(c)
    order = np.argsort(pre, kind="stable")
    pre, post, cnt = (np.asarray(a)[order] for a in (pre, post, cnt))
    sign = np.where(rng.random(len(pre)) < 0.7, 1, -1)
    for rows, targets, c in strong:
        sign[np.isin(pre, list(rows)) & np.isin(post, targets) & (cnt == c)] = 1
    pq.write_table(pa.table({
        "Presynaptic_ID": ids[pre], "Postsynaptic_ID": ids[post],
        "Presynaptic_Index": pre.astype(np.int64), "Postsynaptic_Index": post.astype(np.int64),
        "Connectivity": cnt.astype(np.int64), "Excitatory": sign.astype(np.int64),
        "Excitatory x Connectivity": (sign * cnt).astype(np.int64)}), d / "Connectivity_783.parquet",
        row_group_size=500)
    return d


def test_load_connectome_shapes_and_signs(raw):
    indptr, indices, data, ids = flywire.load_connectome(raw)
    assert len(ids) == N and indptr[-1] == len(indices) == len(data)
    assert indices.dtype == np.int32 and data.dtype == np.float32
    assert (data < 0).any() and (data > 0).any()
    assert np.isclose(np.abs(data).min(), 0.275)               # one synapse = w_syn


def test_groups_follow_the_annotation(raw):
    _, _, _, ids = flywire.load_connectome(raw)
    g = flywire.make_groups(flywire.load_annotations(ids, raw))
    assert list(g["GF"]) == [50, 51]
    assert list(g["WALK"]) == [52, 53, 54, 55]                  # DNp09 and DNp71 share hemibrain type DNp09
    assert len(g["LPLC2"]) == 20 and len(g["LC4"]) == 10 and len(g["LOOM"]) == 30
    assert list(g["RETREAT_IN"]) == list(range(60, 70)) and list(g["MDN"]) == [70, 71, 72, 73]
    assert list(g["DNa02_L"]) == [74] and list(g["DNa02_R"]) == [75]
    assert list(g["LC10_L"]) == list(range(76, 86)) and list(g["LC10_R"]) == list(range(86, 96))
    assert len(g["VPN"]) == 80 and len(g["DN"]) == 20
    assert list(g["TOUCH_L"]) == list(range(100, 115)) and list(g["TOUCH_R"]) == list(range(120, 135))
    assert list(g["GROOM"]) == [150, 151, 152, 153]
    assert list(g["SUGAR"]) == list(range(160, 166))            # "sugar/low_salt" is left out
    assert list(g["MN9"]) == [170, 171]


def test_an_annotation_file_without_sub_classes_gives_an_empty_taste_group(raw, tmp_path):
    import shutil

    ann = pd.read_csv(raw / "Supplemental_file1_neuron_annotations.tsv", sep="\t").drop(columns=["cell_sub_class"])
    ann.to_csv(tmp_path / "Supplemental_file1_neuron_annotations.tsv", sep="\t", index=False)
    shutil.copy(raw / "Completeness_783.csv", tmp_path)
    _, _, _, ids = flywire.load_connectome(raw)
    g = flywire.make_groups(flywire.load_annotations(ids, tmp_path))
    assert len(g["SUGAR"]) == 0 and list(g["MN9"]) == [170, 171] and list(g["GF"]) == [50, 51]


def test_the_cache_carries_its_version_and_the_lc10_protocols_stay_under_the_ceiling(raw, tmp_path):
    from neuropest.paths import CACHE_VERSION, cache_version
    from neuropest.states import LC10_MAX_HZ

    net = flywire.build(raw)
    net.save(tmp_path / "c.npz")
    assert cache_version(tmp_path / "c.npz") == CACHE_VERSION
    for proto in flywire.train_protocols() + flywire.test_protocols():
        assert all(hz <= LC10_MAX_HZ for k, hz in proto.items() if k.startswith("LC10"))
    assert any("SUGAR" in p for p in flywire.train_protocols())


def test_build_orders_pinned_first_and_prefix_is_a_valid_network(raw):
    net = flywire.build(raw)
    g = net.groups
    pinned = set()
    for k in flywire.ANCHOR_GROUPS + flywire.INPUT_GROUPS:
        pinned |= set(g[k])
    assert set(net.order[:len(pinned)]) == pinned
    assert sorted(net.order) == list(range(N))
    sub = net.prefix(150)
    assert sub.n == 150 and len(sub.groups["GF"]) == 2 and len(sub.groups["LOOM"]) == 30
    assert len(sub.groups["MDN"]) == 4 and len(sub.groups["LC10_L"]) == 10
    assert net.extra["activity"].shape == (N,) and net.meta["kind"] == "flywire"


def test_dropping_silent_neurons_is_lossless(raw):
    """The model has no spontaneous activity: a tier holding every neuron that spikes is exact."""
    net = flywire.build(raw)
    active = int((net.extra["activity"] > 0).sum())
    sub = net.prefix(active)
    out = []
    for n in (net, sub):
        e = LIFEngine(n, dt=0.5, seed=5, eps=0.0)
        e.add_drive(n.groups["LOOM"], 20.0)
        e.add_drive(n.groups["RETREAT_IN"], 30.0)
        e.advance(400)
        out.append((e.pop_counts(n.groups["GF"]).sum(), e.pop_counts(n.groups["MDN"]).sum()))
    assert out[0][0] > 0 and out[0][1] > 0
    for a, b in zip(*out):
        assert abs(a - b) <= 0.05 * a


def test_cache_roundtrip_keeps_order_extra_and_meta(raw, tmp_path):
    net = flywire.build(raw)
    net.save(tmp_path / "c.npz")
    back = flywire.Network.load(tmp_path / "c.npz")
    assert np.array_equal(back.order, net.order) and back.meta["kind"] == "flywire"
    assert np.array_equal(back.extra["activity"], net.extra["activity"])
    assert back.groups.keys() == net.groups.keys()


@pytest.fixture(scope="module")
def tier(raw):
    return flywire.build(raw).prefix(200)


def _settle(b, ms=600):
    for _ in range(int(ms / 4)):
        b.advance(4.0)


def test_brain_calm_cursor_leaves_the_fly_standing(tier):
    b = Brain(tier)
    b.set_stimulus(900, 0, 0.0, bearing=0.0)
    _settle(b, 1000)
    assert b.state == STAND


def test_brain_fast_approach_escapes_and_moderate_approach_retreats(tier):
    b = Brain(tier)
    b.set_stimulus(150, 3000, 0.0)                      # expansion 20 /s: take-off outranks retreat
    seen = {b.advance(4.0) for _ in range(150)}
    assert FLY in seen
    b = Brain(tier)
    b.set_stimulus(300, 500, 0.0)                       # expansion 1.7 /s: backward walking only
    seen = {b.advance(4.0) for _ in range(250)}
    assert RETREAT in seen and FLY not in seen


def test_brain_gf_spike_event_takes_off_without_the_rate_path(tier):
    never_rate = 1e9                                    # the smoothed-rate path can no longer fire
    b = Brain(tier, spec=replace(FLYWIRE, gf_on_hz=never_rate, gf_event_spikes=2))
    b.set_stimulus(150, 3000, 0.0)
    assert FLY in {b.advance(4.0) for _ in range(150)}
    off = Brain(tier, spec=replace(FLYWIRE, gf_on_hz=never_rate, gf_event_spikes=0))
    off.set_stimulus(150, 3000, 0.0)
    assert FLY not in {off.advance(4.0) for _ in range(150)}


def test_brain_gf_spike_window_covers_whole_chunks_of_at_least_the_event_time(tier):
    b = Brain(tier, spec=replace(FLYWIRE, gf_event_ms=10.0))
    for spikes in (1, 0, 0, 0, 0, 2):                   # 4 ms chunks: the window is 3 chunks (12 ms >= 10 ms)
        b._track_gf_spikes(4.0, spikes)
    assert b.gf_window == 2
    b._track_gf_spikes(4.0, 1)
    b._track_gf_spikes(4.0, 1)
    assert b.gf_window == 4
    for _ in range(3):
        b._track_gf_spikes(4.0, 0)
    assert b.gf_window == 0
    g = Brain(tier, spec=replace(FLYWIRE, gf_event_ms=10.0))
    g._track_gf_spikes(12.0, 3)                         # a GPU chunk longer than the window is its own window
    g._track_gf_spikes(12.0, 1)
    assert g.gf_window == 1


def test_brain_steering_follows_the_side_of_the_cursor(tier):
    right, left = Brain(tier), Brain(tier)
    right.set_stimulus(150, 0, 0.0, bearing=1.5)
    left.set_stimulus(150, 0, 0.0, bearing=-1.5)
    _settle(right, 800)
    _settle(left, 800)
    assert right.steer > 5 and left.steer < -5
    ahead = Brain(tier)
    ahead.set_stimulus(150, 0, 0.0, bearing=0.0)
    _settle(ahead, 800)
    assert abs(ahead.steer) < 1


def test_brain_image_rates_replace_the_cursor_drive(tier):
    loom = tier.groups["LOOM"]
    b = Brain(tier)
    b.set_stimulus(150, 3000, 0.0)                      # the cursor numbers say: fast approach
    b.set_vision(loom, np.zeros(len(loom)))             # the image shows nothing growing, and it wins
    assert FLY not in {b.advance(4.0) for _ in range(150)}
    b.clear_vision()                                    # back to the cursor numbers
    assert FLY in {b.advance(4.0) for _ in range(150)}
    c = Brain(tier)
    c.set_stimulus(900, 0, 0.0)                         # no cursor drive at all, but the image shows a strong expansion
    c.set_vision(loom, np.full(len(loom), 120.0))
    assert FLY in {c.advance(4.0) for _ in range(150)}


def test_brain_hover_makes_the_fly_groom_and_calms_a_take_off(tier):
    b = Brain(tier)
    b.set_stimulus(5, 0, 0.0, bearing=0.5, touch=1.0)    # cursor on the fly, on its right
    seen = {b.advance(4.0) for _ in range(250)}
    assert GROOM in seen and FLY not in seen
    b.set_stimulus(900, 0, 0.0, bearing=0.5, touch=0.0)  # cursor leaves: grooming ends after its minimum
    _settle(b, 2000)
    assert b.state == STAND
    off = Brain(tier)
    off.set_stimulus(5, 0, 0.0, bearing=0.5, touch=0.0)  # near but not touching: no grooming
    assert GROOM not in {off.advance(4.0) for _ in range(250)}


def _decide(b, ms=2000, **rates):
    """Hold the smoothed rates / looming at the given values for `ms` and return the states passed through."""
    loom = rates.pop("loom", None)
    b.rates.update({k: float(v) for k, v in rates.items()})
    if loom is not None:
        b.loom = float(loom)
    seen = []
    for _ in range(int(ms / 4)):
        b._decode(4.0)
        if not seen or seen[-1] != b.state:
            seen.append(b.state)
    return seen


def test_brain_priority_is_take_off_retreat_freeze_groom_then_walk_or_stand(tier):
    assert _decide(Brain(tier), loom=1.0) == [FREEZE]                     # weak looming alone: freeze
    assert _decide(Brain(tier), loom=1.0, MDN=10.0) == [RETREAT]          # retreat outranks freeze
    assert _decide(Brain(tier), GROOM=20.0, MDN=10.0) == [RETREAT]        # ... and grooming
    assert _decide(Brain(tier), GROOM=20.0, loom=1.0) == [FREEZE]
    assert _decide(Brain(tier), GROOM=20.0) == [GROOM]
    assert _decide(Brain(tier), GF=20.0, GROOM=20.0, MDN=10.0, loom=1.0) == [FLY]
    b = Brain(tier)
    assert _decide(b, GROOM=20.0) == [GROOM] and b.state == GROOM
    assert _decide(b, 8, MDN=10.0) == [RETREAT]                           # a retreat input interrupts grooming at once
    assert _decide(b, MDN=0.0, GROOM=20.0, loom=1.0) == [RETREAT, FREEZE]   # leaving retreat falls to the next priority


def test_brain_freeze_lasts_its_minimum_and_ends_when_the_looming_has_passed(tier):
    b = Brain(tier)
    assert _decide(b, 100, loom=1.0) == [FREEZE]
    assert _decide(b, 1000, loom=0.0) == [FREEZE]                         # still frozen: the minimum is 1.2 s
    assert _decide(b, 400, loom=0.0)[-1] == STAND                         # then it ends (walk bias 0: stand)
    off = Brain(tier, spec=replace(FLYWIRE, freeze_on=0.0))
    assert FREEZE not in _decide(off, loom=5.0)


def test_brain_contact_is_not_looming_and_a_slow_approach_freezes_a_standing_fly(tier):
    touching = Brain(tier)
    touching.set_stimulus(10, 2000, 0.0, touch=1.0)                       # cursor on the fly, moving
    assert touching._loom_in == 0.0
    b = Brain(tier)
    b.set_stimulus(200, 200, 0.0)                                         # 1 /s: above the freeze rule, far below retreat
    assert FREEZE in {b.advance(4.0) for _ in range(250)}
    calm = Brain(tier)
    calm.set_stimulus(300, 60, 0.0)                                       # 0.2 /s: a cursor drifting closer, no reaction
    assert FREEZE not in {calm.advance(4.0) for _ in range(250)}


def test_brain_c1_hysteresis_and_decision_smoothing(tier):
    b = Brain(tier)
    # 1. Brief spike/blip of WALK (e.g. 5 Hz for just 20 ms): should NOT start walking
    b.rates["WALK"] = 5.0
    for _ in range(5):  # 20 ms
        b._decode(4.0)
    b.rates["WALK"] = 0.0
    for _ in range(150):  # 600 ms dwell check
        b._decode(4.0)
    assert b.state == STAND

    # 2. Sustained walk rate above walk_on_hz (3.5 Hz): should transition to WALK
    b.rates["WALK"] = 6.0
    for _ in range(200):  # 800 ms
        b._decode(4.0)
    assert b.state == WALK

    # 3. Rate dips to intermediate value (2.8 Hz, between walk_off_hz 2.0 and walk_on_hz 3.5): stays WALK (hysteresis)
    b.rates["WALK"] = 2.8
    for _ in range(200):  # 800 ms
        b._decode(4.0)
    assert b.state == WALK

    # 4. Rate drops below walk_off_hz (2.0 Hz): drops to STAND
    b.rates["WALK"] = 0.5
    for _ in range(200):  # 800 ms
        b._decode(4.0)
    assert b.state == STAND


def test_brain_b3_defensive_arousal(tier):
    b = Brain(tier)
    assert b.arousal == 0.0

    # Fast approach triggers escape (FLY) and surges arousal
    b.set_stimulus(150, 3000, 0.0)
    seen = {b.advance(4.0) for _ in range(150)}
    assert FLY in seen
    assert b.arousal >= 0.5

    # Arousal decays over time
    initial_arousal = b.arousal
    for _ in range(250):  # 1 second of quiet
        b.set_stimulus(900, 0, 0.0)
        b.advance(4.0)
    assert b.arousal < initial_arousal



def test_tier_cache_is_written_once_and_follows_the_full_cache(raw, tmp_path):
    full = flywire.build(raw)
    cache = tmp_path / "full.npz"
    full.save(cache)
    tiers = tmp_path / "tiers"
    a = flywire.load_tier(150, cache, tiers)
    path = flywire.tier_path(150, cache, tiers)
    assert path.exists() and a.n == 150
    assert not list(tiers.glob("*.tmp.npz"))                    # the atomic write left nothing behind
    mtime = path.stat().st_mtime_ns
    b = flywire.load_tier(150, cache, tiers)                    # served from the file
    assert path.stat().st_mtime_ns == mtime
    assert np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.data, b.data) and np.array_equal(a.ids, b.ids)
    assert all(np.array_equal(a.groups[k], b.groups[k]) for k in a.groups)
    ref = full.prefix(150)                                      # and it is the same network as cutting it by hand
    assert np.array_equal(ref.indices, b.indices) and np.array_equal(ref.ids, b.ids)
    # a rebuilt full cache invalidates the tier
    full.save(cache)
    import os
    os.utime(cache, ns=(cache.stat().st_atime_ns, cache.stat().st_mtime_ns + 5_000_000_000))
    flywire.load_tier(150, cache, tiers)
    assert path.stat().st_mtime_ns != mtime
    # a damaged tier file is rebuilt, and the whole brain is never cached
    path.write_bytes(b"not an npz")
    assert flywire.load_tier(150, cache, tiers).n == 150
    assert flywire.load_tier(full.n, cache, tiers).n == full.n
    assert not flywire.tier_path(full.n, cache, tiers).exists()


def test_image_rates_reach_the_cells_unchanged_and_only_the_freeze_input_is_odor_gated_in_the_brain(tier):
    """The worker applies the odor gate to the image path through the detector gain (before the rate caps, see
    test_eyeview); the brain must not scale the already capped rates a second time."""
    from neuropest.brain import odor_gate

    loom, retreat = tier.groups["LOOM"], tier.groups["RETREAT_IN"]
    idx = np.concatenate([loom, retreat])
    rates = np.full(len(idx), 100.0, np.float32)
    for phero, hunger in ((0.0, 1.0), (0.1, 1.0), (0.9, 1.0), (0.9, 0.0)):
        b = Brain(tier)
        b.set_stimulus(900, 0, 0.0, phero_drive=phero, hunger=hunger)
        b.set_vision(idx, rates, expansion=2.0)
        assert np.array_equal(b._last_fi[:len(idx)], idx)
        assert (b._last_fr[:len(idx)] == 100.0).all()
        assert b._loom_in == pytest.approx(2.0 * odor_gate(phero * hunger))


def test_food_odor_scales_the_cursor_retreat_drive_like_the_looming_drive(tier):
    from neuropest.brain import FLYWIRE, odor_gate

    b = Brain(tier)
    sl_loom, sl_ret = b._cursor_slices[0], b._cursor_slices[1]
    assert len(b._cursor_fi[sl_ret]) > 0                              # FlyWire has RETREAT_IN cells (the toy has none)
    for phero in (0.0, 0.1, 0.3):
        b.set_stimulus(100, 200, 0.0, phero_drive=phero, hunger=1.0)  # 2 /s before the gate
        g = odor_gate(phero)
        assert b._cursor_fr[sl_ret] == pytest.approx(min(FLYWIRE.retreat_max_hz, 2.0 * g * FLYWIRE.retreat_gain))
        assert b._cursor_fr[sl_loom] == pytest.approx(min(FLYWIRE.loom_max_hz, 2.0 * g * FLYWIRE.loom_gain))


def test_the_default_pet_does_not_startle_at_its_own_lure(tier):
    """Default start hunger 0.35 with the attractive cursor right on the fly (its own scent ~1): fully calm. This pins
    the calibration of ODOR_GATE_FULL (at 0.5 the gate would stay 0.3 open here and the pet could back away)."""
    b = Brain(tier)
    b.set_stimulus(40, 1500, 0.0, phero_drive=0.99, hunger=0.35)
    assert b._loom_in == 0.0
    assert not b._cursor_fr[b._cursor_slices[0]].any() and not b._cursor_fr[b._cursor_slices[1]].any()


# ---------------------------------------------------------------------- the lab: silencing and stimulation (Brain.set_lab)
def test_lab_silenced_giant_fiber_blocks_take_off_and_release_restores_it(tier):
    b = Brain(tier)
    assert b.set_lab(silenced=["GF"]) == (("GF",), ())
    b.set_stimulus(150, 3000, 0.0)                          # a fast approach: an intact fly takes off
    assert FLY not in {b.advance(4.0) for _ in range(150)}
    assert b.rates["GF"] == 0.0 and b.lab_on
    b.set_lab(silenced=[])
    assert FLY in {b.advance(4.0) for _ in range(150)}      # the same approach, the GF back


def test_lab_gf_pulse_takes_off_through_its_own_spikes_and_ends_by_itself(tier):
    b = Brain(tier)
    b.set_stimulus(900, 0, 0.0)
    _settle(b, 200)
    gf = set(tier.groups["GF"].tolist())
    b.set_lab(stimulated={"GF": 150.0}, stim_ms=300.0)
    assert gf <= set(b.engine.f_idx.tolist())
    modes = set()
    for _ in range(100):                                    # 400 ms
        if b.advance(4.0) == FLY:
            modes.add(b.flight_mode)
        assert b.gf_window == sum(n for _, n in b._gf_hist) >= 0      # the window stays the sum of its history
    assert modes == {"short"}                               # an escape take-off ...
    assert not gf & set(b.engine.f_idx.tolist()) and not b._lab_stim   # ... and the light went off by itself
    assert b.arousal == 0.0                                 # what the lab evoked is no defensive surge


def test_lab_looming_neurons_take_off_through_the_circuit_unless_the_gf_is_silenced(tier):
    b = Brain(tier)
    b.set_stimulus(900, 0, 0.0)
    b.set_lab(stimulated={"LC4": 80.0, "LPLC2": 80.0}, stim_ms=600.0)
    assert FLY in {b.advance(4.0) for _ in range(150)}
    c = Brain(tier)
    c.set_stimulus(900, 0, 0.0)
    c.set_lab(silenced=["GF"], stimulated={"LC4": 80.0, "LPLC2": 80.0}, stim_ms=600.0)
    assert FLY not in {c.advance(4.0) for _ in range(150)}
    assert c.lab_rates["LC4"] > 40.0                        # the looming cells did fire; the GF could not


def test_lab_lpc1_drives_the_moonwalker_and_lc10_on_one_side_turns(tier):
    b = Brain(tier)
    b.set_stimulus(900, 0, 0.0)
    b.set_lab(stimulated={"LPC1": 30.0}, stim_ms=800.0)
    assert RETREAT in {b.advance(4.0) for _ in range(200)}
    r = Brain(tier)
    r.set_stimulus(900, 0, 0.0)
    r.set_lab(stimulated={"LC10_R": 20.0}, stim_ms=1000.0)
    _settle(r, 600)
    assert r.steer > 5.0                                    # the right DNa02: a turn to the right


def test_lab_stimulation_merges_with_the_sensory_drive_one_entry_per_neuron(tier):
    loom, lc4 = tier.groups["LOOM"], tier.groups["LC4"]
    p = {hz: 1.0 - np.exp(-hz * 0.5 / 1000.0) for hz in (50.0, 120.0, 200.0)}
    for lab_hz, want in ((50.0, 120.0), (200.0, 200.0)):    # the larger of the two rates wins
        b = Brain(tier)
        b.set_vision(loom, np.full(len(loom), 120.0))
        b.set_lab(stimulated={"LC4": lab_hz}, stim_ms=1000.0)
        fi, fp = b.engine.f_idx, b.engine.f_p
        assert len(fi) == len(np.unique(fi))
        got = dict(zip(fi.tolist(), fp.tolist()))
        assert all(abs(got[i] - p[want]) < 1e-6 for i in lc4.tolist())


def test_the_lab_does_not_teach_but_a_real_threat_does(tier):
    b = Brain(tier, learning=True)
    b.set_stimulus(120, 0, 0.0)                             # the cursor close and still: a cue is present
    _settle(b, 200)
    b.mb.dirty = False
    for _ in range(3):                                      # three GF pulses: three escape take-offs
        b.set_lab(stimulated={"GF": 150.0}, stim_ms=300.0)
        _settle(b, 800)
    assert not b.mb.dirty and b._punish < 0.05 and b.valence == 0.0
    c = Brain(tier, learning=True)
    c.set_stimulus(150, 3000, 0.0)                          # the control: a real attack punishes
    _settle(c, 600)
    assert c.mb.dirty


def test_the_lab_drops_groups_a_circuit_does_not_have():
    from neuropest.toy_circuit import build

    b = Brain(build(146, 1), learning=False)
    assert b.lab_available() == ("GF", "WALK")
    assert b.set_lab(silenced=["MDN", "GF"], stimulated={"LC4": 80.0, "WALK": 120.0}) == (("GF",), ("WALK",))


def test_a_sandbox_punishment_given_during_the_lab_is_not_learned_after_it(tier):
    b = Brain(tier, learning=True)
    b.set_stimulus(120, 0, 0.0)                             # a cue is present
    _settle(b, 200)
    b.mb.dirty = False
    b.set_lab(stimulated={"GF": 150.0}, stim_ms=300.0)
    _settle(b, 200)
    b._punish = 1.0                                         # the user clicks "punish" mid-pulse
    _settle(b, 3000)                                        # past the pulse, the tail and the 0.8 s decay
    assert not b.mb.dirty and b.valence == 0.0


def test_an_escape_lands_once_the_threat_is_gone_whatever_the_urge_to_forage(tier):
    b = Brain(tier)
    b.set_stimulus(900, 0, 0.0, phero_drive=0.06, hunger=0.6)
    b.set_goal(300.0, 0.0, 0.06)
    b.flight_urge = 0.9                                     # about to fly to the food ...
    b.set_stimulus(150, 3000, 0.0, phero_drive=0.06, hunger=0.6)   # ... when the cursor attacks
    seen = [b.advance(4.0) for _ in range(60)]
    assert FLY in seen and b.flight_urge == 0.0 and b.flight_mode == "short"
    b.set_stimulus(900, 0, 0.0, phero_drive=0.06, hunger=0.6)
    landed = next((i for i in range(400) if b.advance(4.0) != FLY), None)
    assert landed is not None and landed * 4.0 < 1000.0     # within a second of the threat passing (was ~2 s more)


def test_an_attack_under_a_faint_food_odor_still_teaches_fear(tier):
    b = Brain(tier, learning=True)
    b.set_stimulus(900, 0, 0.0, phero_drive=0.06, hunger=0.6)      # odor x hunger 0.036 < ODOR_LEARN_GATE
    b.set_goal(300.0, 0.0, 0.06)
    _settle(b, 800)
    assert b.flight_urge >= 0.5                             # an urge to forage has built up
    b.set_stimulus(150, 3000, 0.0, phero_drive=0.06, hunger=0.6)
    _settle(b, 600)
    assert b.mb.dirty and b.valence < 0.0                   # a real attack is still punished


def test_an_escape_from_a_foraging_flight_is_timed_from_the_attack(tier):
    b = Brain(tier)
    b.state, b.flight_mode, b._dwell = FLY, "long", 4600.0  # late in a foraging flight
    b.rates["GF"] = 50.0
    b._decode(4.0)
    assert b.flight_mode == "short" and b._dwell < 10.0


# ------------------------------------------------------------------ taste, feeding, local search, the LC10 ceiling
def _on_food(b, hunger=1.0, sugar=1.0, ms=1000.0):
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, hunger=hunger, at_target=1.0 if sugar else 0.0, sugar=sugar)
    return {b.advance(4.0) for _ in range(int(ms / 4))}


def test_food_under_a_hungry_fly_fires_mn9_and_it_feeds_until_the_food_is_gone(tier):
    b = Brain(tier)
    assert b.taste
    _on_food(b)
    assert b.state == FEED and b.rates["MN9"] > FLYWIRE.feed_on_hz and b.feed_level > 0.2
    _on_food(b, sugar=0.0, ms=1200.0)                       # MN9 falls, the smoothed reading follows
    assert b.state != FEED and b.feed_level < 0.1


def test_a_sated_fly_on_food_does_not_feed(tier):
    b = Brain(tier)
    assert FEED not in _on_food(b, hunger=0.0)
    assert b.rates["MN9"] == 0.0                            # no drive: the sugar neurons are not sensitive at all


def test_the_sugar_drive_follows_contact_and_hunger(tier):
    b = Brain(tier)
    sl = b._cursor_slices[8]                                # SUGAR, the last of the static drive groups
    for h in (1.0, 0.5, 0.2):
        b.set_stimulus(1e6, 0.0, hunger=h, sugar=1.0)
        assert np.allclose(b._cursor_fr[sl], FLYWIRE.sugar_hz * sugar_gain(h))
    assert sugar_gain(1.0) > sugar_gain(0.5) > sugar_gain(0.2) > sugar_gain(0.0) == 0.0
    b.set_stimulus(1e6, 0.0, hunger=1.0, sugar=0.0)
    assert np.allclose(b._cursor_fr[sl], 0.0)


def test_an_approach_interrupts_feeding(tier):
    b = Brain(tier)
    _on_food(b)
    assert b.state == FEED
    b.set_stimulus(150, 3000, walk_bias=0.0, hunger=1.0, at_target=1.0, sugar=1.0)   # a fast approach: Giant Fiber
    assert FLY in {b.advance(4.0) for _ in range(150)}


def test_the_end_of_a_meal_starts_a_search_that_circles_the_meal_spot_and_a_threat_ends_it(tier):
    b = Brain(tier)
    _on_food(b)
    _on_food(b, sugar=0.0, ms=1200.0)
    assert b.state != FEED and b._search_ms > 0.0
    b.set_meal_spot(100.0, 1.0)                             # the spot on the right: it turns right
    _settle(b, 400)
    assert b.steer > 10.0
    b.set_meal_spot(100.0, -1.0)                            # ... and left once it is on the left
    _settle(b, 400)
    assert b.steer < -10.0
    b.set_stimulus(150, 3000, walk_bias=0.0, hunger=1.0)
    assert FLY in {b.advance(4.0) for _ in range(150)}
    assert b._search_ms == 0.0


def test_a_sated_fly_does_not_search_and_the_lab_pauses_a_search(tier):
    b = Brain(tier)
    b.hunger = 0.1
    b._start_search()
    assert b._search_ms == 0.0
    b.hunger = 1.0
    b._start_search()
    assert b._search_ms > 0.0 and b._search_rate() > 0.0
    b.set_lab(silenced=["GF"])
    assert b._search_rate() == 0.0                          # an experiment runs on the circuit alone


def test_the_lc10_drive_stays_under_the_ceiling_whatever_the_learned_gain(tier):
    import math

    from neuropest.states import LC10_MAX_HZ

    b = Brain(tier)
    b.valence = 1.0                                         # learned desire: x2.5 on the bearing drive
    b.set_stimulus(0.0, 0.0, bearing=math.pi / 2, phero_steer=100.0, phero_drive=0.5, hunger=1.0)
    assert b._cursor_fr[b._cursor_slices[4]].max() == LC10_MAX_HZ
    # the image path: the odor goes onto the image's LC10 cells, one entry per neuron, within the ceiling
    lc = b.group("LC10_R")
    b.set_vision(lc[:5], np.full(5, 50.0, np.float32))
    on = np.isin(b._last_fi, lc)
    assert b._last_fr[on].max() <= LC10_MAX_HZ and len(np.unique(b._last_fi[on])) == on.sum() == len(lc)


def test_a_circuit_without_the_taste_pathway_feeds_by_the_sugar_input():
    from neuropest.toy_circuit import build

    b = Brain(build(146, seed=42))
    assert not b.taste
    assert FEED in _on_food(b)
    assert b.feed_level > 0.0
    assert FEED not in _on_food(Brain(build(146, seed=42)), hunger=0.1)


# ------------------------------------------------------------------ what sets the fly walking (brain.ACTIVITY_REF)
def _walk_drive(b):
    return float(b._curr_cr[b._walk_slice].max())


def test_without_activity_odor_or_search_nothing_drives_walking(tier):
    b = Brain(tier)
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, hunger=1.0)
    _settle(b, 3000)
    assert not b._bout and _walk_drive(b) == 0.0 and b.state == STAND


def test_spontaneous_bouts_come_and_go_and_hunger_makes_them_longer(tier):
    from neuropest.brain import ACTIVITY_REF

    share = {}
    for h in (1.0, 0.0):
        b = Brain(tier, seed=3)
        b.set_stimulus(1e6, 0.0, walk_bias=ACTIVITY_REF, hunger=h)
        on = sum(b._bout for _ in range(30_000) if b.advance(4.0) is not None)      # 120 s
        share[h] = on / 30_000
    assert share[1.0] > 0.5 > 0.25 > share[0.0] > 0.0          # starved ~73%, sated ~9% of the time in a bout


def test_a_food_odor_sets_a_hungry_fly_walking_without_a_bout(tier):
    from neuropest.brain import ODOR_WALK_LEVEL

    b = Brain(tier)
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, phero_drive=0.5, hunger=1.0)
    assert np.isclose(_walk_drive(b), FLYWIRE.walk_drive[0] * ODOR_WALK_LEVEL)
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, phero_drive=0.5, hunger=0.0)              # a sated fly: no pull
    assert _walk_drive(b) == 0.0


def test_losing_the_odor_starts_a_search_toward_where_it_was_stronger_and_finding_it_ends_it(tier):
    b = Brain(tier)
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, phero_drive=0.5, phero_steer=5.0, hunger=1.0)   # stronger on the right
    _settle(b, 2000)
    assert b._search_ms == 0.0
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, phero_drive=0.02, phero_steer=0.0, hunger=1.0)  # out of the plume
    _settle(b, 1500)
    assert b._search_kind == "odor" and b._search_ms > 0.0 and b._search_side == 1 and b.steer > 10.0
    assert _walk_drive(b) > 0.0                                 # it walks the loops
    b.set_stimulus(1e6, 0.0, walk_bias=0.0, phero_drive=0.5, phero_steer=0.0, hunger=1.0)   # back in it
    _settle(b, 1000)
    assert b._search_ms == 0.0


def test_the_brain_reports_the_taste_and_touch_drive_it_applies(tier):
    b = Brain(tier)
    b.set_stimulus(1e6, 0.0, sugar=1.0, hunger=1.0, touch=0.0)
    assert np.isclose(b.drive_sugar_hz, FLYWIRE.sugar_hz) and b.drive_touch_hz == 0.0
    b.set_stimulus(10.0, 0.0, sugar=1.0, hunger=0.0, touch=1.0)
    assert b.drive_sugar_hz == 0.0 and np.isclose(b.drive_touch_hz, FLYWIRE.touch_hz)
