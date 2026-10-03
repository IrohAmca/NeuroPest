"""Loader, ranking, tiers and the Brain on a tiny FlyWire-shaped dataset written to a temp dir."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from neuropest import flywire
from dataclasses import replace

from neuropest.brain import FLY, FLYWIRE, GROOM, RETREAT, STAND, Brain
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
    pd.DataFrame({"Completed": True}, index=pd.Index(ids)).to_csv(d / "Completeness_783.csv")
    ann = pd.DataFrame({"root_id": ids, "flow": "intrinsic", "super_class": sc, "cell_class": cc,
                        "cell_type": types, "hemibrain_type": hb, "side": side})
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
