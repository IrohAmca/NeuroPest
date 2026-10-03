"""FlyWire v783 connectome -> `Network`, with named neuron groups and a relevance ranking.

Raw files (not in git, see README): data/raw/
  Connectivity_783.parquet   Shiu et al. 2024 packaging of FlyWire v783 (Dorkenwald et al. 2024; data license CC BY-NC 4.0, flywire.ai/guidelines)
  Completeness_783.csv       neuron index -> FlyWire root id
  Supplemental_file1_neuron_annotations.tsv   cell types (Schlegel et al. 2024)

The cache written by `tools/build_flywire.py` is data/circuits/flywire_v783.npz.
Memory is tight on small machines, so everything stays int32/float32 and the parquet
is streamed in batches.
"""
from __future__ import annotations

import os
from pathlib import Path

import numba as nb
import numpy as np

from .engine import LIFParams, Network
from .paths import CACHE, RAW_DIR, TIER_DIR

SOURCE = "FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024), CC BY-NC 4.0, via Shiu et al. 2024"

# Anchor descending neurons and what the literature says they do. Used to read behavior out of the
# simulation and as targets of the relevance ranking. Hemibrain type names, as the papers use them.
#   GF   (FlyWire DNp01 "Giant Fiber"): looming escape take-off; driven by LPLC2 (size) and LC4 (velocity)
#        -- Card & Dickinson 2008; von Reyn et al. 2014, 2017; Ache et al. 2019
#   P9   (hemibrain DNp09; FlyWire DNp09 + DNp71): initiates forward walking with ipsilateral turning
#        -- Bidaye et al. 2020
#   MDN  backward walking ("moonwalker") -- Bidaye et al. 2014; visually evoked retreat, Sen et al. 2017
#   DNa01 / DNa02: steering during walking, turning ~ right-left difference
#        -- Rayshubskiy et al., eLife 2025
#   GROOM (FlyWire DNg62 = aDN1, DNge078 = aDN2): antennal / anterior grooming -- Hampel et al. 2015;
#        named by root id in Shiu et al. 2024 (figures.ipynb)
ANCHOR_GROUPS = ("GF", "WALK", "MDN", "DNa01_L", "DNa01_R", "DNa02_L", "DNa02_R", "GROOM")


# --------------------------------------------------------------------------- loading
def load_connectome(raw_dir: Path = RAW_DIR, w_syn: float = LIFParams().w_syn):
    """-> indptr int64, indices int32, data float32 (mV, signed), ids int64 (root ids by index)."""
    import pandas as pd             # build-time only: keeps the worker's startup short
    import pyarrow.parquet as pq

    comp = pd.read_csv(raw_dir / "Completeness_783.csv", index_col=0)
    ids = comp.index.to_numpy(np.int64)
    n = len(ids)
    pf = pq.ParquetFile(raw_dir / "Connectivity_783.parquet")
    rows = pf.metadata.num_rows
    post = np.empty(rows, np.int32)
    data = np.empty(rows, np.float32)
    counts = np.zeros(n, np.int64)
    at, last = 0, -1
    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
    for batch in pf.iter_batches(batch_size=1_000_000, columns=cols):
        k = batch.num_rows
        pre = batch.column(0).to_numpy(zero_copy_only=False)
        if pre[0] < last or np.any(pre[1:] < pre[:-1]):
            raise ValueError("connectivity file is not sorted by presynaptic index")
        last = int(pre[-1])
        counts += np.bincount(pre, minlength=n)
        post[at:at + k] = batch.column(1).to_numpy(zero_copy_only=False)
        data[at:at + k] = batch.column(2).to_numpy(zero_copy_only=False)
        at += k
    assert at == rows and int(post.max()) < n
    data *= np.float32(w_syn)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, post, data, ids


def load_annotations(ids: np.ndarray, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Annotation rows aligned to neuron index (NaN rows for the few ids without annotation)."""
    import pandas as pd

    cols = ["root_id", "flow", "super_class", "cell_class", "cell_type", "hemibrain_type", "side"]
    ann = pd.read_csv(raw_dir / "Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=cols,
                      low_memory=False).drop_duplicates("root_id").set_index("root_id")
    return ann.reindex(ids)


def make_groups(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Named neuron groups (int32 indices) from the annotation table."""
    ct, hb, sc, side = (meta[c].fillna("") for c in ("cell_type", "hemibrain_type", "super_class", "side"))
    cc = meta["cell_class"].fillna("")

    def idx(mask) -> np.ndarray:
        return np.flatnonzero(np.asarray(mask)).astype(np.int32)

    g = {
        "LPLC2": idx(ct == "LPLC2"),
        "LC4": idx(ct == "LC4"),
        "GF": idx(ct == "DNp01"),
        "WALK": idx(hb == "DNp09"),
        "MDN": idx(ct == "MDN"),
        "DNa01_L": idx((hb == "DNa01") & (side == "left")),
        "DNa01_R": idx((hb == "DNa01") & (side == "right")),
        "DNa02_L": idx((hb == "DNa02") & (side == "left")),
        "DNa02_R": idx((hb == "DNa02") & (side == "right")),
        "DN": idx(sc == "descending"),
        "VPN": idx(sc == "visual_projection"),
    }
    g["LOOM"] = np.union1d(g["LPLC2"], g["LC4"]).astype(np.int32)    # driven by the looming stimulus
    # LPC1 is the visual projection type that drives MDN most strongly in the model (tools/probe_inputs.py,
    # tools/probe_retreat.py; raw tables in data/probes/). A FUNCTIONAL PLACEHOLDER, not a biological claim: in the
    # animal LPC1 is a regressive-flow cell whose activation slows the fly down (Isaacson et al. 2023) and MDN's visual
    # input is LC16 (Sen et al. 2017). In the model LC16 does drive MDN too (23.5 Hz at 100 Hz on all 151 neurons, 0 below 50 Hz,
    # 0 when only the frontal ones are driven), LC6 drives the Giant Fiber, LPLC1 drives neither.
    g["RETREAT_IN"] = idx(ct == "LPC1")
    # LC10a/c-2/d drive the DNa02 on their own side (tools/probe_side.py): the bearing input. DNa02 on
    # one side makes the fly turn to that side (Rayshubskiy et al.), so a cursor on the left turns it left.
    lc10 = ct.isin(["LC10a", "LC10c-2", "LC10d"])
    g["LC10_L"] = idx(lc10 & (side == "left"))
    g["LC10_R"] = idx(lc10 & (side == "right"))
    # Touch: the brain holds only the head's mechanosensory neurons (body and leg bristles enter the
    # ventral nerve cord, which is not in the brain connectome): head bristles (BM_*, taste bristles
    # excluded) and Johnston's organ C/E neurons (antennal deflection, Hampel et al. 2015), per side.
    # In the model they drive the grooming DNs aDN1/aDN2 (tools/probe_touch.py).
    touch = ((cc == "mechanosensory") & ct.str.startswith("BM_") & (ct != "BM_Taste")) | ct.str.match(r"JO-[CE]")
    g["TOUCH_L"] = idx(touch & (side == "left"))
    g["TOUCH_R"] = idx(touch & (side == "right"))
    g["GROOM"] = idx(ct.isin(["DNg62", "DNge078"]))
    return g


# --------------------------------------------------------------------------- ranking
def rank_by_pathway(net: Network, sources: np.ndarray, targets: np.ndarray,
                    pinned: np.ndarray | None = None, depth: int = 4, decay: float = 0.7) -> np.ndarray:
    """Order neurons by how much they sit on signal paths from `sources` to `targets`.

    Forward score x: share of a neuron's input (by synapse count) traceable to the sources within
    `depth` steps. Backward score y: share of a neuron's output reaching the targets within `depth`
    steps. Relevance = x * y; ties broken by x + y, then degree. `pinned` neurons (the stimulus
    inputs and the anchor outputs) always come first. Uses absolute weights; the sign of a synapse
    does not change whether a neuron is on a path.
    """
    n = net.n
    col_sum, row_sum = _abs_sums(net.indptr, net.indices, net.data)
    s = np.zeros(n, np.float64)
    s[sources] = 1.0
    t = np.zeros(n, np.float64)
    t[targets] = 1.0
    x, y = s.copy(), t.copy()
    xs, ys = s.copy(), t.copy()
    for k in range(1, depth + 1):
        xs = _spread_forward(net.indptr, net.indices, net.data, col_sum, xs)
        ys = _spread_backward(net.indptr, net.indices, net.data, row_sum, ys)
        x += decay**k * xs
        y += decay**k * ys
    score = x * y
    if pinned is not None:
        score[pinned] = np.inf
    deg = np.diff(net.indptr) + np.bincount(net.indices, minlength=n)
    return np.lexsort((-deg, -(x + y), -score)).astype(np.int32)


@nb.njit(cache=True)
def _abs_sums(indptr, indices, data):
    n = indptr.shape[0] - 1
    col = np.zeros(n)          # total |input| of each neuron
    row = np.zeros(n)          # total |output| of each neuron
    for i in range(n):
        for k in range(indptr[i], indptr[i + 1]):
            a = abs(data[k])
            col[indices[k]] += a
            row[i] += a
    return col, row


@nb.njit(cache=True)
def _spread_forward(indptr, indices, data, col_sum, x):
    """out[post] = sum over pre of x[pre] * |w| / (total |input| of post)."""
    out = np.zeros_like(x)
    for i in range(indptr.shape[0] - 1):
        xi = x[i]
        if xi != 0.0:
            for k in range(indptr[i], indptr[i + 1]):
                j = indices[k]
                out[j] += xi * abs(data[k]) / col_sum[j]
    return out


@nb.njit(cache=True)
def _spread_backward(indptr, indices, data, row_sum, y):
    """out[pre] = sum over post of y[post] * |w| / (total |output| of pre)."""
    out = np.zeros_like(y)
    for i in range(indptr.shape[0] - 1):
        if row_sum[i] > 0.0:
            acc = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                acc += y[indices[k]] * abs(data[k])
            out[i] = acc / row_sum[i]
    return out


# Stimulus protocols the activity ranking is measured at: {group: Poisson rate in Hz}. The cursor
# produces looming input, retreat input and a left/right bearing input, alone or together.
INPUT_GROUPS = ("LOOM", "RETREAT_IN", "LC10_L", "LC10_R", "TOUCH_L", "TOUCH_R")


def train_protocols() -> list[dict[str, float]]:
    p = [{"LOOM": r} for r in (2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0)]
    p += [{"RETREAT_IN": r} for r in (5.0, 10.0, 25.0, 50.0, 80.0)]
    for side in ("LC10_L", "LC10_R"):
        p += [{side: r} for r in (5.0, 10.0, 25.0, 50.0, 100.0)]
    for side in ("TOUCH_L", "TOUCH_R"):
        p += [{side: r} for r in (25.0, 50.0, 100.0, 150.0)]
    p += [{"LOOM": 3.0, "RETREAT_IN": 20.0, "LC10_L": 30.0}, {"LOOM": 6.0, "RETREAT_IN": 40.0, "LC10_R": 60.0}]
    p += [{"TOUCH_L": 100.0, "LOOM": 10.0}, {"TOUCH_R": 80.0, "RETREAT_IN": 20.0}]
    return p


def test_protocols() -> list[dict[str, float]]:
    """Levels the ranking was not built from, for tools/fidelity.py."""
    p = [{"LOOM": r} for r in (3.0, 7.0, 18.0, 35.0, 75.0, 125.0)]
    p += [{"RETREAT_IN": r} for r in (7.0, 18.0, 35.0, 65.0)]
    for side in ("LC10_L", "LC10_R"):
        p += [{side: r} for r in (7.0, 18.0, 35.0, 75.0)]
    for side in ("TOUCH_L", "TOUCH_R"):
        p += [{side: r} for r in (40.0, 75.0, 125.0)]
    p += [{"LOOM": 4.0, "RETREAT_IN": 30.0, "LC10_R": 40.0}, {"LOOM": 2.0, "RETREAT_IN": 12.0, "LC10_L": 20.0}]
    p += [{"TOUCH_R": 100.0, "LOOM": 20.0}]
    return p


def activity_scores(net: Network, protocols=None, dt: float = 0.5, warm_ms: float = 300.0,
                    run_ms: float = 1000.0) -> np.ndarray:
    """Spikes per second of every neuron, summed over the stimulus protocols, in the full network.

    The model has no spontaneous activity, so a neuron that never spikes under a stimulus has no
    effect on the rest: dropping it changes nothing. Ranking by this score therefore gives tiers
    that are exact for the measured stimuli once they contain every active neuron.
    """
    from .engine import LIFEngine

    total = np.zeros(net.n, np.float32)
    every = np.arange(net.n)
    for proto in protocols or train_protocols():
        e = LIFEngine(net, dt=dt, seed=11)
        for name, rate in proto.items():
            e.add_drive(net.groups[name], float(rate))
        e.advance(warm_ms)
        e.pop_counts(every)
        e.advance(run_ms)
        total += e.pop_counts(every) / (run_ms / 1000.0)
    return total


# --------------------------------------------------------------------------- build / load
def build(raw_dir: Path = RAW_DIR, with_activity: bool = True) -> Network:
    indptr, indices, data, ids = load_connectome(raw_dir)
    meta = load_annotations(ids, raw_dir)
    groups = make_groups(meta)
    net = Network(indptr, indices, data, ids=ids, groups=groups,
                  meta={"kind": "flywire", "source": SOURCE, "w_syn": LIFParams().w_syn})
    anchors = np.unique(np.concatenate([groups[k] for k in ANCHOR_GROUPS]))
    pinned = np.unique(np.concatenate([anchors] + [groups[k] for k in INPUT_GROUPS]))
    structural = rank_by_pathway(net, groups["VPN"], anchors, pinned)
    if not with_activity:
        net.order = structural
        return net
    act = activity_scores(net)
    srank = np.empty(net.n, np.int32)
    srank[structural] = np.arange(net.n)
    key = act.astype(np.float64)
    key[pinned] = np.inf                                   # inputs and anchor outputs first
    net.order = np.lexsort((srank, -key)).astype(np.int32)  # most active first, ties by structure
    net.extra = {"activity": act, "struct_rank": srank}
    net.meta["n_active"] = int((act > 0).sum())
    return net


def load_cache(path: Path = CACHE) -> Network:
    if not path.exists():
        raise FileNotFoundError(f"{path} yok. Once: uv run python tools/build_flywire.py")
    return Network.load(path)


TIER_SIZES = (2_000, 5_000, 10_000, 15_000, 20_000, 50_000)      # sizes the control window offers below the full brain


def tier_path(n: int, cache: Path = CACHE, tier_dir: Path = TIER_DIR) -> Path:
    return tier_dir / f"{cache.stem}_n{n}.npz"


def _stamp(cache: Path) -> str:
    st = cache.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def load_tier(n: int, cache: Path = CACHE, tier_dir: Path = TIER_DIR) -> Network:
    """The `n` most relevant neurons, from a per-size cache file next to the full cache.

    Reading the 125 MB full cache and cutting the tier out of it cost ~0.8 s at every worker start; the tier's own
    file loads in ~20 ms. The file is written on first use (atomically: workers may race) and tied to the full
    cache by its size and mtime, so a rebuilt connectome never serves a stale tier."""
    path = tier_path(n, cache, tier_dir)
    stamp = _stamp(cache) if cache.exists() else None
    if stamp is not None and path.exists():
        try:
            net = Network.load(path)
            if net.meta.get("cache_stamp") == stamp:
                return net
        except (OSError, ValueError, KeyError):
            pass                                            # damaged or half-written file: rebuild it
    full = load_cache(cache)
    if n >= full.n:
        return full
    net = full.prefix(n)
    net.meta["cache_stamp"] = stamp
    try:
        tier_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        net.save(tmp)
        os.replace(tmp, path)
    except OSError:
        pass                                                # read-only folder: just run without the cache
    return net
