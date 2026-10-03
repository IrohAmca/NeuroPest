"""FlyWire v783 connectome -> `Network`, with named neuron groups and a relevance ranking.

Raw files (not in git, see README): data/raw/
  Connectivity_783.parquet   Shiu et al. 2024 packaging of FlyWire v783 (Dorkenwald et al. 2024, CC-BY 4.0)
  Completeness_783.csv       neuron index -> FlyWire root id
  Supplemental_file1_neuron_annotations.tsv   cell types (Schlegel et al. 2024)

The cache written by `tools/build_flywire.py` is data/circuits/flywire_v783.npz.
Memory is tight on small machines, so everything stays int32/float32 and the parquet
is streamed in batches.
"""
from __future__ import annotations

from pathlib import Path

import numba as nb
import numpy as np

from .engine import LIFParams, Network
from .paths import CACHE, RAW_DIR

SOURCE = "FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024), CC-BY 4.0, via Shiu et al. 2024"

# Anchor descending neurons and what the literature says they do. Used to read behavior out of the
# simulation and as targets of the relevance ranking. Hemibrain type names, as the papers use them.
#   GF   (FlyWire DNp01 "Giant Fiber"): looming escape take-off; driven by LPLC2 (size) and LC4 (velocity)
#        -- Card & Dickinson 2008; von Reyn et al. 2014, 2017; Ache et al. 2019
#   P9   (hemibrain DNp09; FlyWire DNp09 + DNp71): initiates forward walking with ipsilateral turning
#        -- Bidaye et al. 2020
#   MDN  backward walking ("moonwalker") -- Bidaye et al. 2014; visually evoked retreat, Sen et al. 2017
#   DNa01 / DNa02: steering during walking, turning ~ right-left difference
#        -- Rayshubskiy et al., eLife 2025
ANCHOR_GROUPS = ("GF", "WALK", "MDN", "DNa01_L", "DNa01_R", "DNa02_L", "DNa02_R")


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

    cols =["root_id", "flow", "super_class", "cell_class", "cell_type", "hemibrain_type", "side"]
    ann = pd.read_csv(raw_dir / "Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=cols,
                      low_memory=False).drop_duplicates("root_id").set_index("root_id")
    return ann.reindex(ids)


def make_groups(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Named neuron groups (int32 indices) from the annotation table."""
    ct, hb, sc, side = (meta[c].fillna("") for c in ("cell_type", "hemibrain_type", "super_class", "side"))

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


# Looming-input rates (Hz, Poisson spikes on LPLC2 + LC4) the activity ranking is measured at.
TRAIN_RATES = (2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0)


def activity_scores(net: Network, rates=TRAIN_RATES, dt: float = 0.5, warm_ms: float = 300.0,
                    run_ms: float = 1000.0) -> np.ndarray:
    """Spikes per second of every neuron, summed over the stimulus levels, in the full network.

    The model has no spontaneous activity, so a neuron that never spikes under a stimulus has no
    effect on the rest: dropping it changes nothing. Ranking by this score therefore gives tiers
    that are exact for the measured stimuli once they contain every active neuron.
    """
    from .engine import LIFEngine

    total = np.zeros(net.n, np.float32)
    every = np.arange(net.n)
    for r in rates:
        e = LIFEngine(net, dt=dt, seed=11)
        e.set_drive(net.groups["LOOM"], float(r))
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
    pinned = np.union1d(anchors, groups["LOOM"])
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
