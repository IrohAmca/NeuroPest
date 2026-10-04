"""Build data/circuits/mushroom_flywire.npz: the real FlyWire v783 mushroom-body wiring, small enough to load in a few ms.

Reads the raw files in data/raw (pandas / pyarrow: build time only; the running app reads only the npz with numpy) and
keeps, as plain arrays:

  KC   -> MBON    synapse counts per (Kenyon cell, MBON neuron)            -> the output weights of the learning layer
  PN   -> KC      synapse counts per (projection neuron, Kenyon cell)      -> the calyx wiring (olfactory ALPN, visual PNs)
  DAN  -> MBON    synapse counts per (dopamine neuron, MBON neuron)        -> which MBONs a reward / punishment DAN reaches
  DAN  -> KC      synapses per Kenyon cell, split PAM / PPL1 (stored, not used by the model yet)
  APL  -> KC      synapses per Kenyon cell (stored, not used by the model yet)

Edges below `--min-syn` synapses (default 3) are dropped for the pairwise tables, as is usual for connectome analyses.
MBON -> DAN and KC -> APL / KC -> KC edges are left out. Both hemispheres are kept as they are.

Run: uv run python tools/build_mushroom.py            (prints a census of the cells and how many sit in each tier too)

Data: FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024; packaging Shiu et al. 2024), CC BY-NC 4.0.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from neuropest.paths import CACHE, CIRCUITS, RAW_DIR

OUT = CIRCUITS / "mushroom_flywire.npz"
VERSION = 1
SOURCE = "FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024), CC BY-NC 4.0, via Shiu et al. 2024"

KC, MBON, DAN, APL, PN = 1, 2, 3, 4, 5                  # role codes


def _annotations(ids: np.ndarray, raw_dir: Path):
    import pandas as pd

    want = ["root_id", "super_class", "cell_class", "cell_type", "hemibrain_type", "side", "top_nt"]
    head = pd.read_csv(raw_dir / "Supplemental_file1_neuron_annotations.tsv", sep="\t", nrows=0).columns
    ann = pd.read_csv(raw_dir / "Supplemental_file1_neuron_annotations.tsv", sep="\t",
                      usecols=[c for c in want if c in head], low_memory=False)
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(ids)
    for c in want[1:]:
        ann[c] = ann[c].fillna("") if c in ann else ""
    return ann


def roles(ann) -> np.ndarray:
    """Role code per neuron index: Kenyon cell, MBON, DAN, APL, or a candidate projection neuron into the calyx."""
    cc, ct, sc = (ann[c].to_numpy(str) for c in ("cell_class", "cell_type", "super_class"))
    r = np.zeros(len(ann), np.int8)
    r[(cc == "ALPN") | (sc == "visual_projection")] = PN
    r[ct == "APL"] = APL
    r[cc == "DAN"] = DAN
    r[cc == "MBON"] = MBON
    r[cc == "Kenyon_Cell"] = KC
    return r


def extract(raw_dir: Path = RAW_DIR, min_syn: int = 3) -> dict[str, np.ndarray]:
    import pandas as pd
    import pyarrow.parquet as pq

    ids = pd.read_csv(raw_dir / "Completeness_783.csv", index_col=0).index.to_numpy(np.int64)
    ann = _annotations(ids, raw_dir)
    role = roles(ann)
    pf = pq.ParquetFile(raw_dir / "Connectivity_783.parquet")
    keep = {k: [] for k in ("kc_mbon", "pn_kc", "dan_mbon", "dan_kc", "apl_kc")}
    for b in pf.iter_batches(batch_size=1_000_000, columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity"]):
        pre = b.column(0).to_numpy(zero_copy_only=False)
        post = b.column(1).to_numpy(zero_copy_only=False)
        cnt = b.column(2).to_numpy(zero_copy_only=False)
        rp, rq = role[pre], role[post]
        for name, a, c in (("kc_mbon", KC, MBON), ("pn_kc", PN, KC), ("dan_mbon", DAN, MBON), ("dan_kc", DAN, KC),
                           ("apl_kc", APL, KC)):
            m = (rp == a) & (rq == c)
            if m.any():
                keep[name].append(np.stack([pre[m], post[m], cnt[m]], 1).astype(np.int64))
    edges = {k: (np.concatenate(v) if v else np.zeros((0, 3), np.int64)) for k, v in keep.items()}

    def cells(code):
        return np.flatnonzero(role == code)

    kc, mbon, dan = cells(KC), cells(MBON), cells(DAN)
    # only the projection neurons that reach a Kenyon cell with at least min_syn synapses
    pe = edges["pn_kc"][edges["pn_kc"][:, 2] >= min_syn]
    pn = np.unique(pe[:, 0])

    def index_of(members):                                # neuron index -> row in `members` (-1 elsewhere)
        lut = np.full(len(ids), -1, np.int64)
        lut[members] = np.arange(len(members))
        return lut

    def pairs(name, rows, cols, thresh=min_syn):
        e = edges[name]
        e = e[e[:, 2] >= thresh]
        return (index_of(rows)[e[:, 0]].astype(np.int32), index_of(cols)[e[:, 1]].astype(np.int32),
                e[:, 2].astype(np.int16))

    out: dict[str, np.ndarray] = {"version": np.array(VERSION), "min_syn": np.array(min_syn), "source": np.array(SOURCE)}
    for tag, members in (("kc", kc), ("mbon", mbon), ("pn", pn), ("dan", dan), ("apl", cells(APL))):
        out[f"{tag}_id"] = ids[members]
        out[f"{tag}_type"] = ann["cell_type"].to_numpy(str)[members]
        out[f"{tag}_side"] = ann["side"].to_numpy(str)[members]
    out["mbon_nt"] = ann["top_nt"].to_numpy(str)[mbon]
    out["pn_class"] = np.where(ann["cell_class"].to_numpy(str)[pn] == "ALPN", "olfactory", "visual").astype("U")
    out["kc_mbon_kc"], out["kc_mbon_mbon"], out["kc_mbon_n"] = pairs("kc_mbon", kc, mbon)
    out["pn_kc_pn"], out["pn_kc_kc"], out["pn_kc_n"] = pairs("pn_kc", pn, kc)
    out["dan_mbon_dan"], out["dan_mbon_mbon"], out["dan_mbon_n"] = pairs("dan_mbon", dan, mbon)
    # per-KC totals over all contacts (the single-synapse boutons of the DANs and APL would vanish under min_syn)
    dtype = ann["cell_type"].to_numpy(str)[dan]
    is_pam, is_ppl1 = np.char.startswith(dtype, "PAM"), np.char.startswith(dtype, "PPL1")
    e = edges["dan_kc"]
    di, ki = index_of(dan)[e[:, 0]], index_of(kc)[e[:, 1]]
    out["dan_kc"] = np.zeros((len(kc), 2), np.int32)
    np.add.at(out["dan_kc"], (ki[is_pam[di]], 0), e[is_pam[di], 2])
    np.add.at(out["dan_kc"], (ki[is_ppl1[di]], 1), e[is_ppl1[di], 2])
    e = edges["apl_kc"]
    out["apl_kc"] = np.zeros(len(kc), np.int32)
    np.add.at(out["apl_kc"], index_of(kc)[e[:, 1]], e[:, 2])
    return out


def census(w: dict[str, np.ndarray], rank: np.ndarray | None = None, tiers=(2000, 5000, 10000, 15000, 20000, 50000)):
    """Print what the file holds; with `rank` (relevance rank per root id in `w['*_id']` order) also the tier counts."""
    def line(label, ids_rank, n):
        cols = "".join(f"{int((ids_rank < t).sum()):>8d}" for t in tiers) if ids_rank is not None else ""
        print(f"  {label:22s}{n:>7d}{cols}")

    print(f"  {'':22s}{'total':>7s}" + ("".join(f"{'<' + str(t):>8s}" for t in tiers) if rank is not None else ""))
    for tag, label in (("kc", "Kenyon cells"), ("mbon", "MBONs"), ("dan", "dopamine neurons"), ("pn", "projection neurons"),
                       ("apl", "APL")):
        r = rank(w[f"{tag}_id"]) if rank is not None else None
        line(label, r, len(w[f"{tag}_id"]))
        types, counts = np.unique(w[f"{tag}_type"], return_counts=True)
        if tag in ("kc",):
            for t, c in zip(types, counts):
                line("  " + t, r[w[f"{tag}_type"] == t] if r is not None else None, int(c))
        if tag == "dan":
            fam = np.array([t[:3] if t.startswith(("PAM", "PPL", "PPM")) else "other" for t in w["dan_type"]])
            for f in np.unique(fam):
                line("  " + f, r[fam == f] if r is not None else None, int((fam == f).sum()))
    print(f"  edges: KC->MBON {len(w['kc_mbon_n']):,}, PN->KC {len(w['pn_kc_n']):,}, DAN->MBON {len(w['dan_mbon_n']):,} "
          f"(>= {int(w['min_syn'])} synapses each)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-syn", type=int, default=3)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    t = time.perf_counter()
    w = extract(a.raw, a.min_syn)
    print(f"extracted in {time.perf_counter() - t:.1f} s")
    rank = None
    if CACHE.exists():
        z = np.load(CACHE)
        ids, order = z["ids"], z["order"]
        r = np.empty(len(ids), np.int32)
        r[order] = np.arange(len(ids))
        srt = np.argsort(ids)

        def rank(root_ids, _r=r, _ids=ids, _srt=srt):
            return _r[_srt[np.searchsorted(_ids[_srt], root_ids)]]
    census(w, rank)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **w)
    print("saved", a.out, f"({a.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
