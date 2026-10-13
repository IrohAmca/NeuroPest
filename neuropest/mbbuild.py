"""Extraction of the real FlyWire v783 mushroom-body wiring into data/circuits/mushroom_flywire.npz.

Part of the package (like eyebuild.py) so the in-app download (data_manager.build_all_circuits) can build it in a frozen
or wheel install too; tools/build_mushroom.py is the command line around it (and prints the census). Reads the raw files
in data/raw with pandas / pyarrow (build time only: the running app reads only the npz with numpy). What the arrays
hold is described in tools/build_mushroom.py.

Data: FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024; packaging Shiu et al. 2024), CC BY-NC 4.0.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .paths import CIRCUITS, RAW_DIR

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


def save(w: dict[str, np.ndarray], out: Path = OUT) -> None:
    """Write through a temporary file: an interrupted build must not leave a truncated npz that a saved preference for the
    real cells would then try to load on every start."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez_compressed(tmp, **w)
    os.replace(tmp, out)


def build(raw_dir: Path = RAW_DIR, out: Path = OUT, min_syn: int = 3) -> dict[str, np.ndarray]:
    """Extract the wiring from the raw FlyWire files and write it to `out` (what data_manager.build_all_circuits calls)."""
    w = extract(raw_dir, min_syn)
    save(w, out)
    return w
