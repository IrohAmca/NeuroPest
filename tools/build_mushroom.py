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
The extraction itself is neuropest/mbbuild.py, so the in-app download can run it in a packaged install as well.

Data: FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024; packaging Shiu et al. 2024), CC BY-NC 4.0.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from neuropest.mbbuild import (  # noqa: F401  (re-exported: the extraction lives in the package, see neuropest/mbbuild.py)
    APL,
    DAN,
    KC,
    MBON,
    OUT,
    PN,
    SOURCE,
    VERSION,
    _annotations,
    build,
    extract,
    roles,
    save,
)
from neuropest.paths import CACHE, RAW_DIR


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
    save(w, a.out)
    print("saved", a.out, f"({a.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
