"""What does a touch do? Drive head mechanosensory neurons and read the descending neurons.

The FlyWire brain holds the head's mechanosensory neurons only: Johnston's organ (JO, antennal
deflection and sound) and head bristles (BM_*). Body and leg bristles project to the ventral nerve
cord, which is not in the brain connectome. Each input set is driven alone at a few Poisson rates
in the given network; the anchors, the antennal grooming neurons of Shiu et al. 2024 / Hampel et
al. 2015 (aBN1, aDN1, aDN2) and the most active descending neuron types are printed.

Run: uv run python tools/probe_touch.py [n_neurons] [--rates 50,100,200] [--sets JO-CE,BM_Ant]
"""
from __future__ import annotations

import sys

import numpy as np

from neuropest import flywire
from neuropest.engine import LIFEngine

# antennal grooming neurons, by FlyWire root id (Shiu et al. 2024, figures.ipynb)
ID_aBN1 = 720575940630907434
ID_aDN1 = 720575940616185531
ID_aDN2 = 720575940629806974


def touch_sets(ann) -> dict[str, np.ndarray]:
    """Input set name -> boolean mask over the annotation rows (aligned to neuron index)."""
    ct = ann.cell_type.fillna("")
    cc = ann.cell_class.fillna("")
    side = ann.side.fillna("")
    mech = cc == "mechanosensory"
    jo_ce = ct.str.match(r"JO-(C|E)")
    sets = {
        "JO-CE": jo_ce,
        "JO-F": ct.str.match(r"JO-F"),
        "JO-AB": ct.str.match(r"JO-(A|B)"),
        "BM_Ant": ct == "BM_Ant",
        "BM_InOm": ct == "BM_InOm",
        "BM_head": mech & ct.str.startswith("BM_") & ~ct.isin(["BM_Ant", "BM_InOm", "BM_Taste"]),
        "ANT_all": jo_ce | ct.str.match(r"JO-F") | (ct == "BM_Ant"),
        "JO-CE_L": jo_ce & (side == "left"),
        "JO-CE_R": jo_ce & (side == "right"),
        "BM_InOm_L": (ct == "BM_InOm") & (side == "left"),
        "BM_InOm_R": (ct == "BM_InOm") & (side == "right"),
    }
    return {k: np.asarray(v) for k, v in sets.items()}


def main():
    args = sys.argv[1:]
    rates, only = (50.0, 100.0, 200.0), None
    if "--rates" in args:
        i = args.index("--rates")
        rates = tuple(float(x) for x in args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    if "--sets" in args:
        i = args.index("--sets")
        only = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    full = flywire.load_cache()
    ann = flywire.load_annotations(full.ids)
    net = full.prefix(int(args[0])) if args else full
    g = net.groups
    probes = {"GF": g["GF"], "P9": g["WALK"], "MDN": g["MDN"], "DNa02_L": g["DNa02_L"], "DNa02_R": g["DNa02_R"]}
    # a tier is renumbered, so the named neurons are found by root id
    idx_of = {int(r): i for i, r in enumerate(net.ids)}
    for name, rid in (("aBN1", ID_aBN1), ("aDN1", ID_aDN1), ("aDN2", ID_aDN2)):
        probes[name] = np.array([idx_of[rid]] if rid in idx_of else [], np.int32)
    net_ann = ann.reindex(net.ids)
    dn = net_ann.super_class.fillna("") == "descending"
    dn_type = net_ann.cell_type.fillna("?").where(dn)
    every = np.arange(net.n)
    print(f"network: {net.n:,} neurons")
    for sname, mask in touch_sets(net_ann).items():
        if only and sname not in only:
            continue
        idx = np.flatnonzero(mask).astype(np.int32)
        n_full = int(touch_sets(ann)[sname].sum())
        print(f"\n== {sname}: {len(idx)} of {n_full} neurons in this network")
        if len(idx) == 0:
            continue
        for r in rates:
            e = LIFEngine(net, dt=0.5, seed=5)
            e.set_drive(idx, r)
            e.advance(300)
            e.pop_counts(every)
            e.advance(1000)
            c = e.pop_counts(every).astype(float)
            line = " ".join(f"{k} {c[v].mean():5.1f}" if len(v) else f"{k}   -  " for k, v in probes.items())
            top = (dn_type.dropna().to_frame("t").assign(hz=c[dn.to_numpy()])
                   .groupby("t").hz.mean().sort_values(ascending=False).head(8))
            tops = ", ".join(f"{t} {h:.0f}" for t, h in top.items() if h > 0)
            print(f"  {r:5.0f} Hz | {line} | active {(c > 0).sum():6d}\n           top DN types: {tops}",
                  flush=True)


if __name__ == "__main__":
    main()
