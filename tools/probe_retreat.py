"""Which visual projection neuron (VPN) types drive MDN (backward walking) in the full brain, and why?

The app drives LPC1 into MDN for the retreat response. The literature disagrees: LPC1 is a translational
(T4b/T5b regressive flow) cell and its activation slows a fly down (Isaacson et al. 2023), whereas MDN's visual input
is LC16 (Sen et al. 2017) and the pathway is polysynaptic (Wu et al. 2016). This tool measures it on the connectome:

  1. wiring: signed synapse weight of every VPN type onto MDN in one hop and in two hops (mV per spike, summed over
     the type's neurons; the model's weight is sign x synapse count x 0.275 mV)
  2. simulation: every VPN type with >= 4 neurons driven alone (Poisson, all neurons of the type) at several rates in the
     full network; reports MDN, GF, DNp09 (WALK), DNa02, GROOM and the number of active neurons
  3. frontal variant: only the neurons whose receptive field centre lies within 60 deg of straight ahead (the app
     drives them from an image, so the drive is never uniform over a type)

Raw tables are written to data/probes/ (tracked in git, so the numbers can be checked): retreat_wiring.csv,
retreat_sim.csv. Run: uv run python tools/probe_retreat.py [--rates 10 25 50 100] [--types LC16 LC6 ...] [--no-sim]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from neuropest import flywire
from neuropest.engine import LIFEngine
from neuropest.paths import ROOT

OUT = ROOT / "data" / "probes"
ANCHORS = ("MDN", "GF", "WALK", "DNa02", "GROOM")


def types_of(net, ann):
    """cell type -> neuron indices, visual projection neurons only."""
    vpn = ann[ann.super_class == "visual_projection"]
    by_id = np.argsort(net.ids)
    out = {}
    for ct, sub in vpn.groupby("cell_type"):
        out[str(ct)] = by_id[np.searchsorted(net.ids[by_id], sub.index.values)].astype(np.int32)
    return out


def wiring(net, types, mdn):
    """Signed weight onto MDN in one and two hops, summed per type (mV per spike of every neuron of the type)."""
    a = net.to_csr()                                      # a[pre, post] = mV
    one = np.asarray(a[:, mdn].sum(axis=1)).ravel()       # pre -> MDN, one hop
    two = a @ one                                         # pre -> x -> MDN
    rows = []
    for ct, idx in types.items():
        rows.append(dict(cell_type=ct, n=len(idx), onehop_mv=float(one[idx].sum()), twohop_mv=float(two[idx].sum()),
                         onehop_per_neuron=float(one[idx].mean()), twohop_per_neuron=float(two[idx].mean())))
    return rows


def simulate(net, idx, rate, every, seed=7, warm=300.0, run=1000.0):
    e = LIFEngine(net, dt=0.5, seed=seed)
    e.set_drive(idx, rate)
    e.advance(warm)
    e.pop_counts(every)
    e.advance(run)
    c = e.pop_counts(every).astype(float) / (run / 1000.0)
    g = net.groups
    return dict(MDN=c[g["MDN"]].mean(), GF=c[g["GF"]].mean(), WALK=c[g["WALK"]].mean(),
                DNa02=c[np.concatenate([g["DNa02_L"], g["DNa02_R"]])].mean(), GROOM=c[g["GROOM"]].mean(),
                active=int((c > 0).sum()))


def frontal(net, idx, deg=60.0):
    """The neurons of `idx` whose receptive field centre is within `deg` of straight ahead (needs field.npz)."""
    from neuropest.vision import VisualField

    f = VisualField.load()
    row_of = {int(r): i for i, r in enumerate(net.ids)}
    keep = np.zeros(net.n, bool)
    az = np.degrees(np.arctan2(f.rf_dir[:, 1], f.rf_dir[:, 0]))
    for rid, a in zip(f.rf_id, az):
        if abs(a) <= deg and int(rid) in row_of:
            keep[row_of[int(rid)]] = True
    return idx[keep[idx]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=[10.0, 25.0, 50.0, 100.0])
    ap.add_argument("--types", nargs="*", default=None, help="only these cell types (default: all VPN types)")
    ap.add_argument("--no-sim", action="store_true")
    a = ap.parse_args()
    net = flywire.load_cache()
    ann = flywire.load_annotations(net.ids)
    types = types_of(net, ann)
    OUT.mkdir(parents=True, exist_ok=True)

    w = sorted(wiring(net, types, net.groups["MDN"]), key=lambda r: -abs(r["twohop_mv"]))
    with open(OUT / "retreat_wiring.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(w[0]))
        wr.writeheader()
        wr.writerows(w)
    print("wiring onto MDN (mV per spike summed over the type), strongest two-hop first:")
    for r in w[:12]:
        print(f"  {r['cell_type']:10s} n={r['n']:4d} one-hop {r['onehop_mv']:8.2f}  two-hop {r['twohop_mv']:9.2f}")
    if a.no_sim:
        return

    names = [t for t in (a.types or types) if t in types and len(types[t]) >= 4]
    every = np.arange(net.n)
    rows = []
    t0 = time.perf_counter()
    with open(OUT / "retreat_sim.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["cell_type", "n_neurons", "subset", "n_driven", "rate_hz", *ANCHORS, "active"])
        for ct in names:
            for subset in ("all", "frontal"):
                idx = types[ct] if subset == "all" else frontal(net, types[ct])
                if len(idx) < 2:
                    continue
                for rate in a.rates:
                    r = simulate(net, idx, rate, every)
                    rows.append((ct, subset, rate, r))
                    wr.writerow([ct, len(types[ct]), subset, len(idx), rate, *(f"{r[k]:.3f}" for k in ANCHORS), r["active"]])
                    fh.flush()
            best = [x for x in rows if x[0] == ct and x[1] == "all"]
            print(f"{ct:10s} n={len(types[ct]):4d} MDN at " + " ".join(f"{x[2]:.0f} Hz: {x[3]['MDN']:5.1f}" for x in best)
                  + f" | GF max {max(x[3]['GF'] for x in best):5.1f} | {time.perf_counter() - t0:5.0f} s", flush=True)
    top = sorted({(x[0]) for x in rows}, key=lambda ct: -max(x[3]["MDN"] for x in rows if x[0] == ct))[:12]
    print("\nstrongest MDN drive (max over rates and subsets):")
    for ct in top:
        print(f"  {ct:10s} {max(x[3]['MDN'] for x in rows if x[0] == ct):6.1f} Hz")


if __name__ == "__main__":
    main()
