"""Fidelity and cost of each FlyWire tier against the full-brain simulation.

For stimulus levels the ranking was NOT built from, drive LPLC2 + LC4 at a given Poisson rate in the
full network and in each tier (top-N neurons by the stored relevance order) and compare:
  GF err     mean |GF_tier - GF_full| / max(GF_full, 5 Hz)          (take-off command)
  DN corr    Pearson correlation of per-DN rates, DNs present in the tier, all levels pooled
  DN kept    share of the full network's descending-neuron spiking that lies on retained DNs
  active %   mean share of neurons updated per step; realtime x = simulated s per wall s
Writes data/circuits/tiers.json (read by the GUI).

Run: uv run python tools/fidelity.py [n ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from neuropest import flywire, paths
from neuropest.engine import LIFEngine

TEST_RATES = (3.0, 7.0, 18.0, 35.0, 75.0, 125.0)
DEFAULT_TIERS = (300, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000)
DT = 0.5


def run(net, rates=TEST_RATES, warm_ms=300.0, run_ms=1500.0):
    g = net.groups
    res = []
    for r in rates:
        e = LIFEngine(net, dt=DT, seed=21)
        e.set_drive(g["LOOM"], float(r))
        e.advance(warm_ms)
        e.pop_counts(np.arange(net.n))
        u0, st0 = e.neuron_updates, e.steps
        t = time.perf_counter()
        e.advance(run_ms)
        wall = time.perf_counter() - t
        counts = e.pop_counts(np.arange(net.n)) / (run_ms / 1000.0)
        res.append(dict(rate=r, counts=counts, rt=run_ms / 1000.0 / wall,
                        active=100.0 * (e.neuron_updates - u0) / (e.steps - st0) / net.n))
    return res


def main():
    full = flywire.load_cache()
    tiers = [int(a) for a in sys.argv[1:]] or [n for n in DEFAULT_TIERS if n < full.n]
    ref = run(full)
    gf_full = np.array([r["counts"][full.groups["GF"]].mean() for r in ref])
    dn_full = np.stack([r["counts"][full.groups["DN"]] for r in ref])           # (rates, DN)
    dn_total = dn_full.sum()
    out = [dict(n=full.n, edges=int(full.nnz), gf_err=0.0, dn_corr=1.0, dn_kept=1.0,
                active=float(np.mean([r["active"] for r in ref])), realtime=float(np.mean([r["rt"] for r in ref])),
                rt_min=float(min(r["rt"] for r in ref)))]
    print(f"{'neurons':>8} {'edges':>9} | {'GF err %':>8} {'DN corr':>8} {'DN kept %':>9} | "
          f"{'active %':>8} {'realtime x':>10} {'worst x':>8}")
    print(f"{full.n:>8} {full.nnz:>9} | {0:>8.1f} {1:>8.3f} {100:>9.1f} | "
          f"{out[0]['active']:>8.1f} {out[0]['realtime']:>10.2f} {out[0]['rt_min']:>8.2f}  (full brain)", flush=True)
    dn_ids = full.groups["DN"]
    for n in tiers:
        sub = full.prefix(n)
        res = run(sub)
        gf = np.array([r["counts"][sub.groups["GF"]].mean() for r in res])
        gf_err = float(np.mean(np.abs(gf - gf_full) / np.maximum(gf_full, 5.0))) * 100
        # DNs kept in the tier: map full DN indices -> tier indices through the order prefix
        pos = np.full(full.n, -1, np.int64)
        pos[full.order[:n]] = np.arange(n)
        have = pos[dn_ids] >= 0
        a = dn_full[:, have].ravel()
        b = np.stack([r["counts"][pos[dn_ids[have]]] for r in res]).ravel()
        corr = float(np.corrcoef(a, b)[0, 1]) if have.sum() > 2 and a.std() > 0 and b.std() > 0 else float("nan")
        kept = float(dn_full[:, have].sum() / dn_total)
        row = dict(n=n, edges=int(sub.nnz), gf_err=gf_err, dn_corr=corr, dn_kept=kept,
                   active=float(np.mean([r["active"] for r in res])), realtime=float(np.mean([r["rt"] for r in res])),
                   rt_min=float(min(r["rt"] for r in res)))
        out.append(row)
        print(f"{n:>8} {sub.nnz:>9} | {gf_err:>8.1f} {corr:>8.3f} {100 * kept:>9.1f} | "
              f"{row['active']:>8.1f} {row['realtime']:>10.2f} {row['rt_min']:>8.2f}", flush=True)
    path = paths.TIERS
    path.write_text(json.dumps(dict(dt=DT, rates=TEST_RATES, tiers=sorted(out, key=lambda r: r["n"])), indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
