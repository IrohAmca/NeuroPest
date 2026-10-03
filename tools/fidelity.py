"""Fidelity and cost of each FlyWire tier against the full-brain simulation.

For stimulus levels the ranking was NOT built from (flywire.test_protocols: looming, retreat input,
left/right bearing input, and mixtures), drive the input groups in the full network and in each tier
(top-N neurons by the stored relevance order) and compare:
  GF err     looming protocols:  mean |GF_tier - GF_full| / max(GF_full, 5 Hz)       (take-off command)
  MDN err    retreat protocols:  same for MDN                                         (backward walking)
  DNa02 err  bearing protocols:  same for the DNa02 on the driven side                (steering)
  groom err  touch protocols:    same for the grooming DNs aDN1/aDN2 (GROOM)          (grooming)
  mix err    mixed protocols:    mean over GF, MDN and both DNa02
  DN corr    Pearson correlation of per-DN rates, DNs present in the tier, all protocols pooled
  DN kept    share of the full network's descending-neuron spiking that lies on retained DNs
  active %   mean share of neurons updated per step; realtime x = simulated s per wall s (mean and worst)
Writes data/circuits/tiers.json (read by the GUI).

Run: uv run python tools/fidelity.py [n ...]
"""
from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np

from neuropest import flywire, paths
from neuropest.engine import LIFEngine

DEFAULT_TIERS = (1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 100000)
DT = 0.5
ANCHORS = ("GF", "MDN", "DNa02_L", "DNa02_R", "GROOM")


def machine_name() -> str:
    """CPU the speed columns were measured on (they are machine specific)."""
    name = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo", encoding="utf8"):
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{name}, {platform.system()}"


def run(net, protocols, warm_ms=300.0, run_ms=1000.0):
    g = net.groups
    res = []
    every = np.arange(net.n)
    for proto in protocols:
        e = LIFEngine(net, dt=DT, seed=21)
        for name, rate in proto.items():
            e.add_drive(g[name], float(rate))
        e.advance(warm_ms)
        e.pop_counts(every)
        u0, st0 = e.neuron_updates, e.steps
        t = time.perf_counter()
        e.advance(run_ms)
        wall = time.perf_counter() - t
        counts = e.pop_counts(every) / (run_ms / 1000.0)
        res.append(dict(counts=counts, rt=run_ms / 1000.0 / wall,
                        active=100.0 * (e.neuron_updates - u0) / (e.steps - st0) / net.n))
    return res


def anchor_rates(net, res):
    g = net.groups
    return np.array([[r["counts"][g[a]].mean() for a in ANCHORS] for r in res])      # (protocols, anchors)


def rel_err(tier, full):
    return np.abs(tier - full) / np.maximum(full, 5.0)


def main():
    protocols = flywire.test_protocols()
    fam = {k: [i for i, p in enumerate(protocols) if list(p) == [k]] for k in ("LOOM", "RETREAT_IN", "LC10_L", "LC10_R", "TOUCH_L", "TOUCH_R")}
    mixed = [i for i, p in enumerate(protocols) if len(p) > 1]
    full = flywire.load_cache()
    tiers = [int(a) for a in sys.argv[1:]] or [n for n in DEFAULT_TIERS if n < full.n]
    ref = run(full, protocols)
    a_full = anchor_rates(full, ref)
    dn_full = np.stack([r["counts"][full.groups["DN"]] for r in ref])
    dn_ids = full.groups["DN"]

    def errors(a_tier):
        e = rel_err(a_tier, a_full)
        return dict(
            gf_err=100 * float(e[fam["LOOM"], 0].mean()),
            mdn_err=100 * float(e[fam["RETREAT_IN"], 1].mean()),
            steer_err=100 * float(np.mean([e[fam["LC10_L"], 2].mean(), e[fam["LC10_R"], 3].mean()])),
            groom_err=100 * float(e[fam["TOUCH_L"] + fam["TOUCH_R"], 4].mean()),
            mix_err=100 * float(e[mixed].mean()))

    rts = [r["rt"] for r in ref]
    out = [dict(n=full.n, edges=int(full.nnz), **errors(a_full), dn_corr=1.0, dn_kept=1.0,
                active=float(np.mean([r["active"] for r in ref])), realtime=float(np.mean(rts)), rt_min=float(min(rts)))]
    head = (f"{'neurons':>8} {'edges':>9} | {'GF %':>6} {'MDN %':>6} {'DNa02 %':>7} {'groom %':>7} {'mix %':>6} | {'DN corr':>7} "
            f"{'DN kept %':>9} | {'active %':>8} {'x mean':>7} {'x worst':>7}")
    print(head)
    print(f"{full.n:>8} {full.nnz:>9} | {0:>6.1f} {0:>6.1f} {0:>7.1f} {0:>7.1f} {0:>6.1f} | {1:>7.3f} {100:>9.1f} | "
          f"{out[0]['active']:>8.1f} {out[0]['realtime']:>7.2f} {out[0]['rt_min']:>7.2f}  (full brain)", flush=True)
    for n in tiers:
        sub = full.prefix(n)
        res = run(sub, protocols)
        err = errors(anchor_rates(sub, res))
        pos = np.full(full.n, -1, np.int64)
        pos[full.order[:n]] = np.arange(n)
        have = pos[dn_ids] >= 0
        a = dn_full[:, have].ravel()
        b = np.stack([r["counts"][pos[dn_ids[have]]] for r in res]).ravel()
        corr = float(np.corrcoef(a, b)[0, 1]) if have.sum() > 2 and a.std() > 0 and b.std() > 0 else float("nan")
        kept = float(dn_full[:, have].sum() / dn_full.sum())
        rts = [r["rt"] for r in res]
        row = dict(n=n, edges=int(sub.nnz), **err, dn_corr=corr, dn_kept=kept,
                   active=float(np.mean([r["active"] for r in res])), realtime=float(np.mean(rts)), rt_min=float(min(rts)))
        out.append(row)
        print(f"{n:>8} {sub.nnz:>9} | {err['gf_err']:>6.1f} {err['mdn_err']:>6.1f} {err['steer_err']:>7.1f} {err['groom_err']:>7.1f} {err['mix_err']:>6.1f} | "
              f"{corr:>7.3f} {100 * kept:>9.1f} | {row['active']:>8.1f} {row['realtime']:>7.2f} {row['rt_min']:>7.2f}", flush=True)
    paths.TIERS.write_text(json.dumps(dict(dt=DT, machine=machine_name(), protocols=protocols, tiers=sorted(out, key=lambda r: r["n"])), indent=1))
    print("wrote", paths.TIERS)


if __name__ == "__main__":
    main()
