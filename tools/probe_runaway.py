"""The olfactory / mushroom-body runaway of the model, and whether a network can fall into it.

The LIF model has no spike-frequency adaptation and no short-term depression, the mechanisms that keep the antennal lobe
graded in the animal (Kazama & Wilson 2008; Olsen & Wilson 2008). Driven hard enough (food-odor projection neurons at
10 Hz, the "sugar/low_salt" taste neurons at 100 Hz, LC10 on one side at ~55-70 Hz and above), the full brain falls into a
self-sustained state of ~9,500 neurons (Kenyon cells, ALPNs, olfactory receptor neurons, LH and AL local neurons, DANs)
that keeps firing after the input ends, with DNa02 locked to one side; in the 15,000 tier built with an LC10 protocol at
100 Hz it cost ~30x the CPU and the pet circled to the left. states.LC10_MAX_HZ keeps the brain's LC10 drive under it,
and the ranking protocols stay under it too, so the tier no longer holds that network (tools/build_flywire.py).

Drives each input for --seconds, switches it off, and counts the neurons still spiking 0.5-1.5 s later
(> 1000: the runaway), with DNa02 left / right and the CPU time per simulated second over that window.

Run: uv run python tools/probe_runaway.py [n_neurons] [--inputs LC10_R:40,60,80,100 LC10_L:40,100 PN:10 LB3B:100]
     [--seconds 1] [--seeds 3]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from neuropest import flywire
from neuropest.engine import LIFEngine

FOOD_PN = ("DM1_lPN", "DM2_lPN", "DM3_adPN", "DM4_adPN")


def input_sets(net) -> dict[str, np.ndarray]:
    ann = flywire.load_annotations(net.ids)
    ct, side = ann["cell_type"].fillna(""), ann["side"].fillna("")
    sub = ann["cell_sub_class"].fillna("") if "cell_sub_class" in ann.columns else ct == "\0"
    f = lambda m: np.flatnonzero(np.asarray(m)).astype(np.int32)  # noqa: E731
    out = {k: v for k, v in net.groups.items() if k in ("LC10_L", "LC10_R", "LOOM", "RETREAT_IN", "TOUCH_L", "TOUCH_R", "SUGAR")}
    out["PN"] = f(ct.isin(FOOD_PN) & (side == "left"))          # food-odor projection neurons, left antenna
    out["LB3B"] = f(sub == "sugar/low_salt")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=15_000)
    ap.add_argument("--inputs", nargs="+", default=["LC10_R:40,60,80,100", "LC10_L:40,100", "PN:10", "LB3B:100", "SUGAR:150"])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    net = flywire.load_tier(a.n) if a.n < 138_639 else flywire.load_cache()
    g = net.groups
    sets = input_sets(net)
    every = np.arange(net.n)
    print(f"network: {net.n:,} neurons; walking drive 0.65; input {a.seconds:g} s, then off; {a.seeds} seeds")
    for spec in a.inputs:
        name, rates = spec.split(":")
        idx = sets.get(name, np.zeros(0, np.int32))
        if not len(idx):
            print(f"{name}: not in this network")
            continue
        for hz in (float(x) for x in rates.split(",")):
            rows = []
            for s in range(1, a.seeds + 1):
                e = LIFEngine(net, dt=0.5, seed=7919 * s)
                e.set_current_drive(g["WALK"], 650.0 * 0.65, 3.0)
                e.add_drive(idx, hz)
                e.advance(a.seconds * 1000.0)
                e.set_drive(np.zeros(0, np.int32), 0.0)
                e.advance(500)
                e.pop_counts(every)
                t0 = time.process_time()
                e.advance(1000)
                cpu = time.process_time() - t0
                c = e.pop_counts(every)
                rows.append((int((c > 0).sum()), c[g["DNa02_L"]].mean(), c[g["DNa02_R"]].mean(), cpu * 1000))
            r = np.array(rows, float)
            print(f"{name:7s} ({len(idx):4d}) {hz:5.0f} Hz | still spiking {r[:, 0].astype(int).tolist()} -> runaway "
                  f"{int((r[:, 0] > 1000).sum())}/{a.seeds} | DNa02 L {r[:, 1].mean():5.1f} R {r[:, 2].mean():5.1f} | "
                  f"CPU {r[:, 3].mean():5.0f} ms per simulated s", flush=True)


if __name__ == "__main__":
    main()
