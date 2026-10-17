"""Taste: the sugar receptor neurons (SUGAR group, LB3c) drive the proboscis motor neuron MN9 (Shiu et al. 2024).

Drives SUGAR at a few Poisson rates in the given network with the app's walking drive on and prints MN9 (the feeding
readout, brain.FLYWIRE feed_on_hz / mn9_full_hz), P9 (walking; sugar does not reach it in this model, so standing
while feeding is the decoder's) and how many neurons still spike 0.5-1.5 s after the input ends (0 = no runaway).
With --lb3b it also drives the "sugar/low_salt" neurons (LB3b), which the SUGAR group leaves out: at 100 Hz they start
the olfactory / mushroom-body runaway in the full brain (tools/probe_runaway.py).

Run: uv run python tools/probe_sugar.py [n_neurons] [--rates 40,50,60,80,100,120,150] [--seeds 3] [--lb3b]
"""
from __future__ import annotations

import argparse

import numpy as np

from neuropest import flywire
from neuropest.engine import LIFEngine


def run(net, idx, hz, seed, walk_bias=0.65, secs=1.0):
    g = net.groups
    every = np.arange(net.n)
    e = LIFEngine(net, dt=0.5, seed=seed)
    e.set_current_drive(g["WALK"], 650.0 * walk_bias, 3.0)
    e.add_drive(idx, float(hz))
    e.advance(300)
    e.pop_counts(every)
    e.advance(secs * 1000.0)
    c = e.pop_counts(every) / secs
    e.set_drive(np.zeros(0, np.int32), 0.0)
    e.advance(500)
    e.pop_counts(every)
    e.advance(1000)
    after = int((e.pop_counts(every) > 0).sum())
    return c[g["MN9"]].mean(), c[g["WALK"]].mean(), int((c > 0).sum()), after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=15_000)
    ap.add_argument("--rates", default="40,50,60,80,100,120,150")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lb3b", action="store_true")
    a = ap.parse_args()
    net = flywire.load_tier(a.n) if a.n < 138_639 else flywire.load_cache()
    g = net.groups
    if not len(g.get("SUGAR", [])) or not len(g.get("MN9", [])):
        raise SystemExit("this cache has no SUGAR / MN9 groups: rebuild it (uv run python tools/build_flywire.py)")
    sets = {"SUGAR (LB3c)": g["SUGAR"]}
    if a.lb3b:
        ann = flywire.load_annotations(net.ids)
        lb3b = np.flatnonzero((ann["cell_sub_class"].fillna("") == "sugar/low_salt").to_numpy()).astype(np.int32)
        sets["sugar/low_salt (LB3b)"] = lb3b
    print(f"network: {net.n:,} neurons; SUGAR {len(g['SUGAR'])}, MN9 {len(g['MN9'])}; walking drive 0.65; "
          f"{a.seeds} seeds, 1 s each")
    for name, idx in sets.items():
        print(f"== {name}: {len(idx)} neurons")
        if not len(idx):
            print("  not in this network (the 15,000 tier holds only the SUGAR group; use the full brain: n = 138639)")
            continue
        for hz in (float(x) for x in a.rates.split(",")):
            r = np.array([run(net, idx, hz, 7919 * s) for s in range(1, a.seeds + 1)], float)
            print(f"  {hz:5.0f} Hz | MN9 {r[:, 0].mean():6.1f} Hz | P9 {r[:, 1].mean():5.1f} Hz | spiking {r[:, 2].mean():6.0f} "
                  f"| still spiking 0.5-1.5 s after {r[:, 3].astype(int).tolist()}", flush=True)


if __name__ == "__main__":
    main()
