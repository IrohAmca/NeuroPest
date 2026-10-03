"""Stimulus -> take-off latency inside the brain (simulated time), rate path against spike-event path.

For a few expansion rates (cursor numbers: closing speed / distance) the stimulus switches on at t = 0 and
the time to the FLY state is measured over several seeds, with the spike event off (rate path only, the old
behaviour) and on (`BrainSpec.gf_event_spikes`). Also prints the time of the first GF spike.

Run: uv run python tools/latency.py [--n 15000] [--seeds 8] [--chunk 4]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from neuropest import flywire
from neuropest.brain import FLYWIRE, Brain, FLY


def trial(net, spec, exp, seed, chunk):
    b = Brain(net, seed=seed, spec=spec)
    b.walk_bias = 0.0
    b.set_stimulus(150.0, exp * 150.0, 0.0)
    gf = b.group("GF")
    first_spike = None
    for i in range(int(400 / chunk)):
        before = int(b.engine.counts[gf].sum())
        state = b.advance(chunk)
        if first_spike is None and int(b.engine.counts[gf].sum()) != before or b.rates["GF"] > 0 and first_spike is None:
            first_spike = (i + 1) * chunk
        if state == FLY:
            return (i + 1) * chunk, first_spike
    return None, first_spike


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--chunk", type=float, default=4.0)
    a = ap.parse_args()
    net = flywire.load_cache().prefix(a.n)
    print("expansion /s | take-off after (ms): median [min..max], never | first GF spike (ms)")
    for exp in (3.0, 5.0, 7.0, 10.0, 20.0, 40.0):
        row = []
        for label, spec in (("rate", replace(FLYWIRE, gf_event_spikes=0)), ("event", FLYWIRE)):
            res = [trial(net, spec, exp, s, a.chunk) for s in range(a.seeds)]
            t = [r[0] for r in res if r[0] is not None]
            fs = [r[1] for r in res if r[1] is not None]
            row.append(f"{label}: " + (f"{np.median(t):4.0f} [{min(t):3.0f}..{max(t):3.0f}]" if t else "  -  ")
                       + f", never {len(res) - len(t)}/{len(res)}")
        print(f"{exp:11.1f} | " + " | ".join(row) + f" | first spike {np.median(fs) if fs else float('nan'):.0f}")


if __name__ == "__main__":
    main()
