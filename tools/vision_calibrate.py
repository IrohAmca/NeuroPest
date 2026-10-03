"""Image -> retinotopic features -> projection neuron rates -> behavior, on synthetic scenes.

A fly at the origin faces +x (heading 0). Scenes are drawn on the screen plane. Prints feature strength and
what the Brain does, so gains and thresholds can be tuned against the cursor-based behavior.

Run: uv run python tools/vision_calibrate.py [--tier 15000] [--h 60] [--r 30] [--gain-e 40 --gain-r 20 --gain-o 40]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from neuropest import flywire
from neuropest.brain import Brain
from neuropest.vision import Features, VisualField, sample_scene

DT = 1.0 / 60.0


def disk(cx, cy, radius, dark=0.0, back=0.5):
    def scene(px, py):
        return np.where((px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2, dark, back).astype(np.float32)
    return scene


def make_scenarios(r):
    """name -> (function(t) -> scene, duration s). The event runs from t = 0.5 s."""
    def approach(speed, d0=400.0, y=0.0, stop=50.0):
        return lambda t: disk(max(stop, d0 - speed * max(0.0, t - 0.5)), y, r)

    def slide(speed, d=150.0):
        return lambda t: disk(d, -400 + speed * max(0.0, t - 0.5), r)

    return {
        "still disk, ahead, 150 px": (lambda t: disk(150, 0, r), 1.5),
        "still disk, right, 150 px": (lambda t: disk(0, 150, r), 1.5),
        "still disk, left, 150 px": (lambda t: disk(0, -150, r), 1.5),
        "approach 150 px/s": (approach(150), 2.6),
        "approach 400 px/s": (approach(400), 1.5),
        "approach 800 px/s": (approach(800), 1.2),
        "approach 1500 px/s": (approach(1500), 1.0),
        "slide 400 px/s at 150 px": (slide(400), 2.0),
        "recede 400 px/s": (lambda t: disk(50 + 400 * max(0.0, t - 0.5), 0, r), 1.5),
        "recede after 3 s near": (lambda t: disk(50 + 400 * max(0.0, t - 3.0), 0, r), 4.0),
    }


def run_scenario(net, field, retina, feats, idx, maps, a, fn, dur):
    """Run one scene; returns the strongest behavior reached (stand < retreat < fly), when, and peaks."""
    loom_col, ret_col, lc10_col, n_loom, n_ret = maps
    brain = Brain(net, dt=0.5, seed=1)
    brain.walk_bias = 0.0
    feats.reset()
    seen = Counter()
    first = {}
    peak = dict(E=0.0, GF=0.0, MDN=0.0, steer=0.0, steer_min=0.0, loomHz=0.0)
    for f in range(int(dur / DT)):
        t = f * DT
        lum_now = sample_scene(retina, fn(t), 0.0, 0.0, 0.0, a.h)
        lum_prev = sample_scene(retina, fn(t - DT), 0.0, 0.0, 0.0, a.h)
        ft = feats.update(lum_now, lum_prev, DT)
        e, o, ee = ft["expansion_pooled"], ft["object_pooled"], ft["expansion_eye"]
        retreat = np.minimum(a.gain_r * ee[ret_col], 20.0) * np.clip((a.flee_hi - ee[ret_col]) / (a.flee_hi - a.flee_lo), 0.0, 1.0)
        rates = np.concatenate([np.minimum(a.gain_e * e[loom_col], a.max_hz), retreat, np.minimum(a.gain_o * o[lc10_col], a.max_o)])
        brain.set_vision(idx, rates)
        state = brain.advance(DT * 1000.0)
        seen[state] += 1
        if t >= 0.5 and state not in first:
            first[state] = t - 0.5
        peak["E"] = max(peak["E"], float(e.max()))
        peak["GF"] = max(peak["GF"], brain.rates["GF"])
        peak["MDN"] = max(peak["MDN"], brain.rates["MDN"])
        peak["steer"] = max(peak["steer"], brain.steer)
        peak["steer_min"] = min(peak["steer_min"], brain.steer)
        peak["loomHz"] = max(peak["loomHz"], float(rates[:n_loom].max()))
    outcome = "fly" if "fly" in first else ("retreat" if "retreat" in first else "stand")
    return outcome, first, seen, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=15000)
    ap.add_argument("--h", type=float, default=100.0)
    ap.add_argument("--r", type=float, default=30.0)
    ap.add_argument("--gain-e", type=float, default=25.0, help="Hz of LPLC2/LC4 drive per unit of expansion")
    ap.add_argument("--gain-r", type=float, default=10.0, help="Hz of LPC1 drive per unit of expansion")
    ap.add_argument("--gain-o", type=float, default=100.0, help="Hz of LC10 drive per unit of object feature")
    ap.add_argument("--max-hz", type=float, default=150.0)
    ap.add_argument("--max-o", type=float, default=100.0)
    ap.add_argument("--flee-lo", type=float, default=3.5, help="retreat drive fades out between flee-lo and flee-hi")
    ap.add_argument("--flee-hi", type=float, default=5.0)
    ap.add_argument("--tau-adapt", type=float, default=2.0, help="seconds of the slow baseline (photoreceptor adaptation)")
    ap.add_argument("--only", default="")
    ap.add_argument("--grid", action="store_true", help="search gains for: 150 stand, 400 retreat, 800 and 1500 fly, rest stand")
    a = ap.parse_args()

    net = flywire.load_cache().prefix(a.tier)
    field = VisualField.load()
    retina = field.as_retina()
    feats = Features(field, tau_adapt=a.tau_adapt)
    loom_idx, loom_col = field.neurons(net, ["LPLC2", "LC4"])
    ret_idx, ret_col = field.neurons(net, ["LPC1"])
    lc10_idx, lc10_col = field.neurons(net, ["LC10a", "LC10c-2", "LC10d"])
    idx = np.concatenate([loom_idx, ret_idx, lc10_idx])
    maps = (loom_col, ret_col, lc10_col, len(loom_idx), len(ret_idx))
    print(f"tier {net.n} neurons; driven: {len(loom_idx)} LPLC2+LC4, {len(ret_idx)} LPC1, {len(lc10_idx)} LC10")
    scen = make_scenarios(a.r)
    if a.grid:
        target = {"approach 150 px/s": "stand", "approach 400 px/s": "retreat", "approach 800 px/s": "fly",
                  "approach 1500 px/s": "fly", "slide 400 px/s at 150 px": "stand", "recede 400 px/s": "stand",
                  "recede after 3 s near": "stand",
                  "still disk, ahead, 150 px": "stand"}
        best = []
        for ge in (15.0, 25.0, 40.0):
            for gr in (6.0, 12.0, 24.0):
                for lo, hi in ((2.5, 4.0), (3.5, 5.0), (4.5, 6.5)):
                    a.gain_e, a.gain_r, a.flee_lo, a.flee_hi = ge, gr, lo, hi
                    res = {n: run_scenario(net, field, retina, feats, idx, maps, a, *scen[n])[0] for n in target}
                    score = sum(res[n] == target[n] for n in target)
                    best.append((score, ge, gr, lo, hi, res))
                    print(f"gain_e {ge:4.0f} gain_r {gr:4.0f} flee {lo}-{hi}: score {score}/{len(target)} "
                          + " ".join(f"{n.split()[0][:3]}{n.split()[1][:4]}={res[n][0]}" for n in target), flush=True)
        best.sort(key=lambda x: -x[0])
        print("best:", best[0][:5])
        return
    for name, (fn, dur) in scen.items():
        if a.only and a.only not in name:
            continue
        outcome, first, seen, peak = run_scenario(net, field, retina, feats, idx, maps, a, fn, dur)
        n = sum(seen.values())
        print(f"{name:28s} -> {outcome:7s} | E {peak['E']:5.1f} loom {peak['loomHz']:4.0f} Hz | GF {peak['GF']:4.0f} MDN {peak['MDN']:3.0f} "
              f"steer {peak['steer_min']:4.0f}..{peak['steer']:3.0f} | " + " ".join(f"{k} {100 * v / n:3.0f}%" for k, v in seen.items())
              + "".join(f" | {k} at {v:.2f} s" for k, v in first.items() if k != "stand"), flush=True)


if __name__ == "__main__":
    main()
