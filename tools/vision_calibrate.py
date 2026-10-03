"""Cursor scenes -> the REAL VisionDrive -> projection-neuron rates -> Brain, scanned over eye height and speed.

A fly at the origin faces +x (heading 0); the cursor moves along a path. For every frame (20 ms, the worker's real
vision period; the Brain advances in its real 4 ms chunks) the drive is the same `visual.VisionDrive` the worker
runs, with its halo / height scaling and its `VisionParams` gains, so a number here is what the app does.
(The first version of this tool re-implemented the drive with its own gains and skipped the scaling; its results
did not carry over to the app.)

Prints the outcome (stand / retreat / fly) per scenario, per eye height, per cursor model:
  sphere  the cursor is a disc at eye level (VisionParams.cursor_model, the default; height independent)
  disk    the cursor lies on the screen plane and is seen through the funnel (the first model; height dependent)

Run: uv run python tools/vision_calibrate.py [--tier 15000] [--model sphere|disk|both] [--h 20 40 100 200 300]
         [--speeds 150 400 800 1500] [--grid] [--gain-loom 25 --gain-retreat 18 ...]
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, replace

import numpy as np

from neuropest import flywire
from neuropest.brain import Brain
from neuropest.visual import DISK_PLANE, VisionDrive, VisionParams, cursor_scene

FRAME_S = 0.02               # the worker's vision period (runner.VISION_PERIOD_MS)
CHUNK_MS = 4.0               # the worker's engine chunk on the CPU
EVENT_S = 0.5                # the event of every scenario starts here


def paths(speeds):
    """name -> (cursor position at time t, duration s)."""
    def approach(speed, d0=400.0, stop=50.0, deg=0.0):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return lambda t: (c * max(stop, d0 - speed * max(0.0, t - EVENT_S)), s * max(stop, d0 - speed * max(0.0, t - EVENT_S)))

    out = {"still, ahead, 150 px": (lambda t: (150.0, 0.0), 1.5),
           "still, right, 150 px": (lambda t: (0.0, 150.0), 1.5)}
    for v in speeds:
        out[f"approach {v:g} px/s"] = (approach(v), max(1.0, 0.5 + 350.0 / v + 0.6))
    out["approach 800 px/s from 60 deg right"] = (approach(800.0, deg=60.0), 1.8)
    out["approach 800 px/s from 100 deg left"] = (approach(800.0, deg=-100.0), 1.8)
    out["slide 400 px/s at 150 px"] = (lambda t: (150.0, -400.0 + 400.0 * max(0.0, t - EVENT_S)), 2.0)
    out["recede 400 px/s"] = (lambda t: (50.0 + 400.0 * max(0.0, t - EVENT_S), 0.0), 1.5)
    out["recede after 3 s near"] = (lambda t: (50.0 + 400.0 * max(0.0, t - 3.0), 0.0), 4.0)
    return out


def run_scenario(net, drive, path, duration, height):
    brain = Brain(net)
    brain.walk_bias = 0.0
    drive.reset()
    drive.eye_height = height
    seen, first = Counter(), {}
    peak = dict(GF=0.0, MDN=0.0, steer=0.0, steer_min=0.0)
    sphere = drive.params.cursor_model == "sphere"
    for f in range(int(duration / FRAME_S)):
        t = f * FRAME_S
        now, prev = path(t), path(t - FRAME_S)
        if sphere:
            idx, rates = drive.step_cursor(now, prev, 0.0, 0.0, 0.0, FRAME_S)
        else:
            r = drive.halo_px
            idx, rates = drive.step(cursor_scene(*now, r, drive.params.background),
                                    cursor_scene(*prev, r, drive.params.background), 0.0, 0.0, 0.0, FRAME_S)
        brain.set_vision(idx, rates)
        for _ in range(int(FRAME_S * 1000 / CHUNK_MS)):
            state = brain.advance(CHUNK_MS)
            seen[state] += 1
            if t >= EVENT_S and state not in first:
                first[state] = t - EVENT_S
            peak["GF"] = max(peak["GF"], brain.rates["GF"])
            peak["MDN"] = max(peak["MDN"], brain.rates["MDN"])
            peak["steer"], peak["steer_min"] = max(peak["steer"], brain.steer), min(peak["steer_min"], brain.steer)
    outcome = "fly" if "fly" in first else ("retreat" if "retreat" in first else "stand")
    return outcome, first, seen, peak


TARGET = {"still, ahead, 150 px": "stand", "slide 400 px/s at 150 px": "stand", "recede 400 px/s": "stand",
          "recede after 3 s near": "stand", "approach 150 px/s": "stand", "approach 400 px/s": "retreat",
          "approach 800 px/s": "fly", "approach 1500 px/s": "fly"}


def scan(net, field, params, heights, scen, label=""):
    drive = VisionDrive(net, params, field)
    print(f"\n== {label} (loom {params.gain_loom:g} max {params.max_loom:g}, retreat {params.gain_retreat:g} max "
          f"{params.max_retreat:g}, flee {params.flee_lo:g}-{params.flee_hi:g}, object {params.gain_object:g}) ==")
    names = list(scen)
    print(f"{'h px':>5} | " + " | ".join(f"{n.replace('approach ', 'app ').replace(' px/s', '')[:20]:>20}" for n in names))
    score = 0
    results = {}
    for h in heights:
        cells = []
        for name in names:
            path, dur = scen[name]
            outcome, first, seen, peak = run_scenario(net, drive, path, dur, h)
            at = first.get("fly" if outcome == "fly" else "retreat")
            ok = TARGET.get(name) in (None, outcome)
            score += TARGET.get(name) == outcome
            cells.append(f"{outcome + (f' {at:.2f}s' if at is not None else ''):>16}{'' if ok else ' !!'}"[:20].rjust(20))
            results[(h, name)] = (outcome, at, peak)
        print(f"{h:5.0f} | " + " | ".join(cells), flush=True)
    return score, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=15000)
    ap.add_argument("--model", default="both", choices=["sphere", "disk", "both"])
    ap.add_argument("--h", type=float, nargs="+", default=[20.0, 40.0, 100.0, 200.0, 300.0])
    ap.add_argument("--speeds", type=float, nargs="+", default=[150.0, 400.0, 800.0, 1500.0])
    ap.add_argument("--only", default="")
    ap.add_argument("--grid", action="store_true", help="search gain_loom, gain_retreat and the flee range for the best score")
    for f in fields(VisionParams):
        if f.type in ("float", float):
            ap.add_argument("--" + f.name.replace("_", "-"), type=float, default=None)
    a = ap.parse_args()

    net = flywire.load_cache().prefix(a.tier)
    field = __import__("neuropest.vision", fromlist=["VisualField"]).VisualField.load()
    scen = {n: v for n, v in paths(a.speeds).items() if a.only in n}
    over = {f.name: getattr(a, f.name) for f in fields(VisionParams) if getattr(a, f.name, None) is not None}
    models = ["sphere", "disk"] if a.model == "both" else [a.model]
    d = VisionDrive(net, field=field)
    print(f"tier {net.n} neurons; driven: {len(d.loom_idx)} LPLC2+LC4, {len(d.ret_idx)} LPC1, {len(d.obj_idx)} LC10")
    for model in models:
        base = replace(VisionParams() if model == "sphere" else DISK_PLANE, **over)
        if not a.grid:
            scan(net, field, base, a.h, scen, label=f"cursor model {model}")
            continue
        best = []
        for gl in (15.0, 25.0, 40.0):
            for gr in (6.0, 12.0, 24.0):
                for lo, hi in ((2.5, 4.0), (3.5, 5.0), (4.5, 6.5)):
                    p = replace(base, gain_loom=gl, gain_retreat=gr, flee_lo=lo, flee_hi=hi)
                    score, _ = scan(net, field, p, a.h, scen, label=f"grid {model}")
                    best.append((score, gl, gr, lo, hi))
        best.sort(reverse=True)
        print("best (score, gain_loom, gain_retreat, flee_lo, flee_hi):", best[:5])


if __name__ == "__main__":
    main()
