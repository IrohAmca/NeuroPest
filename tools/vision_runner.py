"""The real worker process with the visual input on, driven by scripted cursor motions.

The fly stands still at the middle of a screen and faces +x; the cursor (a dark disk in its view) moves along a
path. Prints what the fly does and what the worker costs, so the visual input can be judged without the GUI.

Run: uv run python tools/vision_runner.py [--n 15000] [--backend cpu|gpu] [--h 100]
"""
from __future__ import annotations

import argparse
import time

from neuropest.runner import EngineConfig, Runner

FLY = (1000.0, 700.0, 0.0)             # x, y, heading


def cursor_path(kind: str, t: float):
    """Cursor position (screen px) t seconds into a scenario; the event starts at t = 0.5 s."""
    s = max(0.0, t - 0.5)
    if kind == "far":
        return 1000.0 + 900.0, 700.0
    if kind == "approach slow":
        return max(FLY[0] + 50.0, FLY[0] + 400.0 - 150.0 * s), FLY[1]
    if kind == "approach medium":
        return max(FLY[0] + 50.0, FLY[0] + 400.0 - 400.0 * s), FLY[1]
    if kind == "approach fast":
        return max(FLY[0] + 50.0, FLY[0] + 400.0 - 1500.0 * s), FLY[1]
    if kind == "slide":
        return FLY[0] + 150.0, FLY[1] - 400.0 + 400.0 * s
    if kind == "recede":
        return FLY[0] + 50.0 + 400.0 * s, FLY[1]
    raise ValueError(kind)


def run(r: Runner, kind: str, seconds: float):
    seen, first, peak_gf, peak_mdn, peak_walk, steer = {}, {}, 0.0, 0.0, 0.0, [0.0, 0.0]
    t0 = time.perf_counter()
    while (t := time.perf_counter() - t0) < seconds:
        cx, cy = cursor_path(kind, t)
        d = ((cx - FLY[0]) ** 2 + (cy - FLY[1]) ** 2) ** 0.5
        r.send(d, 0.0, 0.0, FLY, (cx, cy))
        st = r.stats()
        seen[r.state] = seen.get(r.state, 0) + 1
        if t >= 0.5:
            first.setdefault(r.state, t - 0.5)
        peak_gf, peak_mdn, peak_walk = max(peak_gf, st["gf"]), max(peak_mdn, st["mdn"]), max(peak_walk, st["walk"])
        steer = [min(steer[0], st["steer"]), max(steer[1], st["steer"])]
        time.sleep(0.016)
    n = sum(seen.values())
    print(f"{kind:16s} | " + " ".join(f"{k} {100 * v / n:3.0f}%" for k, v in seen.items())
          + f" | GF {peak_gf:4.0f} MDN {peak_mdn:3.0f} WALK {peak_walk:3.0f} steer {steer[0]:+4.0f}..{steer[1]:+3.0f}"
          + "".join(f" | {k} at {v:.2f} s" for k, v in first.items() if k != "stand"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--h", type=float, default=100.0, help="eye height, px")
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    r = Runner(EngineConfig("flywire", a.n, a.dt, backend=a.backend))
    try:
        t0 = time.time()
        while not r.ready:
            if r.failed or time.time() - t0 > 180:
                raise SystemExit("worker did not start")
            time.sleep(0.1)
        r.bias = 0.0
        r.eye_height = a.h
        r.vision = True
        r.send(1e6, 0.0, 0.0, FLY, cursor_path("far", 0))
        time.sleep(1.5)
        print(f"vision flag in the worker: {r.stats()['vision']:+.0f} (1 = on, -1 = unusable)")
        for kind in ("far", "approach slow", "approach medium", "approach fast", "slide", "recede"):
            if a.only and a.only not in kind:
                continue
            r.vision = False                                    # off and on again: the eye starts every scenario fresh
            r.send(1e6, 0.0, 0.0, FLY, cursor_path("far", 0))
            time.sleep(2.0)                                     # let the previous event fade in the brain
            x0, y0 = cursor_path(kind, 0.0)                     # the scene is already in place when the eye opens
            r.send(1e6, 0.0, 0.0, FLY, (x0, y0))
            r.vision = True
            r.send(1e6, 0.0, 0.0, FLY, (x0, y0))
            time.sleep(0.8)
            run(r, kind, a.seconds)
        st = r.stats()
        print(f"worker: x{st['rt']:.1f} real time, cpu {100 * st['cpu']:.0f}% of one core, lag {st['lag_ms']:.0f} ms, "
              f"{st['n']:,} neurons, {st['spikes']:,.0f} spikes/s")
    finally:
        r.stop()


if __name__ == "__main__":
    main()
