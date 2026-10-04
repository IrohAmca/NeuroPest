"""Closed-loop probe: does the fly learn the CURSOR through the visual projection neurons that reach the Kenyon cells?

The real `visual.VisionDrive` (the cells' drive from the image through their connectome receptive fields) feeds the
real `Brain` on a connectome tier, 20 ms vision frames and 4 ms engine chunks as in the worker. The fly stands at the
origin facing +x. Training: the cursor (a dark disc, moving a little, as a hand does) is the source of an attractive
pheromone and the fly feeds at it (reward), then the cursor goes away; 4 trials. Test: the cursor alone, no pheromone, at
100 / 250 / 500 px, plus looming, a static desktop and a moving window, for a trained and a naive fly.

Run: uv run python tools/probe_mb_vision.py [--tier 15000] [--wiring random|flywire] [--trials 4]
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import numpy as np

from neuropest import flywire
from neuropest.brain import Brain
from neuropest.vision import image_scene
from neuropest.visual import VisionDrive

FRAME_S, CHUNK_MS = 0.02, 4.0
FAR = (2000.0, 0.0)


def wiggle(dist: float, amp: float | None = None, hz: float = 1.0):
    """Cursor ahead at `dist` px, moving sideways like a hand (amplitude ~0.6 of the distance by default)."""
    amp = 0.6 * dist if amp is None else amp
    return lambda t: (dist, amp * np.sin(2 * np.pi * hz * t))


def approach(speed: float, d0: float = 400.0, stop: float = 50.0):
    return lambda t: (max(stop, d0 - speed * max(0.0, t - 0.5)), 0.0)


@dataclass
class Trace:
    valence: np.ndarray
    near: np.ndarray
    loom: np.ndarray
    walk_drive: np.ndarray
    lc10: np.ndarray


class Rig:
    def __init__(self, net, wiring: str):
        self.net, self.wiring = net, wiring
        self.drive = VisionDrive(net)
        self.new_fly()

    def new_fly(self):
        self.brain = Brain(self.net, mb_wiring=self.wiring)
        self.brain.walk_bias = 0.0
        self.drive.reset()

    def _lc10(self) -> float:
        b = self.brain
        if b._last_fi is None:
            return 0.0
        m = np.isin(b._last_fi, np.concatenate([b.group("LC10_R"), b.group("LC10_L")]))
        return float(b._last_fr[m].mean()) if m.any() else 0.0

    def phase(self, path, seconds: float, phero: float = 0.0, feed: bool = False, scene=None) -> Trace:
        """`path`: cursor position at time t; `scene(t)`: a grayscale image (static desktop / moving window) or None."""
        b, d = self.brain, self.drive
        b.set_stimulus(1e6, 0.0, 0.0, phero_drive=phero, at_target=1.0 if feed else 0.0)
        rows = []
        for f in range(int(seconds / FRAME_S)):
            t = f * FRAME_S
            if scene is None:
                idx, rates = d.step_cursor(path(t), path(t - FRAME_S), 0.0, 0.0, 0.0, FRAME_S)
            else:
                idx, rates = d.step(image_scene(scene(t), outside=0.5), image_scene(scene(t - FRAME_S), outside=0.5),
                                    800.0, 450.0, 0.0, FRAME_S)
            b.set_vision(idx, rates, d.expansion, (d.mb_cue["cursor_near"], d.mb_cue["looming"]))
            for _ in range(int(FRAME_S * 1000 / CHUNK_MS)):
                b.advance(CHUNK_MS)
            walk = float(b._last_cr.max()) if b._last_cr is not None else 0.0
            rows.append((b.valence, d.mb_cue["cursor_near"], d.mb_cue["looming"], walk, self._lc10()))
        b.clear_vision()
        a = np.array(rows).T
        return Trace(*a)

    def train(self, trials: int):
        for _ in range(trials):
            self.phase(wiggle(100.0), 3.0, phero=0.9, feed=True)           # feeding at the cursor
            self.phase(lambda t: FAR, 6.0)                                  # then it is gone


def report(label: str, tr: Trace, after: float = 1.0):
    k = int(after / FRAME_S)
    seen = tr.near[k:] > 0.1
    print(f"  {label:30s} valence mean {tr.valence[k:].mean():+.2f} min {tr.valence[k:].min():+.2f} max {tr.valence[k:].max():+.2f}"
          f" (while cue>0.1: {tr.valence[k:][seen].mean() if seen.any() else float('nan'):+.2f}) | near mean {tr.near[k:].mean():.2f}"
          f" loom mean {tr.loom[k:].mean():.2f} | walk drive max {tr.walk_drive[k:].max():5.0f} | LC10 {tr.lc10[k:].mean():5.1f} Hz")


def battery(rig: Rig, rng_img: np.ndarray):
    """Every test starts from the same memory (the fly keeps learning inside a test, as it would, but not across them)."""
    snap = copy.deepcopy(rig.brain.mb)

    def test(label, *args, **kw):
        rig.brain.mb = copy.deepcopy(snap)
        rig.brain.valence = 0.0
        report(label, rig.phase(*args, **kw))

    for dist in (100, 250, 500):
        test(f"cursor wiggling at {dist} px", wiggle(dist), 4.0)
    test("cursor far (2000 px)", lambda t: FAR, 3.0)
    test("looming 800 px/s 400->50", approach(800.0), 3.0)

    def window(t):
        im = rng_img.copy()
        x = int(200 + 300 * max(t, 0.0))
        im[300:700, x:x + 400] = 0.05
        return im
    test("static desktop", lambda t: FAR, 3.0, scene=lambda t: rng_img)
    test("window dragged 300 px/s", lambda t: FAR, 3.0, scene=window)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", type=int, default=15000)
    ap.add_argument("--wiring", default="random", choices=("random", "flywire"))
    ap.add_argument("--trials", type=int, default=4)
    a = ap.parse_args()
    net = flywire.load_tier(a.tier)
    img = (np.random.default_rng(0).random((900, 1600)) * 0.6 + 0.2).astype(np.float32)
    rig = Rig(net, a.wiring)
    print(f"tier {a.tier}, mushroom body: {rig.brain.mb.kind}; cursor VPNs {len(rig.drive.mb_near_col)}, looming VPNs "
          f"{len(rig.drive.mb_loom_col)}")
    print("\nnaive fly")
    battery(rig, img)
    rig.new_fly()
    print(f"\nafter {a.trials} feedings at the cursor")
    rig.train(a.trials)
    battery(rig, img)


if __name__ == "__main__":
    main()
