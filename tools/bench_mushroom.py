"""Per-step cost of the mushroom-body layer: the random model (default) and the real FlyWire cells.

Run: uv run python tools/bench_mushroom.py     (the real-cell rows need data/circuits/mushroom_flywire.npz)
"""
from __future__ import annotations

import time

import numpy as np

from neuropest.mushroom import MushroomBody
from neuropest.paths import MUSHROOM

CUES = np.array([0.5, 0.2, 0.7, 0.1, 0.0])


def per_step_us(mb, reward: float, n: int = 2000) -> float:
    for _ in range(100):
        mb.step(CUES, reward, 0.0, 0.05)
    t = time.perf_counter()
    for _ in range(n):
        mb.step(CUES, reward, 0.0, 0.05)
    return (time.perf_counter() - t) / n * 1e6


def main():
    rows = [("random, 2000 KCs", MushroomBody)]
    if MUSHROOM.exists():
        rows.append(("FlyWire, 5177 KCs", MushroomBody.from_flywire))
    print(f"{'':20s}{'idle us/step':>14s}{'learning us/step':>18s}")
    for label, make in rows:
        t = time.perf_counter()
        mb = make()
        built = (time.perf_counter() - t) * 1e3
        print(f"{label:20s}{per_step_us(mb, 0.0):14.0f}{per_step_us(mb, 1.0):18.0f}   (built in {built:.0f} ms)")


if __name__ == "__main__":
    main()
