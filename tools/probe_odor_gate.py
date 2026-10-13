"""How strongly may a food odor silence the startle? Calibration of brain.ODOR_GATE_FULL against the old hard gate.

The attractive cursor (default cursor mode: its own scent, a Gaussian plume of radius 200 px) comes at a standing fly
and stops on it, like a user bringing the "food" to the pet. For hunger levels and approach speeds this prints, per gate
variant, the share of time in RETREAT / FLY / FREEZE with the cursor distance at which each state first appeared, and how
many 4 ms chunks of RETREAT or FLY fall within 150 px of the cursor (the lure zone).

  OLD    the former rule: no looming / retreat / freeze input at all once odor x hunger > 0.05
  E=x    brain.odor_gate with ODOR_GATE_FULL = x: the reflexes scale with 1 - (odor x hunger) / x

A good gate keeps the startle for a fast approach from afar (outside the plume), never starts a retreat or take-off at
the lure (the cursor is food there), and is not blind across the plume edge the way the old cliff was. Measured on the
15,000-neuron tier (2026-10-14): E = 0.15 and 0.2 start no startle inside the lure zone at any hunger (as OLD); E = 0.3
(hunger 0.15) and E = 0.5 (hunger 0.15 and 0.35) do: a retreat starting at ~80-100 px or a take-off at the cursor
itself. ODOR_GATE_FULL = 0.2 was chosen: the
fully blind radius shrinks from ~340 px to ~250 px at hunger 1 and from ~270 px to ~150 px at hunger 0.35 (food plume
r = 220 px), with partial reactivity in between.

Run: uv run python tools/probe_odor_gate.py [--tier N] [--gates 0.15,0.2,0.3,0.5]
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

from neuropest import brain as B
from neuropest import flywire

R = 200.0          # the cursor's own plume (pheromone.PheromoneField.update_cursor)
CHUNK = 4.0        # ms, the CPU worker's chunk
LURE_PX = 150.0


def plume(d: float) -> float:
    return math.exp(-(d * d) / (0.8 * R * R)) if d <= 2.5 * R else 0.0      # pheromone.PheromoneSource.concentration_at


def old_gate(strength: float) -> float:
    return 0.0 if strength > 0.05 else 1.0


def approach(net, hunger: float, speed: float, start=450.0, stop_at=30.0, hold_s=1.0, seed=3):
    b = B.Brain(net, dt=0.5, seed=seed)
    b.set_stimulus(start, 0.0, phero_drive=plume(start), hunger=hunger)
    for _ in range(100):                                   # settle for 0.4 s with the cursor standing far away
        b.advance(CHUNK)
    d, hold = start, 0.0
    states, first, lure = Counter(), {}, Counter()
    while hold < hold_s:
        moving = d > stop_at
        if moving:
            d = max(stop_at, d - speed * CHUNK / 1000.0)
        else:
            hold += CHUNK / 1000.0
        b.set_stimulus(d, speed if moving else 0.0, phero_drive=plume(d), hunger=hunger,
                       at_target=1.0 if d <= 28.0 else 0.0)
        st = b.advance(CHUNK)
        states[st] += 1
        first.setdefault(st, round(d))
        if d < LURE_PX:
            lure[st] += 1
    return states, first, lure


def cell(states, first, lure) -> str:
    tot = sum(states.values())
    shown = " ".join(f"{k[:3]}{100 * states[k] / tot:.0f}%@{first[k]}" for k in ("retreat", "fly", "freeze") if k in states)
    return f"{shown or '-':<28} lure:{lure['retreat'] + lure['fly']:>3}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", type=int, default=15_000)
    ap.add_argument("--gates", default="0.15,0.2,0.3,0.5")
    a = ap.parse_args()
    net = flywire.load_tier(a.tier)
    graded, full = B.odor_gate, B.ODOR_GATE_FULL
    variants = [("OLD", None)] + [(f"E={g}", float(g)) for g in a.gates.split(",")]
    print(f"{'hunger':>6} {'px/s':>5} | " + " | ".join(f"{name:<38}" for name, _ in variants))
    try:
        for hunger in (0.15, 0.35, 0.7, 1.0):
            for speed in (300.0, 900.0, 2500.0):
                cells = []
                for _, e in variants:
                    B.odor_gate = old_gate if e is None else graded
                    B.ODOR_GATE_FULL = full if e is None else e
                    cells.append(cell(*approach(net, hunger, speed)))
                print(f"{hunger:>6.2f} {speed:>5.0f} | " + " | ".join(f"{c:<38}" for c in cells))
    finally:
        B.odor_gate, B.ODOR_GATE_FULL = graded, full


if __name__ == "__main__":
    main()
