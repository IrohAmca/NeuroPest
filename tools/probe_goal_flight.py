"""Does a hungry fly get to a food source, and where does a foraging flight land? Closed loop, headless.

The world side is the app's own code (pheromone.PheromoneField.smell, GoalTracker, fly.Fly.update on a PlayArea), the
worker side is what runner._loop does per iteration (set_stimulus when the rounded inputs change, set_goal, the meal
counter, 4 ms chunks), with the brain in this process. One 1920x1040 screen; the fly starts in the middle, one food source
at --dist px in one of 4 directions, the cursor far away (so only the food acts), hunger fixed, the source not respawned.

Per variant it prints: trials that ate and the median time to eat, foraging (long) and escape (short) take-offs, the gap
to the food at each foraging landing (px past the 28 px eating radius; <= 0 = landed on it), wall hits during foraging
flights, and the peak Giant Fiber rate during them (the flight must not use it).

  goal       the code as it is (foraging flight, brain.GOAL_*)
  walk-only  GOAL_URGE_RATE = 0: no foraging flight, the fly has to walk there

FINDINGS #18 (before): flights toward food went through the Giant Fiber and landed 580-775 px from the food.

Run: uv run python tools/probe_goal_flight.py [--tier 15000] [--seeds 6] [--seconds 25] [--dist 300] [--hunger 0.6]
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time

from neuropest import brain as B
from neuropest import flywire
from neuropest.fly import Fly, PlayArea
from neuropest.pheromone import GoalTracker, PheromoneField
from neuropest.states import FLY

SCREEN = (0.0, 0.0, 1920.0, 1040.0)
MARGIN = 24.0               # app: FLY_MARGIN_PX x scale
FRAME_S = 1.0 / 60.0
CHUNKS_PER_FRAME = 4        # 4 x 4 ms ~ one 60 Hz frame of simulated time
RADIUS = 28.0               # eating radius at scale 1
BIAS = 0.65                 # runner.Runner.bias (walking drive)


def trial(net, seed: int, direction: int, dist: float, hunger: float, seconds: float) -> dict:
    random.seed(seed)
    l, t, r, b = SCREEN
    area = PlayArea.from_rect((l + MARGIN, t + MARGIN, r - MARGIN, b - MARGIN), margin=MARGIN)
    cx, cy = (l + r) / 2, (t + b) / 2
    fly = Fly(cx, cy)                                          # random heading (seeded)
    ang = direction * math.pi / 2
    field = PheromoneField(border_margin=30.0, border_strength=0.4)
    food = field.add_source(cx + dist * math.cos(ang), cy + dist * math.sin(ang), radius=220.0)
    goal = GoalTracker()
    brain = B.Brain(net, dt=0.5, seed=seed, learning=True)
    cursor = (-5000.0, -5000.0)
    last_in, fed = None, 0
    out = dict(ate=False, t_eat=None, long=0, short=0, gaps=[], wall=0, gf_peak=0.0)
    prev_state, prev_mode = brain.state, ""
    frames = int(seconds / FRAME_S)
    for k in range(frames):
        state, gear = brain.state, brain.gear
        long_flight = state == FLY and gear == "fly_long"
        steer_ph, drive, repel, delta = field.smell(fly.x, fly.y, fly.heading, area, 1.0)
        at_target = any(math.hypot(s.x - fly.x, s.y - fly.y) < RADIUS for s in field.sources)
        if at_target and field.consume_at(fly.x, fly.y, consume_radius=RADIUS):
            fed += 1
            if not out["ate"]:
                out["ate"], out["t_eat"] = True, k * FRAME_S
        goal.update(field, fly.x, fly.y, area, flying_to_it=long_flight)
        fly.update(FRAME_S, state, cursor, area, brain.steer, mode="long" if long_flight else "",
                   goal=goal.target(RADIUS))
        if long_flight:
            out["wall"] += int(fly.hit_wall)
            out["gf_peak"] = max(out["gf_peak"], brain.rates["GF"])
        gap, bearing, odor = goal.reading(fly.x, fly.y, fly.heading, RADIUS)
        dist_c = math.hypot(cursor[0] - fly.x, cursor[1] - fly.y)
        # the worker: inputs rounded as in runner._loop, applied when they change
        cur = (round(dist_c, 1), 0.0, BIAS, 1.0, round(fly.bearing_of(cursor), 2), 0.0, round(steer_ph, 1),
               round(drive, 2), round(repel, 2), 1.0 if at_target else 0.0, round(hunger, 2), float(fly.hit_wall),
               0.0, round(delta, 4))
        if cur != last_in:
            brain.set_stimulus(*cur)
            last_in = cur
        if fed:
            brain.note_fed(fed)
            fed = 0
        brain.set_goal(gap, bearing, odor)
        for _ in range(CHUNKS_PER_FRAME):
            brain.advance(4.0)
        # take-offs and landings
        if brain.state == FLY and prev_state != FLY:
            out["long" if brain.flight_mode == "long" else "short"] += 1
        if prev_state == FLY and brain.state != FLY and prev_mode == "long":
            out["gaps"].append(math.hypot(food.x - fly.x, food.y - fly.y) - RADIUS)
        prev_state, prev_mode = brain.state, brain.flight_mode
        if out["ate"] and brain.state != FLY:
            break
    return out


def report(name: str, rows: list[dict]) -> None:
    ate = [r for r in rows if r["ate"]]
    gaps = [g for r in rows for g in r["gaps"]]
    t_eat = statistics.median(r["t_eat"] for r in ate) if ate else float("nan")
    gap_txt = (f"median {statistics.median(gaps):6.1f}, max {max(gaps):6.1f}, landed on it {sum(g <= 0 for g in gaps)}"
               f"/{len(gaps)}") if gaps else "-"
    print(f"{name:10s} ate {len(ate):2d}/{len(rows)} (median {t_eat:5.1f} s) | take-offs long {sum(r['long'] for r in rows):3d}"
          f" short {sum(r['short'] for r in rows):3d} | foraging landing gap px: {gap_txt} | wall hits "
          f"{sum(r['wall'] for r in rows)} | GF peak {max(r['gf_peak'] for r in rows):5.1f} Hz")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", type=int, default=15_000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--dist", type=float, default=300.0)
    ap.add_argument("--hunger", type=float, default=0.6)
    ap.add_argument("--variants", default="goal,walk-only")
    a = ap.parse_args()
    net = flywire.load_tier(a.tier)
    rate = B.GOAL_URGE_RATE
    print(f"tier {a.tier}, food {a.dist:.0f} px away, hunger {a.hunger}, {a.seeds} seeds x 4 directions, {a.seconds:.0f} s")
    try:
        for name in a.variants.split(","):
            B.GOAL_URGE_RATE = 0.0 if name == "walk-only" else rate
            t0 = time.perf_counter()
            rows = [trial(net, 1_000_003 * s + d, d, a.dist, a.hunger, a.seconds) for s in range(a.seeds) for d in range(4)]
            report(name, rows)
            print(f"           ({time.perf_counter() - t0:.0f} s)")
    finally:
        B.GOAL_URGE_RATE = rate


if __name__ == "__main__":
    main()
