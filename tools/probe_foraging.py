"""Foraging in the app's world, closed loop and headless: how often and how soon does a fly find and eat food, how much
of its time does it walk, and what does it cost?

The world is the app's own code (PheromoneField with FOOD_COUNT sized foods, its eating loop: FEED + eat_at at app.EAT_RATE
x feed level, MetabolicState, the meal spot for the local search, Fly.update on a PlayArea); the worker side is what
runner._loop does per iteration (set_stimulus when the rounded inputs change, the meal counter, the meal spot, 4 ms
chunks). One 1920x1040 screen, the fly in the middle, the cursor far away (only the food acts), --seconds per trial.

  A      the code as it is: spontaneous bouts, the odor drives walking, a search when the plume is lost (brain.ACTIVITY_REF)
  old    an approximation of the architecture before it: always in a walking bout (the constant drive), no odor-loss
         search

Per variant: trials that ate, median time to the first meal, meals per trial, share of time walking / feeding / in a
search, CPU ms per simulated second (the brain only).

Run: uv run python tools/probe_foraging.py [--tier 15000] [--seeds 6] [--seconds 90] [--hunger 0.7] [--variants A,old]
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time

from neuropest import brain as B
from neuropest import flywire
from neuropest.app import EAT_RATE
from neuropest.fly import Fly, PlayArea
from neuropest.metabolism import MetabolicState
from neuropest.pheromone import FOOD_COUNT, PheromoneField
from neuropest.states import FEED, FLY, WALK

SCREEN = (0.0, 0.0, 1920.0, 1040.0)
MARGIN = 24.0
FRAME_S = 1.0 / 60.0
CHUNKS_PER_FRAME = 4
RADIUS = 28.0
BIAS = 0.65


def trial(net, seed: int, hunger: float, seconds: float) -> dict:
    random.seed(seed)
    l, t, r, b = SCREEN
    area = PlayArea.from_rect((l + MARGIN, t + MARGIN, r - MARGIN, b - MARGIN), margin=MARGIN)
    fly = Fly((l + r) / 2, (t + b) / 2)
    field = PheromoneField(border_margin=30.0, border_strength=0.4)
    field.spawn_random_sources(area.screens, count=FOOD_COUNT)
    meta = MetabolicState(energy=1.0 - hunger)
    brain = B.Brain(net, dt=0.5, seed=seed, learning=False)
    cursor = (-5000.0, -5000.0)
    last_in, was_feeding, spot = None, False, None
    out = dict(t_eat=None, meals=0, walk=0, feed=0, search=0, frames=0, cpu=0.0)
    for k in range(int(seconds / FRAME_S)):
        state = brain.state
        long_flight = state == FLY and brain.gear == "fly_long"
        steer_ph, drive, repel = field.smell(fly.x, fly.y, fly.heading, area, 1.0)
        src = field.food_at(fly.x, fly.y, RADIUS)
        feeding = state == FEED and src is not None
        if feeding:
            meta.eat(field.eat_at(fly.x, fly.y, EAT_RATE * brain.feed_level * FRAME_S, radius=RADIUS,
                                  screen_boxes=area.screens))
            spot = (fly.x, fly.y)
            if not was_feeding:
                brain.note_fed(1)
                out["meals"] += 1
                out["t_eat"] = out["t_eat"] if out["t_eat"] is not None else k * FRAME_S
        was_feeding = feeding
        meta.update(FRAME_S, state)
        fly.update(FRAME_S, state, cursor, area, brain.steer, mode="long" if long_flight else "")
        dist_c = math.hypot(cursor[0] - fly.x, cursor[1] - fly.y)
        h = meta.hunger
        cur = (round(dist_c, 1), 0.0, BIAS, 1.0, round(fly.bearing_of(cursor), 2), 0.0, round(steer_ph, 1),
               round(drive, 2), round(repel, 2), 1.0 if src is not None else 0.0, round(h, 2), float(fly.hit_wall),
               1.0 if src is not None else 0.0)
        if cur != last_in:
            brain.set_stimulus(*cur)
            last_in = cur
        if spot is not None:
            brain.set_meal_spot(math.hypot(spot[0] - fly.x, spot[1] - fly.y), fly.bearing_of(spot))
        c0 = time.process_time()
        for _ in range(CHUNKS_PER_FRAME):
            brain.advance(4.0)
        out["cpu"] += time.process_time() - c0
        out["frames"] += 1
        out["walk"] += brain.state == WALK
        out["feed"] += brain.state == FEED
        out["search"] += brain._search_ms > 0.0
    return out


def report(name: str, rows: list[dict], seconds: float) -> None:
    ate = [r for r in rows if r["t_eat"] is not None]
    t_eat = statistics.median(r["t_eat"] for r in ate) if ate else float("nan")
    frames = sum(r["frames"] for r in rows)
    share = lambda k: 100.0 * sum(r[k] for r in rows) / frames  # noqa: E731
    cpu = 1000.0 * sum(r["cpu"] for r in rows) / (frames * FRAME_S)
    print(f"{name:4s} ate {len(ate):2d}/{len(rows)} (first meal median {t_eat:5.1f} s) | meals/trial "
          f"{statistics.mean(r['meals'] for r in rows):4.1f} | walking {share('walk'):4.0f}% feeding {share('feed'):4.0f}% "
          f"searching {share('search'):4.0f}% | brain CPU {cpu:5.0f} ms per simulated s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", type=int, default=15_000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--hunger", type=float, nargs="+", default=[0.7, 0.1])
    ap.add_argument("--variants", default="A,old")
    a = ap.parse_args()
    net = flywire.load_tier(a.tier)
    saved = (B.BOUT_WALK_S, B.BOUT_REST_S, B.ODOR_LOSS_MIN)
    print(f"tier {a.tier}, {FOOD_COUNT} foods, {a.seeds} seeds x {a.seconds:.0f} s")
    try:
        for hunger in a.hunger:
            print(f"== starting hunger {hunger}")
            for name in a.variants.split(","):
                if name == "old":                       # always in a bout, no odor-loss search
                    B.BOUT_WALK_S, B.BOUT_REST_S, B.ODOR_LOSS_MIN = (1e6, 1e6), (1e-6, 1e-6), 1e9
                else:
                    B.BOUT_WALK_S, B.BOUT_REST_S, B.ODOR_LOSS_MIN = saved
                rows = [trial(net, 1_000_003 * s + 7, hunger, a.seconds) for s in range(a.seeds)]
                report(name, rows, a.seconds)
    finally:
        B.BOUT_WALK_S, B.BOUT_REST_S, B.ODOR_LOSS_MIN = saved


if __name__ == "__main__":
    main()
