"""Foraging flight (FINDINGS #18): a hungry fly flies to a food source it is not reaching on foot, without the Giant
Fiber, and lands on it. Decision (brain.Brain._decode), flight physics (fly.Fly.update), goal choice (pheromone)."""
import math
import random

import numpy as np
import pytest

from neuropest.brain import GOAL_LOST_MS, GOAL_MAX_FLIGHT_MS, Brain
from neuropest.fly import Fly, PlayArea
from neuropest.mushroom import MushroomBody
from neuropest.pheromone import GoalTracker, PheromoneField
from neuropest.states import FLY, WALK
from neuropest.toy_circuit import build


def _forager(hunger=0.8, learning=False, **stim):
    b = Brain(build(146, 1), learning=learning)
    b.set_stimulus(900, 0, 0.0, phero_drive=0.3, hunger=hunger, **stim)
    return b


def _until_flight(b, gap=300.0, odor=0.5, ms=4000.0):
    b.set_goal(gap, 0.0, odor)
    for _ in range(int(ms / 4)):
        if b.advance(4.0) == FLY:
            return True
    return False


def test_a_foraging_flight_takes_off_without_the_giant_fiber():
    b = _forager()
    gf_peak = 0.0
    assert _until_flight(b)
    for _ in range(50):
        b.advance(4.0)
        gf_peak = max(gf_peak, b.rates["GF"])
    assert b.state == FLY and b.flight_mode == "long" and b.gear == "fly_long"
    assert gf_peak < 3.0 and b.arousal == 0.0               # no Giant Fiber, no defensive surge


@pytest.mark.parametrize("case", ["within_reach", "sated", "no_odor", "alarm", "afraid", "walking_there"])
def test_no_foraging_flight_when_walking_works_or_something_says_no(case):
    b = _forager(hunger=0.2 if case == "sated" else 0.8, phero_repel=0.5 if case == "alarm" else 0.0)
    gap = 100.0 if case == "within_reach" else 300.0
    odor = 0.0 if case == "no_odor" else 0.5
    for _ in range(1000):                                   # 4 s
        if case == "afraid":
            b.valence = -0.5
        if case == "walking_there":
            gap -= 70.0 * 0.004                             # closing in at walking speed
            b.state = WALK
        b.set_goal(gap, 0.0, odor)
        assert b.advance(4.0) != FLY, case


def test_a_foraging_flight_lands_on_the_meal_counter_without_an_at_target_pulse():
    b = _forager()
    assert _until_flight(b)
    for _ in range(50):
        b.advance(4.0)
    assert b.state == FLY
    b.note_fed(1)
    b.advance(4.0)
    assert b.state != FLY and b._goal_refractory_ms > 7000.0       # and it does not hop off again at once
    b.set_goal(300.0, 0.0, 0.5)
    assert FLY not in {b.advance(4.0) for _ in range(1000)}


def test_a_foraging_flight_lands_when_the_goal_is_lost_or_after_its_time_cap():
    b = _forager()
    assert _until_flight(b)
    b.set_goal(1e6, 0.0, 0.0)                              # eaten by someone else / gone
    seen = [b.advance(4.0) for _ in range(int(GOAL_LOST_MS / 4) + 10)]
    assert seen[0] == FLY and seen[-1] != FLY
    c = _forager()
    assert _until_flight(c)
    c.set_goal(300.0, 0.0, 0.5)                             # never reached
    seen = [c.advance(4.0) for _ in range(int(GOAL_MAX_FLIGHT_MS / 4) + 10)]
    assert seen[-1] != FLY


def test_an_attack_during_a_foraging_flight_turns_it_into_an_escape():
    b = _forager()
    assert _until_flight(b)
    b.rates["GF"] = 50.0                                    # the Giant Fiber fires on the way
    b._decode(4.0)
    assert b.state == FLY and b.flight_mode == "short" and b.arousal >= 0.5


def test_a_giant_fiber_take_off_is_an_escape_even_with_a_strong_urge():
    b = _forager()
    b.flight_urge = 1.0
    b.rates["GF"] = 50.0
    b._decode(4.0)
    assert b.state == FLY and b.flight_mode == "short"


def test_no_fear_is_learned_from_a_wall_hit_during_a_foraging_flight():
    b = _forager(learning=True)
    assert _until_flight(b)
    b.wall_bump = 1.0
    for _ in range(20):
        b.advance(4.0)
    assert b._punish == 0.0


# ------------------------------------------------------------------------------------------- flight physics (fly.py)
AREA = PlayArea.from_rect((24.0, 24.0, 1896.0, 1016.0), margin=24.0)
DT = 1.0 / 60.0


def _fly_to(f, goal, seconds=4.0):
    for _ in range(int(seconds / DT)):
        f.update(DT, FLY, (0.0, 0.0), AREA, 0.0, mode="long", goal=goal)
        assert not f.hit_wall
        if math.hypot(goal[0] - f.x, goal[1] - f.y) <= goal[2]:
            return True
    return False


def test_a_foraging_flight_reaches_food_in_every_direction_without_touching_a_wall():
    random.seed(1)
    for k in range(12):
        f = Fly(960.0, 520.0)
        a = 2 * math.pi * k / 12
        assert _fly_to(f, (960.0 + 350.0 * math.cos(a), 520.0 + 350.0 * math.sin(a), 28.0)), k


def test_a_goal_close_to_the_side_is_reached_without_orbiting_it():
    f = Fly(960.0, 520.0)
    f.heading = 0.0
    assert _fly_to(f, (960.0, 560.0, 28.0), seconds=2.0)    # 40 px to the right of the heading


def test_without_a_goal_a_foraging_fly_hovers_and_an_escape_keeps_its_speed():
    f = Fly(960.0, 520.0)
    f.update(DT, FLY, (0.0, 0.0), AREA, 0.0, mode="long", goal=None)
    assert (f.x, f.y) == (960.0, 520.0)
    e = Fly(960.0, 520.0)
    x0, y0 = e.x, e.y
    e.update(DT, FLY, (0.0, 0.0), AREA, 0.0)
    assert math.hypot(e.x - x0, e.y - y0) == pytest.approx(420.0 * DT, rel=1e-6)


# ------------------------------------------------------------------------------------------- goal choice (pheromone)
class _Box:
    def __init__(self, l, t, r, b):
        self.raw_l, self.raw_t, self.raw_r, self.raw_b = l, t, r, b

    def contains_point(self, x, y):
        return self.raw_l <= x <= self.raw_r and self.raw_t <= y <= self.raw_b


def test_the_goal_is_the_strongest_source_on_the_flys_screen_and_never_the_cursor():
    field = PheromoneField()
    near_other_screen = field.add_source(1950.0, 500.0)    # closest, but on the next monitor
    mine = field.add_source(1600.0, 500.0)
    field.update_cursor(1890.0, 500.0, kind="attract")     # the cursor lure, closer still
    src, odor = field.goal_for(1880.0, 500.0, _Box(0, 0, 1920, 1080))
    assert src is mine and odor > 0.0
    assert field.goal_for(1880.0, 500.0)[0] is near_other_screen    # without a screen: any source


def test_the_goal_is_kept_during_the_flight_and_dropped_once_eaten():
    field = PheromoneField()
    a = field.add_source(500.0, 500.0)
    g = GoalTracker()
    assert g.update(field, 300.0, 500.0) is a
    b = field.add_source(320.0, 500.0)                     # a nearer one appears during the flight
    assert g.update(field, 300.0, 500.0, flying_to_it=True) is a
    gap, bearing, odor = g.reading(300.0, 500.0, 0.0, 28.0)
    assert gap == pytest.approx(200.0 - 28.0) and bearing == pytest.approx(0.0) and odor > 0.0
    field.consume_at(500.0, 500.0, consume_radius=28.0)
    assert g.update(field, 480.0, 500.0, flying_to_it=True) is None and g.reading(0, 0, 0, 28)[0] == 1e6
    assert g.update(field, 300.0, 500.0) is b              # landed: chosen again


# ------------------------------------------------------------------------------------------- the lab's plasticity pause
def test_a_paused_mushroom_body_reads_out_but_learns_nothing():
    mb = MushroomBody()
    cue = np.array([0.9, 0.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        mb.step(cue, 1.0, 0.0, 0.05, plastic=False)        # rewarded, but plasticity is paused
    assert not mb.dirty and mb.read(cue) == 0.0 and mb.pam > 0.0
    for _ in range(20):
        mb.step(cue, 1.0, 0.0, 0.05)
    assert mb.dirty and mb.read(cue) > 0.1
