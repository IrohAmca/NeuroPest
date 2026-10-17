"""Tests for the pheromone field, bilateral antenna sampling, and tropotaxis."""
import math

from neuropest.fly import ScreenBox, PlayArea
from neuropest.pheromone import PheromoneField, PheromoneSource


def test_pheromone_source_concentration():
    src = PheromoneSource(x=100.0, y=100.0, kind="attract", radius=200.0, strength=1.0)
    # Peak at center
    assert math.isclose(src.concentration_at(100.0, 100.0), 1.0, rel_tol=1e-3)
    # Drops with distance
    c_near = src.concentration_at(150.0, 100.0)
    c_far = src.concentration_at(300.0, 100.0)
    assert 0.0 < c_far < c_near < 1.0


def test_bilateral_antenna_tropotaxis():
    field = PheromoneField(border_margin=18.0)
    # Place an attractive food source to the right of a fly facing north (heading = -pi/2)
    # Fly at (200, 200), source at (300, 200) -> right side of fly
    field.add_source(300.0, 200.0, kind="attract", radius=200.0, strength=1.0)

    # Heading pointing up (North: -pi/2 in screen coordinates)
    heading_north = -math.pi / 2.0
    ant = field.sample_antennae(200.0, 200.0, heading_north)

    # Right antenna should sense more than left antenna
    assert ant["ar_attr"] > ant["al_attr"]
    assert ant["delta_attr"] > 0.0


def test_slim_border_repulsion():
    # Screen box from 0 to 1000
    scr = ScreenBox(0, "scr", 0, 0, 1000, 1000, 0, 0, 1000, 1000)
    area = PlayArea([scr])
    field = PheromoneField(border_margin=20.0, border_strength=2.0)

    # Point in the middle: zero border repulsion
    _, rep_center = field.sample_point(500.0, 500.0, play_area=area)
    assert rep_center == 0.0

    # Point very close to border (e.g., x=5 px): strong border repulsion
    _, rep_edge = field.sample_point(5.0, 500.0, play_area=area)
    assert rep_edge > 0.5


def test_spawn_random_sources():
    scr = ScreenBox(0, "scr", 0, 0, 1920, 1080, 0, 0, 1920, 1080)
    field = PheromoneField()
    field.spawn_random_sources([scr], count=6)

    assert len(field.sources) == 6
    # All desktop food sources are positive/attractive
    for s in field.sources:
        assert s.kind == "attract"


def test_consume_food_source():
    scr = ScreenBox(0, "scr", 0, 0, 1920, 1080, 0, 0, 1920, 1080)
    field = PheromoneField()
    field.add_source(200.0, 200.0)
    assert len(field.sources) == 1

    # Far away -> not consumed
    consumed = field.consume_at(500.0, 500.0, consume_radius=30.0, screen_boxes=[scr])
    assert not consumed
    assert len(field.sources) == 1

    # At food location -> consumed, disappears, and new one spawns
    consumed = field.consume_at(205.0, 205.0, consume_radius=30.0, screen_boxes=[scr])
    assert consumed
    # Re-spawned elsewhere so count stays replenished
    assert len(field.sources) == 1
    assert not (field.sources[0].x == 200.0 and field.sources[0].y == 200.0)



def test_dynamic_landing_and_takeoff():
    from neuropest.brain import Brain
    from neuropest.toy_circuit import build
    from neuropest.states import FLY, STAND, WALK

    net = build(146, seed=42)
    brain = Brain(net)

    # 1. High alarm repel triggers flight
    brain.set_stimulus(dist=1000.0, closing_speed=0.0, phero_repel=0.95)
    for _ in range(5):  # ~100 ms for smoothed rate to cross threshold
        brain.advance(20.0)
    assert brain.state == FLY

    # 2. While threat is active, fly stays airborne past old 300 ms boundary
    for _ in range(25):  # 25 * 20ms = 500ms
        brain.set_stimulus(dist=100.0, closing_speed=500.0, phero_repel=0.9)
        brain.advance(20.0)
    assert brain.state == FLY  # Still flying, not forced down at 300 ms!

    # 3. Arrived at target: fly lands promptly
    brain.set_stimulus(dist=1000.0, closing_speed=0.0, phero_repel=0.0, at_target=1.0)
    for _ in range(6):  # ~120 ms to let EMA rate settle
        brain.advance(20.0)
    assert brain.state != FLY  # Successfully landed!


def test_reward_plus_and_feeding_effect():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication
    from neuropest.render import RewardEffect, draw_reward_plus, draw_fly

    _ = QApplication.instance() or QApplication([])

    # 1. Test RewardEffect lifecycle
    eff = RewardEffect(100.0, 100.0, duration=0.8)
    assert eff.progress == 0.0
    alive = eff.update(0.4)
    assert alive is True
    assert math.isclose(eff.progress, 0.5, rel_tol=1e-2)
    alive = eff.update(0.5)
    assert alive is False
    assert eff.progress == 1.0

    # 2. Test rendering draw_reward_plus onto QImage
    img = QImage(200, 200, QImage.Format_ARGB32)
    p = QPainter(img)
    draw_reward_plus(p, 100.0, 100.0, progress=0.2, scale=1.0)
    draw_reward_plus(p, 100.0, 100.0, progress=0.8, scale=1.0)

    # 3. Test draw_fly with feed_glow
    draw_fly(p, 100.0, 100.0, 0.0, "walk", 0.5, scale=1.0, feed_glow=0.8)
    p.end()


def test_positive_cursor_approach_does_not_flee():
    from neuropest.brain import Brain
    from neuropest.toy_circuit import build
    from neuropest.fly import Fly

    net = build(146, seed=42)
    brain = Brain(net)

    # 1. A strong food odor (phero_drive x hunger >= brain.ODOR_GATE_FULL; hunger defaults to 1):
    # Approaching with high closing speed must NOT trigger retreat or loom (a weaker odor only weakens them)
    brain.set_stimulus(dist=80.0, closing_speed=400.0, phero_drive=0.8, phero_steer=15.0)
    assert brain._loom_in == 0.0
    for _ in range(5):
        brain.advance(20.0)
    # Fly should NOT flee or retreat into MDN panic
    assert brain.rates["MDN"] < 4.0
    assert brain.rates["GF"] < 10.0

    # 2. Test physical fly in flight: when flying near cursor, does NOT force 180-degree escape vector
    fly = Fly(100.0, 100.0)
    fly.heading = 0.0  # facing right (+x)
    cursor = (150.0, 100.0)  # cursor is directly ahead (+x)
    # Update fly in FLY state with steer=0
    fly.update(0.1, "fly", cursor, (0, 0, 1000, 1000), steer=0.0)
    # Heading should remain facing towards cursor (near 0.0), NOT flipped 180 degrees to math.pi!
    assert abs(fly.heading) < 0.2


def test_corner_sampling_zero_division_guard():
    scr = ScreenBox(0, "scr", 0, 0, 1920, 1080, 0, 0, 1920, 1080, margin=0.0)
    area = PlayArea([scr])
    # When border_margin is 0.0, sampling at (0, 0) must not throw ZeroDivisionError
    field = PheromoneField(border_margin=0.0, border_strength=0.0)
    res = field.sample_antennae(0.0, 0.0, 0.0, play_area=area)
    assert res["al_rep"] == 0.0 and res["ar_rep"] == 0.0






# ------------------------------------------------------------------------------ food odor and the startle reflexes
def test_odor_gate_falls_smoothly_from_full_sensitivity_to_none():
    import numpy as np

    from neuropest.brain import ODOR_GATE_FULL, odor_gate

    assert odor_gate(0.0) == 1.0
    assert 0.5 < odor_gate(0.05) < 1.0                   # the old cliff (0.05): a faint plume edge costs a part, not all
    values = [odor_gate(x) for x in np.linspace(0.0, 1.5, 31)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert odor_gate(ODOR_GATE_FULL) == 0.0 and odor_gate(5.0) == 0.0
    assert odor_gate(0.2 * 1.0) == 0.0                   # the cursor's own lure at full hunger: silenced
    assert 0.0 < odor_gate(0.15 * 0.35) < 1.0            # the plume edge at the default start hunger: partly reactive


def test_food_odor_weakens_the_looming_input_in_proportion_to_its_strength():
    import pytest

    from neuropest.brain import Brain, odor_gate
    from neuropest.toy_circuit import build

    brain = Brain(build(146, seed=42))
    brain.set_stimulus(dist=100.0, closing_speed=400.0)                       # 4 /s, no odor
    assert brain._loom_in == pytest.approx(4.0)
    for phero, hunger in ((0.05, 1.0), (0.1, 1.0), (0.2, 0.5), (0.3, 0.35)):
        brain.set_stimulus(dist=100.0, closing_speed=400.0, phero_drive=phero, hunger=hunger)
        assert brain._loom_in == pytest.approx(4.0 * odor_gate(phero * hunger))
    brain.set_stimulus(dist=100.0, closing_speed=400.0, phero_drive=0.5, hunger=1.0)   # strong odor: fully calm
    assert brain._loom_in == 0.0

    # the looming drive on the cells follows the same factor (toy: loom_gain Hz per 1/s)
    brain.set_stimulus(dist=100.0, closing_speed=400.0, phero_drive=0.1, hunger=1.0)
    assert 0.0 < odor_gate(0.1) < 1.0
    loom_cells = brain._cursor_fr[brain._cursor_slices[0]]
    assert len(loom_cells) and loom_cells == pytest.approx(brain.spec.loom_gain * 4.0 * odor_gate(0.1))


def test_a_satiated_fly_keeps_the_full_startle_whatever_it_smells():
    import pytest

    from neuropest.brain import Brain
    from neuropest.toy_circuit import build

    brain = Brain(build(146, seed=42))
    brain.set_stimulus(dist=100.0, closing_speed=400.0, phero_drive=0.9, hunger=0.0)
    assert brain._loom_in == pytest.approx(4.0)


def test_a_startle_under_food_odor_does_not_teach_fear():
    """The reflex is only weakened under a faint odor, but an escape that happens next to food is still not read as
    punishment (the learning gate keeps its hard threshold)."""
    from neuropest.brain import Brain, ODOR_LEARN_GATE
    from neuropest.toy_circuit import build

    brain = Brain(build(146, seed=42))
    brain.set_stimulus(dist=900.0, closing_speed=0.0, phero_drive=0.3, hunger=0.4)     # 0.12 > ODOR_LEARN_GATE
    assert 0.3 * 0.4 > ODOR_LEARN_GATE
    brain.rates["GF"] = 120.0                                                          # a strong escape
    for _ in range(20):
        brain._learn(50.0)
    assert brain._punish == 0.0


def test_a_bigger_food_smells_further_and_stronger_and_shrinks_as_it_is_eaten():
    from neuropest.pheromone import FOOD_RADIUS_PX

    big = PheromoneSource(0.0, 0.0, radius=FOOD_RADIUS_PX, strength=1.0, size=1.4)
    small = PheromoneSource(0.0, 0.0, radius=FOOD_RADIUS_PX, strength=1.0, size=0.6)
    assert big.radius > small.radius and big.strength > small.strength
    assert big.concentration_at(350.0, 0.0) > small.concentration_at(350.0, 0.0)
    r0 = big.radius
    big.set_size(0.35)                                       # a quarter of it left: half the spread and strength
    assert math.isclose(big.radius, r0 / 2.0) and math.isclose(big.strength, math.sqrt(0.35))


def test_the_fly_eats_a_food_bit_by_bit_and_a_new_one_appears_once_it_is_gone():
    from neuropest.pheromone import FOOD_COUNT

    scr = ScreenBox(0, "scr", 0, 0, 1920, 1080, 0, 0, 1920, 1080)
    field = PheromoneField()
    field.spawn_random_sources([scr])
    assert len(field.sources) == FOOD_COUNT == 3
    food = field.sources[0]
    food.set_size(1.0)
    assert field.eat_at(food.x + 5.0, food.y, 0.4, radius=28.0, screen_boxes=[scr]) == 0.4
    assert math.isclose(food.size, 0.6) and food in field.sources
    assert field.eat_at(food.x + 500.0, food.y + 500.0, 0.4, radius=28.0, screen_boxes=[scr]) == 0.0   # not on it
    eaten = field.eat_at(food.x, food.y, 0.58, radius=28.0, screen_boxes=[scr])     # 0.02 left: under FOOD_MIN_SIZE
    assert math.isclose(eaten, 0.6) and food not in field.sources and len(field.sources) == FOOD_COUNT


def test_new_food_comes_in_sizes():
    from neuropest.pheromone import FOOD_SIZE_RANGE

    scr = ScreenBox(0, "scr", 0, 0, 1920, 1080, 0, 0, 1920, 1080)
    field = PheromoneField()
    field.spawn_random_sources([scr], count=20)
    sizes = [s.size for s in field.sources]
    assert all(FOOD_SIZE_RANGE[0] <= x <= FOOD_SIZE_RANGE[1] for x in sizes) and max(sizes) > min(sizes)
