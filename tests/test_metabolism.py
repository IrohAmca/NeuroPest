import pytest

from neuropest.metabolism import MetabolicConfig, MetabolicState
from neuropest.states import FLY, GROOM, STAND, WALK


def test_metabolic_state_initial_and_bounds():
    m = MetabolicState(energy=0.65, enabled=True)
    assert m.energy == pytest.approx(0.65)
    assert m.hunger == pytest.approx(0.35)
    assert m.hunger_pct == 35
    assert m.energy_pct == 65


def test_disabled_hunger_defaults_to_perpetual_hunger():
    m = MetabolicState(energy=1.0, enabled=False)
    assert m.hunger == 1.0
    assert m.hunger_pct == 100


def test_depletion_rates_depend_on_behavior_state():
    cfg = MetabolicConfig(burn_stand=0.01, burn_walk=0.05, burn_fly=0.20, rate_mult=1.0)
    m = MetabolicState(energy=1.0, enabled=True, cfg=cfg)

    # 1 second of standing
    m.update(1.0, STAND)
    assert m.energy == pytest.approx(0.99)

    # 1 second of walking
    m.update(1.0, WALK)
    assert m.energy == pytest.approx(0.94)

    # 1 second of flying (much higher burn)
    m.update(1.0, FLY)
    assert m.energy == pytest.approx(0.74)


def test_feeding_and_clamping():
    cfg = MetabolicConfig(feed_amount=0.35)
    m = MetabolicState(energy=0.5, enabled=True, cfg=cfg)

    m.feed()
    assert m.energy == pytest.approx(0.85)

    m.feed()
    assert m.energy == 1.0  # clamped at 1.0
    assert m.hunger == 0.0


def test_starve_and_satiate_actions():
    m = MetabolicState(energy=0.8, enabled=True)
    m.starve()
    assert m.energy == 0.0
    assert m.hunger == 1.0

    m.satiate()
    assert m.energy == 1.0
    assert m.hunger == 0.0
