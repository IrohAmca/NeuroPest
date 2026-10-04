"""Mushroom-body learning: selectivity, forgetting, memory file, and the valence's effect on the brain's drive."""
import time

import numpy as np

from neuropest.brain import Brain
from neuropest.mushroom import MBParams, MushroomBody
from neuropest.toy_circuit import build

FOOD = np.array([0.9, 0, 0, 0, 0.])
CURSOR = np.array([0, 0, 0.8, 0, 0.])
QUIET = np.zeros(5)


def _pair(mb, cue, reward=0.0, punish=0.0, trials=4):
    for _ in range(trials):
        for _ in range(40):                              # 2 s of cue with the unconditioned stimulus
            mb.step(cue, reward, punish, 0.05)
        for _ in range(100):
            mb.step(QUIET, 0, 0, 0.05)


def test_naive_fly_has_no_valence():
    mb = MushroomBody()
    assert mb.read(FOOD) == 0.0 and mb.read(CURSOR) == 0.0 and mb.step(FOOD, 0, 0, 0.05) == 0.0


def test_reward_makes_the_paired_cue_desired_and_leaves_another_cue_alone():
    mb = MushroomBody()
    _pair(mb, FOOD, reward=1.0)
    assert mb.read(FOOD) > 0.5
    assert abs(mb.read(CURSOR)) < 0.05


def test_punishment_makes_the_paired_cue_feared():
    mb = MushroomBody()
    _pair(mb, CURSOR, punish=1.0)
    assert mb.read(CURSOR) < -0.5 and abs(mb.read(FOOD)) < 0.05


def test_no_dopamine_no_learning():
    mb = MushroomBody()
    _pair(mb, FOOD)
    assert mb.read(FOOD) == 0.0


def test_memory_fades_and_reset_clears_it():
    mb = MushroomBody(MBParams(tau_forget_s=60.0))
    _pair(mb, FOOD, reward=1.0)
    v0 = mb.read(FOOD)
    for _ in range(int(120 / 0.5)):                      # two minutes later
        mb.step(QUIET, 0, 0, 0.5)
    assert 0 < mb.read(FOOD) < v0 * 0.25
    mb.reset()
    assert mb.read(FOOD) == 0.0


def test_save_and_load_roundtrip_and_layout_check(tmp_path):
    mb = MushroomBody()
    _pair(mb, FOOD, reward=1.0)
    f = tmp_path / "mem.npz"
    mb.save(f)
    other = MushroomBody()
    assert other.load(f) and abs(other.read(FOOD) - mb.read(FOOD)) < 1e-6
    assert not MushroomBody(MBParams(n_kc=500)).load(f)
    assert not MushroomBody().load(tmp_path / "missing.npz")


def test_step_is_cheap():
    mb = MushroomBody()
    cues = np.array([0.5, 0.2, 0.7, 0.1, 0.0])
    t = time.perf_counter()
    for _ in range(200):
        mb.step(cues, 0.0, 0.0, 0.05)
    assert (time.perf_counter() - t) / 200 < 2e-3        # budget: one 50 ms step is far below 1 ms on a laptop


def _toy_brain(learning=True):
    net = build(146, 1)
    net.groups["LC10_R"] = np.array([5], np.int32)
    net.groups["LC10_L"] = np.array([6], np.int32)
    return Brain(net, learning=learning)


def _lc10(b):
    drive = dict(zip(b._last_fi.tolist(), b._last_fr.tolist()))
    return drive[5], drive[6]                            # right, left


def test_brain_learns_to_want_the_cue_that_came_with_food():
    b = _toy_brain()
    for _ in range(4):
        b.set_stimulus(900, 0, 0.0, phero_drive=0.9, at_target=1.0)    # feeding with the food odor
        for _ in range(30):
            b.advance(50.0)
        b.set_stimulus(900, 0, 0.0)
        for _ in range(60):
            b.advance(50.0)
    b.set_stimulus(900, 0, 0.0, phero_drive=0.9)         # the odor alone
    b.advance(50.0)
    assert b.valence > 0.3
    naive = _toy_brain()
    naive.set_stimulus(900, 0, 0.0, phero_drive=0.9)
    naive.advance(50.0)
    assert naive.valence == 0.0


def test_valence_scales_and_reverses_the_turn_toward_a_pheromone():
    b = _toy_brain()
    b.set_stimulus(900, 0, 0.0, phero_steer=10.0)
    r0, l0 = _lc10(b)
    assert r0 == 10.0 and l0 == 0.0                      # naive: unchanged
    b.valence = 0.8
    b._drive()
    assert _lc10(b)[0] > r0
    b.valence = -0.8
    b._drive()
    r, l = _lc10(b)
    assert l > 0 and r == 0                               # fear of the odor: turns away (the other side is driven)


def test_learning_can_be_switched_off():
    b = _toy_brain(learning=False)
    b.set_stimulus(900, 0, 0.0, phero_drive=0.9, at_target=1.0)
    for _ in range(40):
        b.advance(50.0)
    assert b.mb is None and b.valence == 0.0


def test_escape_punishes_the_cue_but_a_chase_does_not():
    b = _toy_brain()
    for _ in range(3):                                   # the cursor rushes the fly: the Giant Fiber fires
        b.set_stimulus(120, 3000, 0.0)
        for _ in range(20):
            b.advance(50.0)
        b.set_stimulus(900, 0, 0.0)
        for _ in range(60):
            b.advance(50.0)
    b.set_stimulus(120, 0, 0.0)                          # the cursor close again, not moving
    b.advance(50.0)
    assert b.valence < -0.2
    c = _toy_brain()
    c.set_stimulus(900, 0, 0.0, phero_drive=0.9)
    c.flight_urge = 1.0                                  # voluntary pursuit drives the Giant Fiber too
    for _ in range(40):
        c.advance(50.0)
    assert c._punish == 0.0


def test_runner_remembers_across_restarts_and_forgets_on_request(tmp_path):
    import time as _t

    from neuropest.runner import EngineConfig, Runner

    f = tmp_path / "mem.npz"
    cfg = EngineConfig(n=500, dt=0.5, memory_path=str(f))
    r = Runner(cfg)
    try:
        t0 = _t.time()
        while not r.ready and _t.time() - t0 < 60:
            _t.sleep(0.1)
        assert r.ready
        for _ in range(60):                              # feed with the food odor for a few seconds
            r.send(900, 0, phero_drive=0.9, at_target=True)
            _t.sleep(0.1)
    finally:
        r.stop()
    assert f.exists() and MushroomBody().load(f)         # saved on exit
    mb = MushroomBody()
    mb.load(f)
    assert mb.read(np.array([0.9, 0, 0, 0, 0.])) > 0.1
    r2 = Runner(cfg)
    try:
        t0 = _t.time()
        while not r2.ready and _t.time() - t0 < 60:
            _t.sleep(0.1)
        r2.forget()
        _t.sleep(0.5)
    finally:
        r2.stop()
    mb2 = MushroomBody()
    mb2.load(f)
    assert mb2.read(np.array([0.9, 0, 0, 0, 0.])) == 0.0
