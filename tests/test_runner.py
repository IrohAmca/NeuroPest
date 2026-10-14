import time

import numpy as np
import pytest

from neuropest.paths import CACHE, EYE, FIELD
from neuropest.runner import (I_CX, I_CY, I_STAMP, I_X, I_Y, EngineConfig, Runner, _Approach)


def _wait(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.skipif(not CACHE.exists(), reason="needs the FlyWire cache (tools/build_flywire.py)")
def test_worker_runs_the_real_connectome_tier():
    r = Runner(EngineConfig("flywire", 5_000, 0.5))
    try:
        assert _wait(lambda: r.ready, 120), "worker did not become ready"
        r.bias = 0.0
        r.send(900, 0)
        time.sleep(1.5)
        assert r.state == "stand" and r.stats()["n"] == 5_000
        r.send(150, 3000)                       # fast approach: LPLC2/LC4 -> Giant Fiber
        assert _wait(lambda: r.state == "fly", 3.0)
        assert r.stats()["gf"] > 15
    finally:
        r.stop()


def test_worker_process_reacts_to_looming_and_reports_telemetry():
    r = Runner(EngineConfig(n=2_000, dt=0.5))
    try:
        assert _wait(lambda: r.ready, 90), "worker did not become ready"
        r.bias = 0.0
        r.send(900, 0)
        time.sleep(1.5)
        assert r.state == "stand"
        r.send(150, 3000)                       # fast approach
        assert _wait(lambda: r.state == "fly", 3.0)
        st = r.stats()
        assert st["n"] == 2_000 and st["rt"] > 1.0 and st["gf"] > 5
        # restart with another size
        r.start(EngineConfig(n=10_000, dt=1.0))
        assert _wait(lambda: r.ready and r.stats()["n"] == 10_000, 60)
    finally:
        r.stop()
    assert not r.alive


FLY_POSE = (1000.0, 700.0, 0.0)             # x, y, heading of a fly standing on the screen


@pytest.mark.skipif(not (CACHE.exists() and EYE.exists() and FIELD.exists()),
                    reason="needs the FlyWire cache and the eye data (tools/build_eye.py)")
def test_worker_sees_the_cursor_through_the_funnel():
    r = Runner(EngineConfig("flywire", 5_000, 0.5))
    try:
        assert _wait(lambda: r.ready, 120), "worker did not become ready"
        r.bias, r.vision = 0.0, True
        for _ in range(90):                                     # the cursor is far away: nothing to see
            r.send(900.0, 0.0, 0.0, pose=FLY_POSE, cursor=(1900.0, 700.0))
            time.sleep(0.016)
        assert r.stats()["vision"] == 1 and r.state == "stand"
        t0 = time.time()
        while time.time() - t0 < 8.0 and r.state != "fly":      # then it dashes at the fly (closing speed is not sent)
            elapsed = (time.time() - t0) % 1.5
            cx = max(1050.0, 1400.0 - 1500.0 * max(0.0, elapsed - 0.2))
            r.send(abs(cx - FLY_POSE[0]), 0.0, 0.0, pose=FLY_POSE, cursor=(cx, 700.0))
            time.sleep(0.016)
        assert r.state == "fly"
        r.vision = False                                        # back to the cursor numbers
        r.send(900.0, 0.0, 0.0, pose=FLY_POSE, cursor=(1900.0, 700.0))
        assert _wait(lambda: r.stats()["vision"] == 0, 5.0)
    finally:
        r.stop()


@pytest.mark.skipif(not CACHE.exists(), reason="needs the FlyWire cache (tools/build_flywire.py)")
def test_worker_grooms_when_the_cursor_touches_the_fly():
    r = Runner(EngineConfig("flywire", 15_000, 0.5))
    try:
        assert _wait(lambda: r.ready, 120), "worker did not become ready"
        r.bias = 0.0
        r.send(900, 0)
        time.sleep(1.0)
        assert r.state == "stand"
        r.send(5, 0, 0.5, touch=1.0)            # hover on the fly: head touch -> aDN1/aDN2
        assert _wait(lambda: r.state == "groom", 4.0)
        assert r.stats()["groom"] > 8
        r.send(900, 0)
        assert _wait(lambda: r.state == "stand", 6.0)
    finally:
        r.stop()


def test_vision_switch_is_harmless_on_a_circuit_without_receptive_fields():
    r = Runner(EngineConfig(n=2_000, dt=0.5))
    try:
        assert _wait(lambda: r.ready, 90), "worker did not become ready"
        r.bias, r.vision = 0.0, True
        r.send(900.0, 0.0, 0.0, pose=(500.0, 500.0, 0.0), cursor=(900.0, 500.0))
        assert _wait(lambda: r.stats()["vision"] < 0, 10.0)     # unusable here, reported once, no retry loop
        r.send(150.0, 3000.0, 0.0, pose=(500.0, 500.0, 0.0), cursor=(650.0, 500.0))
        assert _wait(lambda: r.state == "fly", 3.0)             # the cursor numbers still drive the fly
    finally:
        r.stop()


@pytest.mark.skipif(not CACHE.exists(), reason="needs the FlyWire cache (tools/build_flywire.py)")
def test_touch_suppresses_take_off_but_retreat_outranks_grooming():
    """Real connectome, 15,000 neurons: head touch drives aDN1/aDN2 and suppresses the Giant Fiber; a retreat input
    now interrupts grooming (priority take-off, retreat, freeze, groom), so only a still hover grooms."""
    from neuropest import flywire
    from neuropest.brain import Brain

    net = flywire.load_cache().prefix(15_000)
    seen = {}
    for tag, closing, dist, touch in (("moderate", 1000, 150, 0.0), ("moderate+touch", 1000, 150, 1.0),
                                      ("fast+touch", 3000, 150, 1.0), ("hover", 0, 10, 1.0)):
        b = Brain(net)
        b.set_stimulus(dist, closing, 0.0, bearing=0.5, touch=touch)
        seen[tag] = {b.advance(4.0) for _ in range(250)}
    assert "fly" in seen["moderate"]
    assert "retreat" in seen["moderate+touch"] and "fly" not in seen["moderate+touch"]     # touch still calms the take-off
    assert "fly" in seen["fast+touch"]
    assert "groom" in seen["hover"] and not seen["hover"] & {"fly", "retreat", "freeze"}


def test_restart_returns_at_once_and_the_last_request_wins():
    r = Runner(EngineConfig(n=2_000, dt=0.5))
    try:
        assert _wait(lambda: r.ready, 90), "worker did not become ready"
        t0 = time.perf_counter()
        r.start(EngineConfig(n=5_000, dt=0.5))                  # the GUI thread must not wait for the old worker
        r.start(EngineConfig(n=10_000, dt=0.5))
        took = time.perf_counter() - t0
        assert took < 0.5, f"start blocked for {took:.2f} s"
        assert not r.ready and r.alive and not r.failed         # restarting is not an error
        assert _wait(lambda: r.ready and r.stats()["n"] == 10_000, 90)
        time.sleep(0.5)
        assert r.stats()["n"] == 10_000 and r.cfg.n == 10_000   # the 5,000 one never replaced it
    finally:
        r.stop()
    assert not r.alive


def test_stop_during_a_restart_leaves_no_worker_behind():
    r = Runner(EngineConfig(n=2_000, dt=0.5))
    assert _wait(lambda: r.ready, 90)
    r.start(EngineConfig(n=5_000, dt=0.5))
    r.stop()
    time.sleep(0.5)
    assert not r.alive and r._proc is None


def test_closing_speed_comes_from_real_time_between_samples_and_ignores_own_steps():
    a = _Approach()
    inp = [0.0] * 16

    def frame(stamp, cursor, fly=(0.0, 0.0)):
        inp[I_STAMP], inp[I_CX], inp[I_CY], inp[I_X], inp[I_Y] = stamp, *cursor, *fly
        return a.update(inp)

    assert a.update([0.0] * 16) is None                         # no stamp: the caller uses the GUI's own number
    assert frame(1.000, (500.0, 0.0)) == 0.0                    # first sample: nothing to compare with
    assert frame(1.016, (484.0, 0.0)) == pytest.approx(1000.0)
    assert frame(1.016, (400.0, 0.0)) == pytest.approx(1000.0)  # the same stamp is not a new sample
    assert frame(1.216, (284.0, 0.0)) == pytest.approx(1000.0)  # a 200 ms stall: 200 px in 0.2 s (clamping to 50 ms said 4000)
    assert frame(2.000, (100.0, 0.0)) == 0.0                    # a gap this long says nothing about its speed
    frame(3.000, (300.0, 0.0))
    assert frame(3.016, (300.0, 0.0), fly=(5.0, 0.0)) == pytest.approx(0.0)    # the fly stepped toward a standing cursor
    assert frame(3.032, (290.0, 0.0), fly=(5.0, 0.0)) == pytest.approx(10.0 / 0.016)   # the cursor itself moved 10 px


@pytest.mark.skipif(not CACHE.exists(), reason="needs the FlyWire cache (tools/build_flywire.py)")
def test_walking_up_to_a_standing_cursor_is_not_a_threat():
    """The GUI's own closing number includes the fly's steps (400 px/s here, 2.7 /s at 150 px: a retreat); with
    stamps the worker measures the cursor alone."""
    def run(stamped):
        r = Runner(EngineConfig("flywire", 5_000, 0.5))
        seen = set()
        try:
            assert _wait(lambda: r.ready, 120), "worker did not become ready"
            r.bias = 0.0
            t0 = time.perf_counter()
            while (t := time.perf_counter() - t0) < 2.0:
                fx = 1000.0 + min(400.0 * t, 100.0)             # the fly walks toward a cursor at x = 1250
                r.send(1250.0 - fx, 400.0 if t < 0.25 else 0.0, 0.0, pose=(fx, 700.0, 0.0), cursor=(1250.0, 700.0),
                       stamp=time.perf_counter() if stamped else 0.0)
                seen.add(r.state)
                time.sleep(0.016)
            return seen
        finally:
            r.stop()

    # (without stamps this scene used to alarm the fly; LPLC2 now reads size too, so it no longer does on a 5k circuit,
    # and the unit test above is what pins the number itself)
    assert run(True) == {"stand"}


def _bare_runner():
    """A Runner without a worker: shared arrays as plain lists, reported ready."""
    from neuropest.runner import INP_LEN, O_READY, OUT_LEN, Runner

    r = Runner.__new__(Runner)
    r.inp = [0.0] * INP_LEN
    r.out = [0.0] * OUT_LEN
    r.out[O_READY] = 1.0
    r._lab_init()
    return r


def test_every_slot_is_unique_and_inside_its_array():
    from neuropest import runner as rn
    from neuropest.states import LAB_GROUPS

    ins = {k: v for k, v in vars(rn).items() if k.startswith("I_") and isinstance(v, int)}
    outs = {k: v for k, v in vars(rn).items() if k.startswith("O_") and isinstance(v, int)}
    lab_rates = set(range(rn.I_LAB_RATE0, rn.I_LAB_RATE0 + len(LAB_GROUPS)))
    assert rn.I_LAB_RATE0 + len(LAB_GROUPS) <= rn.I_LAB_RATE_END <= rn.INP_LEN
    named = {v for k, v in ins.items() if k not in ("I_LAB_RATE0", "I_LAB_RATE_END")}
    assert len(named) == len(ins) - 2 and not named & lab_rates          # no two inputs share a slot
    assert all(0 <= v < rn.INP_LEN for v in named | lab_rates)
    assert len(set(outs.values())) == len(outs) and all(0 <= v < rn.OUT_LEN for v in outs.values())
    assert set(rn.LAB_RATE_SLOTS.values()) <= set(outs.values())


def test_sandbox_injections_are_counters_and_none_is_lost():
    from neuropest.runner import I_INJECT_CUE_FOOD, I_INJECT_PUNISH, I_INJECT_REWARD, _Counters

    r = _bare_runner()
    r.inject_reward(1.0)
    r.inject_reward(1.0)                                    # a second click before the worker read the first
    r.inject_punish(0.4)
    r.inject_optogenetic("food", 0.5)
    c = _Counters()
    assert c.take(r.inp, I_INJECT_REWARD) == 2.0            # both seen (the old read-and-clear lost one)
    assert c.take(r.inp, I_INJECT_REWARD) == 0.0            # and only once
    assert c.take(r.inp, I_INJECT_PUNISH) == pytest.approx(0.4)
    assert c.take(r.inp, I_INJECT_CUE_FOOD) == pytest.approx(0.5)


def test_lab_requests_are_written_under_the_sequence_lock():
    from neuropest.runner import I_LAB_RATE0, I_LAB_SEQ, I_LAB_SILENCE, I_LAB_STIM_ID, I_LAB_STIM_MS
    from neuropest.states import LAB_FULL_HZ, LAB_GROUPS, lab_names

    r = _bare_runner()
    assert r.stimulate("GF", 1.0, ms=300.0)
    assert r.inp[I_LAB_SEQ] == 2.0 and r.inp[I_LAB_SEQ] % 2 == 0          # whole (even) after the write
    assert r.inp[I_LAB_STIM_ID] == 1.0 and r.inp[I_LAB_STIM_MS] == 300.0
    assert r.inp[I_LAB_RATE0 + LAB_GROUPS.index("GF")] == LAB_FULL_HZ["GF"]
    r.silence("MDN")
    assert lab_names(r.inp[I_LAB_SILENCE]) == ("MDN",) and r.inp[I_LAB_STIM_ID] == 1.0   # silencing is no new pulse
    r.inject_optogenetic("takeoff", 0.5)                    # the old API now stimulates the real neurons
    assert r.inp[I_LAB_STIM_ID] == 2.0 and r.inp[I_LAB_RATE0] == 0.5 * LAB_FULL_HZ["GF"]
    assert not r.stimulate("NOPE")
    r.clear_lab()
    assert r.inp[I_LAB_SILENCE] == 0.0 and not any(r.inp[I_LAB_RATE0:I_LAB_RATE0 + len(LAB_GROUPS)])
    r.out[0] = 0.0                                          # not ready: nothing is sent
    seq = r.inp[I_LAB_SEQ]
    assert not r.stimulate("GF") and r.inp[I_LAB_SEQ] == seq


def _toy_loop(inp, out, iterations, brain=None, **kw):
    from neuropest.brain import Brain
    from neuropest.runner import EngineConfig, _loop
    from neuropest.toy_circuit import build

    class StopAfter:
        def __init__(self, n):
            self.n = n

        def is_set(self):
            self.n -= 1
            return self.n < 0

    net = build(146, 1)
    brain = brain or Brain(net, learning=False)
    _loop(EngineConfig("toy", 146, 0.5), brain, net, inp, out, StopAfter(iterations), **kw)
    return brain


def test_the_worker_applies_a_lab_request_once_and_reports_what_it_applied():
    from neuropest.runner import (I_DIST, I_LAB_SEQ, I_LAB_SILENCE, INP_LEN, O_LAB_AVAIL, O_LAB_SEQ, O_LAB_SILENCED,
                                  O_LAB_STIM, OUT_LEN)
    from neuropest.states import lab_mask, lab_names

    r = _bare_runner()
    r.inp = np.zeros(INP_LEN)
    r.out = np.zeros(OUT_LEN)
    r.out[0] = 1.0
    r.inp[I_DIST] = 1e6
    r.silence("MDN")                                        # the toy circuit has no MDN: dropped, and reported so
    r.stimulate("GF", 1.0, ms=1000.0)
    brain = _toy_loop(r.inp, r.out, 5)
    assert r.out[O_LAB_SEQ] == r.inp[I_LAB_SEQ]             # acknowledged
    assert lab_names(r.out[O_LAB_AVAIL]) == ("GF", "WALK")
    assert lab_names(r.out[O_LAB_STIM]) == ("GF",) and lab_names(r.out[O_LAB_SILENCED]) == ()
    assert brain.lab_on and brain._lab_stim == {"GF": 150.0}
    seq = r.inp[I_LAB_SEQ]
    r.inp[I_LAB_SEQ] = seq + 1                              # the GUI is in the middle of a write (odd) ...
    r.inp[I_LAB_SILENCE] = float(lab_mask(["GF"]))
    _toy_loop(r.inp, r.out, 3, brain=brain)
    assert brain._lab_sil == ()                             # ... so the worker leaves the half-written block alone
    r.inp[I_LAB_SEQ] = seq + 2                              # whole again
    _toy_loop(r.inp, r.out, 3, brain=brain)
    assert brain._lab_sil == ("GF",)


def test_no_memory_is_saved_while_the_lab_drives_the_circuit(monkeypatch, tmp_path):
    from neuropest import runner as rn
    from neuropest.brain import Brain
    from neuropest.toy_circuit import build

    monkeypatch.setattr(rn, "MEMORY_SAVE_S", 0.0)
    brain = Brain(build(146, 1), learning=True)
    saved = []
    brain.mb.save = lambda path: saved.append(path)
    brain.mb.dirty = True
    brain.set_lab(silenced=["GF"])
    inp, out = np.zeros(rn.INP_LEN), np.zeros(rn.OUT_LEN)
    inp[rn.I_DIST] = 1e6
    _toy_loop(inp, out, 5, brain=brain, memory=tmp_path / "m.npz")
    assert saved == []
    brain.set_lab(silenced=[])
    brain._lab_tail_ms = 0.0
    brain.mb.dirty = True
    _toy_loop(inp, out, 3, brain=brain, memory=tmp_path / "m.npz")
    assert saved


def test_meals_reach_the_brain_once_per_count_and_never_from_a_fresh_array():
    from neuropest.runner import I_DIST, I_FED, INP_LEN, OUT_LEN

    inp, out = np.zeros(INP_LEN), np.zeros(OUT_LEN)
    inp[I_DIST] = 1e6
    brain = _toy_loop(inp, out, 3)
    assert brain._fed_events == 0
    inp[I_FED] += 1.0
    inp[I_FED] += 1.0
    _toy_loop(inp, out, 3, brain=brain)                     # (a new loop starts its counters at 0 again)
    assert brain._fed_events == 2


def test_a_silenced_group_survives_an_engine_restart_and_a_stimulation_does_not():
    r = Runner(EngineConfig(n=2_000, dt=0.5))
    try:
        assert _wait(lambda: r.ready, 90), "worker did not become ready"
        r.silence("GF")
        r.stimulate("WALK", 1.0, ms=10_000)
        assert _wait(lambda: r.lab_status()["acked"] and r.lab_status()["silenced"] == ("GF",), 5.0)
        assert r.lab_status()["stimulated"] == ("WALK",) and r.lab_status()["gate"]
        r.start(EngineConfig(n=2_000, dt=0.5, seed=2))
        assert _wait(lambda: r.ready, 90), "restarted worker did not become ready"
        assert _wait(lambda: r.lab_status()["acked"] and r.lab_status()["silenced"] == ("GF",), 5.0)
        assert r.lab_status()["stimulated"] == ()
    finally:
        r.stop()


# ------------------------------------------------------------------------- mushroom-body wiring choice and buffer
def test_wiring_env_var_wins_over_the_preference_and_the_real_cells_need_their_file(monkeypatch, tmp_path):
    from neuropest import paths, runner

    from neuropest import mushroom

    built = tmp_path / "mushroom_flywire.npz"
    built.write_text("x", encoding="utf-8")
    monkeypatch.setattr(paths, "MUSHROOM", built)
    monkeypatch.setattr(mushroom, "wiring_usable", lambda path=None: path.exists())   # "built" stands for usable here
    monkeypatch.delenv("NEUROPEST_MB", raising=False)
    assert runner.effective_mb_wiring("random") == "random"
    assert runner.effective_mb_wiring("flywire") == "flywire"
    assert runner.effective_mb_wiring("nonsense") == "random"

    monkeypatch.setenv("NEUROPEST_MB", "random")                    # forced both ways
    assert runner.effective_mb_wiring("flywire") == "random"
    monkeypatch.setenv("NEUROPEST_MB", "FlyWire")
    assert runner.effective_mb_wiring("random") == "flywire"
    if CACHE.exists():                                              # without the cache default_config() is the toy circuit
        assert runner.default_config().mb_wiring == "flywire"

    monkeypatch.setattr(paths, "MUSHROOM", tmp_path / "missing.npz")
    assert runner.effective_mb_wiring("flywire") == "random"        # no file, no real cells (the worker would fall back)


def test_weight_buffer_holds_every_kenyon_cell_of_the_real_model():
    from types import SimpleNamespace

    from neuropest import runner

    assert runner.MB_WEIGHTS_LEN >= 5177                            # the real wiring has 5,177 KCs
    zeros = runner.Runner.get_mb_weights(SimpleNamespace(mb_weights=None))
    assert len(zeros) == runner.MB_WEIGHTS_LEN and not zeros.any()


def test_runner_reports_the_firing_kenyon_cells_of_the_running_model():
    import ctypes
    from types import SimpleNamespace

    import numpy as np

    from neuropest import runner as rn

    out = np.zeros(rn.OUT_LEN)
    buf = (ctypes.c_byte * 8)(0, 1, 0, 1, 1, 0, 0, 0)
    fake = SimpleNamespace(mb_active=buf, out=out)
    assert rn.Runner.get_kc_active(fake) is None             # nothing published yet
    out[rn.O_KC_N] = 5
    got = rn.Runner.get_kc_active(fake)
    assert got.dtype == bool and got.tolist() == [False, True, False, True, True]
    assert rn.Runner.get_kc_active(SimpleNamespace(mb_active=None, out=out)) is None


def test_worker_publishes_which_kenyon_cells_fire_for_which_cue():
    r = Runner(EngineConfig(n=2_000, dt=0.5))                  # the designed model: 2,000 cells, 5% active
    try:
        assert _wait(lambda: r.ready, 90), "worker did not become ready"
        from neuropest.runner import MB_WEIGHTS_LEN

        assert len(np.frombuffer(r.mb_weights, np.float32)) == MB_WEIGHTS_LEN    # the buffers start() really allocates
        assert len(np.frombuffer(r.mb_active, np.int8)) == MB_WEIGHTS_LEN
        r.bias = 0.0
        r.send(120, 0)                                        # the cursor close: the cursor cue is on
        assert _wait(lambda: r.get_kc_active() is not None and r.get_kc_active().any(), 8.0)
        near = r.get_kc_active()
        assert len(near) == 2000 and 0 < near.sum() <= 200
        r.inject_cue_food(1.0)                                # a food cue: a different pattern separates from it
        r.send(120, 0)
        assert _wait(lambda: r.get_kc_active() is not None and not np.array_equal(r.get_kc_active(), near), 8.0)
    finally:
        r.stop()


def test_real_cells_beyond_the_first_2000_reach_the_3d_view():
    """The shared valence buffer was 2,000 floats: the real model's Kenyon cells 2000..5176 never reached the view."""
    from neuropest.paths import MUSHROOM

    if not MUSHROOM.exists():
        pytest.skip("data/circuits/mushroom_flywire.npz not built here")
    r = Runner(EngineConfig(n=2_000, dt=0.5, mb_wiring="flywire"))
    try:
        assert _wait(lambda: r.ready, 90), "worker did not become ready"
        assert _wait(lambda: r.get_kc_active() is not None, 10.0)
        assert len(r.get_kc_active()) == 5177                 # the real model's cell count is published
        r.send(900, 0)
        for _ in range(6):                                     # food cue + reward: the real cells learn appetitive valence
            r.inject_cue_food(1.0)
            r.inject_reward(1.0)
            time.sleep(0.3)
        assert _wait(lambda: np.abs(r.get_mb_weights()[2000:5177]).max() > 0.0, 10.0)
    finally:
        r.stop()


def test_stats_report_the_heading_the_gui_sent():
    """The 3D compass reads stats()['heading']; without it the bump silently falls back to the steering integral."""
    import math

    from neuropest.runner import INP_LEN, O_READY, OUT_LEN, Runner

    r = Runner.__new__(Runner)
    r.inp = [0.0] * INP_LEN
    r.out = [0.0] * OUT_LEN
    r.out[O_READY] = 1.0
    r.vision, r.preview, r.eye_height, r.bias, r.skittish = False, False, 100.0, 0.65, 1.0
    r.capture = None
    r.send(500.0, 0.0, pose=(10.0, 20.0, 1.25))
    assert r.stats()["heading"] == 1.25
    r.send(500.0, 0.0, pose=(10.0, 20.0, -math.pi / 2))
    assert r.stats()["heading"] == -math.pi / 2


def test_worker_passes_the_raw_closing_speed_and_the_brain_gates_it_by_odor():
    """The worker used to zero the closing speed itself whenever odor x hunger > 0.05 (a second copy of the old cliff)."""
    import numpy as np

    from neuropest.brain import Brain, odor_gate
    from neuropest.runner import (I_CLOSING, I_DIST, I_HUNGER, I_PHERO_DRIVE, I_SKITTISH, INP_LEN, OUT_LEN,
                                  EngineConfig, _loop)
    from neuropest.toy_circuit import build

    class StopAfter:
        def __init__(self, n):
            self.n = n

        def is_set(self):
            self.n -= 1
            return self.n < 0

    net = build(146, 1)
    brain = Brain(net, learning=False)
    inp, out = np.zeros(INP_LEN), np.zeros(OUT_LEN)
    inp[I_DIST], inp[I_CLOSING], inp[I_SKITTISH] = 100.0, 400.0, 1.0       # no stamp: the GUI number is used as given
    inp[I_PHERO_DRIVE], inp[I_HUNGER] = 0.1, 1.0
    _loop(EngineConfig("toy", 146, 0.5), brain, net, inp, out, StopAfter(3))
    assert brain._stim[1] == 400.0
    arousal = 1.0 + 0.5 * brain.arousal                                    # a startle may raise arousal meanwhile
    assert brain._loom_in == pytest.approx(4.0 * arousal * odor_gate(0.1))
    assert 0.0 < odor_gate(0.1) < 1.0


def test_a_damaged_wiring_file_runs_the_designed_model(monkeypatch, tmp_path):
    from neuropest import paths, runner

    bad = tmp_path / "mushroom_flywire.npz"
    bad.write_bytes(b"")
    monkeypatch.setattr(paths, "MUSHROOM", bad)
    monkeypatch.delenv("NEUROPEST_MB", raising=False)
    assert runner.effective_mb_wiring("flywire") == "random"
