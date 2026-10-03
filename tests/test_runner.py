import time

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
        while time.time() - t0 < 5.0 and r.state != "fly":      # then it dashes at the fly (closing speed is not sent)
            cx = max(1050.0, 1400.0 - 1500.0 * max(0.0, time.time() - t0 - 0.3))
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
def test_touch_calms_a_moderate_approach_but_not_a_fast_one():
    """Real connectome, 15,000 neurons: head touch drives aDN1/aDN2 and suppresses the Giant Fiber."""
    from neuropest import flywire
    from neuropest.brain import Brain

    net = flywire.load_cache().prefix(15_000)
    seen = {}
    for tag, closing, touch in (("moderate", 1000, 0.0), ("moderate+touch", 1000, 1.0), ("fast+touch", 3000, 1.0)):
        b = Brain(net)
        b.set_stimulus(150, closing, 0.0, bearing=0.5, touch=touch)
        seen[tag] = {b.advance(4.0) for _ in range(250)}
    assert "fly" in seen["moderate"]
    assert "groom" in seen["moderate+touch"] and "fly" not in seen["moderate+touch"]
    assert "fly" in seen["fast+touch"]


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

    unstamped = run(False)
    assert "retreat" in unstamped or "fly" in unstamped         # the old number alone does alarm it
    assert run(True) == {"stand"}
