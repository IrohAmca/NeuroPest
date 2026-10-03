import time

import pytest

from neuropest.paths import CACHE
from neuropest.runner import EngineConfig, Runner


def _wait(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.skipif(not CACHE.exists(), reason="needs the FlyWire cache (tools/build_flywire.py)")
def test_worker_runs_the_real_connectome_tier():
    r = Runner(EngineConfig("flywire", 2_000, 0.5))
    try:
        assert _wait(lambda: r.ready, 120), "worker did not become ready"
        r.bias = 0.0
        r.send(900, 0)
        time.sleep(1.5)
        assert r.state == "stand" and r.stats()["n"] == 2_000
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
