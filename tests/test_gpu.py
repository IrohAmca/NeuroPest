"""GPU engine against the NumPy reference engine. Skipped without wgpu or a GPU adapter."""
import numpy as np
import pytest

pytest.importorskip("wgpu")
from neuropest.engine import Network, ReferenceEngine
from neuropest.engine.lif_wgpu import WGPUEngine, gpu_available

pytestmark = pytest.mark.skipif(not gpu_available(), reason="no GPU adapter")


def _drive_schedule(engine, idx, steps, every, current=False):
    for t in range(steps):
        on = 1e9 if t % every == 0 else 0.0                 # p == 1.0: deterministic forced spike
        if current:
            engine.set_current_drive(idx, on, 60.0)
        else:
            engine.set_drive(idx, on)
        engine.advance(engine.dt)


def _counts(e, n):
    return np.asarray(e.pop_counts(np.arange(n)))


@pytest.mark.parametrize("dt", [0.1, 0.5, 1.0])
def test_matches_reference(dt):
    net = Network.random(300, mean_deg=15, seed=3, gain=1.4)
    drive = np.arange(0, 300, 12)
    gpu, ref = WGPUEngine(net, dt=dt), ReferenceEngine(net, dt=dt)
    steps, every = int(150 / dt), max(1, int(6 / dt))
    _drive_schedule(gpu, drive, steps, every)
    _drive_schedule(ref, drive, steps, every)
    g, r = _counts(gpu, 300), ref.counts
    assert r.sum() > 300, "test network should propagate activity"
    assert abs(g.sum() - r.sum()) <= 0.05 * r.sum()
    assert np.corrcoef(g, r)[0, 1] > 0.97


def test_current_drive_matches_reference():
    net = Network.random(300, mean_deg=15, seed=4, gain=1.4)
    idx = np.arange(0, 300, 7)
    gpu, ref = WGPUEngine(net, dt=0.5), ReferenceEngine(net, dt=0.5)
    _drive_schedule(gpu, idx, 300, 10, current=True)
    _drive_schedule(ref, idx, 300, 10, current=True)
    g, r = _counts(gpu, 300), ref.counts
    assert r.sum() > 100
    assert abs(g.sum() - r.sum()) <= 0.05 * r.sum() and np.corrcoef(g, r)[0, 1] > 0.97


def test_synaptic_delay_and_threshold():
    net = Network.from_coo([0], [1], [80.0], 2)            # a lone 80 mV kick crosses threshold ~2.35 ms after arrival
    e = WGPUEngine(net, dt=0.1)
    e.set_drive([0], 1e9)
    e.advance(0.1)
    e.set_drive([0], 0.0)
    first = None
    for step in range(1, 200):
        e.advance(0.1)
        if first is None and e.pop_counts([1])[0]:
            first = step
    assert first is not None and 38 <= first <= 46, first


def test_inhibition_blocks_spiking():
    def target_spikes(pre, post, w):
        e = WGPUEngine(Network.from_coo(pre, post, w, 4), dt=0.5)
        e.set_drive([0, 1, 3], 1e9)
        e.advance(0.5)
        e.set_drive([0, 1, 3], 0.0)
        e.advance(50)
        return int(e.pop_counts([2])[0])

    assert target_spikes([0, 1], [2, 2], [50.0, 50.0]) >= 1
    assert target_spikes([0, 1, 3], [2, 2, 2], [50.0, 50.0, -400.0]) == 0


def test_poisson_rate_and_pop_counts_reset():
    net = Network.random(500, seed=1)
    e = WGPUEngine(net, dt=0.5, seed=2)
    idx = np.arange(200)
    e.set_drive(idx, 100.0)
    e.advance(1000.0)
    c = e.pop_counts(idx)
    assert 90 < c.mean() < 110                              # ~100 Hz for 1 s
    assert e.pop_counts(idx).sum() == 0                     # counters were reset by the pop


def test_brain_and_worker_on_the_gpu():
    import time

    from neuropest.brain import FLY, Brain
    from neuropest.runner import EngineConfig, Runner, list_gpus, pick_gpu
    from neuropest.toy_circuit import build

    b = Brain(build(146), backend="gpu")
    b.set_stimulus(900, 0, 0.0)
    for _ in range(100):
        b.advance(4.0)
    assert b.state != FLY
    b.set_stimulus(150, 3000, 0.0)
    assert FLY in {b.advance(4.0) for _ in range(100)}
    b.engine.close()

    gpus = list_gpus()
    assert gpus and pick_gpu(gpus) is not None
    r = Runner(EngineConfig("toy", 146, 0.5, backend="gpu", adapter=pick_gpu(gpus)))
    try:
        end = time.time() + 60
        while not (r.ready or r.failed) and time.time() < end:
            time.sleep(0.05)
        assert r.ready
        r.bias = 0.0
        r.send(150, 3000)
        end = time.time() + 5
        while r.state != "fly" and time.time() < end:
            time.sleep(0.02)
        assert r.state == "fly"
        end = time.time() + 8
        while (r.stats()["rt"] <= 0 or r.stats()["spikes"] <= 0) and time.time() < end:   # published every 250 ms
            time.sleep(0.05)
        assert r.stats()["active"] == -1.0 and r.stats()["spikes"] > 0
    finally:
        r.stop()


def test_silent_network_stays_silent():
    e = WGPUEngine(Network.random(1000, seed=1), dt=0.5)
    e.advance(200.0)
    assert e.total_spikes == 0 and _counts(e, 1000).sum() == 0
