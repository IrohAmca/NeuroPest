import numpy as np
import pytest

from neuropest.engine import LIFEngine, LIFParams, Network, ReferenceEngine


def _run_schedule(engine, drive_idx, steps, every):
    for t in range(steps):
        if t % every == 0:
            engine.set_drive(drive_idx, 1e9)   # p == 1.0: deterministic forced spike
        else:
            engine.set_drive(drive_idx, 0.0)
        engine.advance(engine.dt)


@pytest.mark.parametrize("dt", [0.1, 0.5, 1.0])
def test_matches_reference(dt):
    net = Network.random(300, mean_deg=15, seed=3, gain=1.4)
    drive = np.arange(0, 300, 12)
    fast = LIFEngine(net, dt=dt, eps=0.0)
    ref = ReferenceEngine(net, dt=dt)
    steps = int(150 / dt)
    _run_schedule(fast, drive, steps, every=max(1, int(6 / dt)))
    _run_schedule(ref, drive, steps, every=max(1, int(6 / dt)))
    assert ref.total_spikes > 300, "test network should actually propagate activity"
    assert abs(fast.total_spikes - ref.total_spikes) <= 0.02 * ref.total_spikes
    corr = np.corrcoef(fast.counts, ref.counts)[0, 1]
    assert corr > 0.98


def test_silent_network_costs_nothing():
    net = Network.random(2000, seed=1)
    e = LIFEngine(net)
    e.advance(200)
    assert e.total_spikes == 0 and e.n_active == 0 and e.neuron_updates == 0


def test_activity_dies_out_and_leaves_active_set():
    net = Network.random(2000, seed=1, gain=1.2)
    e = LIFEngine(net, eps=0.01)
    e.set_drive(np.arange(50), 150.0)
    e.advance(100)
    assert e.total_spikes > 0 and e.n_active > 0
    e.set_drive(np.arange(0), 0.0)
    e.advance(2000)
    assert e.n_active == 0


def test_synaptic_delay_and_threshold():
    # neuron 0 -> neuron 1 with a kick large enough to reach threshold
    # A lone 80 mV kick on g: u(t) = 80/3 (exp(-t/20) - exp(-t/5)) crosses the 7 mV
    # threshold ~2.35 ms after it arrives, which is t_dly = 1.8 ms after the spike.
    net = Network.from_coo([0], [1], [80.0], 2)
    for cls in (LIFEngine, ReferenceEngine):
        e = cls(net, dt=0.1, eps=0.0) if cls is LIFEngine else cls(net, dt=0.1)
        e.set_drive([0], 1e9)
        e.advance(0.1)             # neuron 0 spikes in step 0
        e.set_drive([0], 0.0)
        first = None
        for step in range(1, 200):
            e.advance(0.1)
            if e.counts[1] and first is None:
                first = step
        assert e.counts[0] == 1 and e.counts[1] == 1
        assert 38 <= first <= 46, (cls.__name__, first)


def test_inhibition_blocks_spiking():
    def target_spikes(pre, post, w):
        e = LIFEngine(Network.from_coo(pre, post, w, 4), dt=0.5, eps=0.0)
        e.set_drive([0, 1, 3], 1e9)
        e.advance(0.5)
        e.set_drive([0, 1, 3], 0.0)
        e.advance(50)
        return e.counts[2]

    assert target_spikes([0, 1], [2, 2], [50.0, 50.0]) >= 1
    assert target_spikes([0, 1, 3], [2, 2, 2], [50.0, 50.0, -400.0]) == 0


def test_induced_keeps_edges_and_groups():
    net = Network.random(100, seed=5)
    net.groups = {"a": np.array([3, 7, 50], np.int32)}
    keep = np.array([7, 3, 10, 20, 50])
    sub = net.induced(keep)
    assert sub.n == 5
    assert list(sub.groups["a"]) == [1, 0, 4]
    full = net.to_csr().toarray()
    assert np.allclose(sub.to_csr().toarray(), full[np.ix_(keep, keep)])


def test_save_load_roundtrip(tmp_path):
    net = Network.random(50, seed=2)
    net.groups = {"x": np.array([1, 2], np.int32)}
    net.ids = np.arange(50, dtype=np.int64) + 10**17
    net.save(tmp_path / "n.npz")
    back = Network.load(tmp_path / "n.npz")
    assert np.array_equal(back.data, net.data) and np.array_equal(back.ids, net.ids)
    assert list(back.groups["x"]) == [1, 2]
