"""Fly's-eye preview: snapshot transport and the three pictures (neuropest/eyeview.py)."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from neuropest import eyeview as ev
from neuropest.paths import FIELD


def _dirs(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]


@pytest.fixture
def layout():
    # four columns: ahead on the horizon, 90 deg right on the horizon, straight ahead 45 deg down, behind-left up
    return ev.EyeLayout(_dirs([0.0, 90.0, 0.0, -150.0], [0.0, 0.0, -45.0, 40.0]))


def _snap(lum=(0.9, 0.1, 0.5, 0.0), exp=(0, 0, 0, 0), obj=(0, 0, 0, 0), cols=(0, 1, 2, 3), rates=(0, 0, 0, 0),
          sizes=(1, 1, 1, 1)):
    f = np.float32
    return ev.Snapshot(1, np.array(lum, f), np.array(exp, f), np.array(obj, f), np.array(cols, np.int32),
                       np.array(rates, f), tuple(sizes))


def _shared():
    return np.frombuffer((ctypes.c_float * ev.PREVIEW_LEN)(), dtype=np.float32)


def test_nothing_published_reads_as_none():
    assert ev.read_snapshot(_shared()) is None


def test_snapshot_round_trips_through_the_shared_array():
    buf = _shared()
    lum, exp, obj = np.linspace(0, 1, 7, dtype=np.float32), np.arange(7, dtype=np.float32), np.ones(7, np.float32) * 0.5
    cols, rates = np.array([6, 0, 3, 2, 5]), np.array([10, 0, 55, 150, 1], np.float32)
    ev.write_snapshot(buf, 41, lum, exp, obj, cols, rates, (2, 1, 1, 1))
    s = ev.read_snapshot(buf)
    assert s.seq == 41 and s.sizes == (2, 1, 1, 1)
    np.testing.assert_allclose(s.lum, lum)
    np.testing.assert_allclose(s.expansion, exp)
    np.testing.assert_array_equal(s.cols, cols)
    cols_lc4, rates_lc4 = s.group(1)
    assert list(cols_lc4) == [3] and list(rates_lc4) == [55.0]
    assert len(s.group(3)[0]) == 1


def test_snapshot_is_a_copy_not_a_view():
    buf = _shared()
    ev.write_snapshot(buf, 1, np.ones(3, np.float32), np.zeros(3, np.float32), np.zeros(3, np.float32),
                      np.array([0]), np.array([5.0], np.float32), (1, 0, 0, 0))
    s = ev.read_snapshot(buf)
    ev.write_snapshot(buf, 2, np.zeros(3, np.float32), np.zeros(3, np.float32), np.zeros(3, np.float32),
                      np.array([0]), np.array([9.0], np.float32), (1, 0, 0, 0))
    assert s.seq == 1 and s.lum.max() == 1.0 and s.rates[0] == 5.0


def test_a_corrupt_header_is_ignored():
    buf = _shared()
    buf[0], buf[1] = 5, ev.MAX_COLS + 10
    assert ev.read_snapshot(buf) is None


def test_columns_land_at_their_azimuth_and_elevation(layout):
    w, h = 440, 200
    xs, ys = layout.pixels(w, h)
    assert xs[0] == round(180 / 360 * (w - 1))                    # straight ahead is the middle column of the picture
    assert xs[1] > xs[0] > xs[3]                                  # right of ahead, left of ahead
    assert ys[3] < ys[0] < ys[2]                                  # up is towards the top of the picture
    img = ev.render_eye(_snap(), layout, w, h)
    assert img.shape == (h, w, 3) and img.dtype == np.uint8
    assert img[ys[0], xs[0]].tolist() == [229, 229, 229]          # luminance 0.9 -> grey 229
    assert img[ys[1], xs[1]].tolist() == [25, 25, 25]
    assert img[ys[3], xs[3]].tolist() == [0, 0, 0]


def test_eye_view_draws_the_horizon_and_the_ahead_meridian(layout):
    w, h = 440, 200
    img = ev.render_eye(_snap(lum=(0.5,) * 4), layout, w, h)
    y0 = round(ev.EL_TOP / (ev.EL_TOP - ev.EL_BOTTOM) * (h - 1))
    assert img[y0, 0].tolist() == list(ev.GRID_HORIZON)


def test_motion_picture_is_orange_for_expansion_and_cyan_for_a_small_object(layout):
    w, h = 440, 200
    xs, ys = layout.pixels(w, h)
    quiet = ev.render_motion(_snap(), layout, w, h)
    assert quiet[ys[2], xs[2]].tolist() == [34, 34, 42]
    img = ev.render_motion(_snap(exp=(0, 0, 3.0, 0), obj=(0, 1.0, 0, 0)), layout, w, h)
    r, g, b = img[ys[2], xs[2]]
    assert r > 200 and r > g > b                                  # orange
    r, g, b = img[ys[1], xs[1]]
    assert b > 200 and b > g > r                                  # cyan


def test_neuron_tiles_show_only_their_own_group(layout):
    w, h = 440, 200
    # one active LC4 neuron on column 0 (second group = right-hand top tile); everything else silent
    snap = _snap(cols=(0, 0, 1, 2), rates=(0.0, 100.0, 0.0, 0.0))
    img = ev.render_neurons(snap, layout, w, h)
    x0, y0, tw, th = ev.tile_rect(1, w, h)
    tile = img[y0:y0 + th, x0:x0 + tw].astype(int)
    lit = (tile[..., 0] > 150) & (tile[..., 1] > 100)
    assert lit.any()
    assert not ((img[:h // 2, :w // 2].astype(int)[..., 0] > 150).any())      # LPLC2 tile stays dark
    assert ev.group_summary(snap, 1) == "1/1 active · 100 Hz"


def test_neuron_tile_clips_neurons_to_its_own_rectangle(layout):
    w, h = 440, 200
    snap = _snap(cols=(0, 1, 2, 3), rates=(150, 150, 150, 150), sizes=(4, 0, 0, 0))
    img = ev.render_neurons(snap, layout, w, h)
    assert (img[:, w // 2:, 0] < 100).all() and (img[h // 2:, :, 0] < 100).all()


def test_empty_group_summary_and_empty_snapshot_render(layout):
    snap = _snap(cols=(), rates=(), sizes=(0, 0, 0, 0))
    assert ev.group_summary(snap, 0) == "none in this circuit"
    for render in ev.RENDERERS:
        assert render(snap, layout, 220, 100).shape == (100, 220, 3)


@pytest.mark.skipif(not FIELD.exists(), reason="needs data/circuits/field.npz")
def test_real_layout_renders_a_full_size_snapshot():
    layout = ev.EyeLayout.load()
    m = len(layout)
    assert 1000 < m <= ev.MAX_COLS
    rng = np.random.default_rng(0)
    cols = rng.integers(0, m, 900)
    snap = ev.Snapshot(3, rng.random(m).astype(np.float32), rng.random(m).astype(np.float32) * 3,
                       rng.random(m).astype(np.float32), cols.astype(np.int32),
                       (rng.random(900) * 100).astype(np.float32), (210, 104, 164, 422))
    for render in ev.RENDERERS:
        img = render(snap, layout, 440, 200)
        assert img.shape == (200, 440, 3) and img.max() > 100
    # the dataset's eyes: both together look below the horizon more than above it
    assert (layout.el < 0).sum() > (layout.el > 0).sum()


def test_worker_publishes_what_the_drive_saw_only_when_asked():
    """_Eye._publish with a stand-in drive: the arrays end up in the shared preview in the group order."""
    from neuropest.runner import _Eye

    shared = (ctypes.c_float * ev.PREVIEW_LEN)()
    eye = _Eye(None, None, None, shared)
    d = SimpleNamespace(
        last=(np.array([0.2, 0.8, 0.5], np.float32), np.array([0.0, 2.0, 0.0], np.float32),
              np.array([0.0, 0.0, 1.0], np.float32)),
        loom_idx=np.arange(3), n_lplc2=2, ret_idx=np.arange(1), obj_idx=np.arange(2),
        loom_col=np.array([0, 1, 2]), ret_col=np.array([1]), obj_col=np.array([2, 0]))
    rates = np.array([10, 20, 30, 5, 40, 50], np.float32)
    eye._publish(d, rates)
    s = ev.read_snapshot(np.frombuffer(shared, dtype=np.float32))
    assert s.sizes == (2, 1, 1, 2) and s.seq == 1
    np.testing.assert_allclose(s.lum, [0.2, 0.8, 0.5])
    assert list(s.group(0)[0]) == [0, 1] and list(s.group(1)[1]) == [30.0]
    assert list(s.group(2)[0]) == [1] and list(s.group(3)[1]) == [40.0, 50.0]
    eye._publish(SimpleNamespace(last=None), rates)                 # no frame yet: nothing is written
    assert ev.read_snapshot(np.frombuffer(shared, dtype=np.float32)).seq == 1


def test_preview_flag_has_its_own_free_input_slot():
    from neuropest.runner import I_INJECT_CUE_NEAR, I_PREVIEW, INP_LEN

    assert I_INJECT_CUE_NEAR < I_PREVIEW < INP_LEN


@pytest.mark.skipif(not FIELD.exists(), reason="needs data/circuits/field.npz and eye.npz")
def test_real_eye_in_the_worker_loop_publishes_a_preview_of_an_approaching_cursor():
    """_Eye with the real VisionDrive (every receptive-field neuron of the dataset as the 'circuit'): switch the eye
    on, let a cursor rush at the fly, read the preview back like the GUI does."""
    import time

    from neuropest.runner import (I_CX, I_CY, I_HEAD, I_HEIGHT, I_PREVIEW, I_SKITTISH, I_STAMP, I_VISION, I_X, I_Y,
                                  INP_LEN, O_VISION, OUT_LEN, VISION_PERIOD_MS, _Eye)
    from neuropest.vision import VisualField

    field = VisualField.load()
    net = SimpleNamespace(ids=field.rf_id.copy())
    brain = SimpleNamespace(vision_calls=0, set_vision=lambda *a: None, clear_vision=lambda: None)
    shared = (ctypes.c_float * ev.PREVIEW_LEN)()
    eye = _Eye(net, None, None, shared)
    inp, out = np.zeros(INP_LEN), np.zeros(OUT_LEN)
    inp[I_VISION], inp[I_PREVIEW], inp[I_HEIGHT], inp[I_SKITTISH] = 1.0, 1.0, 140.0, 1.0
    inp[I_X], inp[I_Y], inp[I_HEAD] = 500.0, 500.0, 0.0
    sim_ms, t = 0.0, time.perf_counter()
    deadline = t + 60.0
    while not eye.on and time.perf_counter() < deadline:            # the drive is built on a thread
        eye.update(brain, inp, out, sim_ms)
        time.sleep(0.02)
    assert eye.on and out[O_VISION] == 1.0
    stamp = 1.0
    for k in range(60):                                             # parked far away, then a 800 px/s dash
        d = 400.0 if k < 20 else max(40.0, 400.0 - 16.0 * (k - 20) * 1.0)
        inp[I_CX], inp[I_CY] = 500.0 + d, 500.0
        stamp += VISION_PERIOD_MS / 1000.0
        inp[I_STAMP] = stamp
        sim_ms += VISION_PERIOD_MS
        eye.update(brain, inp, out, sim_ms)
        if k == 10:
            quiet = ev.read_snapshot(np.frombuffer(shared, dtype=np.float32))
    snap = ev.read_snapshot(np.frombuffer(shared, dtype=np.float32))
    assert quiet is not None and snap is not None and snap.seq > quiet.seq
    assert snap.sizes[0] > 0 and snap.sizes[1] > 0 and snap.sizes[3] > 0
    assert quiet.group(0)[1].max() == 0.0 and quiet.group(1)[1].max() == 0.0     # nothing approaches: no looming
    assert snap.group(0)[1].max() > 10 and snap.group(1)[1].max() > 10     # LPLC2 and LC4 fire for the dash
    assert snap.expansion.max() > 0.5 and snap.lum.min() < 0.2        # the cursor disc is dark on the panorama
    layout = ev.EyeLayout.load()
    for render in ev.RENDERERS:
        assert render(snap, layout).any()


@pytest.mark.skipif(not FIELD.exists(), reason="needs data/circuits/field.npz and eye.npz")
def test_eye_applies_the_food_odor_gate_through_the_detector_gain_before_the_rate_caps():
    """Scaling the capped rates (max_retreat 20 Hz) by a partial gate would keep the retreat input under the ~12 Hz
    LPC1 -> MDN threshold for good while the looming input could still reach take-off: no retreat step any more. Gating
    the gain keeps the freeze -> retreat -> take-off order; a strong enough looming still reaches the retreat cap."""
    import time

    from neuropest.brain import odor_gate
    from neuropest.runner import (I_CX, I_CY, I_HEAD, I_HEIGHT, I_HUNGER, I_PHERO_DRIVE, I_SKITTISH, I_STAMP, I_VISION,
                                  I_X, I_Y, INP_LEN, OUT_LEN, VISION_PERIOD_MS, _Eye)
    from neuropest.vision import VisualField

    field = VisualField.load()

    def dash(phero, hunger):
        seen = []
        brain = SimpleNamespace(set_vision=lambda idx, rates, *a: seen.append(np.array(rates)), clear_vision=lambda: None)
        eye = _Eye(SimpleNamespace(ids=field.rf_id.copy()), None, None, None)
        inp, out = np.zeros(INP_LEN), np.zeros(OUT_LEN)
        inp[I_VISION], inp[I_HEIGHT], inp[I_SKITTISH] = 1.0, 140.0, 1.0
        inp[I_PHERO_DRIVE], inp[I_HUNGER] = phero, hunger
        inp[I_X], inp[I_Y], inp[I_HEAD] = 500.0, 500.0, 0.0
        sim_ms, deadline = 0.0, time.perf_counter() + 60.0
        while not eye.on and time.perf_counter() < deadline:
            eye.update(brain, inp, out, sim_ms)
            time.sleep(0.02)
        assert eye.on
        stamp = 1.0
        for k in range(60):                                         # parked, then a 1600 px/s dash to 30 px
            d = 400.0 if k < 20 else max(30.0, 400.0 - 32.0 * (k - 20))
            inp[I_CX], inp[I_CY] = 500.0 + d, 500.0
            stamp += VISION_PERIOD_MS / 1000.0
            inp[I_STAMP] = stamp
            sim_ms += VISION_PERIOD_MS
            eye.update(brain, inp, out, sim_ms)
        n_loom, n_ret = len(eye.drive.loom_idx), len(eye.drive.ret_idx)
        r = np.array(seen)
        return eye.drive, r[:, :n_loom].max(), r[:, n_loom:n_loom + n_ret].max()

    _, loom0, ret0 = dash(0.0, 1.0)
    g = odor_gate(0.1)
    drive, loom_g, ret_g = dash(0.1, 1.0)
    assert 0.0 < g < 1.0 and drive.skittish == pytest.approx(g)     # the gate went into the detector gain
    p = drive.params
    assert ret0 == pytest.approx(p.max_retreat)                     # the dash saturates the retreat cells ...
    assert ret_g > p.max_retreat * g + 1.0                          # ... and still goes past cap x gate under the odor
    assert loom_g < loom0                                           # while the looming drive is weakened
    _, loom_full, ret_full = dash(0.9, 1.0)
    assert loom_full == 0.0 and ret_full == 0.0                     # a strong odor silences both
