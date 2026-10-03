"""Funnel geometry, retinotopic detectors and the image-driven input.

The geometry and detector tests run on synthetic columns; the ones that need the connectome-derived eye data
(tools/build_eye.py) are skipped when it has not been built.
"""
import warnings

import numpy as np
import pytest

from neuropest import flywire
from neuropest.paths import CACHE, EYE, FIELD
from neuropest.vision import Features, Retina, VisualField, image_scene, plane_hits, sample_scene, sample_scenes
from neuropest.visual import VisionDrive, cursor_scene

DT = 1 / 60
needs_eye = pytest.mark.skipif(not (CACHE.exists() and EYE.exists() and FIELD.exists()),
                               reason="needs the FlyWire cache and the eye data (tools/build_eye.py)")


def retina_of(az_deg, el_deg) -> Retina:
    az, el = np.radians(np.asarray(az_deg, np.float32)), np.radians(np.asarray(el_deg, np.float32))
    n = len(az)
    return Retina(np.arange(n, dtype=np.int32), az, el, np.zeros(n, np.uint8), np.zeros(n, np.uint8))


# ------------------------------------------------------------------------------------------- the funnel
def test_funnel_geometry_and_columns_above_the_horizon_do_not_warn():
    r = retina_of([0.0, 90.0, 0.0, 0.0], [-45.0, -45.0, 10.0, -1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # inf * cos(psi) used to raise a RuntimeWarning
        hx, hy, hits = plane_hits(r, 500.0, 300.0, 0.0, 100.0, 2200.0)
        turned = plane_hits(r, 500.0, 300.0, np.pi / 2, 100.0, 2200.0)
    assert list(hits) == [True, True, False, False]         # looking up, or too close to the horizon: no screen
    assert np.allclose([hx[0], hy[0]], [600.0, 300.0], atol=0.5)     # 45 deg down from 100 px: 100 px ahead
    assert np.allclose([hx[1], hy[1]], [500.0, 400.0], atol=0.5)     # to the fly's right is +y on the screen
    assert np.isfinite(hx).all() and np.isfinite(hy).all()
    assert np.allclose([turned[0][0], turned[1][0]], [500.0, 400.0], atol=0.5)    # facing +y: ahead is +y


def test_sample_scenes_matches_sample_scene():
    r = retina_of(np.linspace(-60, 60, 9), np.full(9, -30.0))
    a, b = cursor_scene(660.0, 300.0, 40.0), cursor_scene(700.0, 330.0, 40.0)
    both = sample_scenes(r, (a, b), 500.0, 300.0, 0.0, 100.0)
    assert np.array_equal(both[0], sample_scene(r, a, 500.0, 300.0, 0.0, 100.0))
    assert np.array_equal(both[1], sample_scene(r, b, 500.0, 300.0, 0.0, 100.0))
    assert both[0].min() == 0.0 and both[0].max() == 0.5   # the disk is seen by some columns, the rest is screen


def test_image_scene_is_bilinear_and_neutral_outside():
    s = image_scene(np.array([[0.0, 1.0], [0.0, 1.0]], np.float32), outside=0.25)
    assert np.allclose(s(np.array([0.5, -5.0, 9.0]), np.array([0.5, 0.0, 0.0])), [0.5, 0.25, 0.25])


# ------------------------------------------------------------------------------------------- detectors
def synthetic_field(step_deg=6.0) -> VisualField:
    """Two patches of medulla columns on the sphere (left eye az -120..20, right eye az -20..120)."""
    dirs, eyes = [], []
    for eye, (a0, a1) in enumerate(((-120.0, 20.0), (-20.0, 120.0))):
        for az in np.arange(a0, a1 + 1e-6, step_deg):
            for el in np.arange(-60.0, 40.0 + 1e-6, step_deg):
                a, e = np.radians(az), np.radians(el)
                dirs.append([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
                eyes.append(eye)
    none = np.zeros(0)
    return VisualField(np.array(dirs, np.float32), np.array(eyes, np.uint8), none.astype(np.int64),
                       np.zeros((0, 3), np.float32), none.astype(np.float32), none.astype(np.int16),
                       none.astype(np.int32), none.astype(np.uint8), ["LPLC2"])


def direction(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


@pytest.fixture(scope="module")
def field():
    return synthetic_field()


def disk(field, centre, radius_deg, dark=0.0, back=0.5):
    ang = np.degrees(np.arccos(np.clip(field.col_dir.astype(np.float64) @ centre, -1, 1)))
    return np.where(ang <= radius_deg, dark, back).astype(np.float32)


def peaks(field, scene, frames, settle=30):
    """Strongest expansion and object feature while `scene(i)` plays (frame i < 0: settling on frame 0)."""
    f = Features(field)
    f.reset(np.full(len(field.col_dir), 0.5, np.float32))
    top = dict(expansion=0.0, object=0.0)
    for i in range(-settle, frames):
        out = f.update(scene(i), scene(i - 1), DT)
        if i >= 0:
            top["expansion"] = max(top["expansion"], float(out["expansion"].max()))
            top["object"] = max(top["object"], float(out["object"].max()))
    return top


AHEAD_BELOW = direction(0.0, -20.0)


def ramp(i, frames=24):
    return min(max(i, 0), frames) / frames


def test_expansion_fires_for_a_growing_object_of_either_polarity(field):
    dark = peaks(field, lambda i: disk(field, AHEAD_BELOW, 6.0 + 18.0 * ramp(i)), 40)
    bright = peaks(field, lambda i: disk(field, AHEAD_BELOW, 6.0 + 18.0 * ramp(i), dark=1.0, back=0.3), 40)
    assert dark["expansion"] > 1.0 and bright["expansion"] > 1.0


def test_expansion_stays_silent_for_contraction_translation_and_wide_field_change(field):
    shrinking = peaks(field, lambda i: disk(field, AHEAD_BELOW, 24.0 - 18.0 * ramp(i)), 40)
    sliding = peaks(field, lambda i: disk(field, direction(-40.0 + max(i, 0), -20.0), 12.0), 60)    # 60 deg/s
    darkening = peaks(field, lambda i: np.full(len(field.col_dir), 0.5 - 0.5 * ramp(i), np.float32), 40)
    still = peaks(field, lambda i: disk(field, AHEAD_BELOW, 20.0), 40)
    for name, p in (("shrinking", shrinking), ("sliding", sliding), ("darkening", darkening), ("still", still)):
        assert p["expansion"] < 0.05, name


def test_small_object_feature_prefers_a_small_contrasting_spot(field):
    spot = peaks(field, lambda i: disk(field, AHEAD_BELOW, 4.0), 20)
    flat = peaks(field, lambda i: np.full(len(field.col_dir), 0.2, np.float32), 20)
    assert spot["object"] > 0.5 and flat["object"] < 0.01


def test_pooled_outputs_follow_the_requested_rows(field):
    f = Features(field)
    f.reset(np.full(len(field.col_dir), 0.5, np.float32))
    rows = np.array([3, 10, 100], np.int32)
    out = {}
    for i in range(-30, 30):
        out = f.update(disk(field, AHEAD_BELOW, 6.0 + 18.0 * ramp(i)), disk(field, AHEAD_BELOW, 6.0 + 18.0 * ramp(i - 1)),
                       DT, rows, rows)
    full = Features(field)
    full.reset(np.full(len(field.col_dir), 0.5, np.float32))
    for i in range(-30, 30):
        ref = full.update(disk(field, AHEAD_BELOW, 6.0 + 18.0 * ramp(i)), disk(field, AHEAD_BELOW, 6.0 + 18.0 * ramp(i - 1)), DT)
    assert np.allclose(out["expansion_pooled"][rows], ref["expansion_pooled"][rows])
    assert np.allclose(out["object_pooled"][rows], ref["object_pooled"][rows])
    others = np.setdiff1d(np.arange(len(field.col_dir)), rows)
    assert not out["expansion_pooled"][others].any()              # columns nobody reads are left alone


# --------------------------------------------------------------------------- the cursor through the funnel
@needs_eye
def test_cursor_disk_drives_looming_neurons_only_while_it_approaches():
    net = flywire.load_cache().prefix(5000)
    drive = VisionDrive(net)
    assert drive.usable and len(drive.idx) > 500
    fly = (1000.0, 700.0, 0.0)

    def loom_peak(path, seconds):
        drive.reset()
        top = 0.0
        for k in range(int(seconds * 60)):
            now, before = path(k / 60), path((k - 1) / 60)
            idx, rates = drive.step(cursor_scene(*now, 30.0), cursor_scene(*before, 30.0), *fly, DT)
            assert len(idx) == len(rates) and np.isfinite(rates).all() and rates.min() >= 0.0
            top = max(top, float(rates[:len(drive.loom_idx)].max()))
        return top

    still = loom_peak(lambda t: (1150.0, 700.0), 1.0)
    dash = loom_peak(lambda t: (max(1050.0, 1400.0 - 1500.0 * max(0.0, t - 0.3)), 700.0), 1.0)
    assert still < 1.0 and dash > 50.0
    receding = loom_peak(lambda t: (1050.0 + 400.0 * max(0.0, t - 3.0), 700.0), 4.0)    # near for 3 s, then away
    assert receding < 10.0                                      # leaving is not looming (slow baseline)
