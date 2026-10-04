"""Funnel geometry, retinotopic detectors and the image-driven input.

The geometry and detector tests run on synthetic columns; the ones that need the connectome-derived eye data
(tools/build_eye.py) are skipped when it has not been built.
"""
import warnings

import numpy as np
import pytest

from neuropest import flywire
from neuropest.paths import CACHE, EYE, FIELD
from neuropest.vision import (Features, Retina, VisualField, image_scene, plane_hits, sample_scene, sample_scenes,
                              sphere_luminance)
from neuropest.visual import DISK_PLANE, VisionDrive, VisionParams, cursor_scene

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
    spot = peaks(field, lambda i: disk(field, direction(max(i, 0) * 1.5, -20.0), 4.0), 20)
    flat = peaks(field, lambda i: np.full(len(field.col_dir), 0.2, np.float32), 20)
    still = peaks(field, lambda i: disk(field, AHEAD_BELOW, 4.0), 20)
    assert spot["object"] > 0.5 and flat["object"] < 0.01 and still["object"] < 0.01


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
    drive = VisionDrive(net, DISK_PLANE)
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


@needs_eye
def test_looming_survives_a_more_vertical_view():
    """From 250 px up a 30 px disk was smaller than a column and the fly never fled (E = 0 for every approach)."""
    net = flywire.load_cache().prefix(5000)
    drive = VisionDrive(net, DISK_PLANE)
    fly = (1000.0, 700.0, 0.0)
    assert drive.halo_px == drive.params.halo_px                   # unchanged at the calibrated height

    def loom_peak(speed, height):
        drive.eye_height = height
        drive.reset()
        path = lambda t: (max(1050.0, 1400.0 - speed * max(0.0, t - 0.3)), 700.0)    # noqa: E731
        top = 0.0
        for k in range(75):
            idx, rates = drive.step(cursor_scene(*path(k / 60), drive.halo_px),
                                    cursor_scene(*path((k - 1) / 60), drive.halo_px), *fly, DT)
            top = max(top, float(rates[:len(drive.loom_idx)].max()))
        return top

    for height in (100.0, 250.0, 350.0):
        assert loom_peak(1500.0, height) > 50.0, height            # a fast dash still drives the looming neurons
        assert loom_peak(150.0, height) < 30.0, height             # a slow walk up to the fly still does not


# --------------------------------------------------------------------------- the eye-level disc
def test_sphere_disc_sits_at_the_cursor_azimuth_and_subtends_atan_r_over_d(field):
    cd = field.col_dir
    lum = sphere_luminance(cd, (1100.0, 700.0), 1000.0, 700.0, 0.0, 30.0)         # 100 px straight ahead
    ang = np.degrees(np.arccos(np.clip(cd.astype(np.float64) @ direction(0.0, 0.0), -1, 1)))
    half = np.degrees(np.arctan2(30.0, 100.0))                                      # 16.7 deg
    assert (lum[ang < half - 3.0] == 0.0).all() and (lum[ang > half + 3.0] == 0.5).all()
    right = sphere_luminance(cd, (1000.0, 800.0), 1000.0, 700.0, 0.0, 30.0)         # +y on screen is the fly's right
    assert cd[right < 0.25][:, 1].mean() > 0.9
    turned = sphere_luminance(cd, (1000.0, 800.0), 1000.0, 700.0, np.pi / 2, 30.0)  # heading +y: it is straight ahead
    assert np.array_equal(turned, lum)
    # the rim is soft: a disc smaller than a column still dims the nearest column by its coverage
    far = sphere_luminance(cd, (1700.0, 728.0), 1000.0, 700.0, 0.0, 30.0)         # 2.3 deg off a column, radius 2.5 deg
    assert 0.0 < (0.5 - far).max() < 0.5


def test_sphere_expansion_does_not_depend_on_the_eye_height(field):
    """The disc faces the fly, so eye height does not enter at all (the plane disk it replaces shrank into a thin
    band as the eye rose and the detectors fell silent)."""
    def peak(path):
        f = Features(field)
        f.reset(np.full(len(field.col_dir), 0.5, np.float32))
        top = 0.0
        for k in range(-30, 70):
            t = k / 60
            now = sphere_luminance(field.col_dir, path(t), 0.0, 0.0, 0.0, 30.0)
            before = sphere_luminance(field.col_dir, path(t - 1 / 60), 0.0, 0.0, 0.0, 30.0)
            out = f.update(now, before, DT)
            top = max(top, float(out["expansion"].max()))
        return top

    dash = lambda t: (max(60.0, 400.0 - 1500.0 * max(0.0, t)), 0.0)                  # noqa: E731
    slide = lambda t: (150.0, -400.0 + 400.0 * max(0.0, t))                           # noqa: E731
    still = lambda t: (150.0, 0.0)                                                    # noqa: E731
    top = peak(dash)
    assert top > 3.0 and peak(slide) < 0.25 * top and peak(still) < 0.05


@needs_eye
def test_sphere_cursor_drives_looming_at_any_azimuth_and_height_does_not_matter():
    net = flywire.load_cache().prefix(5000)
    drive = VisionDrive(net)
    assert drive.params.cursor_model == "sphere" and drive.halo_px == drive.params.halo_px
    fly = (1000.0, 700.0, 0.0)
    nl = len(drive.loom_idx)

    def loom_peak(deg, height, speed=1500.0):
        drive.eye_height = height
        drive.reset()
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        path = lambda t: (1000.0 + c * max(60.0, 400.0 - speed * max(0.0, t - 0.3)),     # noqa: E731
                          700.0 + s * max(60.0, 400.0 - speed * max(0.0, t - 0.3)))
        top = 0.0
        for k in range(75):
            idx, rates = drive.step_cursor(path(k / 60), path((k - 1) / 60), *fly, DT)
            assert np.isfinite(rates).all() and rates.min() >= 0.0
            top = max(top, float(rates[:nl].max()))
        return top

    results = {(deg, h): loom_peak(deg, h) for deg in (0.0, 60.0, -100.0) for h in (20.0, 100.0, 300.0)}
    assert min(results.values()) > 50.0
    for deg in (0.0, 60.0, -100.0):                                 # the same at every height
        assert results[(deg, 20.0)] == results[(deg, 300.0)]
    assert loom_peak(0.0, 100.0, speed=150.0) < 60.0                # a slow walk up to the fly drives far less
    behind = loom_peak(180.0, 100.0)
    assert behind < 1.0                                             # nothing is seen behind (no columns there)


@needs_eye
def test_lplc2_reads_the_size_of_the_object_and_lc4_its_speed():
    from dataclasses import replace

    net = flywire.load_cache().prefix(5000)
    fly = (1000.0, 700.0, 0.0)

    def peaks_of(params, radius, speed):
        drive = VisionDrive(net, replace(params, halo_px=radius))
        n2, nl = drive.n_lplc2, len(drive.loom_idx)
        assert 0 < n2 < nl
        path = lambda t: (1000.0 + max(50.0, 400.0 - speed * max(0.0, t - 0.3)), 700.0)     # noqa: E731
        top2 = top4 = 0.0
        for k in range(90):
            idx, rates = drive.step_cursor(path(k / 60), path((k - 1) / 60), *fly, DT)
            top2, top4 = max(top2, float(rates[:n2].max())), max(top4, float(rates[n2:nl].max()))
        return top2, top4

    base = VisionParams()
    small2, small4 = peaks_of(base, 15.0, 800.0)
    big2, big4 = peaks_of(base, 60.0, 800.0)
    assert small4 > 40.0 and small2 < 0.7 * small4                  # a small fast object: LC4 leads, LPLC2 is held back
    assert big2 > 0.9 * big4 > 40.0                                 # a big one fills LPLC2's field: both respond
    assert big2 > 2.0 * small2                                      # LPLC2 grows with size ...
    slow2, slow4 = peaks_of(base, 60.0, 150.0)
    assert slow4 < 0.6 * big4                                       # ... and LC4 with the expansion speed
    s2, s4 = peaks_of(replace(base, loom_split=False), 15.0, 800.0)  # the first version: one rate for both
    assert s2 == pytest.approx(s4)


def test_receptive_field_weights_are_flat_by_default_and_gaussian_on_request(field):
    flat, tuned = Features(field), Features(field, rf_sigma_deg=10.0)
    assert (flat.pool_w == 1.0).all() and len(flat.pool_w) == len(flat.pool[1])
    assert tuned.pool_w.max() == 1.0 and 0.0 < tuned.pool_w.min() < 0.1       # own column 1, 30 deg away exp(-4.5)
