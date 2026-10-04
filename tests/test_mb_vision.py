"""The mushroom body reads the image through the visual projection neurons that reach the Kenyon cells.

Synthetic columns and receptive fields (no connectome data needed), like tests/test_vision.py."""
from types import SimpleNamespace

import numpy as np
import pytest

from neuropest.brain import Brain
from neuropest.mushroom import CUE_PN_TYPES, MBParams, MushroomBody
from neuropest.toy_circuit import build
from neuropest.vision import VisualField
from neuropest.visual import VisionDrive

from test_vision import direction, synthetic_field

FR = 0.02
FLY = (0.0, 0.0, 0.0)


def field_with_cells() -> VisualField:
    """Synthetic columns plus a few cells with receptive fields: LPLC2 / LC4 / LPC1 / LC10 (what the engine drives) and
    two of the cursor group, two of the looming group of the mushroom body, all looking ahead."""
    f = synthetic_field()
    types = ["LPLC2", "LC4", "LPC1", "LC10a", "aMe12", "MTe32", "aMe26", "MTe37", "LTe72", "XYZ"]
    cells = [("LPLC2", 1), ("LC4", 2), ("LPC1", 3), ("LC10a", 4), ("aMe12", 5), ("MTe32", 6), ("aMe26", 7),
             ("MTe37", 8), ("XYZ", 9)]
    col = int(np.argmax(f.col_dir @ direction(0.0, 0.0)))
    k = len(cells)
    f.rf_id = np.array([100 + i for _, i in cells], np.int64)
    f.rf_dir = np.tile(f.col_dir[col], (k, 1)).astype(np.float32)
    f.rf_conc = np.full(k, 0.9, np.float32)
    f.rf_type = np.array([types.index(t) for t, _ in cells], np.int16)
    f.rf_col = np.full(k, col, np.int32)
    f.rf_side = np.zeros(k, np.uint8)
    f.types = types
    return f


@pytest.fixture(scope="module")
def drive():
    net = SimpleNamespace(ids=np.array([100, 101, 102, 103, 104], np.int64))      # only LPLC2..LC10a are "in the tier"
    return VisionDrive(net, field=field_with_cells())


def play(drive, path, seconds):
    drive.reset()
    near, loom = [], []
    for k in range(int(seconds / FR)):
        t = k * FR
        drive.step_cursor(path(t), path(t - FR), *FLY, FR)
        near.append(drive.mb_cue["cursor_near"])
        loom.append(drive.mb_cue["looming"])
    return np.array(near), np.array(loom)


def test_cells_outside_the_tier_still_have_their_receptive_field_columns(drive):
    f = drive.field
    assert len(drive.mb_near_col) == 2 and len(drive.mb_loom_col) == 2          # aMe12, MTe32 / aMe26, MTe37 (LTe72: none)
    assert len(drive.field.cell_columns(["nothing"])) == 0
    assert len(drive.idx) == len(drive.loom_idx) + len(drive.ret_idx) + len(drive.obj_idx)   # no forced spikes for them
    assert set(CUE_PN_TYPES["cursor_near"]) >= {"aMe12", "MTe32"} and f.types.index("aMe12") >= 0


def test_a_moving_cursor_close_by_drives_the_cursor_group_a_still_one_does_not(drive):
    moving, _ = play(drive, lambda t: (100.0, 60.0 * np.sin(2 * np.pi * t)), 2.0)
    still, still_loom = play(drive, lambda t: (100.0, 0.0), 2.0)
    far, _ = play(drive, lambda t: (2000.0, 0.0), 2.0)
    assert moving.max() > 0.5 and moving.mean() > 0.1
    assert still.max() == 0.0 and still_loom.max() == 0.0                       # change detectors: a static scene is silent
    assert far.max() == 0.0


def test_a_dash_drives_the_looming_group_and_a_slide_barely_does(drive):
    dash = lambda t: (max(50.0, 400.0 - 800.0 * max(0.0, t - 0.5)), 0.0)        # noqa: E731
    slide = lambda t: (150.0, -300.0 + 200.0 * t)                               # noqa: E731
    _, loom_dash = play(drive, dash, 2.0)
    _, loom_slide = play(drive, slide, 2.0)
    assert loom_dash.max() > 0.5 and loom_slide.max() < 0.2


def test_levels_stay_in_0_1_and_follow_the_skittishness(drive):
    dash = lambda t: (max(50.0, 400.0 - 800.0 * max(0.0, t - 0.5)), 0.0)        # noqa: E731
    drive.skittish = 1.0
    _, a = play(drive, dash, 2.0)
    drive.skittish = 0.3
    _, b = play(drive, dash, 2.0)
    drive.skittish = 1.0
    assert 0.0 <= b.max() < a.max() <= 1.0


# ------------------------------------------------------------------------------------------------ the Brain
def toy_brain():
    net = build(146, 1)
    net.groups["LC10_R"] = np.array([5], np.int32)
    net.groups["LC10_L"] = np.array([6], np.int32)
    return Brain(net)


def spy_cues(brain):
    seen = []
    step = brain.mb.step

    def spy(cues, reward, punishment, dt):
        seen.append(tuple(float(c) for c in cues))
        return step(cues, reward, punishment, dt)

    brain.mb.step = spy
    return seen


def test_the_mushroom_body_reads_the_image_cells_while_vision_is_on_and_the_numbers_otherwise():
    b = toy_brain()
    seen = spy_cues(b)
    b.set_stimulus(250.0, 0.0, 0.0)                                              # cursor numbers: 250 px away
    b.advance(50.0)
    assert seen[-1][2] == pytest.approx(0.5) and seen[-1][3] == 0.0
    b.set_vision(np.array([5], np.int32), np.array([10.0], np.float32), 0.0, (0.8, 0.3))
    b.advance(50.0)
    assert seen[-1][2] == pytest.approx(0.8) and seen[-1][3] == pytest.approx(0.3)   # the cells' own drive
    b.set_vision(np.array([5], np.int32), np.array([10.0], np.float32), 0.0)         # vision on, no cell cues given
    b.advance(50.0)
    assert seen[-1][2] == pytest.approx(0.5)
    b.set_vision(np.array([5], np.int32), np.array([10.0], np.float32), 0.0, (0.8, 0.3))
    b.clear_vision()
    b.advance(50.0)
    assert seen[-1][2] == pytest.approx(0.5)                                         # off again: the numbers


@pytest.mark.parametrize("wiring", ["random", "flywire"])
def test_seeing_the_cursor_while_feeding_makes_the_cursor_alone_wanted(wiring):
    if wiring == "flywire":
        from neuropest.paths import MUSHROOM
        if not MUSHROOM.exists():
            pytest.skip("real mushroom-body wiring not built")
    b = toy_brain()
    if wiring == "flywire":
        b.mb = MushroomBody.from_flywire()
    idx, rates = np.array([5], np.int32), np.array([1.0], np.float32)
    for _ in range(4):
        b.set_stimulus(1e6, 0.0, 0.0, phero_drive=0.9, at_target=1.0)               # feeding at the cursor's pheromone
        b.set_vision(idx, rates, 0.0, (0.9, 0.0))
        for _ in range(30):
            b.advance(50.0)
        b.set_stimulus(1e6, 0.0, 0.0)
        b.set_vision(idx, rates, 0.0, (0.0, 0.0))                                   # the cursor is gone
        for _ in range(60):
            b.advance(50.0)
    b.set_vision(idx, rates, 0.0, (0.9, 0.0))                                       # the cursor alone, no pheromone
    b.advance(50.0)
    assert b.valence > (0.3 if wiring == "random" else 0.1)      # the real gamma-d Kenyon cells learn less (probe_mb_vision)
    naive = toy_brain()
    naive.set_vision(idx, rates, 0.0, (0.9, 0.0))
    naive.advance(50.0)
    assert naive.valence == 0.0
    b.set_vision(idx, rates, 0.0, (0.0, 0.0))                                       # a silent image: nothing learned is read
    b.advance(50.0)
    assert b.valence == 0.0
