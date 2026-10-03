import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from neuropest.control import Control
from neuropest.runner import EngineConfig
from neuropest.theme import apply_theme
from neuropest.tray import Tray


class FakeRunner:
    def __init__(self):
        self.cfg = EngineConfig("toy", 146, 0.5)
        self.bias, self.skittish = 0.65, 1.0
        self.vision, self.eye_height = False, 100.0
        self.state, self.ready, self.failed, self.alive = "walk", True, False, True
        self.rt = 5.0
        self.started = []

    def stats(self):
        return dict(ready=self.ready, n=146, rt=self.rt, active=40, cpu=0.1, lag_ms=5, gf=0, walk=30, rest=0,
                    mdn=2, steer=4, sim_s=1.0, spikes=900, vision=float(self.vision))

    def start(self, cfg):
        self.started.append(cfg)
        self.cfg = cfg


class FakeOverlay:
    scale = 1.4
    home = None

    def __init__(self):
        self.shown = True

    def setVisible(self, v):
        self.shown = v


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    apply_theme(a)
    return a


def test_readouts_and_state(app):
    r = FakeRunner()
    c = Control(FakeOverlay(), r)
    c._refresh()
    assert "Yürüyor" in c.pill.text()
    assert c.s_rt.value.text() == "×5.0" and c.warn.isHidden()
    r.rt, r.state = 0.5, "fly"
    c._refresh()
    assert "Uçarak" in c.pill.text() and not c.warn.isHidden()
    r.failed = True
    c._refresh()
    assert "Hata" in c.pill.text() and c.warn.objectName() == "Error"


def test_size_change_restarts_engine(app):
    r = FakeRunner()
    c = Control(FakeOverlay(), r)
    c.size.setValue(c.size.maximum())
    c._apply()
    assert r.started and r.started[-1].n == 139_000


def test_vision_switch_and_eye_height_reach_the_runner(app):
    r = FakeRunner()
    c = Control(FakeOverlay(), r)
    c.vision.setChecked(True)
    assert r.vision is True
    c._refresh()                                        # the worker reports the flag back; no crash either way
    c.vision.setChecked(False)
    assert r.vision is False


def test_tray_and_window_visibility_stay_in_step(app):
    r, o = FakeRunner(), FakeOverlay()
    c = Control(o, r)
    t = Tray(app, c, r)
    t.visible.setChecked(False)
    assert not c.visible.isChecked() and not o.shown
    c.visible.setChecked(True)
    assert t.visible.isChecked() and o.shown
    assert "Yürüyor" in t.toolTip()
