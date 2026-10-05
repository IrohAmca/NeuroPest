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
        self.valence = 0.0
        self.forgotten = 0
        self.started = []

    def stats(self):
        return dict(ready=self.ready, n=146, rt=self.rt, active=40, cpu=0.1, lag_ms=5, gf=0, walk=30, rest=0,
                    mdn=2, steer=4, sim_s=1.0, spikes=900, vision=float(self.vision))

    def start(self, cfg):
        self.started.append(cfg)
        self.cfg = cfg

    def forget(self):
        self.forgotten += 1


class FakeOverlay:
    scale = 1.4
    home = None
    touch_groom_enabled = False

    def __init__(self):
        self.shown = False

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
    c.hw.setCurrentIndex(1)
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


def test_pheromone_controls_reach_overlay(app):
    r, o = FakeRunner(), FakeOverlay()
    c = Control(o, r)
    # Check default enabled and mode
    assert c.phero_enable.isChecked() is True
    c.phero_enable.setChecked(False)
    assert o.pheromone_enabled is False
    c.phero_enable.setChecked(True)
    assert o.pheromone_enabled is True

    # Check cursor pheromone mode switching
    c.cursor_phero_combo.setCurrentIndex(1)  # repel
    assert o.cursor_phero_mode == "repel"
    c.cursor_phero_combo.setCurrentIndex(2)  # none
    assert o.cursor_phero_mode == "none"
    c.cursor_phero_combo.setCurrentIndex(0)  # attract
    assert o.cursor_phero_mode == "attract"



def test_tray_tooltip_and_open_control(app):
    r, o = FakeRunner(), FakeOverlay()
    c = Control(o, r)
    t = Tray(app, c, r)
    t._refresh()
    assert "Yürüyor" in t.toolTip()
    assert t.status.text() == "Sinek: Yürüyor"
    t._open_control()
    assert c.isVisible()


def test_gpus_are_listed_only_when_asked_for_or_needed(app, monkeypatch):
    import time

    from neuropest import control

    calls = []
    gpu = dict(index=3, name="Fake GPU", backend="Vulkan", type="DiscreteGPU")
    monkeypatch.setattr(control, "list_gpus", lambda: calls.append(1) or [gpu])
    monkeypatch.setattr(control.importlib.util, "find_spec", lambda name, *a: object() if name == "wgpu" else None)

    def finish(c):
        end = time.time() + 5
        while not c._gpu_scanned and time.time() < end:
            c._gpus_found()
            time.sleep(0.02)
        assert c._gpu_scanned

    r = FakeRunner()
    c = control.Control(FakeOverlay(), r, auto_scan_gpus=False)
    assert not calls and c.hw.itemText(2) == control.SCAN_ITEM      # manual mode: opening window scans nothing
    c.size.setValue(0)
    c._apply()
    assert not calls                                                # a small size on "Otomatik" does not need it
    c.hw.setCurrentIndex(2)                                         # picking the entry asks for the scan
    c._hw_activated(2)
    assert c.hw.currentIndex() == 0
    finish(c)
    assert len(calls) == 1 and c.hw.itemText(2).startswith("GPU: Fake GPU") and c.hw.count() == 3

    r2 = FakeRunner()                                               # a big size on "Otomatik" waits for the scan
    c2 = control.Control(FakeOverlay(), r2, auto_scan_gpus=False)
    c2.size.setValue(c2.size.maximum())
    c2._apply()
    assert not r2.started and c2._gpu_scanning
    finish(c2)
    assert r2.started[-1].backend == "gpu" and r2.started[-1].adapter == 3 and len(calls) == 2


def test_gpus_auto_scan_and_preference_restore(app, monkeypatch, tmp_path):
    import time
    from neuropest import control
    from neuropest.preferences import Preferences

    calls = []
    gpus = [
        dict(index=0, name="Fast GPU", backend="Vulkan", type="DiscreteGPU"),
        dict(index=1, name="Integrated GPU", backend="Vulkan", type="IntegratedGPU"),
    ]
    monkeypatch.setattr(control, "list_gpus", lambda: calls.append(1) or gpus)
    monkeypatch.setattr(control.importlib.util, "find_spec", lambda name, *a: object() if name == "wgpu" else None)

    # Saved preference chooses Fast GPU
    pref_file = tmp_path / "pref.json"
    prefs = Preferences(hardware="gpu", gpu_name="Fast GPU", gpu_backend="Vulkan", gpu_index=0)
    prefs.save(pref_file)

    r = FakeRunner()
    c = control.Control(FakeOverlay(), r, prefs=prefs, auto_scan_gpus=True)
    assert len(calls) == 1                                         # automatic scan triggered on init!

    # Wait for scan to complete and simulate timeout
    c._gpus_found()
    assert c._gpu_scanned
    assert c.hw.currentIndex() == 2                                # Fast GPU selected at index 2
    assert "Fast GPU" in c.hw.currentText()
    assert r.cfg.backend == "gpu" and r.cfg.adapter == 0           # Runner started with preferred GPU!


def test_learning_page_shows_valence_forgets_and_switches_learning(app):
    r = FakeRunner()
    r.forgot = 0
    r.forget = lambda: setattr(r, "forgot", r.forgot + 1)
    base = r.stats
    r.stats = lambda: dict(base(), valence=-0.6, v_motor=1.25, gear="fly_long")
    c = Control(FakeOverlay(), r)
    c._refresh()
    assert "korku" in c.s_val.value.text() and c.s_vm.value.text() == "1.25" and "kovalama" in c.s_gear.value.text()
    c.forget_btn.click()
    assert r.forgot == 1
    c.learn_enable.setChecked(False)
    assert r.started and r.started[-1].learning is False
