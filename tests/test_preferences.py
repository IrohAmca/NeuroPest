import json
import os
import pytest
from PySide6.QtWidgets import QApplication

from neuropest.control import Control
from neuropest.preferences import Preferences
from neuropest.runner import EngineConfig
from neuropest.theme import apply_theme


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    apply_theme(a)
    return a


class FakeRunner:
    def __init__(self):
        self.cfg = EngineConfig("toy", 146, 0.5)
        self.bias, self.skittish, self.wall_pain = 0.65, 1.0, 1.0
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


class FakeOverlay:
    scale = 1.0
    skin = "classic"
    home = None
    touch_groom_enabled = False
    pheromone_enabled = True
    cursor_phero_mode = "attract"

    def __init__(self):
        self.shown = False

    def setVisible(self, v):
        self.shown = v

    def set_home(self, h):
        self.home = h

    def update(self):
        pass


def test_preferences_defaults():
    p = Preferences()
    assert p.circuit == "flywire"
    assert p.neurons == 15_000
    assert p.dt == 0.5
    assert p.symmetry == "individual"
    assert p.learning is True
    assert p.hardware == "auto"
    assert p.scale == 1.0
    assert p.skin == "classic"
    assert p.hunger_enabled is True
    assert p.metabolic_rate == 1.0
    assert p.touch_groom_enabled is False
    assert p.bias == 0.65
    assert p.skittish == 1.0
    assert p.wall_pain == 1.0
    assert p.vision_enabled is False
    assert p.eye_height == 140.0
    assert p.pheromone_enabled is True
    assert p.cursor_phero_mode == "attract"


def test_preferences_save_and_load(tmp_path):
    f = tmp_path / "custom_prefs.json"
    p = Preferences(
        circuit="toy",
        neurons=5000,
        dt=0.1,
        symmetry="symmetric",
        learning=False,
        hardware="gpu",
        gpu_name="RTX 4090",
        gpu_backend="Vulkan",
        gpu_index=2,
        scale=2.5,
        skin="cyber",
        hunger_enabled=False,
        metabolic_rate=2.0,
        touch_groom_enabled=True,
        bias=0.8,
        skittish=1.5,
        wall_pain=0.5,
        vision_enabled=True,
        eye_height=180.0,
        pheromone_enabled=False,
        cursor_phero_mode="repel",
    )
    p.save(f)
    assert f.exists()

    loaded = Preferences.load(f)
    assert loaded.circuit == "toy"
    assert loaded.neurons == 5000
    assert loaded.dt == 0.1
    assert loaded.symmetry == "symmetric"
    assert loaded.learning is False
    assert loaded.hardware == "gpu"
    assert loaded.gpu_name == "RTX 4090"
    assert loaded.gpu_backend == "Vulkan"
    assert loaded.gpu_index == 2
    assert loaded.scale == 2.5
    assert loaded.skin == "cyber"
    assert loaded.hunger_enabled is False
    assert loaded.metabolic_rate == 2.0
    assert loaded.touch_groom_enabled is True
    assert loaded.bias == 0.8
    assert loaded.skittish == 1.5
    assert loaded.wall_pain == 0.5
    assert loaded.vision_enabled is True
    assert loaded.eye_height == 180.0
    assert loaded.pheromone_enabled is False
    assert loaded.cursor_phero_mode == "repel"


def test_preferences_corrupt_file_fallback(tmp_path):
    f = tmp_path / "corrupt.json"
    f.write_text("{invalid_json: true", encoding="utf-8")
    loaded = Preferences.load(f)
    assert loaded.scale == 1.0
    assert loaded.circuit == "flywire"


def test_preferences_sanitization():
    p = Preferences(
        scale=99.0,          # should clamp to 4.0
        metabolic_rate=0.01, # should clamp to 0.2
        bias=-0.5,           # should clamp to 0.0
        skittish=0.01,       # should clamp to 0.1
        wall_pain=10.0,      # should clamp to 3.0
        eye_height=10.0,     # should clamp to 40.0
        circuit="unknown",   # fallback to flywire
        symmetry="invalid",  # fallback to individual
        hardware="invalid",  # fallback to auto
        cursor_phero_mode="?", # fallback to attract
    )
    p._sanitize()
    assert p.scale == 4.0
    assert p.metabolic_rate == 0.2
    assert p.bias == 0.0
    assert p.skittish == 0.1
    assert p.wall_pain == 3.0
    assert p.eye_height == 40.0
    assert p.circuit == "flywire"
    assert p.symmetry == "individual"
    assert p.hardware == "auto"
    assert p.cursor_phero_mode == "attract"


def test_control_ui_changes_persist_to_file(app, tmp_path):
    pref_file = tmp_path / "ui_prefs.json"
    prefs = Preferences()
    prefs.save(pref_file)

    r = FakeRunner()
    o = FakeOverlay()
    c = Control(o, r, prefs=prefs, auto_scan_gpus=False)

    # Change skin
    c._on_skin_selected("dark")
    assert prefs.skin == "dark"
    reloaded = Preferences.load(pref_file)
    assert reloaded.skin == "dark"

    # Change scale
    c.stack.setCurrentIndex(5)
    c._on_scale_changed(20)  # scale 2.0
    assert prefs.scale == 2.0
    reloaded = Preferences.load(pref_file)
    assert reloaded.scale == 2.0

    # Change hunger and touch
    c.hunger_enable_box.setChecked(False)
    assert prefs.hunger_enabled is False
    reloaded = Preferences.load(pref_file)
    assert reloaded.hunger_enabled is False

    c.touch_groom_box.setChecked(True)
    assert prefs.touch_groom_enabled is True
    reloaded = Preferences.load(pref_file)
    assert reloaded.touch_groom_enabled is True

    # Change pheromone mode
    c.cursor_phero_combo.setCurrentIndex(1)  # repel
    assert prefs.cursor_phero_mode == "repel"
    reloaded = Preferences.load(pref_file)
    assert reloaded.cursor_phero_mode == "repel"

    # Change advanced sliders
    c._on_bias_changed(85)  # 0.85
    assert prefs.bias == 0.85
    reloaded = Preferences.load(pref_file)
    assert reloaded.bias == 0.85

    c._on_wall_pain_changed(150)  # 1.5
    assert prefs.wall_pain == 1.5
    reloaded = Preferences.load(pref_file)
    assert reloaded.wall_pain == 1.5


def test_preferences_save_when_file_locked(tmp_path):
    pref_file = tmp_path / "locked_prefs.json"
    p = Preferences(skin="classic")
    p.save(pref_file)
    assert pref_file.exists()

    # Hold the file open for reading (simulating Windows file watcher / antivirus / editor)
    with open(pref_file, "r", encoding="utf-8") as _holder:
        p.skin = "cyborg"
        p.save(pref_file)

    reloaded = Preferences.load(pref_file)
    assert reloaded.skin == "cyborg"

