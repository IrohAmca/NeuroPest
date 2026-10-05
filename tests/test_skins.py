"""Tests for multi-skin rendering and skin selection."""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from neuropest.paths import SKINS_DIR
from neuropest.render import AVAILABLE_SKINS, SkinManager, draw_fly
from neuropest.states import FLY, FREEZE, GROOM, RETREAT, STAND, WALK


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_available_skins():
    assert "classic" in AVAILABLE_SKINS
    assert "chubby" in AVAILABLE_SKINS
    assert "cyborg" in AVAILABLE_SKINS
    assert "candy" in AVAILABLE_SKINS
    assert "cartoon" in AVAILABLE_SKINS
    assert len(AVAILABLE_SKINS) >= 5


def test_skin_manager_assets_exist(qapp):
    mgr = SkinManager(SKINS_DIR)
    for key in ("chubby", "cyborg", "candy", "cartoon"):
        # Test loading each skin
        for state in (STAND, WALK, FLY, RETREAT, GROOM, FREEZE):
            pm = mgr.get_frame(key, state, phase=0.5)
            assert pm is not None, f"Skin {key} returned None for state {state}"
            assert not pm.isNull(), f"Skin {key} returned null pixmap for state {state}"
            assert pm.width() > 0 and pm.height() > 0


def test_skin_manager_fallback(qapp):
    mgr = SkinManager(SKINS_DIR)
    # Non-existent skin returns None
    assert mgr.get_frame("non_existent_skin", WALK, phase=0.0) is None


def test_draw_fly_all_skins(qapp):
    img = QImage(200, 200, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)

    for skin in AVAILABLE_SKINS.keys():
        for state in (STAND, WALK, FLY, GROOM):
            draw_fly(p, 100.0, 100.0, 0.0, state, phase=0.5, scale=1.0, feed_glow=0.0, skin=skin)
            draw_fly(p, 100.0, 100.0, 0.5, state, phase=1.5, scale=1.5, feed_glow=0.7, skin=skin)

    p.end()


def test_skin_thumbnails_exist(qapp):
    from neuropest.paths import SKIN_THUMBNAILS
    assert SKIN_THUMBNAILS.is_dir()
    for key in AVAILABLE_SKINS.keys():
        thumb_p = SKIN_THUMBNAILS / f"{key}.png"
        assert thumb_p.exists(), f"Missing thumbnail for {key}"
        img = QImage(str(thumb_p))
        assert not img.isNull()
        assert img.width() == 64 and img.height() == 64


def test_control_skin_selection(qapp):
    from neuropest.app import Overlay
    from neuropest.control import Control
    from neuropest.runner import Runner

    runner = Runner()
    overlay = Overlay(runner)
    assert overlay.skin == "classic"

    ctrl = Control(overlay, runner)
    assert hasattr(ctrl, "skin_buttons")
    assert len(ctrl.skin_buttons) == len(AVAILABLE_SKINS)

    # Click each visual skin button
    for key, btn in ctrl.skin_buttons.items():
        assert not btn.icon().isNull()
        btn.click()
        assert overlay.skin == key
        assert btn.isChecked()
        assert key in ctrl.skin_desc_label.text() or AVAILABLE_SKINS[key] in ctrl.skin_desc_label.text()

    runner.stop()

