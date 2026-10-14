"""Tests for 3D Mushroom Body Geometry and Visualization (FlyWire v783).

Verifies coordinate extraction, normalization, 3D perspective projection,
interactive camera controls, color modes, telemetry binding, and UI rendering.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from neuropest.control import Control
from neuropest.mb_geometry import (
    DAN_PAM,
    DAN_PPL1,
    LOBE_ALPHABETA,
    LOBE_GAMMA,
    Mushroom3DData,
    build_fallback_model,
    load_mushroom_3d,
    norm3d,
)
from neuropest.mb_view3d import (
    MODE_ACTIVITY,
    MODE_ANATOMY,
    MODE_PLASTICITY,
    MODE_VALENCE,
    PRESET_ANTERIOR,
    PRESET_DORSAL,
    PRESET_ISOMETRIC,
    PRESET_SAGITTAL,
    MushroomBody3DView,
)
from neuropest.paths import MUSHROOM_3D
from neuropest.runner import EngineConfig, Runner
from neuropest.theme import apply_theme


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    apply_theme(a)
    return a


# ----------------------------------------------------------------------
# Geometry Tests
# ----------------------------------------------------------------------

def test_norm3d_centers_coordinates_at_midline():
    raw_coords = np.array([
        [133200.0, 38000.0, 28000.0],  # exactly at midline / center
        [133200.0 + 38000.0, 38000.0 - 38000.0, 28000.0 + 38000.0],
    ], dtype=np.float32)

    norm = norm3d(raw_coords)
    assert np.allclose(norm[0], [0.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(norm[1], [1.0, -1.0, 1.0], atol=1e-5)


def test_load_mushroom_3d_cached_or_fallback():
    data = load_mushroom_3d()
    assert isinstance(data, Mushroom3DData)
    assert data.n_kc >= 1000
    assert data.n_mbon >= 40
    assert data.n_dan >= 100
    assert len(data.brain_wireframe) > 0
    assert "calyx_left" in data.lobe_centers
    assert "gamma_left" in data.lobe_centers
    assert "pam_reward" in data.lobe_centers

    # Check coordinate ranges are within reasonable brain bounds [-2.5, 2.5]
    assert np.abs(data.kc_soma).max() < 2.5
    assert np.abs(data.kc_ped).max() < 2.5
    assert np.abs(data.kc_target).max() < 2.5
    assert np.abs(data.mbon_pos).max() < 2.5
    assert np.abs(data.dan_pos).max() < 2.5


def test_build_fallback_model_structure():
    fallback = build_fallback_model()
    assert isinstance(fallback, Mushroom3DData)
    assert fallback.n_kc == 2000
    assert fallback.n_mbon == 96
    assert fallback.n_dan == 120
    assert len(fallback.brain_wireframe) > 0
    assert fallback.kc_lobe.shape == (2000,)
    assert fallback.kc_side.shape == (2000,)
    assert fallback.dan_cluster.shape == (120,)


# ----------------------------------------------------------------------
# 3D View Widget & Projection Tests
# ----------------------------------------------------------------------

def test_widget_initialization(app):
    data = load_mushroom_3d()
    view = MushroomBody3DView(data=data)
    assert view.data is data
    assert view.view_mode == MODE_VALENCE
    assert view.yaw == PRESET_ISOMETRIC[0]
    assert view.pitch == PRESET_ISOMETRIC[1]
    assert view.zoom == 220.0
    assert not view.auto_rotate


def test_project_points_math(app):
    view = MushroomBody3DView()
    view.resize(600, 400)
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ], dtype=np.float32)

    sx, sy, z, persp = view._project_points(pts)
    assert len(sx) == 4
    assert len(sy) == 4
    assert np.all(np.isfinite(sx))
    assert np.all(np.isfinite(sy))
    assert np.all(persp > 0.0)


def test_camera_presets_and_reset(app):
    view = MushroomBody3DView()

    view.set_preset(*PRESET_DORSAL)
    assert view.yaw == PRESET_DORSAL[0]
    assert view.pitch == PRESET_DORSAL[1]
    assert view.pan_x == 0.0 and view.pan_y == 0.0

    view.set_preset(*PRESET_ANTERIOR)
    assert view.yaw == PRESET_ANTERIOR[0]
    assert view.pitch == PRESET_ANTERIOR[1]

    view.set_preset(*PRESET_SAGITTAL)
    assert view.yaw == PRESET_SAGITTAL[0]
    assert view.pitch == PRESET_SAGITTAL[1]

    # Orbit, pan, zoom and reset
    view.pan_x = 50.0
    view.pan_y = -30.0
    view.zoom = 450.0
    view.auto_rotate = True

    view.reset_camera()
    assert view.yaw == PRESET_ISOMETRIC[0]
    assert view.pitch == PRESET_ISOMETRIC[1]
    assert view.zoom == 220.0
    assert view.pan_x == 0.0
    assert view.pan_y == 0.0
    assert not view.auto_rotate


def test_view_modes(app):
    view = MushroomBody3DView()
    for mode in (MODE_VALENCE, MODE_PLASTICITY, MODE_ANATOMY, MODE_ACTIVITY):
        view.set_view_mode(mode)
        assert view.view_mode == mode

    # Invalid mode should be ignored
    view.set_view_mode("invalid_mode")
    assert view.view_mode == MODE_ACTIVITY


def test_auto_rotate_and_animation_tick(app):
    view = MushroomBody3DView()
    view.resize(500, 400)
    view.setVisible(True)
    view.auto_rotate = True
    initial_yaw = view.yaw

    # Simulate tick
    view._on_tick()
    assert view.yaw > initial_yaw or abs(view.yaw - initial_yaw) > 1e-4

    # Pulse phase advancement
    p0 = view._pulse_phase
    view._on_tick()
    assert view._pulse_phase != p0


def test_rendering_all_modes_without_crash(app):
    """Render the widget to an offscreen QImage across all 4 modes."""
    data = load_mushroom_3d()
    view = MushroomBody3DView(data=data)
    view.resize(640, 480)

    # Set some synthetic telemetry to exercise all HUD and render branches
    view.valence = 0.65
    view.pam = 0.8
    view.ppl1 = 0.2
    view.kc_active_count = 120
    view.kc_weights[:200] = 0.5
    view.kc_weights[200:400] = -0.7
    view.active_kc_mask[:50] = True

    img = QImage(640, 480, QImage.Format_ARGB32)

    for mode in (MODE_VALENCE, MODE_PLASTICITY, MODE_ANATOMY, MODE_ACTIVITY):
        view.set_view_mode(mode)
        painter = QPainter(img)
        view.paintEvent(None)
        painter.end()
        assert not img.isNull()


def test_mouse_interactions(app):
    view = MushroomBody3DView()
    view.resize(600, 400)

    class FakeMouseEvent:
        def __init__(self, pos, button=Qt.LeftButton, modifiers=Qt.NoModifier):
            self._pos = pos
            self._btn = button
            self._mod = modifiers

        def pos(self):
            return self._pos

        def button(self):
            return self._btn

        def modifiers(self):
            return self._mod

    # Left click & drag -> orbit
    view.mousePressEvent(FakeMouseEvent(QPoint(100, 100), Qt.LeftButton))
    yaw0, pitch0 = view.yaw, view.pitch
    view.mouseMoveEvent(FakeMouseEvent(QPoint(120, 110)))
    assert view.yaw != yaw0
    assert view.pitch != pitch0
    view.mouseReleaseEvent(FakeMouseEvent(QPoint(120, 110)))

    # Right click & drag -> pan
    view.mousePressEvent(FakeMouseEvent(QPoint(100, 100), Qt.RightButton))
    px0, py0 = view.pan_x, view.pan_y
    view.mouseMoveEvent(FakeMouseEvent(QPoint(130, 80)))
    assert view.pan_x == px0 + 30
    assert view.pan_y == py0 - 20
    view.mouseReleaseEvent(FakeMouseEvent(QPoint(130, 80)))

    # Double click -> reset
    view.mouseDoubleClickEvent(FakeMouseEvent(QPoint(100, 100)))
    assert view.yaw == PRESET_ISOMETRIC[0]
    assert view.pan_x == 0.0


# ----------------------------------------------------------------------
# Control UI Integration Tests
# ----------------------------------------------------------------------

class MockRunner:
    def __init__(self):
        self.cfg = EngineConfig("toy", 146, 0.5)
        self.bias, self.skittish = 0.65, 1.0
        self.vision, self.eye_height = False, 100.0
        self.state, self.ready, self.failed, self.alive = "walk", True, False, True
        self.rt = 5.0
        self.injected = []

    def stats(self):
        return dict(
            ready=self.ready, n=146, rt=self.rt, active=40, cpu=0.1, lag_ms=5, gf=0, walk=30,
            rest=0, mdn=2, steer=4, sim_s=1.0, spikes=900, vision=0.0,
            valence=0.45, v_motor=1.1, gear="walk",
            pam=0.75, ppl1=0.1, mbon_app=1.2, mbon_av=0.8,
            kc_active=85, cue_food=0.8, cue_near=0.0, weights_dirty=1,
        )

    def inject_reward(self, mag=1.0):
        self.injected.append(("reward", mag))

    def inject_punish(self, mag=1.0):
        self.injected.append(("punish", mag))

    def inject_cue_food(self, mag=1.0):
        self.injected.append(("food", mag))

    def inject_cue_near(self, mag=1.0):
        self.injected.append(("near", mag))

    def get_mb_weights(self):
        return np.full(100, 0.3, dtype=np.float32)

    def forget(self):
        self.injected.append(("forget", 1.0))


class MockOverlay:
    scale = 1.4
    home = None
    pheromone_enabled = True
    cursor_phero_mode = "attract"

    def setVisible(self, _):
        pass


def test_control_learning_tab_and_sandbox_injection(app):
    runner = MockRunner()
    ctrl = Control(MockOverlay(), runner)
    ctrl.resize(760, 560)
    ctrl.show()

    # Switch to Learning & Memory tab (index 6)
    ctrl._switch_tab(6)
    app.processEvents()

    assert ctrl.mb_view3d is not None
    # 3D view is optional: initially hidden to save system resources
    assert not ctrl.mb_view3d.isVisible()

    # Toggle view ON
    ctrl._toggle_3d_view(True)
    app.processEvents()
    assert ctrl.mb_view3d.isVisible()

    # Verify telemetry fetching from runner into 3D view
    ctrl.mb_view3d._fetch_telemetry()
    assert abs(ctrl.mb_view3d.valence - 0.45) < 1e-4
    assert abs(ctrl.mb_view3d.pam - 0.75) < 1e-4
    assert ctrl.mb_view3d.kc_active_count == 85

    # Test test actions
    ctrl._inject_test_action("reward")
    assert ("reward", 1.0) in runner.injected
    assert ("food", 1.0) in runner.injected

    ctrl._inject_test_action("punish")
    assert ("punish", 1.0) in runner.injected
    assert ("near", 1.0) in runner.injected

    ctrl._inject_test_action("food")
    assert ("food", 1.0) in runner.injected

    ctrl._inject_test_action("cursor")
    assert ("near", 1.0) in runner.injected

    ctrl._forget()
    assert ("forget", 1.0) in runner.injected


def test_help_overlay_toggle_and_rendering(app):
    data = load_mushroom_3d()
    view = MushroomBody3DView(data=data)
    view.resize(700, 500)
    view.show()
    app.processEvents()

    # Pre-render to build button geometry
    img = QImage(700, 500, QImage.Format_ARGB32)
    p = QPainter(img)
    view.paintEvent(None)
    p.end()

    assert "help" in view._btn_rects
    help_rect = view._btn_rects["help"]
    assert help_rect.width() > 0

    class FakeMouseEvent:
        def __init__(self, pos, button=Qt.LeftButton, modifiers=Qt.NoModifier):
            self._pos = pos
            self._btn = button
            self._mod = modifiers

        def pos(self):
            return self._pos

        def button(self):
            return self._btn

        def modifiers(self):
            return self._mod

    # 1. Click help button to toggle overlay ON
    assert not view._show_help
    view.mousePressEvent(FakeMouseEvent(help_rect.center()))
    assert view._show_help

    # Render with help overlay visible
    p = QPainter(img)
    view.paintEvent(None)
    p.end()
    assert not img.isNull()

    # Verify close_help button was registered
    assert "close_help" in view._btn_rects
    close_rect = view._btn_rects["close_help"]

    # 2. Click close button to toggle overlay OFF
    view.mousePressEvent(FakeMouseEvent(close_rect.center()))
    assert not view._show_help

    # 3. Toggle back on, then click outside card to dismiss
    view.mousePressEvent(FakeMouseEvent(help_rect.center()))
    assert view._show_help
    # Click at (2, 2) which is outside card_rect
    view.mousePressEvent(FakeMouseEvent(QPoint(2, 2)))
    assert not view._show_help

    # 4. Hover detection for help button
    view._check_hover(help_rect.center())
    assert "Guide" in view._hover_text


def test_adaptive_timer_throttling(app):
    view = MushroomBody3DView()
    view.resize(600, 400)
    view.show()
    app.processEvents()

    # Idle tick -> timer interval adjusts to 200ms
    view.auto_rotate = False
    view._is_panning = False
    view.pam = 0.0
    view.ppl1 = 0.0
    view.kc_active_count = 0
    view._on_tick()
    assert view._timer.interval() == 200

    # Active tick (e.g. dopamine burst) -> timer interval throttles to 33ms
    view.pam = 0.75
    view._on_tick()
    assert view._timer.interval() == 33


def test_neural_network_module_optional_and_system_load_warning(app):
    """Verify that neural network visualization is optional, explains the circuit, and notes system load."""
    runner = MockRunner()
    ctrl = Control(MockOverlay(), runner)
    ctrl.show()
    app.processEvents()

    # 1. Check Learning & Memory Tab (index 6)
    ctrl._switch_tab(6)
    app.processEvents()

    # Explanation text exists and describes the network
    assert hasattr(ctrl, "neural_desc_label")
    assert "Mushroom Body" in ctrl.neural_desc_label.text()
    assert "Kenyon" in ctrl.neural_desc_label.text()

    # System load warning note exists and mentions additional load / CPU/GPU
    assert hasattr(ctrl, "lbl_system_load_warn")
    warn_text = ctrl.lbl_system_load_warn.text()
    assert "system load" in warn_text.lower()
    assert "cpu/gpu" in warn_text.lower()

    # By default, visualization is optional (hidden to save system load)
    assert not ctrl.mb_view3d_container.isVisible()
    assert not ctrl.mb_view3d.isVisible()
    assert "View" in ctrl.btn_toggle_3d.text()

    # Toggle view on
    ctrl.btn_toggle_3d.click()
    app.processEvents()
    assert ctrl.mb_view3d_container.isVisible()
    assert ctrl.mb_view3d.isVisible()
    assert "Hide" in ctrl.btn_toggle_3d.text()

    # Toggle view off
    ctrl.btn_toggle_3d.click()
    app.processEvents()
    assert not ctrl.mb_view3d_container.isVisible()
    assert not ctrl.mb_view3d.isVisible()

    # 2. Check Circuit & Hardware Tab (index 4): 3D visualizer is kept out of hardware tab
    ctrl._switch_tab(4)
    app.processEvents()

    assert not hasattr(ctrl, "circ_3d_desc")
    assert not hasattr(ctrl, "btn_circ_view_3d")


def test_whole_brain_circuits_data_structure():
    """Verify CX, VPN, ALPN, and DN circuits in geometry data."""
    data = load_mushroom_3d()
    assert data.n_cx > 0
    assert data.n_vpn > 0
    assert data.n_alpn > 0
    assert data.n_dn >= 18
    assert len(data.cx_sub) == data.n_cx
    assert len(data.vpn_side) == data.n_vpn
    assert len(data.alpn_side) == data.n_alpn
    assert len(data.dn_type) == data.n_dn

    fallback = build_fallback_model()
    assert fallback.n_cx > 0
    assert fallback.n_vpn > 0
    assert fallback.n_alpn > 0
    assert fallback.n_dn >= 18


def test_anatomical_layer_toggles(app):
    """Test layer toggle methods, button clicks, and rendering with selective layers."""
    data = load_mushroom_3d()
    view = MushroomBody3DView(data=data)
    view.resize(640, 480)

    # Initial layer states all True
    assert view.layer_mb
    assert view.layer_cx
    assert view.layer_visual
    assert view.layer_olfactory
    assert view.layer_motor
    assert view.layer_wireframe

    # Toggle individual layers
    view.toggle_layer("cx")
    assert not view.layer_cx
    view.toggle_layer("cx")
    assert view.layer_cx

    view.set_layer_visible("motor", False)
    assert not view.layer_motor

    # Render with various layer combinations
    img = QImage(640, 480, QImage.Format_ARGB32)
    p = QPainter(img)
    view.paintEvent(None)
    p.end()

    # Toggle via simulated button clicks
    class FakeClick:
        def __init__(self, pos):
            self._pos = pos
        def pos(self):
            return self._pos
        def button(self):
            return Qt.LeftButton
        def modifiers(self):
            return Qt.NoModifier

    if "layer_visual" in view._btn_rects:
        rect = view._btn_rects["layer_visual"]
        initial_val = view.layer_visual
        view.mousePressEvent(FakeClick(rect.center()))
        assert view.layer_visual != initial_val


def test_sensorimotor_signal_flow_and_rates(app):
    """Test action potential propagation when sensory cues and motor rates fire."""
    data = load_mushroom_3d()
    view = MushroomBody3DView(data=data)
    view.resize(640, 480)
    view.setVisible(True)

    # Inject visual cue & motor take-off rates
    view.cue_near = 0.8
    view.rate_gf = 15.0
    view.rate_walk = 22.0
    view.rate_steer = 8.0
    view.rate_mdn = 4.0
    view.cue_food = 0.7

    # Render frame with active signals
    img = QImage(640, 480, QImage.Format_ARGB32)
    p = QPainter(img)
    view.paintEvent(None)
    p.end()

    # Advance tick and verify pulse propagation & heading updates
    view._pulse_phase = 0.1
    view._on_tick()
    assert view._pulse_phase > 0.1

class _Click:
    def __init__(self, pos, btn=Qt.LeftButton, mod=Qt.NoModifier):
        self._pos, self._btn, self._mod = pos, btn, mod

    def pos(self):
        return self._pos

    def button(self):
        return self._btn

    def modifiers(self):
        return self._mod


class _LabRunner:
    """Runner double with the lab API: records the calls, reports what it was asked as applied."""
    ready = True

    def __init__(self):
        self.calls = []
        self.silenced = set()
        self.stimulated = ()

    def stats(self):
        return dict(ready=True, valence=0.0, pam=0.0, ppl1=0.0, mbon_app=1.0, mbon_av=1.0, kc_active=0, gf=0.0,
                    walk=0.0, mdn=0.0, steer=0.0, groom=0.0, cue_food=0.0, cue_near=0.0, weights_dirty=0,
                    heading=0.0, state="stand")

    def get_mb_weights(self):
        return np.zeros(0, dtype=np.float32)

    def stimulate(self, group, intensity=1.0, ms=300.0):
        self.calls.append(("stimulate", group, ms))
        self.stimulated = (group,)
        return True

    def stop_stimulation(self):
        self.calls.append(("stop",))
        self.stimulated = ()

    def silence(self, group, on=True):
        self.calls.append(("silence", group, on))
        (self.silenced.add if on else self.silenced.discard)(group)

    def clear_lab(self):
        self.calls.append(("clear",))
        self.silenced.clear()

    def inject_reward(self, amount=1.0):
        self.calls.append(("reward", amount))

    def lab_status(self):
        from neuropest.states import LAB_GROUPS

        sil = tuple(g for g in LAB_GROUPS if g in self.silenced)
        return dict(available=LAB_GROUPS, silenced=sil, stimulated=self.stimulated, left_ms=0.0, gate=bool(sil),
                    acked=True, asked_silenced=sil)

    def lab_rates(self):
        return {"LC4": 12.5}


def _soma_pos(view, group):
    x, y, _ = view._members_xy(group)
    return QPoint(int(round(float(x[0]))), int(round(float(y[0]))))


def _lab_view(runner=None):
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=runner or _LabRunner())
    view.resize(900, 600)
    view.layer_mb = view.layer_cx = view.layer_olfactory = False   # only the lab's neurons can be picked
    return view


def test_lab_targets_are_the_circuit_groups_with_their_drawn_neurons(app):
    view = _lab_view()
    assert {"GF", "MDN", "WALK", "DNa02_L", "DNa02_R", "GROOM", "LC4", "LPLC2", "LPC1", "LC10_L", "LC10_R"} <= set(
        view._lab_members)
    for group, (system, idx) in view._lab_members.items():
        assert len(idx) and system in ("dn", "vpn")
        assert view._target_at(_soma_pos(view, group), threshold_px=3.0)[0] == group


def test_a_press_on_a_target_lights_it_while_held_and_a_miss_never_stimulates(app):
    runner = _LabRunner()
    view = _lab_view(runner)
    view.set_tool_mode("opto")
    view.mousePressEvent(_Click(QPoint(5, view.height() - 5)))      # empty space (no HUD there)
    view.mouseReleaseEvent(_Click(QPoint(5, view.height() - 5)))
    assert runner.calls == []                                        # a miss does nothing (it used to fire the GF)
    pos = _soma_pos(view, "GF")
    view.mousePressEvent(_Click(pos))
    assert runner.calls == [("stimulate", "GF", 10_000.0)] and view._held[0] == "GF"
    view._held = ("GF", view._held[1] - 1.0)                        # held for a second
    view.mouseReleaseEvent(_Click(pos))
    assert runner.calls[-1] == ("stop",) and view._held is None
    assert "Giant Fiber" in view._status[0]


def test_stimulation_and_silencing_are_sent_only_for_real_targets_of_a_running_engine(app):
    runner = _LabRunner()
    view = _lab_view(runner)
    assert view.stimulate_target("MDN") and runner.calls[-1] == ("stimulate", "MDN", 300.0)
    assert not view.stimulate_target("EB") and not view.stimulate_target("nope")
    assert view.toggle_silence("LC4") and runner.calls[-1] == ("silence", "LC4", True)
    view._fetch_telemetry()
    assert view._lab["silenced"] == ("LC4",)
    assert not view.stimulate_target("LC4")                         # silenced: light cannot make it spike
    assert "silenced" in view._status[0]
    assert view.toggle_silence("LC4") and runner.calls[-1] == ("silence", "LC4", False)
    assert view.stimulate_target("reward") and runner.calls[-1] == ("reward", 1.0)   # a sandbox signal
    n = len(runner.calls)
    runner.ready = False
    assert not view.stimulate_target("GF") and len(runner.calls) == n and view._status[2]   # warned, nothing sent


def test_the_silence_tool_toggles_and_the_chips_release(app):
    runner = _LabRunner()
    view = _lab_view(runner)
    view.set_tool_mode("silence")
    pos = _soma_pos(view, "LPLC2")
    view.mousePressEvent(_Click(pos))
    view.mouseReleaseEvent(_Click(pos))
    assert runner.calls[-1] == ("silence", "LPLC2", True)
    view._fetch_telemetry()
    view.grab()
    assert "chip_LPLC2" in view._btn_rects and "release_all" in view._btn_rects
    view.mousePressEvent(_Click(view._btn_rects["release_all"].center()))
    assert runner.calls[-1] == ("clear",)


def test_right_drag_pans_and_left_drag_orbits_in_every_tool_without_touching_the_lab(app):
    for tool in ("camera", "opto", "silence"):
        runner = _LabRunner()
        view = _lab_view(runner)
        view.set_tool_mode(tool)
        start = QPoint(30, view.height() - 30)
        view.mousePressEvent(_Click(start, Qt.RightButton))
        view.mouseMoveEvent(_Click(QPoint(start.x() + 30, start.y() - 20)))
        view.mouseReleaseEvent(_Click(QPoint(start.x() + 30, start.y() - 20), Qt.RightButton))
        assert (view.pan_x, view.pan_y) == (30.0, -20.0), tool
        yaw = view.yaw
        view.mousePressEvent(_Click(start))
        view.mouseMoveEvent(_Click(QPoint(start.x() + 40, start.y())))
        view.mouseReleaseEvent(_Click(QPoint(start.x() + 40, start.y())))
        assert view.yaw != yaw and runner.calls == [], tool


def test_hovering_a_target_rings_its_group_and_says_what_a_press_does(app):
    view = _lab_view()
    view.set_tool_mode("opto")
    view._fetch_telemetry()
    view._check_hover(_soma_pos(view, "LC4"))
    assert view._hover_target == "LC4" and "LC4" in view._hover_text and "12.5 Hz" in view._hover_text
    view.grab()                                                      # the overlay draws the ring and the markers


def test_hud_controls_never_overlap_at_any_width(app):
    inside_card = {"close_inspector", "card_stim", "card_silence", "card_sandbox"}
    for w in range(480, 1141, 60):
        for silenced in (("GF", "LC4"), ("GF", "LC4", "MDN", "WALK", "LPC1")):
            view = MushroomBody3DView(data=load_mushroom_3d())
            view.resize(w, 410)
            view._lab["silenced"] = view._lab["asked_silenced"] = silenced
            x, y, _ = view._members_xy("GF")
            view._pin(view._find_neuron_at(QPoint(int(x[0]), int(y[0])), 3.0, systems=("dn",)))
            view.grab()
            rects = list(view._btn_rects.items())
            assert "card" in view._btn_rects and "release_all" in view._btn_rects
            for i, (a, ra) in enumerate(rects):
                assert ra.right() < w and ra.bottom() < 410, (w, a)
                for b, rb in rects[i + 1:]:
                    if {a, b} & inside_card and "card" in (a, b):
                        continue                            # the card's own buttons sit on the card
                    assert not ra.intersects(rb), (w, silenced, a, b)


def test_vectorised_picking_matches_a_brute_force_search(app):
    import math as _m

    view = MushroomBody3DView(data=load_mushroom_3d())
    view.resize(800, 520)
    rng = np.random.default_rng(5)
    for yaw in (-0.45, 0.8, 2.2):
        view.yaw = yaw
        systems = [(s, pts) for s, pts, vis in view._pick_sources() if vis and len(pts)]
        for fx, fy in rng.uniform([0, 0], [800, 520], (60, 2)):
            x, y = int(fx), int(fy)                                 # the same point for both searches
            best = None
            dist = {}
            for system, pts in systems:
                sx, sy, _, _ = view._project_points(pts)
                for i in range(len(pts)):
                    d = _m.hypot(x - sx[i], y - sy[i])
                    dist[(system, i)] = d
                    if d <= 20.0 and (best is None or d < best[0]):
                        best = (d, system, i)
            got = view._pick(QPoint(x, y), 20.0)
            if best is None:
                assert got is None
            else:
                assert got is not None and abs(dist[got] - best[0]) < 1e-6    # the nearest (a tie may pick either)


def test_the_scene_at_rest_is_rendered_once_per_camera_pose(app):
    view = MushroomBody3DView(data=load_mushroom_3d())
    view.resize(700, 420)
    calls = []
    real = view._paint_base
    view._paint_base = lambda p: (calls.append(1), real(p))
    view.grab()
    view.grab()
    assert len(calls) == 1                                          # the camera stood still
    view.yaw += 0.2
    view.grab()
    assert len(calls) == 2


def test_frame_pacing_backs_off_when_a_frame_is_expensive(app):
    view = MushroomBody3DView()
    view.resize(600, 400)
    view.show()
    app.processEvents()
    view.pam = 0.75                                                  # something lives: fast frames ...
    view._paint_ms = 5.0
    view._on_tick()
    assert view._timer.interval() == 33
    view._paint_ms = 60.0                                            # ... unless a frame costs this much
    view._on_tick()
    assert view._timer.interval() >= 180


def test_neuron_connectome_inspector_search_and_reticle(app):
    """Test connectome inspection, metadata resolution, reticle rendering, and pinning."""
    data = load_mushroom_3d()
    view = MushroomBody3DView(data=data)
    view.resize(640, 480)
    view.setVisible(True)

    class FakeClick:
        def __init__(self, pos, btn=Qt.LeftButton, mod=Qt.NoModifier):
            self._pos = pos
            self._btn = btn
            self._mod = mod
        def pos(self):
            return self._pos
        def button(self):
            return self._btn
        def modifiers(self):
            return self._mod

    # 1. Inspect finding neurons by projecting known soma coordinates
    gf_mask = view._dn_type == "GF"
    assert np.any(gf_mask)
    gf_soma = view._dn_soma[gf_mask][0]
    sx, sy, _, valid = view._project_points(np.array([gf_soma], dtype=np.float32))
    assert valid[0]

    found_neuron = view._find_neuron_at(QPoint(int(sx[0]), int(sy[0])), threshold_px=30.0)
    assert found_neuron is not None
    assert "Giant Fiber" in found_neuron["name"]
    assert "ACh" in found_neuron["nt"]
    assert "Motor Descending" in found_neuron["system"]
    assert "partners" in found_neuron

    # 2. "inspect" is Explore: a click pins the card
    view.set_tool_mode("inspect")
    assert view.tool_mode == "camera"

    # Click on the neuron position -> pins it
    view.mousePressEvent(FakeClick(QPoint(int(sx[0]), int(sy[0]))))
    view.mouseReleaseEvent(FakeClick(QPoint(int(sx[0]), int(sy[0]))))
    assert view._inspected_pinned
    assert view._inspected_neuron is not None

    # 3. Render offscreen with active inspection reticle & holographic card
    img = QImage(640, 480, QImage.Format_ARGB32)
    p = QPainter(img)
    view.paintEvent(None)
    p.end()
    assert not img.isNull()

    # 4. Dynamic firing rate refresh on tick
    view.rate_gf = 42.5
    view._on_tick()
    assert "42.5 Hz" in view._inspected_neuron["rate_hz"]

    # 5. Dismiss inspector via close button
    if "close_inspector" in view._btn_rects:
        c_rect = view._btn_rects["close_inspector"]
        view.mousePressEvent(FakeClick(c_rect.center()))
        assert view._inspected_neuron is None
        assert not view._inspected_pinned


def test_translucent_neuropil_hulls_and_toggles(app):
    """Test 3D translucent volumetric neuropil hulls generation, depth sorting, and layer toggles."""
    data = load_mushroom_3d()
    view = MushroomBody3DView(data=data)
    view.resize(640, 480)

    # 1. Verify volumetric envelopes exist for all major brain regions
    assert hasattr(view, "_neuropil_volumes")
    assert len(view._neuropil_volumes) >= 7
    ids = {nv["id"] for nv in view._neuropil_volumes}
    assert {"al_l", "al_r", "ca_l", "ca_r", "eb", "ol_l", "ol_r"}.issubset(ids)

    for nv in view._neuropil_volumes:
        assert len(nv["vertices"]) > 0
        assert len(nv["faces"]) > 0
        assert nv["vertices"].shape[1] == 3
        assert nv["faces"].shape[1] == 3

    # 2. Render with hulls on
    view.layer_hulls = True
    img1 = QImage(640, 480, QImage.Format_ARGB32)
    p = QPainter(img1)
    view.paintEvent(None)
    p.end()
    assert not img1.isNull()

    # 3. Toggle layer off
    view.toggle_layer("hulls")
    assert not view.layer_hulls

    img2 = QImage(640, 480, QImage.Format_ARGB32)
    p = QPainter(img2)
    view.paintEvent(None)
    p.end()
    assert not img2.isNull()









# ----------------------------------------------------------------------
# Central-complex compass
# ----------------------------------------------------------------------

class _HeadingRunner:
    """Runner double that reports the fly's heading the way Runner.stats() does."""
    ready = True

    def __init__(self, heading=0.0, steer=0.0):
        self.heading = heading
        self.steer = steer

    def stats(self):
        return dict(ready=True, valence=0.0, pam=0.0, ppl1=0.0, mbon_app=1.0, mbon_av=1.0, kc_active=0,
                    gf=0.0, walk=0.0, mdn=0.0, steer=self.steer, groom=0.0, cue_food=0.0, cue_near=0.0,
                    weights_dirty=0, heading=self.heading)

    def get_mb_weights(self):
        return np.zeros(0, dtype=np.float32)


def _ring_node_nearest(n, angle):
    return int(np.argmin(np.abs((np.arange(n) * 2 * math.pi / n - angle + math.pi) % (2 * math.pi) - math.pi)))


def test_compass_bump_follows_the_fly_heading_not_the_animation_clock(app):
    runner = _HeadingRunner(1.0)
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=runner)
    n = len(view._eb_ring)
    assert n >= 3
    view._fetch_telemetry()

    def lit():
        return set(np.flatnonzero(view._eb_lit(n)).tolist())

    first = lit()
    assert _ring_node_nearest(n, 1.0) in first
    for clock in (0.0, 0.37, 0.8):                 # the pulse clock drives the other animations, not the compass
        view._pulse_phase = clock
        assert lit() == first

    runner.heading = 4.0                           # the fly turned: the bump moves with it
    view._fetch_telemetry()
    assert lit() != first
    assert _ring_node_nearest(n, 4.0) in lit()
    assert abs(view._compass_phase() - 4.0) < 1e-9


def test_compass_without_a_runner_heading_integrates_steering_at_the_body_turn_gain(app):
    from neuropest.fly import TURN_GAIN

    view = MushroomBody3DView(data=load_mushroom_3d())
    view.setVisible(True)
    view.rate_steer = 10.0
    h0 = view.heading_angle
    view._last_tick = view._last_tick - 0.1        # one 0.1 s tick
    view._on_tick()
    gained = (view.heading_angle - h0) % (2 * math.pi)
    assert 0.0 < gained <= 10.0 * 0.12 * TURN_GAIN
    assert view._fly_heading is None


def test_a_real_heading_stops_the_steering_integral(app):
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=_HeadingRunner(2.5, steer=20.0))
    view.setVisible(True)
    view._fetch_telemetry()
    h0 = view.heading_angle
    view._last_tick = view._last_tick - 0.1
    view._on_tick()
    assert view.rate_steer == 20.0                 # steering is on: without the real heading this would integrate
    assert view.heading_angle == h0
    assert abs(view._compass_phase() - 2.5) < 1e-9


def test_eb_nodes_are_ordered_around_the_ring_whatever_the_table_order(app):
    view = MushroomBody3DView(data=load_mushroom_3d())
    k = 12
    ang = np.linspace(0.0, 2 * math.pi, k, endpoint=False)
    # a tilted ring (mostly in the x-z plane), so ordering by the raw x-y angle would not do
    ring = np.stack([np.cos(ang), 0.1 * np.cos(ang), 0.6 * np.sin(ang)], axis=1).astype(np.float32)
    perm = np.random.default_rng(3).permutation(k)
    others = np.random.default_rng(4).normal(size=(5, 3)).astype(np.float32)
    view._cx_pts = np.vstack([ring[perm], others])
    view._cx_sub = np.array([0] * k + [1] * 5, dtype=np.int8)

    order = view._eb_ring_order()
    assert sorted(order.tolist()) == list(range(k))            # only the EB nodes
    a = ang[perm][order]
    steps = np.diff(np.concatenate([a, a[:1]]))
    assert np.allclose(np.abs((steps + math.pi) % (2 * math.pi) - math.pi), 2 * math.pi / k)


def test_real_eb_ring_is_a_proper_ring(app):
    """The cached FlyWire table lists the EB nodes by cell type; the ring order must be positional."""
    view = MushroomBody3DView(data=load_mushroom_3d())
    ring = view._eb_ring
    if len(ring) < 6:
        pytest.skip("no real central-complex data")
    p = view._cx_pts[ring].astype(np.float64)
    p -= p.mean(axis=0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    plane = p @ vt[:2].T
    ang = np.arctan2(plane[:, 1], plane[:, 0])
    assert np.all(np.diff(ang) >= 0)                           # angle increases along the drawn order


# ----------------------------------------------------------------------
# Kenyon-cell activity: the worker's real firing set, not a fixed mask
# ----------------------------------------------------------------------

class _KcRunner(_HeadingRunner):
    """Runner double that publishes the model's firing set like Runner.get_kc_active()."""

    def __init__(self, n_model=2000):
        super().__init__(0.0)
        self.n_model = n_model
        self.fired = np.zeros(n_model, dtype=bool)

    def get_kc_active(self):
        return self.fired.copy() if self.n_model else None


def test_active_kenyon_cells_are_the_models_own_firing_set(app):
    runner = _KcRunner(2000)
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=runner)
    assert view.data.n_kc >= 2000

    runner.fired[[10, 11, 500, 1999]] = True
    view._fetch_telemetry()
    assert view._kc_from_model and view._kc_model_n == 2000
    assert set(np.flatnonzero(view.active_kc_mask).tolist()) == {10, 11, 500, 1999}

    runner.fired[:] = False                          # another stimulus: another set, not the same fixed subset
    runner.fired[[7, 8, 9]] = True
    view._fetch_telemetry()
    assert {7, 8, 9} <= set(np.flatnonzero(view.active_kc_mask).tolist())       # (the earlier cells are still fading)


def test_a_firing_cell_stays_lit_for_a_moment_and_then_fades(app):
    import time as _time

    runner = _KcRunner(2000)
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=runner)
    view.setVisible(True)
    runner.fired[42] = True
    view._fetch_telemetry()
    assert view.active_kc_mask[42]
    runner.fired[:] = False                          # it stops firing
    view._last_tick = _time.perf_counter() - 0.05
    view._on_tick()
    assert view.active_kc_mask[42]                   # still lit 50 ms later
    for _ in range(10):
        view._last_tick = _time.perf_counter() - 0.1
        view._on_tick()
    assert not view.active_kc_mask.any()


def _patch_brightness(view, x, y, r=3):
    img = view.grab().toImage()
    return sum(img.pixelColor(x + dx, y + dy).red() + img.pixelColor(x + dx, y + dy).green()
               + img.pixelColor(x + dx, y + dy).blue() for dx in range(-r, r + 1) for dy in range(-r, r + 1))


def test_firing_cells_off_the_drawing_stride_are_still_painted(app):
    runner = _KcRunner(2000)
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=runner)
    view.resize(700, 520)
    view.layer_hulls = False
    step_s, step_t = view._soma_step, view._tract_step
    on_both = [i for i in range(1, 400) if i % step_s != 0 and i % step_t != 0][:5]
    assert on_both
    cell = on_both[0]
    runner.fired[on_both] = True
    view._fetch_telemetry()
    assert set(on_both) <= set(view._active_off_soma.tolist())
    assert set(on_both) <= set(view._active_off_tract.tolist())
    soma_tract = view._active_off_soma.copy(), view._active_off_tract.copy()

    # the soma pass alone: a spark on the cell's soma
    sx, sy, _, _ = view._project_points(view.data.kc_soma[[cell]])
    at_soma = int(round(float(sx[0]))), int(round(float(sy[0])))
    view._active_off_tract = np.zeros(0, dtype=np.int64)
    with_soma = _patch_brightness(view, *at_soma)
    view._active_off_soma = np.zeros(0, dtype=np.int64)
    assert with_soma > _patch_brightness(view, *at_soma) + 1000

    # the tract pass alone: the action-potential pulse halfway along the cell's own axon
    view._active_off_soma, view._active_off_tract = np.zeros(0, dtype=np.int64), soma_tract[1]
    view._pulse_phase = 0.5
    pts = [view._project_points(a[[cell]]) for a in (view.data.kc_soma, view.data.kc_ped, view.data.kc_target)]
    (s0, s1), (p0, p1), (t0, t1) = [(float(q[0][0]), float(q[1][0])) for q in pts]
    mid = int(round(0.25 * s0 + 0.5 * p0 + 0.25 * t0)), int(round(0.25 * s1 + 0.5 * p1 + 0.25 * t1))
    with_tract = _patch_brightness(view, *mid, r=2)
    view._active_off_tract = np.zeros(0, dtype=np.int64)
    assert with_tract > _patch_brightness(view, *mid, r=2) + 300


def test_hovering_a_firing_cell_between_the_samples_shows_that_cell_and_its_card_follows_it(app):
    import time as _time

    runner = _KcRunner(2000)
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=runner)
    view.resize(700, 520)
    view.setVisible(True)
    for layer in ("layer_visual", "layer_olfactory", "layer_motor", "layer_cx"):
        setattr(view, layer, False)
    cell = next(i for i in range(1, 400) if i % view._soma_step != 0)
    runner.fired[cell] = True
    view._fetch_telemetry()
    sx, sy, _, _ = view._project_points(view.data.kc_soma[[cell]])
    card = view._find_neuron_at(QPoint(int(round(float(sx[0]))), int(round(float(sy[0])))), threshold_px=1.5)
    assert card is not None and card["kc_index"] == cell
    assert card["rate_hz"].startswith("Active")

    view._inspected_neuron = card                    # pinned; the cell stops firing
    runner.fired[:] = False
    view._last_tick = _time.perf_counter() - 0.05
    view._on_tick()
    assert view._inspected_neuron["rate_hz"].startswith("Fading")
    for _ in range(10):
        view._last_tick = _time.perf_counter() - 0.1
        view._on_tick()
    assert view._inspected_neuron["rate_hz"].startswith("Silent")


def test_a_hidden_view_does_not_come_back_with_a_stale_firing_set(app):
    runner = _KcRunner(2000)
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=runner)
    view.setVisible(True)
    runner.fired[42] = True
    view._fetch_telemetry()
    assert view.active_kc_mask[42]
    view.setVisible(False)                           # hidden: no ticks, so no fading either
    assert not view.active_kc_mask.any() and not view._kc_glow.any()


def test_the_published_set_gives_the_models_cell_count_and_nothing_lights_without_a_model(app):
    import time as _time

    runner = _KcRunner(2000)
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=runner)
    view.setVisible(True)
    runner.fired[[3, 4, 5]] = True
    view.kc_active_count = 100
    view._fetch_telemetry()
    view.kc_active_count = 100
    assert view._kc_model_n == 2000 and view.active_kc_mask[[3, 4, 5]].all()
    assert view._kc_hud_text() == "100 / 2,000 (5.0%)"           # the model's own cells, not the 3D anatomy's 5,177
    runner.n_model = 0                               # no model published (learning off / starting)
    runner.fired = np.zeros(0, dtype=bool)
    for _ in range(10):
        view._last_tick = _time.perf_counter() - 0.1
        view._on_tick()
    assert view._kc_model_n == 0 and not view.active_kc_mask.any()
    view.kc_active_count = 0
    assert view._kc_hud_text() == f"0 / {view.data.n_kc:,} (0.0%)"


def test_a_runner_without_the_set_still_gets_the_illustrative_mask(app):
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=_HeadingRunner(0.0))
    st_runner = view.runner
    st_runner.stats = lambda: dict(valence=0.0, pam=0.0, ppl1=0.0, mbon_app=1.0, mbon_av=1.0, kc_active=100, gf=0.0,
                                   walk=0.0, mdn=0.0, steer=0.0, groom=0.0, cue_food=0.8, cue_near=0.0,
                                   weights_dirty=0)
    view._fetch_telemetry()
    assert not view._kc_from_model
    assert view.active_kc_mask.any()


def _cx_only(view):
    for layer in ("layer_mb", "layer_visual", "layer_olfactory", "layer_motor", "layer_wireframe", "layer_hulls"):
        setattr(view, layer, False)
    view.layer_cx = True


def test_the_drawn_ring_lights_the_nodes_at_the_fly_heading(app):
    runner = _HeadingRunner(1.0)
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=runner)
    view.resize(700, 520)
    _cx_only(view)
    ring = view._eb_ring
    n = len(ring)
    if n < 6:
        pytest.skip("no real central-complex ring")

    def brightness(k, r=2):                          # summed RGB around ring node k on the rendered widget
        img = view.grab().toImage()
        sx, sy, _, _ = view._project_points(view._cx_pts[[ring[k]]])
        x, y = int(round(float(sx[0]))), int(round(float(sy[0])))
        return sum(img.pixelColor(x + dx, y + dy).red() + img.pixelColor(x + dx, y + dy).green()
                   + img.pixelColor(x + dx, y + dy).blue() for dx in range(-r, r + 1) for dy in range(-r, r + 1))

    a, b = _ring_node_nearest(n, 1.0), _ring_node_nearest(n, 1.0 + math.pi)
    view._fetch_telemetry()
    lit_a, dark_b = brightness(a), brightness(b)
    runner.heading = 1.0 + math.pi                   # the fly turns around: the bump goes to the other side
    view._fetch_telemetry()
    dark_a, lit_b = brightness(a), brightness(b)
    assert lit_a > dark_a and lit_b > dark_b


def test_inspector_rate_of_an_eb_node_follows_the_bump_also_on_a_pinned_card(app):
    import time as _time

    runner = _HeadingRunner(1.0)
    view = MushroomBody3DView(data=load_mushroom_3d(), runner=runner)
    view.resize(700, 520)
    view.setVisible(True)
    _cx_only(view)
    ring = view._eb_ring
    n = len(ring)
    if n < 6:
        pytest.skip("no real central-complex ring")
    view._fetch_telemetry()
    a = _ring_node_nearest(n, 1.0)
    sx, sy, _, _ = view._project_points(view._cx_pts[[ring[a]]])
    card = view._find_neuron_at(QPoint(int(round(float(sx[0]))), int(round(float(sy[0])))))
    assert card is not None and card["cx_index"] == int(ring[a])
    assert card["rate_hz"] == "16.0 Hz"                  # in the bump

    view._inspected_neuron = card                        # pinned; the fly then turns away
    runner.heading = 1.0 + math.pi
    view._last_tick = _time.perf_counter() - 0.05
    view._on_tick()
    assert view._inspected_neuron["rate_hz"] == "4.0 Hz"


def test_hover_compares_inspector_cards_without_numpy_ambiguity(app):
    """Two KC cards of the same type differ only in their numpy position; `card_a != card_b` on the dicts would raise."""
    view = MushroomBody3DView(data=load_mushroom_3d())
    a, b = view._kc_card(0), view._kc_card(3)
    a["name"] = b["name"] = "Kenyon Cell (same type)"
    assert view._same_card(a, dict(a)) and not view._same_card(a, b)
    assert view._same_card(None, None) and not view._same_card(a, None) and not view._same_card(None, b)


def test_keyboard_switches_tools_and_home_resets_the_camera(app):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent

    view = MushroomBody3DView(data=load_mushroom_3d())
    for key, tool in ((Qt.Key_2, "opto"), (Qt.Key_3, "silence"), (Qt.Key_1, "camera")):
        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))
        assert view.tool_mode == tool
    view.yaw += 1.0
    view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Home, Qt.NoModifier))
    assert view.yaw == PRESET_ISOMETRIC[0]



def test_the_open_guide_blocks_the_controls_under_it(app):
    runner = _LabRunner()
    view = _lab_view(runner)
    view._pin(view._find_neuron_at(_soma_pos(view, "GF"), 3.0, systems=("dn",)))
    view.grab()
    stim = view._btn_rects["card_stim"].center()
    tool = view._btn_rects["tool_silence"].center()
    view._show_help = True
    view.grab()
    for pos in (stim, tool):
        view.mousePressEvent(_Click(pos))
        view.mouseReleaseEvent(_Click(pos))
    assert runner.calls == [] and view.tool_mode == "camera" and view._show_help
    view.mousePressEvent(_Click(view._btn_rects["close_help"].center()))
    assert not view._show_help


def test_the_card_takes_the_clicks_on_it(app):
    runner = _LabRunner()
    view = _lab_view(runner)
    for g in ("GF", "LC4", "MDN"):
        runner.silence(g, True)
    runner.calls.clear()
    view._fetch_telemetry()
    view._pin(view._find_neuron_at(_soma_pos(view, "GF"), 3.0, systems=("dn",)))
    view.grab()
    card = view._btn_rects["card"]
    pos = QPoint(card.left() + 20, card.top() + 12)         # the card's header
    view.mousePressEvent(_Click(pos))
    view.mouseReleaseEvent(_Click(pos))
    assert runner.calls == [] and view._inspected_neuron is not None   # no chip under it fired, the card stayed


def test_switching_tool_mid_hold_ends_the_light_and_the_press(app):
    runner = _LabRunner()
    view = _lab_view(runner)
    view.set_tool_mode("opto")
    pos = _soma_pos(view, "GF")
    view.mousePressEvent(_Click(pos))
    view._held = ("GF", view._held[1] - 1.0)
    view.set_tool_mode("silence")                           # key 3 while the button is down
    view.mouseReleaseEvent(_Click(pos))
    assert ("silence", "GF", True) not in runner.calls and runner.calls[-1] == ("stop",)


def test_a_deferred_stop_does_not_end_a_newer_pulse(app):
    runner = _LabRunner()
    view = _lab_view(runner)
    view.set_tool_mode("opto")
    scheduled = []
    import neuropest.mb_view3d as mv

    real = mv.QTimer.singleShot
    mv.QTimer.singleShot = lambda ms, fn: scheduled.append(fn)
    try:
        pos = _soma_pos(view, "GF")
        view.mousePressEvent(_Click(pos))
        view.mouseReleaseEvent(_Click(pos))                 # a quick click: its stop is deferred
        assert view.stimulate_target("MDN")                 # a new pulse meanwhile
        n = len(runner.calls)
        for fn in scheduled:
            fn()
        assert len(runner.calls) == n                       # the old deferred stop did not end MDN
    finally:
        mv.QTimer.singleShot = real


def test_a_quick_second_click_on_a_target_is_a_click(app):
    runner = _LabRunner()
    view = _lab_view(runner)
    view.set_tool_mode("silence")
    pos = _soma_pos(view, "LC4")
    view.mousePressEvent(_Click(pos))
    view.mouseReleaseEvent(_Click(pos))
    view._fetch_telemetry()
    view.mouseDoubleClickEvent(_Click(pos))                 # Qt sends the second press as a double-click
    view.mouseReleaseEvent(_Click(pos))
    assert runner.calls == [("silence", "LC4", True), ("silence", "LC4", False)]


def test_a_view_hidden_mid_drag_does_not_orbit_on_plain_moves_later(app):
    view = _lab_view()
    view.show()
    app.processEvents()
    view.mousePressEvent(_Click(QPoint(30, view.height() - 30)))
    view.hide()                                             # the release never comes
    view.show()
    app.processEvents()
    yaw = view.yaw
    view.mouseMoveEvent(_Click(QPoint(90, view.height() - 30)))
    assert view.yaw == yaw and view._press is None
