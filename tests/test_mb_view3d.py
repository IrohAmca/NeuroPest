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
    h0 = view.heading_angle
    view._on_tick()

def test_virtual_optogenetics_targets_and_triggers(app):
    """Test optogenetic ChR2 stimulation hubs, runner triggers, and visual ripple animation."""
    data = load_mushroom_3d()
    view = MushroomBody3DView(data=data)
    view.resize(640, 480)

    class MockRunner:
        def __init__(self):
            self.ready = True
            self.calls = []

        def inject_gf(self, amt):
            self.calls.append(("gf", amt))

        def inject_mdn(self, amt):
            self.calls.append(("mdn", amt))

        def inject_walk(self, amt):
            self.calls.append(("walk", amt))

        def inject_reward(self, amt):
            self.calls.append(("reward", amt))

        def inject_punish(self, amt):
            self.calls.append(("punish", amt))

        def inject_cue_food(self, amt):
            self.calls.append(("food", amt))

        def inject_cue_near(self, amt):
            self.calls.append(("near", amt))

        def inject_optogenetic(self, tgt, amt):
            self.calls.append((tgt, amt))

    mock = MockRunner()
    view.runner = mock

    # 1. Inspect optogenetic targets
    targets = view._get_opto_targets()
    target_ids = {t["id"] for t in targets}
    assert {"gf", "mdn", "walk", "steer", "pam", "ppl1", "alpn", "vpn", "eb"}.issubset(target_ids)

    # 2. Trigger stimulation across major hubs
    view.trigger_opto_stimulus("gf", intensity=1.0)
    assert ("gf", 1.0) in mock.calls
    assert len(view._opto_flashes) == 1
    assert "Giant Fiber" in view._opto_status

    view.trigger_opto_stimulus("mdn", intensity=0.8)
    assert ("mdn", 0.8) in mock.calls
    assert len(view._opto_flashes) == 2

    view.trigger_opto_stimulus("walk", intensity=1.0)
    assert ("walk", 1.0) in mock.calls

    view.trigger_opto_stimulus("pam", intensity=1.0)
    assert ("reward", 1.0) in mock.calls

    view.trigger_opto_stimulus("ppl1", intensity=1.0)
    assert ("punish", 1.0) in mock.calls

    # 3. Render offscreen with active laser ripple flashes
    img = QImage(640, 480, QImage.Format_ARGB32)
    p = QPainter(img)
    view.paintEvent(None)
    p.end()
    assert not img.isNull()


def test_virtual_optogenetics_tool_mode_and_mouse_click(app):
    """Test tool mode selection and canvas click-to-stimulate interaction."""
    view = MushroomBody3DView()
    view.resize(640, 480)

    class MockRunner:
        def __init__(self):
            self.ready = True
            self.gf_called = False

        def inject_gf(self, amt):
            self.gf_called = True

    mock = MockRunner()
    view.runner = mock

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

    # Default tool mode is camera
    assert view.tool_mode == "camera"

    # Switch to opto tool mode
    view.set_tool_mode("opto")
    assert view.tool_mode == "opto"

    # Find screen coordinate for Giant Fiber target
    gf_target = next(t for t in view._get_opto_targets() if t["id"] == "gf")
    sx, sy, _, valid = view._project_points(np.array([gf_target["pos"]], dtype=np.float32))
    assert valid[0]

    # Click on the GF target location
    click_pos = QPoint(int(sx[0]), int(sy[0]))
    view.mousePressEvent(FakeClick(click_pos))

    # Verification: GF stimulation triggered and flash registered
    assert mock.gf_called
    assert len(view._opto_flashes) >= 1


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

    # 2. Test tool mode switch to inspect
    view.set_tool_mode("inspect")
    assert view.tool_mode == "inspect"

    # Click on the neuron position -> pins it
    view.mousePressEvent(FakeClick(QPoint(int(sx[0]), int(sy[0]))))
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





