"""FlyWire v783 3D Mushroom Body Interactive Visualization.

Provides a 3D neural circuit and mental map visualizer showing:
  * Bilateral Mushroom Body (Kenyon cells, Calyx, Pedunculus, Lobes)
  * Real-time synaptic memory formation (Appetitive emerald green vs Aversive ruby red)
  * Dopamine neuron activation (PAM reward vs PPL1 punishment)
  * Live sparse Kenyon cell firing (~5% code) and axonal action potential propagation
  * Drosophila brain neuropil 3D wireframe envelope for anatomical context
  * Interactive camera (Orbit, Pan, Zoom, Presets, Auto-rotation)
  * HUD telemetry (Valence gauge, Dopamine bars, Active KCs, MBON balance)
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QLineF, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .mb_geometry import (
    DAN_PAM,
    DAN_PPL1,
    LOBE_ALPHABETA,
    LOBE_ALPHAP_BETAP,
    LOBE_GAMMA,
    Mushroom3DData,
    load_mushroom_3d,
)

if TYPE_CHECKING:
    from .runner import Runner

# Visualization Modes
MODE_VALENCE = "valence"        # Learned mental map (Valence: Red=Aversive, Green=Appetitive, Slate=Naive)
MODE_PLASTICITY = "plasticity"  # Synaptic depression magnitude |Δw|
MODE_ANATOMY = "anatomy"        # FlyWire biological lobe compartments & cell classes
MODE_ACTIVITY = "activity"      # Live neural spikes & dopamine bursts

# Camera Presets (yaw, pitch in radians)
PRESET_ISOMETRIC = (-0.45, 0.35)
PRESET_DORSAL = (0.0, 1.55)     # Top-down FAFB view (Calyx & Lobes)
PRESET_ANTERIOR = (0.0, 0.0)    # Frontal view (Vertical & Medial Lobes)
PRESET_SAGITTAL = (1.57, 0.0)   # Lateral profile (Calyx -> Pedunculus -> Lobes)


class MushroomBody3DView(QWidget):
    """High-performance 3D interactive viewer for the Drosophila Mushroom Body."""

    def __init__(self, runner: Runner | None = None,
                 data: Mushroom3DData | None = None,
                 parent: QWidget | None = None,
                 pip_mode: bool = False):
        super().__init__(parent)
        self.runner = runner
        self.data = data or load_mushroom_3d()
        self.pip_mode = pip_mode

        if self.pip_mode:
            self.setMinimumSize(180, 180)
            self.auto_rotate = True
            self.zoom = 180.0
        else:
            self.setMinimumSize(480, 360)
            self.auto_rotate = False
            self.zoom = 220.0

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)

        # 3D Camera attributes
        self.yaw, self.pitch = PRESET_ISOMETRIC
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._cam_R: np.ndarray | None = None


        # Visual mode
        self.view_mode = MODE_VALENCE

        # Mouse interaction state
        self._last_mouse_pos: QPoint | None = None
        self._is_panning = False
        self._hover_text = ""

        # Help guide overlay state
        self._show_help = False
        self._help_card_rect = QRect()

        # Dynamic telemetry state
        self.valence = 0.0
        self.pam = 0.0
        self.ppl1 = 0.0
        self.mbon_app = 1.0
        self.mbon_av = 1.0
        self.kc_active_count = 0
        self.weights_dirty = 0

        # Live motor and sensory rate readouts
        self.rate_gf = 0.0
        self.rate_walk = 0.0
        self.rate_mdn = 0.0
        self.rate_steer = 0.0
        self.rate_groom = 0.0
        self.cue_food = 0.0
        self.cue_near = 0.0
        self.heading_angle = 0.0

        # Memory buffer (per KC valence delta)
        self.kc_weights = np.zeros(self.data.n_kc, dtype=np.float32)
        self.active_kc_mask = np.zeros(self.data.n_kc, dtype=bool)

        # Pre-sliced geometry cache for fast rendering
        n_kc = self.data.n_kc
        # Somas: sample step=3 (1,725 somas for 5,177 KCs, fast and visually dense)
        self._soma_step = 3 if n_kc > 3000 else 1
        self._soma_indices = np.arange(0, n_kc, self._soma_step)
        self._soma_pts = self.data.kc_soma[self._soma_indices]
        self._soma_lobes = self.data.kc_lobe[self._soma_indices]

        # Axon tracts: sample ~160 representative streamlines
        tract_step = max(1, n_kc // 160)
        self._tract_indices = np.arange(0, n_kc, tract_step)
        self._tract_soma = self.data.kc_soma[self._tract_indices]
        self._tract_ped = self.data.kc_ped[self._tract_indices]
        self._tract_target = self.data.kc_target[self._tract_indices]
        self._tract_lobes = self.data.kc_lobe[self._tract_indices]

        # Wireframe pre-sliced segments
        self._wf_p1 = self.data.brain_wireframe[:, 0] if len(self.data.brain_wireframe) > 0 else np.zeros((0, 3), dtype=np.float32)
        self._wf_p2 = self.data.brain_wireframe[:, 1] if len(self.data.brain_wireframe) > 0 else np.zeros((0, 3), dtype=np.float32)

        # Anatomical Layer Visibility Toggles
        self.layer_mb = True        # Mushroom Body
        self.layer_cx = True        # Central Complex (EB Compass Ring, PB Bridge)
        self.layer_visual = True    # Optic Lobes & Visual Projections (LC4, LC10, LPLC2, LPC1)
        self.layer_olfactory = True # Antennal Lobes & Odor Projections (ALPN)
        self.layer_motor = True     # Descending Motor Neurons (GF, MDN, DNp09, DNa02, aDN)
        self.layer_wireframe = True # Brain Envelope Wireframe
        self.layer_hulls = True     # Translucent Neuropil Volumetric Envelopes (AL, MB, CX, Optic)

        # Pre-cached whole-brain circuits
        self._cx_pts = getattr(self.data, "cx_pts", np.zeros((0, 3), dtype=np.float32))
        self._cx_sub = getattr(self.data, "cx_sub", np.zeros(0, dtype=np.int8))
        self._cx_type = getattr(self.data, "cx_type", np.empty(0, dtype=str))

        self._vpn_soma = getattr(self.data, "vpn_soma", np.zeros((0, 3), dtype=np.float32))
        self._vpn_target = getattr(self.data, "vpn_target", np.zeros((0, 3), dtype=np.float32))
        self._vpn_type = getattr(self.data, "vpn_type", np.empty(0, dtype=str))
        self._vpn_side = getattr(self.data, "vpn_side", np.zeros(0, dtype=np.int8))

        self._alpn_soma = getattr(self.data, "alpn_soma", np.zeros((0, 3), dtype=np.float32))
        self._alpn_target = getattr(self.data, "alpn_target", np.zeros((0, 3), dtype=np.float32))
        self._alpn_type = getattr(self.data, "alpn_type", np.empty(0, dtype=str))
        self._alpn_side = getattr(self.data, "alpn_side", np.zeros(0, dtype=np.int8))

        self._dn_soma = getattr(self.data, "dn_soma", np.zeros((0, 3), dtype=np.float32))
        self._dn_target = getattr(self.data, "dn_target", np.zeros((0, 3), dtype=np.float32))
        self._dn_type = getattr(self.data, "dn_type", np.empty(0, dtype=str))
        self._dn_side = getattr(self.data, "dn_side", np.zeros(0, dtype=np.int8))

        # Animation state (traveling pulses, dopamine glow ripple)
        self._pulse_phase = 0.0
        self._last_tick = time.perf_counter()

        # Render timer (adaptive: 33ms active, 200ms idle)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)

        # Interactive UI buttons on canvas (rectangles in screen coordinates)
        self._btn_rects: dict[str, QRect] = {}

        # Interactive Tools & Virtual Optogenetics
        self.tool_mode = "camera"   # "camera", "opto", "inspect"
        self._opto_flashes: list[dict] = []
        self._opto_status = ""
        self._opto_status_time = 0.0

        # Neuron & Connectome Inspector
        self._inspected_neuron: dict | None = None
        self._inspected_pinned: bool = False

        # Translucent Neuropil Volumetric Envelopes
        self._neuropil_volumes: list[dict] = []
        self._build_neuropil_volumes()

    def _build_neuropil_volumes(self):
        """Construct 3D volumetric polygonal envelopes for key brain neuropils."""
        self._neuropil_volumes = []

        def build_ellipsoid(center, rx, ry, rz, n_lat=8, n_lon=10):
            verts = []
            for i in range(n_lat + 1):
                lat = -math.pi / 2.0 + math.pi * (i / n_lat)
                z = rz * math.sin(lat)
                r_xy = math.cos(lat)
                for j in range(n_lon):
                    lon = 2.0 * math.pi * (j / n_lon)
                    x = rx * r_xy * math.cos(lon)
                    y = ry * r_xy * math.sin(lon)
                    verts.append([center[0] + x, center[1] + y, center[2] + z])
            verts = np.array(verts, dtype=np.float32)

            faces = []
            for i in range(n_lat):
                for j in range(n_lon):
                    j_next = (j + 1) % n_lon
                    p0 = i * n_lon + j
                    p1 = i * n_lon + j_next
                    p2 = (i + 1) * n_lon + j_next
                    p3 = (i + 1) * n_lon + j
                    faces.append([p0, p1, p2])
                    faces.append([p0, p2, p3])
            faces = np.array(faces, dtype=np.int32)
            return verts, faces

        # 1. Antennal Lobes (AL-L & AL-R)
        al_l_center = np.mean(self._alpn_soma[self._alpn_side == -1], axis=0) if np.any(self._alpn_side == -1) else np.array([-0.20, -0.15, -0.28], dtype=np.float32)
        v, f = build_ellipsoid(al_l_center, 0.11, 0.11, 0.11)
        self._neuropil_volumes.append({"id": "al_l", "name": "Left Antennal Lobe", "color": QColor(16, 185, 129), "vertices": v, "faces": f})

        al_r_center = np.mean(self._alpn_soma[self._alpn_side == 1], axis=0) if np.any(self._alpn_side == 1) else np.array([0.20, -0.15, -0.28], dtype=np.float32)
        v, f = build_ellipsoid(al_r_center, 0.11, 0.11, 0.11)
        self._neuropil_volumes.append({"id": "al_r", "name": "Right Antennal Lobe", "color": QColor(16, 185, 129), "vertices": v, "faces": f})

        # 2. Mushroom Body Calyces (MB-CA-L & MB-CA-R)
        ca_l_center = np.array(self.data.lobe_centers.get("calyx_left", (-0.25, -0.10, 0.28)), dtype=np.float32)
        v, f = build_ellipsoid(ca_l_center, 0.13, 0.12, 0.11)
        self._neuropil_volumes.append({"id": "ca_l", "name": "Left MB Calyx", "color": QColor(245, 158, 11), "vertices": v, "faces": f})

        ca_r_center = np.array(self.data.lobe_centers.get("calyx_right", (0.25, -0.10, 0.28)), dtype=np.float32)
        v, f = build_ellipsoid(ca_r_center, 0.13, 0.12, 0.11)
        self._neuropil_volumes.append({"id": "ca_r", "name": "Right MB Calyx", "color": QColor(245, 158, 11), "vertices": v, "faces": f})

        # 3. Mushroom Body Vertical & Medial Lobes
        gamma_l_center = np.array(self.data.lobe_centers.get("gamma_left", (-0.10, 0.12, 0.05)), dtype=np.float32)
        v, f = build_ellipsoid(gamma_l_center, 0.11, 0.14, 0.10)
        self._neuropil_volumes.append({"id": "mb_lobe_l", "name": "Left MB Lobes", "color": QColor(6, 182, 212), "vertices": v, "faces": f})

        gamma_r_center = np.array(self.data.lobe_centers.get("gamma_right", (0.10, 0.12, 0.05)), dtype=np.float32)
        v, f = build_ellipsoid(gamma_r_center, 0.11, 0.14, 0.10)
        self._neuropil_volumes.append({"id": "mb_lobe_r", "name": "Right MB Lobes", "color": QColor(6, 182, 212), "vertices": v, "faces": f})

        # 4. Central Complex: Ellipsoid Body (EB)
        cx_eb = self._cx_pts[self._cx_sub == 0] if len(self._cx_sub) == len(self._cx_pts) else self._cx_pts
        eb_center = np.mean(cx_eb, axis=0) if len(cx_eb) > 0 else np.array([0.0, 0.05, 0.15], dtype=np.float32)
        v, f = build_ellipsoid(eb_center, 0.10, 0.05, 0.10)
        self._neuropil_volumes.append({"id": "eb", "name": "Ellipsoid Body", "color": QColor(56, 189, 248), "vertices": v, "faces": f})

        # 5. Optic Lobes (Left & Right)
        ol_l_center = np.mean(self._vpn_soma[self._vpn_side == -1], axis=0) if np.any(self._vpn_side == -1) else np.array([-0.52, 0.05, 0.0], dtype=np.float32)
        v, f = build_ellipsoid(ol_l_center, 0.16, 0.22, 0.17)
        self._neuropil_volumes.append({"id": "ol_l", "name": "Left Optic Lobe", "color": QColor(168, 85, 247), "vertices": v, "faces": f})

        ol_r_center = np.mean(self._vpn_soma[self._vpn_side == 1], axis=0) if np.any(self._vpn_side == 1) else np.array([0.52, 0.05, 0.0], dtype=np.float32)
        v, f = build_ellipsoid(ol_r_center, 0.16, 0.22, 0.17)
        self._neuropil_volumes.append({"id": "ol_r", "name": "Right Optic Lobe", "color": QColor(168, 85, 247), "vertices": v, "faces": f})

    def set_pip_mode(self, enabled: bool):
        """Toggle compact Picture-in-Picture display mode."""
        self.pip_mode = bool(enabled)
        if self.pip_mode:
            self.auto_rotate = True
            self.zoom = 180.0
        self.update()

    def set_tool_mode(self, mode: str):
        """Set active canvas tool mode ('camera', 'opto', 'inspect')."""
        if mode in ("camera", "opto", "inspect"):
            self.tool_mode = mode
            if mode == "opto":
                self.setCursor(Qt.CrossCursor)
            elif mode == "inspect":
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.update()

    def _get_opto_targets(self) -> list[dict]:
        """Return list of stimulatable circuit targets with authentic 3D positions and evoked behaviors."""
        targets = []
        # 1. Giant Fiber (DNp01 escape takeoff)
        gf_mask = self._dn_type == "GF"
        if np.any(gf_mask):
            pos = np.mean(self._dn_soma[gf_mask], axis=0)
        else:
            pos = np.array([0.0, 0.45, 0.25], dtype=np.float32)
        targets.append({
            "id": "gf",
            "name": "Giant Fiber (DNp01)",
            "action": "Takeoff Escape Reflex",
            "pos": pos,
            "color": QColor(0, 229, 255),
            "inject": "gf",
        })

        # 2. Moonwalker (MDN backward moonwalk retreat)
        mdn_mask = self._dn_type == "MDN"
        if np.any(mdn_mask):
            pos = np.mean(self._dn_soma[mdn_mask], axis=0)
        else:
            pos = np.array([-0.05, 0.35, 0.15], dtype=np.float32)
        targets.append({
            "id": "mdn",
            "name": "Moonwalker DN (MDN)",
            "action": "Backward Moonwalk Retreat",
            "pos": pos,
            "color": QColor(255, 23, 68),
            "inject": "mdn",
        })

        # 3. Walking Command (DNp09 / WALK)
        walk_mask = np.isin(self._dn_type, ["DNp09", "WALK"])
        if np.any(walk_mask):
            pos = np.mean(self._dn_soma[walk_mask], axis=0)
        else:
            pos = np.array([0.08, 0.38, 0.18], dtype=np.float32)
        targets.append({
            "id": "walk",
            "name": "Walking DN (DNp09)",
            "action": "Forward Locomotion",
            "pos": pos,
            "color": QColor(118, 255, 3),
            "inject": "walk",
        })

        # 4. Steering Command (DNa02)
        steer_mask = self._dn_type == "DNa02"
        if np.any(steer_mask):
            pos = np.mean(self._dn_soma[steer_mask], axis=0)
        else:
            pos = np.array([0.15, 0.40, 0.22], dtype=np.float32)
        targets.append({
            "id": "steer",
            "name": "Steering DN (DNa02)",
            "action": "Asymmetric Steering Torque",
            "pos": pos,
            "color": QColor(255, 145, 0),
            "inject": "steer",
        })

        # 5. PAM Dopamine Cluster (Reward)
        if "pam_reward" in self.data.lobe_centers:
            pos = np.array(self.data.lobe_centers["pam_reward"], dtype=np.float32)
            targets.append({
                "id": "pam",
                "name": "PAM Cluster (DAN)",
                "action": "Appetitive Sugar Reward",
                "pos": pos,
                "color": QColor(255, 215, 0),
                "inject": "reward",
            })

        # 6. PPL1 Dopamine Cluster (Punishment)
        if "ppl1_punish" in self.data.lobe_centers:
            pos = np.array(self.data.lobe_centers["ppl1_punish"], dtype=np.float32)
            targets.append({
                "id": "ppl1",
                "name": "PPL1 Cluster (DAN)",
                "action": "Nociceptive Pain Shock",
                "pos": pos,
                "color": QColor(255, 61, 113),
                "inject": "punish",
            })

        # 7. Antennal Glomeruli / Food Odor PN
        if len(self._alpn_soma) > 0:
            pos = np.mean(self._alpn_soma, axis=0)
        else:
            pos = np.array([-0.25, -0.15, -0.3], dtype=np.float32)
        targets.append({
            "id": "alpn",
            "name": "Antennal Glomeruli (ALPN)",
            "action": "Appetitive Food Odor Cue",
            "pos": pos,
            "color": QColor(0, 230, 118),
            "inject": "food",
        })

        # 8. Visual Looming Lobes (LC4 / LPLC2)
        if len(self._vpn_soma) > 0:
            pos = np.mean(self._vpn_soma, axis=0)
        else:
            pos = np.array([0.45, 0.1, -0.1], dtype=np.float32)
        targets.append({
            "id": "vpn",
            "name": "Optic Lobes (LC4 / LPLC2)",
            "action": "Predator Looming Alert",
            "pos": pos,
            "color": QColor(224, 64, 251),
            "inject": "near",
        })

        # 9. Ellipsoid Body Ring (EB Compass)
        cx_eb = self._cx_pts[self._cx_sub == 0] if len(self._cx_sub) == len(self._cx_pts) else self._cx_pts
        if len(cx_eb) > 0:
            pos = np.mean(cx_eb, axis=0)
        else:
            pos = np.array([0.0, 0.05, 0.15], dtype=np.float32)
        targets.append({
            "id": "eb",
            "name": "Ellipsoid Body (EB)",
            "action": "Compass Heading Kick",
            "pos": pos,
            "color": QColor(0, 176, 255),
            "inject": "eb",
        })

        return targets

    def _find_opto_target_at(self, pos: QPoint, threshold_px: float = 30.0) -> dict | None:
        """Find the closest optogenetic target within threshold_px of screen position."""
        targets = self._get_opto_targets()
        best_target = None
        best_dist = float("inf")
        for t in targets:
            sx_arr, sy_arr, _, valid_arr = self._project_points(np.array([t["pos"]], dtype=np.float32))
            if valid_arr[0]:
                d = math.hypot(pos.x() - sx_arr[0], pos.y() - sy_arr[0])
                if d <= threshold_px and d < best_dist:
                    best_dist = d
                    best_target = t
        return best_target

    def trigger_opto_stimulus(self, target: dict | str, intensity: float = 1.0):
        """Trigger virtual optogenetic micro-stimulation on target circuit with ripple animation."""
        if isinstance(target, str):
            tid = target.lower()
            all_t = self._get_opto_targets()
            matched = next((t for t in all_t if t["id"] == tid), None)
            if matched is None:
                matched = next((t for t in all_t if tid in t["name"].lower()), None)
            if matched is None:
                matched = all_t[0] if all_t else None
            target = matched

        if target is None:
            return

        # Deliver physiological stimulation to runner
        inj = target.get("inject", "")
        if self.runner is not None and getattr(self.runner, "ready", False):
            if inj == "gf":
                self.runner.inject_gf(intensity)
            elif inj == "mdn":
                self.runner.inject_mdn(intensity)
            elif inj == "walk":
                self.runner.inject_walk(intensity)
            elif inj == "reward":
                self.runner.inject_reward(intensity)
            elif inj == "punish":
                self.runner.inject_punish(intensity)
            elif inj == "food":
                self.runner.inject_cue_food(intensity)
            elif inj == "near":
                self.runner.inject_cue_near(intensity)
            elif inj == "steer":
                if hasattr(self.runner, "inject_optogenetic"):
                    self.runner.inject_optogenetic("steer", intensity)
            elif inj == "eb":
                self.heading_angle = (self.heading_angle + math.pi * 0.4) % (2 * math.pi)

        # Register visual flash ripple
        self._opto_flashes.append({
            "pos": target["pos"],
            "color": target["color"],
            "name": target["name"],
            "action": target["action"],
            "time": time.perf_counter(),
            "duration": 0.9,
            "intensity": intensity,
        })

        self._opto_status = f"ChR2: {target['name']} -> {target['action']}"
        self._opto_status_time = time.perf_counter()

        if self._timer.interval() != 33:
            self._timer.setInterval(33)
        self.update()

    def _trigger_opto_at_screen_pos(self, pos: QPoint):
        """Fallback stimulation at screen coordinate by selecting nearest brain centroid."""
        target = self._find_opto_target_at(pos, threshold_px=80.0)
        if target is not None:
            self.trigger_opto_stimulus(target)
        else:
            targets = self._get_opto_targets()
            if targets:
                self.trigger_opto_stimulus(targets[0])

    def _find_neuron_at(self, pos: QPoint, threshold_px: float = 24.0) -> dict | None:
        """Find the closest neuron or circuit hub across all systems within screen threshold_px."""
        best = None
        best_dist = float("inf")

        def check_pts(pts_3d, callback):
            nonlocal best, best_dist
            if len(pts_3d) == 0:
                return
            sx_arr, sy_arr, _, valid = self._project_points(pts_3d)
            for i in range(len(pts_3d)):
                if valid[i]:
                    d = math.hypot(pos.x() - sx_arr[i], pos.y() - sy_arr[i])
                    if d <= threshold_px and d < best_dist:
                        cand = callback(i, (sx_arr[i], sy_arr[i]))
                        if cand is not None:
                            best_dist = d
                            best = cand

        # 1. Descending Motor Neurons (DNs)
        if len(self._dn_soma) > 0 and self.layer_motor:
            def dn_cb(i, s_pos):
                dn_t = str(self._dn_type[i]) if i < len(self._dn_type) else "DN"
                target_pt = self._dn_target[i] if i < len(self._dn_target) else None
                soma_pt = self._dn_soma[i]

                if dn_t == "GF":
                    name = "Giant Fiber (DNp01 / GF)"
                    role = "Ballistic takeoff escape reflex & jump initiation (< 5 ms)"
                    hz = f"{self.rate_gf:.1f} Hz"
                    nt = "ACh (Excitatory)"
                    col = QColor(0, 229, 255)
                    part = "In: LPLC2, LC4 Looming • Out: TTM Jump Motor, PSI Wing"
                elif dn_t == "MDN":
                    name = "Moonwalker DN (MDN)"
                    role = "Evokes coordinated backward locomotion / retreat"
                    hz = f"{self.rate_mdn:.1f} Hz"
                    nt = "GABA (Inhibitory CPG)"
                    col = QColor(255, 23, 68)
                    part = "In: LPC1 visual • Out: Thoracic backward walking CPG"
                elif dn_t in ("DNp09", "WALK"):
                    name = f"Walking Command ({dn_t})"
                    role = "Initiates and sustains forward walking locomotion"
                    hz = f"{self.rate_walk:.1f} Hz"
                    nt = "ACh (Excitatory)"
                    col = QColor(118, 255, 3)
                    part = "In: Central complex • Out: Leg central pattern generators"
                elif dn_t == "DNa02":
                    name = "Steering Command (DNa02)"
                    role = "Bilateral asymmetric steering torque and body turning"
                    hz = f"{abs(self.rate_steer):.1f} Hz"
                    nt = "ACh (Excitatory)"
                    col = QColor(255, 145, 0)
                    part = "In: LC10 visual tracking • Out: Asymmetric leg motor bias"
                else:
                    name = f"Descending Motor Neuron ({dn_t})"
                    role = "Motor command descending to ventral nerve cord"
                    hz = f"{self.rate_groom:.1f} Hz" if "aDN" in dn_t else "Active"
                    nt = "ACh (Cholinergic)"
                    col = QColor(239, 68, 68)
                    part = "In: Brain pre-motor • Out: Ventral nerve cord effector"

                return {
                    "name": name,
                    "type": dn_t,
                    "system": "Motor Descending (DN)",
                    "nt": nt,
                    "nt_color": QColor(56, 189, 248) if "ACh" in nt else QColor(249, 115, 22),
                    "pos_3d": soma_pt,
                    "target_3d": target_pt,
                    "rate_hz": hz,
                    "role": role,
                    "partners": part,
                    "color": col,
                }
            check_pts(self._dn_soma, dn_cb)

        # 2. Dopaminergic Reinforcement Clusters (DANs)
        if len(self.data.dan_pos) > 0 and self.layer_mb:
            def dan_cb(i, s_pos):
                dan_t = str(self.data.dan_type[i]) if i < len(self.data.dan_type) else "DAN"
                clus = self.data.dan_cluster[i] if i < len(self.data.dan_cluster) else 2
                pos_pt = self.data.dan_pos[i]
                if clus == DAN_PAM:
                    name = f"PAM Dopamine Cluster ({dan_t})"
                    role = "Appetitive reinforcement: sugar reward & learned attraction"
                    hz = f"{self.pam * 40.0:.1f} Hz"
                    col = QColor(255, 215, 0)
                    part = "In: Gustatory sugar ORNs • Out: KC Kenyon cell synapses"
                elif clus == DAN_PPL1:
                    name = f"PPL1 Dopamine Cluster ({dan_t})"
                    role = "Aversive reinforcement: electric shock, heat & wall collisions"
                    hz = f"{self.ppl1 * 40.0:.1f} Hz"
                    col = QColor(255, 61, 113)
                    part = "In: Nociceptive threat sensors • Out: KC Kenyon cell synapses"
                else:
                    name = f"Dopaminergic Neuron ({dan_t})"
                    role = "Neuromodulatory arousal and behavioral state gating"
                    hz = "10.0 Hz"
                    col = QColor(217, 70, 239)
                    part = "In: Sensory neuropils • Out: Central brain"

                return {
                    "name": name,
                    "type": dan_t,
                    "system": "Mushroom Body Dopamine (DAN)",
                    "nt": "DA (Dopaminergic)",
                    "nt_color": QColor(255, 215, 0),
                    "pos_3d": pos_pt,
                    "target_3d": None,
                    "rate_hz": hz,
                    "role": role,
                    "partners": part,
                    "color": col,
                }
            check_pts(self.data.dan_pos, dan_cb)

        # 3. Mushroom Body Output Neurons (MBONs)
        if len(self.data.mbon_pos) > 0 and self.layer_mb:
            def mbon_cb(i, s_pos):
                mbon_t = str(self.data.mbon_type[i]) if i < len(self.data.mbon_type) else "MBON"
                sign = self.data.mbon_sign[i] if i < len(self.data.mbon_sign) else 0
                nt = str(self.data.mbon_nt[i]) if i < len(self.data.mbon_nt) else "ACh"
                pos_pt = self.data.mbon_pos[i]

                if sign > 0:
                    role = "Approach output pool: drives attraction toward learned odor"
                    hz = f"{self.mbon_app * 18.0:.1f} Hz"
                    col = QColor(16, 185, 129)
                elif sign < 0:
                    role = "Avoidance output pool: drives escape from learned danger"
                    hz = f"{self.mbon_av * 18.0:.1f} Hz"
                    col = QColor(239, 68, 68)
                else:
                    role = "Neutral / context-dependent behavioral gating"
                    hz = "12.0 Hz"
                    col = QColor(148, 163, 184)

                nt_map = {"ACh": "ACh (Excitatory)", "GABA": "GABA (Inhibitory)", "Glu": "Glu (Glutamatergic)"}
                nt_full = nt_map.get(nt, nt)
                nt_col = QColor(56, 189, 248) if nt == "ACh" else (QColor(249, 115, 22) if nt == "GABA" else QColor(234, 179, 8))

                return {
                    "name": f"Output Neuron ({mbon_t})",
                    "type": mbon_t,
                    "system": "Mushroom Body Output (MBON)",
                    "nt": nt_full,
                    "nt_color": nt_col,
                    "pos_3d": pos_pt,
                    "target_3d": None,
                    "rate_hz": hz,
                    "role": role,
                    "partners": "In: KC Kenyon cells • Out: Lateral horn & descending pre-motor",
                    "color": col,
                }
            check_pts(self.data.mbon_pos, mbon_cb)

        # 4. Central Complex Compass (CX)
        if len(self._cx_pts) > 0 and self.layer_cx:
            def cx_cb(i, s_pos):
                cx_t = str(self._cx_type[i]) if i < len(self._cx_type) else "CX"
                sub = self._cx_sub[i] if i < len(self._cx_sub) else 0
                pos_pt = self._cx_pts[i]
                if sub == 0:
                    sub_name = "Ellipsoid Body (EB Ring)"
                    role = "Internal compass heading: E-PG bump tracking fly orientation"
                    part = "In: Ring neurons, PB • Out: Steering pre-motor"
                elif sub == 1:
                    sub_name = "Protocerebral Bridge (PB)"
                    role = "Angular velocity integration & bilateral compass bridge"
                    part = "In: EB heading ring • Out: Central fan-shaped body"
                else:
                    sub_name = "Fan-shaped Body (FB)"
                    role = "Navigation vector calculation and goal-directed heading"
                    part = "In: Protocerebral bridge • Out: Premotor hubs"

                return {
                    "name": f"{sub_name} ({cx_t})",
                    "type": cx_t,
                    "system": "Central Complex (CX Navigation)",
                    "nt": "ACh / GABA",
                    "nt_color": QColor(56, 189, 248),
                    "pos_3d": pos_pt,
                    "target_3d": None,
                    "rate_hz": f"{10.0 + 6.0 * math.cos(self.heading_angle):.1f} Hz",
                    "role": role,
                    "partners": part,
                    "color": QColor(56, 189, 248),
                }
            check_pts(self._cx_pts, cx_cb)

        # 5. Visual Projection Neurons (VPN)
        if len(self._vpn_soma) > 0 and self.layer_visual:
            def vpn_cb(i, s_pos):
                vpn_t = str(self._vpn_type[i]) if i < len(self._vpn_type) else "VPN"
                pos_pt = self._vpn_soma[i]
                target_pt = self._vpn_target[i] if i < len(self._vpn_target) else None

                if "LC4" in vpn_t or "LPLC2" in vpn_t:
                    role = "Detects expanding looming predators (direct input to Giant Fiber)"
                    hz = f"{self.cue_near * 55.0:.1f} Hz"
                    col = QColor(224, 64, 251)
                    part = "In: Lobula complex • Out: Giant Fiber (DNp01 escape)"
                elif "LC10" in vpn_t:
                    role = "Tracks small high-contrast moving targets (drives DNa02 steering)"
                    hz = f"{abs(self.rate_steer) * 2.5:.1f} Hz"
                    col = QColor(249, 115, 22)
                    part = "In: Lobula plate • Out: DNa02 steering command"
                else:
                    role = "Visual feature projection from optic lobe to central brain"
                    hz = "14.0 Hz"
                    col = QColor(249, 115, 22)
                    part = "In: Optic lobes • Out: Posterior lateral protocerebrum"

                return {
                    "name": f"Visual Projection ({vpn_t})",
                    "type": vpn_t,
                    "system": "Optic Lobes (Visual VPN)",
                    "nt": "ACh (Excitatory)",
                    "nt_color": QColor(56, 189, 248),
                    "pos_3d": pos_pt,
                    "target_3d": target_pt,
                    "rate_hz": hz,
                    "role": role,
                    "partners": part,
                    "color": col,
                }
            check_pts(self._vpn_soma, vpn_cb)

        # 6. Antennal Lobe Projection Neurons (ALPN)
        if len(self._alpn_soma) > 0 and self.layer_olfactory:
            def alpn_cb(i, s_pos):
                alpn_t = str(self._alpn_type[i]) if i < len(self._alpn_type) else "ALPN"
                pos_pt = self._alpn_soma[i]
                target_pt = self._alpn_target[i] if i < len(self._alpn_target) else None
                return {
                    "name": f"Antennal Glomerulus ({alpn_t})",
                    "type": alpn_t,
                    "system": "Antennal Lobe (Olfactory PN)",
                    "nt": "ACh (Excitatory)",
                    "nt_color": QColor(56, 189, 248),
                    "pos_3d": pos_pt,
                    "target_3d": target_pt,
                    "rate_hz": f"{self.cue_food * 45.0:.1f} Hz",
                    "role": "Conveys uniglomerular odor identity from antenna to MB Calyx & LH",
                    "partners": "In: Olfactory receptor ORNs • Out: Kenyon Cells & Lateral Horn",
                    "color": QColor(16, 185, 129),
                }
            check_pts(self._alpn_soma, alpn_cb)

        # 7. Kenyon Cells (KC)
        if len(self._soma_pts) > 0 and self.layer_mb:
            def kc_cb(i, s_pos):
                real_idx = self._soma_indices[i]
                kc_t = str(self.data.kc_type[real_idx]) if real_idx < len(self.data.kc_type) else "KC"
                lobe = self._soma_lobes[i]
                soma_pt = self._soma_pts[i]
                target_pt = self.data.kc_target[real_idx] if real_idx < len(self.data.kc_target) else None
                lobe_name = "Gamma Lobe" if lobe == 0 else ("Alpha/Beta Lobe" if lobe == 1 else "Alpha'/Beta' Lobe")

                is_act = self.active_kc_mask[real_idx % len(self.active_kc_mask)] if len(self.active_kc_mask) > 0 else False
                hz = "24.0 Hz (Spiking!)" if is_act else "0.5 Hz (Silent/Sparse)"

                return {
                    "name": f"Kenyon Cell ({kc_t})",
                    "type": kc_t,
                    "system": f"Mushroom Body ({lobe_name})",
                    "nt": "ACh (Cholinergic)",
                    "nt_color": QColor(56, 189, 248),
                    "pos_3d": soma_pt,
                    "target_3d": target_pt,
                    "rate_hz": hz,
                    "role": f"Sparse olfactory memory encoding in {lobe_name}",
                    "partners": "In: Antennal PNs (Calyx) • Out: MBONs & DANs (Compartment)",
                    "color": QColor(6, 182, 212) if lobe == 0 else (QColor(245, 158, 11) if lobe == 1 else QColor(217, 70, 239)),
                }
            check_pts(self._soma_pts, kc_cb)

        return best

    def toggle_layer(self, name: str):
        """Toggle an anatomical layer on or off."""
        attr = f"layer_{name}"
        if hasattr(self, attr):
            setattr(self, attr, not getattr(self, attr))
            self.update()

    def set_layer_visible(self, name: str, visible: bool):
        """Explicitly set an anatomical layer's visibility."""
        attr = f"layer_{name}"
        if hasattr(self, attr):
            setattr(self, attr, bool(visible))
            self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self._pulse_phase = 0.0
        self._last_tick = time.perf_counter()
        if not self._timer.isActive():
            self._timer.start()
        self._fetch_telemetry()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._timer.stop()

    def set_view_mode(self, mode: str):
        if mode in (MODE_VALENCE, MODE_PLASTICITY, MODE_ANATOMY, MODE_ACTIVITY):
            self.view_mode = mode
            self.update()

    def set_preset(self, yaw: float, pitch: float):
        self.yaw = yaw
        self.pitch = pitch
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._cam_R = None
        self.update()

    def toggle_auto_rotate(self):
        self.auto_rotate = not self.auto_rotate
        if self.auto_rotate and self._timer.interval() != 33:
            self._timer.setInterval(33)
        self.update()

    def reset_camera(self):
        self.yaw, self.pitch = PRESET_ISOMETRIC
        self.zoom = 220.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.auto_rotate = False
        self._cam_R = None
        self.update()

    # ------------------------------------------------------------------ tick
    def _on_tick(self):
        if not self.isVisible():
            return

        now = time.perf_counter()
        dt = min(0.1, now - self._last_tick)
        self._last_tick = now

        # Auto rotation
        if self.auto_rotate:
            self.yaw += 0.5 * dt
            if self.yaw > math.pi:
                self.yaw -= 2 * math.pi
            self._cam_R = None

        dirty = self._fetch_telemetry()

        # Dynamic heading angle tracking
        if abs(self.rate_steer) > 0.1:
            self.heading_angle = (self.heading_angle + self.rate_steer * dt * 0.15) % (2 * math.pi)

        # Refresh live firing rate on inspected card if present
        if self._inspected_neuron is not None:
            t = str(self._inspected_neuron.get("type", ""))
            if t == "GF":
                self._inspected_neuron["rate_hz"] = f"{self.rate_gf:.1f} Hz"
            elif t == "MDN":
                self._inspected_neuron["rate_hz"] = f"{self.rate_mdn:.1f} Hz"
            elif t in ("DNp09", "WALK"):
                self._inspected_neuron["rate_hz"] = f"{self.rate_walk:.1f} Hz"
            elif t == "DNa02":
                self._inspected_neuron["rate_hz"] = f"{abs(self.rate_steer):.1f} Hz"
            elif "PAM" in t:
                self._inspected_neuron["rate_hz"] = f"{self.pam * 40.0:.1f} Hz"
            elif "PPL1" in t:
                self._inspected_neuron["rate_hz"] = f"{self.ppl1 * 40.0:.1f} Hz"
            elif "LC4" in t or "LPLC2" in t:
                self._inspected_neuron["rate_hz"] = f"{self.cue_near * 55.0:.1f} Hz"
            elif "ALPN" in str(self._inspected_neuron.get("system", "")):
                self._inspected_neuron["rate_hz"] = f"{self.cue_food * 45.0:.1f} Hz"

        # Check if animation or telemetry is active
        is_active = (
            self.auto_rotate
            or self._is_panning
            or bool(self._opto_flashes)
            or (time.perf_counter() - self._opto_status_time < 3.0)
            or self.pam > 0.02
            or self.ppl1 > 0.02
            or self.kc_active_count > 0
            or self.rate_gf > 1.5
            or self.rate_walk > 1.5
            or self.rate_mdn > 1.5
            or abs(self.rate_steer) > 1.5
            or self.rate_groom > 1.5
            or self.cue_food > 0.02
            or self.cue_near > 0.02
            or self.view_mode == MODE_ACTIVITY
        )

        if is_active:
            # High-rate rendering (33 ms ~ 30 FPS) with wave propagation
            if self._timer.interval() != 33:
                self._timer.setInterval(33)
            self._pulse_phase = (self._pulse_phase + dt * 1.5) % 1.0
            self.update()
        else:
            # Idle throttle: reduce timer to 200 ms (5 Hz) to save CPU
            if self._timer.interval() != 200:
                self._timer.setInterval(200)
            if dirty:
                self.update()

    def _fetch_telemetry(self) -> bool:
        """Update live readouts from the runner if available. Returns True if visual state changed."""
        if self.runner is None or not getattr(self.runner, "ready", False):
            return False

        dirty = False
        st = self.runner.stats()

        val = float(st.get("valence", 0.0))
        pam = float(st.get("pam", 0.0))
        ppl1 = float(st.get("ppl1", 0.0))
        mbon_app = float(st.get("mbon_app", 1.0))
        mbon_av = float(st.get("mbon_av", 1.0))
        kc_act = int(st.get("kc_active", 0))

        r_gf = float(st.get("gf", 0.0))
        r_walk = float(st.get("walk", 0.0))
        r_mdn = float(st.get("mdn", 0.0))
        r_steer = float(st.get("steer", 0.0))
        r_groom = float(st.get("groom", 0.0))
        c_food = float(st.get("cue_food", 0.0))
        c_near = float(st.get("cue_near", 0.0))

        if (
            abs(val - self.valence) > 0.01
            or abs(pam - self.pam) > 0.02
            or abs(ppl1 - self.ppl1) > 0.02
            or abs(mbon_app - self.mbon_app) > 0.02
            or abs(mbon_av - self.mbon_av) > 0.02
            or kc_act != self.kc_active_count
            or abs(r_gf - self.rate_gf) > 0.8
            or abs(r_walk - self.rate_walk) > 0.8
            or abs(r_mdn - self.rate_mdn) > 0.8
            or abs(r_steer - self.rate_steer) > 0.8
            or abs(r_groom - self.rate_groom) > 0.8
            or abs(c_food - self.cue_food) > 0.04
            or abs(c_near - self.cue_near) > 0.04
        ):
            dirty = True

        self.valence = val
        self.pam = pam
        self.ppl1 = ppl1
        self.mbon_app = mbon_app
        self.mbon_av = mbon_av
        prev_kc = self.kc_active_count
        self.kc_active_count = kc_act
        self.rate_gf = r_gf
        self.rate_walk = r_walk
        self.rate_mdn = r_mdn
        self.rate_steer = r_steer
        self.rate_groom = r_groom
        self.cue_food = c_food
        self.cue_near = c_near

        # Check for updated KC weights
        new_dirty = st.get("weights_dirty", 0)
        if new_dirty != self.weights_dirty:
            self.weights_dirty = new_dirty
            w = self.runner.get_mb_weights()
            if len(w) > 0:
                n_cp = min(len(self.kc_weights), len(w))
                self.kc_weights[:n_cp] = w[:n_cp]
                dirty = True

        # Active KC indicators based on cue drives
        cue_food = float(st.get("cue_food", 0.0))
        cue_near = float(st.get("cue_near", 0.0))
        has_active = self.kc_active_count > 0 or cue_food > 0.05 or cue_near > 0.05
        if has_active:
            if dirty or prev_kc != self.kc_active_count:
                rng = np.random.default_rng(123)
                prob = max(0.04, min(0.15, self.kc_active_count / max(1, self.data.n_kc)))
                self.active_kc_mask = rng.uniform(0, 1, self.data.n_kc) < prob
                dirty = True
        else:
            if np.any(self.active_kc_mask):
                self.active_kc_mask.fill(False)
                dirty = True

        return dirty

    # ------------------------------------------------------------------ 3D projection
    def _project_points(self, pts_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project (N, 3) coordinates into screen (x, y) coordinates with depth."""
        if self._cam_R is None:
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            cp, sp = math.cos(self.pitch), math.sin(self.pitch)
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
            self._cam_R = Ry @ Rx

        cam = pts_3d @ self._cam_R

        # Perspective divisor
        persp = 1.0 / np.maximum(0.2, 1.0 + cam[:, 2] * 0.22)

        cx = self.width() * 0.5 + self.pan_x
        cy_screen = self.height() * 0.48 + self.pan_y

        sx = cx + cam[:, 0] * self.zoom * persp
        sy_proj = cy_screen - cam[:, 1] * self.zoom * persp

        return sx, sy_proj, cam[:, 2], persp

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        w, h = self.width(), self.height()

        # Compute camera matrix for this frame once
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        self._cam_R = Ry @ Rx

        # 1. Background
        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0.0, QColor(9, 9, 13))
        bg_grad.setColorAt(1.0, QColor(14, 14, 20))
        p.fillRect(0, 0, w, h, bg_grad)

        # Subtle background grid lines
        p.setPen(QPen(QColor(255, 255, 255, 6), 1, Qt.DotLine))
        grid_step = 60
        for gx in range(0, w, grid_step):
            p.drawLine(gx, 0, gx, h)
        for gy in range(0, h, grid_step):
            p.drawLine(0, gy, w, gy)

        # 2. Draw Drosophila Brain Neuropil Wireframe
        if self.layer_wireframe:
            self._draw_brain_wireframe(p)

        # 2.5. Draw Translucent Neuropil Volumetric Envelopes (AL, MB, CX, Optic)
        if self.layer_hulls:
            self._draw_neuropil_volumes(p)

        # 3. Sensory Input Circuits (Olfactory & Visual)
        if self.layer_olfactory:
            self._draw_antennal_lobes(p)
        if self.layer_visual:
            self._draw_visual_projections(p)

        # 4. Central Complex Navigation & Compass Ring
        if self.layer_cx:
            self._draw_central_complex(p)

        # 5. Mushroom Body Learning Circuit
        if self.layer_mb:
            self._draw_axon_tracts(p)
            self._draw_kenyon_somas(p)
            self._draw_dopamine_clusters(p)
            self._draw_mbons(p)

        # 6. Motor Output System (Descending Neurons)
        if self.layer_motor:
            self._draw_descending_neurons(p)

        # 6.5. Optogenetic Laser Pulses & Flash Ripples
        self._draw_opto_flashes(p)

        # 6.6. Inspected Neuron Reticle & Synaptic Streamline
        self._draw_inspected_reticle(p)

        # 7. Draw HUD Overlay (Valence gauge, stats, dopamine bars, toolbar)
        self._draw_hud(p)

        # 8. Draw Help Guide Overlay (if toggled)
        if self._show_help:
            self._draw_help_overlay(p)

        p.end()

    # ------------------------------------------------------------------ wireframe
    def _draw_brain_wireframe(self, p: QPainter):
        if len(self._wf_p1) == 0:
            return

        x1, y1, _, _ = self._project_points(self._wf_p1)
        x2, y2, _, _ = self._project_points(self._wf_p2)

        p.setBrush(Qt.NoBrush)
        pen = QPen(QColor(56, 189, 248, 22), 1.0)
        p.setPen(pen)

        lines = [QLineF(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])) for i in range(len(x1))]
        p.drawLines(lines)

    # ------------------------------------------------------------------ neuropil hulls
    def _draw_neuropil_volumes(self, p: QPainter):
        """Draw translucent volumetric hulls around major brain neuropils with fresnel contour shading."""
        if not self.layer_hulls or not hasattr(self, "_neuropil_volumes"):
            return

        p.save()
        for nv in self._neuropil_volumes:
            verts_3d = nv["vertices"]
            faces = nv["faces"]
            base_col = nv["color"]

            sx_arr, sy_arr, z_cam, valid = self._project_points(verts_3d)

            # Depth sort faces from back to front
            f_z = (z_cam[faces[:, 0]] + z_cam[faces[:, 1]] + z_cam[faces[:, 2]]) / 3.0
            order = np.argsort(f_z)

            for fi in order:
                idx0, idx1, idx2 = faces[fi]
                if not (valid[idx0] and valid[idx1] and valid[idx2]):
                    continue

                poly = QPolygonF([
                    QPointF(sx_arr[idx0], sy_arr[idx0]),
                    QPointF(sx_arr[idx1], sy_arr[idx1]),
                    QPointF(sx_arr[idx2], sy_arr[idx2]),
                ])

                # 3D normal vector
                v0, v1, v2 = verts_3d[idx0], verts_3d[idx1], verts_3d[idx2]
                norm = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(norm)
                if norm_len < 1e-6:
                    continue
                norm = norm / norm_len

                # Camera space projection for fresnel rim shading
                if self._cam_R is not None:
                    norm_cam = self._cam_R @ norm
                    cos_theta = abs(float(norm_cam[2]))
                else:
                    cos_theta = 0.5

                rim = 1.0 - min(1.0, cos_theta)
                alpha_fill = int(6 + rim * 16)
                alpha_edge = int(18 + rim * 55)

                p.setBrush(QBrush(QColor(base_col.red(), base_col.green(), base_col.blue(), alpha_fill)))
                p.setPen(QPen(QColor(base_col.red(), base_col.green(), base_col.blue(), alpha_edge), 1.0))
                p.drawPolygon(poly)

        p.restore()

    # ------------------------------------------------------------------ tracts
    def _draw_axon_tracts(self, p: QPainter):
        """Draw curved axon bundles connecting Calyx to anterior heel and lobes."""
        if len(self._tract_indices) == 0:
            return

        sx, sy, _, _ = self._project_points(self._tract_soma)
        px, py, _, _ = self._project_points(self._tract_ped)
        tx, ty, _, _ = self._project_points(self._tract_target)

        weights = self.kc_weights[self._tract_indices]
        lobes = self._tract_lobes
        active = self.active_kc_mask[self._tract_indices]

        p.setBrush(Qt.NoBrush)

        for i in range(len(self._tract_indices)):
            w_val = weights[i]
            is_on = active[i]
            lobe = lobes[i]

            # Choose color based on view mode
            col = self._color_for_neuron(w_val, lobe, is_on, alpha=45 if not is_on else 180)
            line_w = 1.0 if not is_on else 2.0
            p.setPen(QPen(col, line_w))

            # Quadratic Bezier through Pedunculus waypoint
            path = QPainterPath()
            path.moveTo(sx[i], sy[i])
            path.quadTo(px[i], py[i], tx[i], ty[i])
            p.drawPath(path)

            # Traveling action potential pulse if neuron is active
            if is_on:
                t = self._pulse_phase
                # Point along Bezier: (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
                bx = (1 - t) ** 2 * sx[i] + 2 * (1 - t) * t * px[i] + t ** 2 * tx[i]
                by = (1 - t) ** 2 * sy[i] + 2 * (1 - t) * t * py[i] + t ** 2 * ty[i]
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(255, 255, 255, 230)))
                p.drawEllipse(QPointF(bx, by), 2.2, 2.2)
                p.setBrush(Qt.NoBrush)

    # ------------------------------------------------------------------ somas
    def _draw_kenyon_somas(self, p: QPainter):
        """Draw thousands of Kenyon cell somas in the bilateral Calyx cups."""
        if len(self._soma_indices) == 0:
            return

        sx, sy, sz, persp = self._project_points(self._soma_pts)
        weights = self.kc_weights[self._soma_indices]
        lobes = self._soma_lobes
        active = self.active_kc_mask[self._soma_indices]

        # Fast path optimization:
        # Group points into naive (batch rendered with drawPoints) and special (rendered individually).
        special_indices: list[int] = []
        naive_pts: list[QPointF] = []

        if self.view_mode == MODE_VALENCE:
            for i in range(len(self._soma_indices)):
                w_val = weights[i]
                is_on = active[i]
                if is_on or abs(w_val) > 0.005:
                    special_indices.append(i)
                else:
                    naive_pts.append(QPointF(sx[i], sy[i]))

            # 1. Batch draw naive somas in one fast drawPoints call
            if naive_pts:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(56, 189, 248, 140), 3.4, Qt.SolidLine, Qt.RoundCap))
                p.drawPoints(naive_pts)

        elif self.view_mode == MODE_ANATOMY:
            # Batch draw by anatomical lobe
            gamma_pts: list[QPointF] = []
            ab_pts: list[QPointF] = []
            abp_pts: list[QPointF] = []
            for i in range(len(self._soma_indices)):
                if active[i]:
                    special_indices.append(i)
                else:
                    l = lobes[i]
                    pt = QPointF(sx[i], sy[i])
                    if l == LOBE_GAMMA:
                        gamma_pts.append(pt)
                    elif l == LOBE_ALPHABETA:
                        ab_pts.append(pt)
                    else:
                        abp_pts.append(pt)

            p.setBrush(Qt.NoBrush)
            if gamma_pts:
                p.setPen(QPen(QColor(6, 182, 212, 140), 3.4, Qt.SolidLine, Qt.RoundCap))
                p.drawPoints(gamma_pts)
            if ab_pts:
                p.setPen(QPen(QColor(245, 158, 11, 140), 3.4, Qt.SolidLine, Qt.RoundCap))
                p.drawPoints(ab_pts)
            if abp_pts:
                p.setPen(QPen(QColor(217, 70, 239, 140), 3.4, Qt.SolidLine, Qt.RoundCap))
                p.drawPoints(abp_pts)

        else:  # MODE_PLASTICITY, MODE_ACTIVITY
            for i in range(len(self._soma_indices)):
                special_indices.append(i)

        # 2. Draw special (active or strongly conditioned) somas with individual halos
        if special_indices:
            p.setPen(Qt.NoPen)
            for i in special_indices:
                w_val = weights[i]
                is_on = active[i]
                lobe = lobes[i]

                base_r = 1.8 * persp[i]
                if is_on:
                    base_r = max(base_r * 1.8, 3.2)

                col = self._color_for_neuron(w_val, lobe, is_on, alpha=140 if not is_on else 255)
                p.setBrush(QBrush(col))

                pt = QPointF(sx[i], sy[i])
                p.drawEllipse(pt, base_r, base_r)

                # Halo for strongly learned or active cells
                if abs(w_val) > 0.04 or is_on:
                    halo_col = QColor(col)
                    halo_col.setAlpha(45)
                    p.setBrush(QBrush(halo_col))
                    p.drawEllipse(pt, base_r * 2.2, base_r * 2.2)

    # ------------------------------------------------------------------ dopamine
    def _draw_dopamine_clusters(self, p: QPainter):
        """Draw PAM (reward) and PPL1 (punishment) dopaminergic clusters."""
        dans = self.data.dan_pos
        dx, dy, dz, persp = self._project_points(dans)

        clusters = self.data.dan_cluster

        # 1. Radiant shockwaves if PAM or PPL1 is actively firing
        if self.pam > 0.05:
            pam_pts = np.flatnonzero(clusters == DAN_PAM)
            if len(pam_pts) > 0:
                mid_x = float(np.mean(dx[pam_pts]))
                mid_y = float(np.mean(dy[pam_pts]))
                self._draw_dopamine_wave(p, mid_x, mid_y, QColor(250, 204, 21), self.pam)

        if self.ppl1 > 0.05:
            ppl_pts = np.flatnonzero(clusters == DAN_PPL1)
            if len(ppl_pts) > 0:
                mid_x = float(np.mean(dx[ppl_pts]))
                mid_y = float(np.mean(dy[ppl_pts]))
                self._draw_dopamine_wave(p, mid_x, mid_y, QColor(239, 68, 68), self.ppl1)

        # 2. Draw individual DAN nodes
        p.setPen(Qt.NoPen)
        for i in range(len(dans)):
            cl = clusters[i]
            if cl == DAN_PAM:
                # Radiant Gold (Reward)
                glow = min(1.0, 0.4 + self.pam * 0.6)
                alpha = int(glow * 255)
                col = QColor(245, 158, 11, alpha)
                r = (2.4 + self.pam * 2.0) * persp[i]
            elif cl == DAN_PPL1:
                # Fiery Scarlet (Punishment / Pain)
                glow = min(1.0, 0.4 + self.ppl1 * 0.6)
                alpha = int(glow * 255)
                col = QColor(239, 68, 68, alpha)
                r = (2.4 + self.ppl1 * 2.0) * persp[i]
            else:
                col = QColor(168, 85, 247, 120)
                r = 1.8 * persp[i]

            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(dx[i], dy[i]), r, r)

    def _draw_dopamine_wave(self, p: QPainter, cx: float, cy: float, col: QColor, strength: float):
        """Draw concentric pulsating dopamine diffusion rings."""
        phase = self._pulse_phase
        max_r = 75.0 * strength

        for ring_idx in range(3):
            r = ((phase + ring_idx * 0.33) % 1.0) * max_r
            if r < 3:
                continue
            fade = max(0.0, 1.0 - r / max_r)
            ring_col = QColor(col)
            ring_col.setAlpha(int(fade * 140 * strength))
            p.setPen(QPen(ring_col, 1.8))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r, r)

    # ------------------------------------------------------------------ mbons
    def _draw_mbons(self, p: QPainter):
        """Draw Mushroom Body Output Neuron (MBON) terminals."""
        mbons = self.data.mbon_pos
        mx, my, mz, persp = self._project_points(mbons)

        signs = self.data.mbon_sign

        p.setPen(QPen(QColor(255, 255, 255, 100), 0.8))

        for i in range(len(mbons)):
            sign = signs[i]
            # Approach MBON (+1) vs Avoidance MBON (-1)
            if sign > 0:
                col = QColor(34, 197, 94, 210) # Spring Green
            elif sign < 0:
                col = QColor(244, 63, 94, 210) # Coral Rose
            else:
                col = QColor(148, 163, 184, 160)

            p.setBrush(QBrush(col))
            r = 3.0 * persp[i]
            p.drawEllipse(QPointF(mx[i], my[i]), r, r)

    # ------------------------------------------------------------------ central complex
    def _draw_central_complex(self, p: QPainter):
        """Draw Central Complex (CX) navigation hub: Ellipsoid Body (EB) compass ring & PB bridge."""
        if len(self._cx_pts) == 0:
            return

        cx, cy, _, persp = self._project_points(self._cx_pts)
        subs = self._cx_sub

        # 1. Ellipsoid Body (EB) Ring: Toroidal compass ring
        eb_mask = subs == 0
        if np.any(eb_mask):
            eb_idx = np.flatnonzero(eb_mask)
            eb_cx = cx[eb_idx]
            eb_cy = cy[eb_idx]

            # Connect perimeter nodes to form the ring outline
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(56, 189, 248, 120), 1.6))
            eb_poly = QPolygonF([QPointF(float(eb_cx[i]), float(eb_cy[i])) for i in range(len(eb_idx))])
            p.drawPolyline(eb_poly)
            if len(eb_idx) > 2:
                p.drawLine(QPointF(float(eb_cx[-1]), float(eb_cy[-1])), QPointF(float(eb_cx[0]), float(eb_cy[0])))

            # Draw E-PG compass neurons on the ring
            heading_phase = (self._pulse_phase * 2 * math.pi) % (2 * math.pi)
            for i in range(len(eb_idx)):
                angle = (i / len(eb_idx)) * 2 * math.pi
                diff = abs((angle - heading_phase + math.pi) % (2 * math.pi) - math.pi)
                is_heading = diff < 0.45

                if is_heading:
                    col = QColor(255, 255, 255, 250)
                    r = 3.6 * persp[eb_idx[i]]
                else:
                    col = QColor(56, 189, 248, 180)
                    r = 2.2 * persp[eb_idx[i]]

                p.setBrush(QBrush(col))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(float(eb_cx[i]), float(eb_cy[i])), r, r)

        # 2. Protocerebral Bridge (PB): Horizontal arched mustache
        pb_mask = subs == 1
        if np.any(pb_mask):
            pb_idx = np.flatnonzero(pb_mask)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(168, 85, 247, 130), 1.4))
            for i in range(len(pb_idx) - 1):
                p.drawLine(QPointF(float(cx[pb_idx[i]]), float(cy[pb_idx[i]])),
                           QPointF(float(cx[pb_idx[i+1]]), float(cy[pb_idx[i+1]])))

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(168, 85, 247, 200)))
            for i in range(len(pb_idx)):
                r = 2.0 * persp[pb_idx[i]]
                p.drawEllipse(QPointF(float(cx[pb_idx[i]]), float(cy[pb_idx[i]])), r, r)

        # 3. Fan-shaped Body (FB):
        fb_mask = subs == 2
        if np.any(fb_mask):
            fb_idx = np.flatnonzero(fb_mask)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(234, 179, 8, 150)))
            for i in range(len(fb_idx)):
                r = 1.8 * persp[fb_idx[i]]
                p.drawEllipse(QPointF(float(cx[fb_idx[i]]), float(cy[fb_idx[i]])), r, r)

    # ------------------------------------------------------------------ visual projections
    def _draw_visual_projections(self, p: QPainter):
        """Draw Visual Projection Neurons (VPN) from Optic Lobes to central brain (LC4, LC10, LPLC2, LPC1)."""
        if len(self._vpn_soma) == 0:
            return

        sx, sy, _, s_persp = self._project_points(self._vpn_soma)
        tx, ty, _, _ = self._project_points(self._vpn_target)
        types = self._vpn_type

        p.setBrush(Qt.NoBrush)

        for i in range(len(self._vpn_soma)):
            vtype = types[i]
            if "LC4" in vtype:
                col = QColor(245, 158, 11, 140)    # Amber (Collision/Looming speed)
            elif "LC10" in vtype:
                col = QColor(14, 165, 233, 140)    # Electric Sky-Blue (Target tracking / Steering)
            elif "LPLC2" in vtype:
                col = QColor(249, 115, 22, 140)    # Coral Orange (Looming size detector)
            elif "LPC1" in vtype:
                col = QColor(168, 85, 247, 140)    # Violet (Retreat drive)
            else:
                col = QColor(56, 189, 248, 100)

            mid_x = (sx[i] + tx[i]) * 0.5
            mid_y = min(sy[i], ty[i]) - 8.0
            p.setPen(QPen(col, 1.2))

            path = QPainterPath()
            path.moveTo(float(sx[i]), float(sy[i]))
            path.quadTo(float(mid_x), float(mid_y), float(tx[i]), float(ty[i]))
            p.drawPath(path)

            # Traveling action potential pulse when visual stimuli or activity is present
            vis_active = (self.cue_near > 0.02 or self.rate_gf > 1.5 or abs(self.rate_steer) > 1.5 or self.view_mode == MODE_ACTIVITY)
            if vis_active:
                t_pulse = (self._pulse_phase + (i % 7) * 0.14) % 1.0
                bx = (1 - t_pulse) ** 2 * sx[i] + 2 * (1 - t_pulse) * t_pulse * mid_x + t_pulse ** 2 * tx[i]
                by = (1 - t_pulse) ** 2 * sy[i] + 2 * (1 - t_pulse) * t_pulse * mid_y + t_pulse ** 2 * ty[i]
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(255, 255, 255, 240)))
                p.drawEllipse(QPointF(float(bx), float(by)), 2.4, 2.4)
                p.setBrush(Qt.NoBrush)

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            r = 2.2 * s_persp[i]
            p.drawEllipse(QPointF(float(sx[i]), float(sy[i])), r, r)

    # ------------------------------------------------------------------ antennal lobes
    def _draw_antennal_lobes(self, p: QPainter):
        """Draw Antennal Lobes (AL) and olfactory projection neurons (ALPN) to Calyx & Lateral Horn."""
        if len(self._alpn_soma) == 0:
            return

        sx, sy, _, s_persp = self._project_points(self._alpn_soma)
        tx, ty, _, _ = self._project_points(self._alpn_target)

        p.setBrush(Qt.NoBrush)
        pen = QPen(QColor(16, 185, 129, 90), 1.0)
        p.setPen(pen)

        odor_active = (self.cue_food > 0.02 or self.pam > 0.05 or self.view_mode == MODE_ACTIVITY)

        for i in range(len(self._alpn_soma)):
            path = QPainterPath()
            path.moveTo(float(sx[i]), float(sy[i]))
            mid_x = (sx[i] + tx[i]) * 0.5
            mid_y = (sy[i] + ty[i]) * 0.5 - 12.0
            path.quadTo(float(mid_x), float(mid_y), float(tx[i]), float(ty[i]))
            p.drawPath(path)

            # Traveling olfactory pulse packet
            if odor_active:
                t_pulse = (self._pulse_phase + (i % 5) * 0.20) % 1.0
                bx = (1 - t_pulse) ** 2 * sx[i] + 2 * (1 - t_pulse) * t_pulse * mid_x + t_pulse ** 2 * tx[i]
                by = (1 - t_pulse) ** 2 * sy[i] + 2 * (1 - t_pulse) * t_pulse * mid_y + t_pulse ** 2 * ty[i]
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(167, 243, 208, 240)))
                p.drawEllipse(QPointF(float(bx), float(by)), 2.2, 2.2)
                p.setBrush(Qt.NoBrush)

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(16, 185, 129, 210)))
        for i in range(len(self._alpn_soma)):
            r = 2.6 * s_persp[i]
            p.drawEllipse(QPointF(float(sx[i]), float(sy[i])), r, r)

    # ------------------------------------------------------------------ descending neurons
    def _draw_descending_neurons(self, p: QPainter):
        """Draw Descending Motor Neurons (DN Hub) plunging through cervical connective toward the VNC."""
        if len(self._dn_soma) == 0:
            return

        sx, sy, _, s_persp = self._project_points(self._dn_soma)
        tx, ty, _, t_persp = self._project_points(self._dn_target)
        types = self._dn_type

        font_tag = QFont("Cascadia Mono", 7, QFont.Bold)

        for i in range(len(self._dn_soma)):
            ntype = types[i]

            if ntype == "GF":
                col = QColor(239, 68, 68, 240) # Giant Fiber (blazing crimson)
                line_w = 2.6
            elif ntype == "MDN":
                col = QColor(217, 70, 239, 230) # Moonwalker (fuchsia)
                line_w = 2.0
            elif ntype == "WALK":
                col = QColor(34, 197, 94, 230) # Forward walk (lime green)
                line_w = 2.0
            elif "DNa" in ntype:
                col = QColor(6, 182, 212, 230) # Steering (cyan)
                line_w = 2.0
            elif ntype == "GROOM":
                col = QColor(139, 92, 246, 230) # Grooming (violet)
                line_w = 2.0
            else:
                col = QColor(148, 163, 184, 180)
                line_w = 1.4

            # Descending axon trunk
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, line_w))
            p.drawLine(QPointF(float(sx[i]), float(sy[i])), QPointF(float(tx[i]), float(ty[i])))

            # Action potentials propagating down motor axon trunks into the VNC
            is_firing = (
                (ntype == "GF" and self.rate_gf > 1.5) or
                (ntype == "WALK" and self.rate_walk > 1.5) or
                (ntype == "MDN" and self.rate_mdn > 1.5) or
                ("DNa" in ntype and abs(self.rate_steer) > 1.5) or
                (ntype == "GROOM" and self.rate_groom > 1.5) or
                (self.view_mode == MODE_ACTIVITY)
            )

            if is_firing:
                t_pulse = (self._pulse_phase * 2.2 + (i % 3) * 0.33) % 1.0
                px = sx[i] + (tx[i] - sx[i]) * t_pulse
                py = sy[i] + (ty[i] - sy[i]) * t_pulse
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(255, 255, 255, 250)))
                p.drawEllipse(QPointF(float(px), float(py)), 3.2, 3.2)
                p.setBrush(Qt.NoBrush)

            # Soma node in central brain
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            r = 3.8 * s_persp[i]
            p.drawEllipse(QPointF(float(sx[i]), float(sy[i])), r, r)

            # Cervical terminal marker
            r_term = 2.0 * t_persp[i]
            p.drawEllipse(QPointF(float(tx[i]), float(ty[i])), r_term, r_term)

            # Labeled tag for primary anchor neurons
            if ntype in ("GF", "MDN", "WALK") and i % 2 == 0:
                p.setFont(font_tag)
                p.setPen(col)
                p.drawText(int(sx[i]) + 6, int(sy[i]) + 3, ntype)
    def _color_for_neuron(self, weight: float, lobe: int, is_active: bool, alpha: int = 180) -> QColor:
        """Map neuron attributes to color based on current visual mode."""
        if is_active:
            # Active firing spark: bright white-hot cyan
            return QColor(255, 255, 255, min(255, alpha + 50))

        if self.view_mode == MODE_VALENCE:
            # Learned mental map:
            # weight > 0: Appetitive memory (Reward -> Emerald Green to Golden Yellow)
            # weight < 0: Aversive memory (Punishment -> Ruby Red to Crimson Violet)
            # weight ~ 0: Naive (Slate Ice-Blue)
            if weight > 0.005:
                t = min(1.0, weight / 0.15)
                # Emerald (#10b981) to Gold (#fbbf24)
                r = int(16 * (1 - t) + 251 * t)
                g = int(185 * (1 - t) + 191 * t)
                b = int(129 * (1 - t) + 36 * t)
                return QColor(r, g, b, min(255, int(alpha * (1.0 + t * 0.4))))
            elif weight < -0.005:
                t = min(1.0, -weight / 0.15)
                # Crimson (#f43f5e) to Ruby Red (#dc2626)
                r = int(244 * (1 - t) + 220 * t)
                g = int(63 * (1 - t) + 38 * t)
                b = int(94 * (1 - t) + 38 * t)
                return QColor(r, g, b, min(255, int(alpha * (1.0 + t * 0.4))))
            else:
                # Naive / neutral
                return QColor(56, 189, 248, alpha)

        elif self.view_mode == MODE_PLASTICITY:
            # Plasticity magnitude |Δw| heatmap
            mag = min(1.0, abs(weight) / 0.12)
            # Blue-violet (#3b82f6) -> Orange (#f97316) -> Radiant Yellow (#fde047)
            if mag < 0.5:
                t = mag / 0.5
                r = int(59 * (1 - t) + 249 * t)
                g = int(130 * (1 - t) + 115 * t)
                b = int(246 * (1 - t) + 22 * t)
            else:
                t = (mag - 0.5) / 0.5
                r = int(249 * (1 - t) + 253 * t)
                g = int(115 * (1 - t) + 224 * t)
                b = int(22 * (1 - t) + 71 * t)
            return QColor(r, g, b, min(255, int(alpha * (0.8 + mag * 0.5))))

        elif self.view_mode == MODE_ANATOMY:
            # Biological anatomical lobes
            if lobe == LOBE_GAMMA:
                return QColor(6, 182, 212, alpha)       # Cyan (Gamma)
            elif lobe == LOBE_ALPHABETA:
                return QColor(245, 158, 11, alpha)      # Amber (Alpha/Beta)
            else:
                return QColor(217, 70, 239, alpha)      # Magenta (Alpha'/Beta')

        else: # MODE_ACTIVITY
            # Base dim cyan, sparks handled above
            return QColor(71, 85, 105, alpha)

    def _draw_opto_flashes(self, p: QPainter):
        """Draw animated optogenetic laser beams, concentric shockwave ripples, and floating badges."""
        if not self._opto_flashes:
            return

        now = time.perf_counter()
        active = []
        for f in self._opto_flashes:
            dt = now - f["time"]
            dur = f.get("duration", 0.9)
            if dt > dur:
                continue
            active.append(f)
            progress = dt / dur
            alpha = int(255 * (1.0 - progress))
            col = f["color"]

            sx_arr, sy_arr, _, valid_arr = self._project_points(np.array([f["pos"]], dtype=np.float32))
            if not valid_arr[0]:
                continue
            sx, sy = float(sx_arr[0]), float(sy_arr[0])

            p.save()
            # 1. Optical laser beam from top of screen into target
            beam_alpha = int(alpha * 0.75)
            beam_pen = QPen(QColor(col.red(), col.green(), col.blue(), beam_alpha), 2.0, Qt.DashLine)
            p.setPen(beam_pen)
            p.drawLine(int(sx), 0, int(sx), int(sy))

            # 2. Concentric expanding shockwave ripple rings
            p.setBrush(Qt.NoBrush)
            for r_idx in range(3):
                phase = (progress + r_idx * 0.25) % 1.0
                radius = 8.0 + phase * 65.0
                ring_alpha = int(alpha * (1.0 - phase))
                if ring_alpha > 5:
                    r_col = QColor(col.red(), col.green(), col.blue(), ring_alpha)
                    p.setPen(QPen(r_col, max(1.2, 3.5 * (1.0 - phase))))
                    p.drawEllipse(QPointF(sx, sy), radius, radius)

            # 3. Bright central photostimulation flare
            flare_grad = QRadialGradient(sx, sy, 24.0)
            flare_grad.setColorAt(0.0, QColor(255, 255, 255, alpha))
            flare_grad.setColorAt(0.35, QColor(col.red(), col.green(), col.blue(), int(alpha * 0.85)))
            flare_grad.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
            p.setBrush(QBrush(flare_grad))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(sx, sy), 24.0, 24.0)

            # 4. Floating holographic badge
            badge_y = sy - 30.0 - progress * 20.0
            badge_text = f"⚡ ChR2: {f['name']} ({f['action']})"
            font_badge = QFont("Segoe UI", 8, QFont.Bold)
            p.setFont(font_badge)
            bw = p.fontMetrics().horizontalAdvance(badge_text) + 16
            bx = int(sx - bw / 2)
            by = int(badge_y - 10)
            b_rect = QRect(bx, by, bw, 20)

            p.setBrush(QBrush(QColor(12, 14, 20, int(alpha * 0.88))))
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), int(alpha * 0.9)), 1.2))
            p.drawRoundedRect(b_rect, 4, 4)

            p.setPen(QColor(255, 255, 255, alpha))
            p.drawText(b_rect, Qt.AlignCenter, badge_text)
            p.restore()

        self._opto_flashes = active

    def _draw_opto_banner(self, p: QPainter, w: int):
        """Draw glowing top status banner when optogenetic pulse is evoked."""
        now = time.perf_counter()
        dt = now - self._opto_status_time
        if dt >= 3.0:
            return

        fade = max(0.0, min(1.0, 3.0 - dt))
        alpha = int(255 * fade)

        font_b = QFont("Segoe UI", 9, QFont.Bold)
        p.setFont(font_b)
        text = f"⚡ OPTOGENETIC PULSE: {self._opto_status}"
        tw = p.fontMetrics().horizontalAdvance(text) + 28
        bx = int((w - tw) / 2)
        by = 12
        rect = QRect(bx, by, tw, 26)

        p.setBrush(QBrush(QColor(10, 18, 28, int(225 * fade))))
        p.setPen(QPen(QColor(0, 229, 255, alpha), 1.5))
        p.drawRoundedRect(rect, 13, 13)

        p.setPen(QColor(230, 248, 255, alpha))
        p.drawText(rect, Qt.AlignCenter, text)

    def _draw_inspected_reticle(self, p: QPainter):
        """Draw 3D animated target reticle and axon connection line for inspected neuron."""
        if self._inspected_neuron is None:
            return

        pos_3d = self._inspected_neuron.get("pos_3d")
        if pos_3d is None:
            return

        sx_arr, sy_arr, _, valid = self._project_points(np.array([pos_3d], dtype=np.float32))
        if not valid[0]:
            return

        sx, sy = float(sx_arr[0]), float(sy_arr[0])
        col = self._inspected_neuron.get("color", QColor(168, 85, 247))

        p.save()
        # 1. Axon projection line to target (if present)
        target_3d = self._inspected_neuron.get("target_3d")
        if target_3d is not None:
            tx_arr, ty_arr, _, t_valid = self._project_points(np.array([target_3d], dtype=np.float32))
            if t_valid[0]:
                tx, ty = float(tx_arr[0]), float(ty_arr[0])
                p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 170), 1.5, Qt.DashLine))
                p.drawLine(int(sx), int(sy), int(tx), int(ty))

                # Terminal end marker
                p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 210)))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(tx, ty), 4.5, 4.5)

        # 2. Pulsing reticle rings
        now = time.perf_counter()
        pulse = 0.5 + 0.5 * math.sin(now * 5.0)
        ring_r = 14.0 + pulse * 4.0

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 190), 1.5))
        p.drawEllipse(QPointF(sx, sy), ring_r, ring_r)

        # Inner solid dot
        p.setBrush(QBrush(QColor(255, 255, 255, 230)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(sx, sy), 3.0, 3.0)

        # 4 Corner reticle tick marks
        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 220), 1.5))
        t_len = 5.0
        gap = ring_r + 2.0
        p.drawLine(int(sx - gap - t_len), int(sy), int(sx - gap), int(sy))
        p.drawLine(int(sx + gap), int(sy), int(sx + gap + t_len), int(sy))
        p.drawLine(int(sx), int(sy - gap - t_len), int(sx), int(sy - gap))
        p.drawLine(int(sx), int(sy + gap), int(sx), int(sy + gap + t_len))

        p.restore()

    def _draw_inspector_card(self, p: QPainter, w: int, h: int):
        """Draw holographic glass card displaying FlyWire connectome metadata and real-time Hz."""
        if self._inspected_neuron is None:
            return

        cw = min(360, w - 32)
        ch = 118
        cx = 16
        cy = h - 52 - ch - 12
        card_rect = QRect(cx, cy, cw, ch)

        p.save()
        # Card background gradient
        bg = QLinearGradient(cx, cy, cx, cy + ch)
        bg.setColorAt(0.0, QColor(14, 16, 24, 240))
        bg.setColorAt(1.0, QColor(10, 11, 16, 248))
        p.setBrush(QBrush(bg))

        col = self._inspected_neuron.get("color", QColor(168, 85, 247))
        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 170), 1.2))
        p.drawRoundedRect(card_rect, 6, 6)

        # Top line: Category + Pin status
        font_cat = QFont("Segoe UI", 7, QFont.Bold)
        p.setFont(font_cat)
        p.setPen(QColor(col.red(), col.green(), col.blue(), 230))
        status_txt = "PINNED" if self._inspected_pinned else "HOVER PREVIEW"
        cat_txt = f"NEURON INSPECTOR • {self._inspected_neuron.get('system', '')} • [{status_txt}]"
        p.drawText(cx + 12, cy + 18, cat_txt)

        # Close button [×] at top right
        close_rect = QRect(cx + cw - 22, cy + 6, 16, 16)
        self._btn_rects["close_inspector"] = close_rect
        p.setPen(QColor(150, 150, 160, 200))
        p.drawText(close_rect, Qt.AlignCenter, "×")

        # Main Name
        font_name = QFont("Segoe UI", 10, QFont.Bold)
        p.setFont(font_name)
        p.setPen(QColor(245, 245, 250))
        p.drawText(cx + 12, cy + 36, self._inspected_neuron.get("name", "Neuron"))

        # Badges row: Neurotransmitter + Rate
        font_badge = QFont("Segoe UI", 7, QFont.DemiBold)
        p.setFont(font_badge)

        bx = cx + 12
        by = cy + 44
        # NT Badge
        nt_txt = self._inspected_neuron.get("nt", "ACh")
        nt_w = p.fontMetrics().horizontalAdvance(nt_txt) + 12
        nt_rect = QRect(bx, by, nt_w, 18)
        nt_c = self._inspected_neuron.get("nt_color", QColor(56, 189, 248))
        p.setBrush(QBrush(QColor(nt_c.red(), nt_c.green(), nt_c.blue(), 55)))
        p.setPen(QPen(QColor(nt_c.red(), nt_c.green(), nt_c.blue(), 180), 1))
        p.drawRoundedRect(nt_rect, 3, 3)
        p.setPen(QColor(245, 245, 250))
        p.drawText(nt_rect, Qt.AlignCenter, nt_txt)
        bx += nt_w + 6

        # Rate Badge
        rate_txt = f"Rate: {self._inspected_neuron.get('rate_hz', '0.0 Hz')}"
        rate_w = p.fontMetrics().horizontalAdvance(rate_txt) + 12
        rate_rect = QRect(bx, by, rate_w, 18)
        p.setBrush(QBrush(QColor(30, 32, 42, 180)))
        p.setPen(QPen(QColor(80, 85, 105, 140), 1))
        p.drawRoundedRect(rate_rect, 3, 3)
        p.setPen(QColor(220, 225, 235))
        p.drawText(rate_rect, Qt.AlignCenter, rate_txt)

        # Role description
        font_role = QFont("Segoe UI", 8, QFont.Normal)
        p.setFont(font_role)
        p.setPen(QColor(180, 185, 195))
        p.drawText(cx + 12, cy + 78, self._inspected_neuron.get("role", ""))

        # Synaptic Partners
        font_part = QFont("Segoe UI", 7, QFont.Normal)
        p.setFont(font_part)
        p.setPen(QColor(130, 135, 145))
        p.drawText(cx + 12, cy + 96, self._inspected_neuron.get("partners", ""))

        p.restore()

    def _draw_pip_hud(self, p: QPainter, w: int, h: int):
        """Draw ultra-compact glass picture-in-picture HUD for desktop mini-brain mode."""
        font_badge = QFont("Segoe UI", 7, QFont.Bold)
        font_mono = QFont("Cascadia Mono", 8, QFont.DemiBold)

        # 1. Top-Left: Mini title & heading degrees
        p.setFont(font_badge)
        deg = int(math.degrees(self.heading_angle)) % 360
        title_txt = f"FlyWire 3D • {deg}°"
        p.setPen(QColor(56, 189, 248, 220))
        p.drawText(10, 16, title_txt)

        # Top-Right: Mini Reset & Rotate buttons
        btn_w, btn_h = 34, 18
        rot_rect = QRect(w - 76, 6, btn_w, btn_h)
        rst_rect = QRect(w - 38, 6, btn_w, btn_h)
        self._btn_rects["rot"] = rot_rect
        self._btn_rects["reset"] = rst_rect

        for b_rect, b_txt, active in [(rot_rect, "Rot", self.auto_rotate), (rst_rect, "Rst", False)]:
            bg_c = QColor(56, 189, 248, 60) if active else QColor(24, 26, 34, 180)
            brd_c = QColor(56, 189, 248, 160) if active else QColor(60, 65, 80, 140)
            p.setBrush(QBrush(bg_c))
            p.setPen(QPen(brd_c, 1))
            p.drawRoundedRect(b_rect, 3, 3)
            p.setPen(QColor(240, 245, 255))
            p.drawText(b_rect, Qt.AlignCenter, b_txt)

        # 2. Bottom: Compact 1-line glass telemetry pill
        bar_h = 22
        bar_y = h - bar_h - 6
        bar_rect = QRect(8, bar_y, w - 16, bar_h)

        p.setBrush(QBrush(QColor(12, 14, 20, 215)))
        p.setPen(QPen(QColor(56, 189, 248, 100), 1))
        p.drawRoundedRect(bar_rect, 4, 4)

        p.setFont(font_mono)
        v_sign = "+" if self.valence >= 0 else ""
        tel_txt = f"V:{v_sign}{self.valence:.2f} | GF:{self.rate_gf:.0f} | W:{self.rate_walk:.0f} | PAM:{int(self.pam*100)}%"
        p.setPen(QColor(230, 235, 245))
        p.drawText(bar_rect, Qt.AlignCenter, tel_txt)

    # ------------------------------------------------------------------ HUD
    def _draw_hud(self, p: QPainter):
        """Draw titles, control buttons, and live telemetry overlays."""
        self._btn_rects.clear()
        w, h = self.width(), self.height()

        if self.pip_mode:
            self._draw_pip_hud(p, w, h)
            return

        font_title = QFont("Segoe UI", 10, QFont.Bold)
        font_small = QFont("Segoe UI", 8, QFont.Normal)
        font_mono = QFont("Cascadia Mono", 9, QFont.DemiBold)

        # 0.5. Top-Center Optogenetic Status Banner
        if self._opto_status:
            self._draw_opto_banner(p, w)

        # 0.6. Neuron Inspector Card
        if self._inspected_neuron is not None:
            self._draw_inspector_card(p, w, h)

        # 1. Top-Left: Header
        p.setFont(font_title)
        p.setPen(QColor(236, 236, 241, 230))
        p.drawText(16, 24, "3D DROSOPHILA CONNECTOME & NEURAL MAP")

        p.setFont(font_small)
        p.setPen(QColor(139, 139, 150, 200))
        p.drawText(16, 38, f"FlyWire v783 • {self.data.n_kc:,} KCs | {self.data.n_mbon} MBONs | {self.data.n_cx} CX Compass | {self.data.n_vpn} Optic | {self.data.n_dn} Motor DNs")

        # 1.5. Layer Toggles Bar (below header)
        self._draw_layer_filters(p, 16, 46)

        # 2. Top-Right: Toolbar buttons
        self._draw_toolbar(p, w)

        # 3. Bottom: Telemetry Bar (Valence Dual-Meter, Dopamine Bars, KC Count)
        self._draw_bottom_telemetry(p, w, h, font_small, font_mono)

        # 4. Hover Tooltip
        if self._hover_text:
            p.setFont(font_small)
            p.setPen(QColor(255, 255, 255, 220))
            p.setBrush(QBrush(QColor(20, 20, 26, 230)))
            rect = QRect(16, h - 84, 280, 24)
            p.drawRoundedRect(rect, 4, 4)
            p.drawText(rect.adjusted(8, 0, 0, 0), Qt.AlignVCenter, self._hover_text)

    def _draw_layer_filters(self, p: QPainter, start_x: int, start_y: int):
        """Draw interactive layer toggle pills on top-left."""
        font_pill = QFont("Segoe UI", 7, QFont.DemiBold)
        p.setFont(font_pill)

        w = self.width()
        is_compact = w < 760

        layers = [
            ("layer_mb", "Mushroom" if is_compact else "Mushroom Body", self.layer_mb, QColor(245, 158, 11)),
            ("layer_cx", "CX Compass" if is_compact else "Central Complex (CX)", self.layer_cx, QColor(56, 189, 248)),
            ("layer_visual", "Vision" if is_compact else "Vision (VPN)", self.layer_visual, QColor(249, 115, 22)),
            ("layer_olfactory", "Odor" if is_compact else "Odor (AL)", self.layer_olfactory, QColor(16, 185, 129)),
            ("layer_motor", "Motor" if is_compact else "Motor (DN)", self.layer_motor, QColor(239, 68, 68)),
            ("layer_hulls", "Hulls" if is_compact else "Neuropils", self.layer_hulls, QColor(168, 85, 247)),
            ("layer_wireframe", "Cage" if is_compact else "Wireframe", self.layer_wireframe, QColor(148, 163, 184)),
        ]

        bx = start_x
        by = start_y
        pill_h = 19
        spacing = 5

        for lid, name, is_on, accent_col in layers:
            text_w = p.fontMetrics().horizontalAdvance(name) + 12
            rect = QRect(bx, by, text_w, pill_h)
            self._btn_rects[lid] = rect

            if is_on:
                bg_col = QColor(accent_col.red(), accent_col.green(), accent_col.blue(), 55)
                border_col = QColor(accent_col.red(), accent_col.green(), accent_col.blue(), 180)
                text_col = QColor(255, 255, 255, 240)
            else:
                bg_col = QColor(24, 24, 30, 160)
                border_col = QColor(55, 55, 65, 130)
                text_col = QColor(120, 120, 130, 180)

            p.setBrush(QBrush(bg_col))
            p.setPen(QPen(border_col, 1))
            p.drawRoundedRect(rect, 4, 4)

            p.setPen(text_col)
            p.drawText(rect, Qt.AlignCenter, name)

            bx += text_w + spacing

    def _draw_toolbar(self, p: QPainter, w: int):
        """Draw interactive buttons on top-right corner."""
        font_btn = QFont("Segoe UI", 8, QFont.DemiBold)
        p.setFont(font_btn)

        if w < 640:
            buttons = [
                ("iso", "Iso"),
                ("dorsal", "Top"),
                ("ant", "Front"),
                ("rot", "Rotate" if not self.auto_rotate else "Stop"),
                ("reset", "Reset"),
                ("help", "?" if not self._show_help else "? Open"),
            ]
            mode_buttons = [
                (MODE_VALENCE, "Valence"),
                (MODE_PLASTICITY, "Plasticity"),
                (MODE_ANATOMY, "Anatomy"),
                (MODE_ACTIVITY, "Activity"),
            ]
        else:
            buttons = [
                ("iso", "3D Isometric"),
                ("dorsal", "Top (Dorsal)"),
                ("ant", "Front (Anterior)"),
                ("rot", "Auto Rotate" if not self.auto_rotate else "Stop Rotation"),
                ("reset", "Reset Camera"),
                ("help", "?" if not self._show_help else "? Open"),
            ]
            mode_buttons = [
                (MODE_VALENCE, "Mental Map (Valence)"),
                (MODE_PLASTICITY, "Plasticity |Δw|"),
                (MODE_ANATOMY, "Anatomy"),
                (MODE_ACTIVITY, "Live Activity"),
            ]

        bx = w - 16
        by = 14
        btn_h = 24
        spacing = 6

        # Render preset buttons right-to-left
        for bid, text in reversed(buttons):
            text_w = p.fontMetrics().horizontalAdvance(text) + 16
            bx -= text_w
            rect = QRect(bx, by, text_w, btn_h)
            self._btn_rects[bid] = rect

            is_active = (bid == "rot" and self.auto_rotate) or (bid == "help" and self._show_help)
            bg_col = QColor(76, 201, 240, 60) if is_active else QColor(28, 28, 33, 190)
            border_col = QColor(76, 201, 240, 140) if is_active else QColor(58, 58, 66, 160)

            p.setBrush(QBrush(bg_col))
            p.setPen(QPen(border_col, 1))
            p.drawRoundedRect(rect, 6, 6)

            p.setPen(QColor(236, 236, 241, 220))
            p.drawText(rect, Qt.AlignCenter, text)

            bx -= spacing

        mbx = w - 16
        mby = by + btn_h + 8
        for mid, mtext in reversed(mode_buttons):
            mw = p.fontMetrics().horizontalAdvance(mtext) + 14
            mbx -= mw
            mrect = QRect(mbx, mby, mw, 22)
            self._btn_rects[f"mode_{mid}"] = mrect

            is_sel = (self.view_mode == mid)
            bg_c = QColor(76, 201, 240, 90) if is_sel else QColor(24, 24, 28, 160)
            border_c = QColor(76, 201, 240, 200) if is_sel else QColor(48, 48, 54, 140)

            p.setBrush(QBrush(bg_c))
            p.setPen(QPen(border_c, 1))
            p.drawRoundedRect(mrect, 5, 5)

            p.setPen(QColor(255, 255, 255, 240) if is_sel else QColor(160, 160, 170, 200))
            p.drawText(mrect, Qt.AlignCenter, mtext)

            mbx -= spacing

        # Separator gap
        mbx -= 8

        # Tool mode selector buttons
        tool_buttons = [
            ("tool_camera", "Orbit" if w < 640 else "📷 Orbit", self.tool_mode == "camera", QColor(148, 163, 184)),
            ("tool_opto", "ChR2" if w < 640 else "⚡ ChR2 Opto", self.tool_mode == "opto", QColor(0, 229, 255)),
            ("tool_inspect", "Inspect" if w < 640 else "🔍 Inspect", self.tool_mode == "inspect", QColor(168, 85, 247)),
        ]

        for tid, ttext, is_act, col in reversed(tool_buttons):
            tw = p.fontMetrics().horizontalAdvance(ttext) + 14
            mbx -= tw
            trect = QRect(mbx, mby, tw, 22)
            self._btn_rects[tid] = trect

            if is_act:
                bg_c = QColor(col.red(), col.green(), col.blue(), 75)
                border_c = QColor(col.red(), col.green(), col.blue(), 210)
                text_c = QColor(255, 255, 255, 250)
            else:
                bg_c = QColor(24, 24, 28, 160)
                border_c = QColor(50, 50, 58, 140)
                text_c = QColor(140, 140, 150, 200)

            p.setBrush(QBrush(bg_c))
            p.setPen(QPen(border_c, 1))
            p.drawRoundedRect(trect, 5, 5)

            p.setPen(text_c)
            p.drawText(trect, Qt.AlignCenter, ttext)

            mbx -= spacing

    def _draw_bottom_telemetry(self, p: QPainter, w: int, h: int, font_small: QFont, font_mono: QFont):
        """Draw bottom glass telemetry card."""
        bar_h = 52
        bar_rect = QRect(16, h - bar_h - 14, w - 32, bar_h)

        # Card container
        p.setBrush(QBrush(QColor(18, 18, 23, 220)))
        p.setPen(QPen(QColor(45, 45, 53, 180), 1))
        p.drawRoundedRect(bar_rect, 10, 10)

        # 1. Valence Dual Meter (-1.00 Fear <-> +1.00 Desire)
        v_width = 170
        vx = bar_rect.x() + 16
        vy = bar_rect.y() + 14

        p.setFont(font_small)
        p.setPen(QColor(139, 139, 150, 220))
        p.drawText(vx, vy, "LEARNED VALENCE (MENTAL MAP)")

        # Meter background
        my = vy + 6
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(28, 28, 33)))
        p.drawRoundedRect(vx, my, v_width, 10, 5, 5)

        # Center line
        mid_vx = vx + v_width // 2
        p.setPen(QPen(QColor(100, 100, 110, 120), 1))
        p.drawLine(mid_vx, my, mid_vx, my + 10)

        # Fill bar
        val = max(-1.0, min(1.0, self.valence))
        if val > 0:
            fill_w = int((val) * (v_width // 2))
            p.setBrush(QBrush(QColor(16, 185, 129, 230))) # Emerald
            p.drawRoundedRect(mid_vx, my, fill_w, 10, 3, 3)
        elif val < 0:
            fill_w = int((-val) * (v_width // 2))
            p.setBrush(QBrush(QColor(239, 68, 68, 230))) # Ruby
            p.drawRoundedRect(mid_vx - fill_w, my, fill_w, 10, 3, 3)

        p.setFont(font_mono)
        v_color = QColor(16, 185, 129) if val > 0.02 else (QColor(239, 68, 68) if val < -0.02 else QColor(220, 220, 230))
        p.setPen(v_color)
        p.drawText(vx + v_width + 10, my + 9, f"{val:+.2f}")

        # 2. Dopamine Bars: PAM (Reward) & PPL1 (Punish)
        d_x = vx + v_width + 75
        p.setFont(font_small)
        p.setPen(QColor(139, 139, 150, 220))
        p.drawText(d_x, vy, "DOPAMINE (PAM REWARD / PPL1 PUNISH)")

        d_bar_w = 80
        # PAM Gold Bar
        p.setBrush(QBrush(QColor(28, 28, 33)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(d_x, my, d_bar_w, 10, 4, 4)
        pam_fill = int(min(1.0, self.pam) * d_bar_w)
        if pam_fill > 0:
            p.setBrush(QBrush(QColor(245, 158, 11, 230)))
            p.drawRoundedRect(d_x, my, pam_fill, 10, 4, 4)

        # PPL1 Red Bar
        ppl_x = d_x + d_bar_w + 10
        p.setBrush(QBrush(QColor(28, 28, 33)))
        p.drawRoundedRect(ppl_x, my, d_bar_w, 10, 4, 4)
        ppl_fill = int(min(1.0, self.ppl1) * d_bar_w)
        if ppl_fill > 0:
            p.setBrush(QBrush(QColor(239, 68, 68, 230)))
            p.drawRoundedRect(ppl_x, my, ppl_fill, 10, 4, 4)

        # 3. Active KCs Count & Sparsity
        k_x = ppl_x + d_bar_w + 35
        if k_x < bar_rect.right() - 140:
            p.setFont(font_small)
            p.setPen(QColor(139, 139, 150, 220))
            p.drawText(k_x, vy, "ACTIVE KC (SPARSE CODE)")

            p.setFont(font_mono)
            p.setPen(QColor(236, 236, 241))
            sparsity = (self.kc_active_count / max(1, self.data.n_kc)) * 100
            p.drawText(k_x, my + 9, f"{self.kc_active_count:,} / {self.data.n_kc:,} ({sparsity:.1f}%)")

        # 4. MBON Balance
        m_x = k_x + 160
        if m_x < bar_rect.right() - 110:
            p.setFont(font_small)
            p.setPen(QColor(139, 139, 150, 220))
            p.drawText(m_x, vy, "MBON BALANCE")

            p.setFont(font_mono)
            p.setPen(QColor(56, 189, 248))
            p.drawText(m_x, my + 9, f"+{self.mbon_app:.2f} / -{self.mbon_av:.2f}")

        # 5. Live Descending Motor Outputs Readout
        mot_x = m_x + 145
        if mot_x < bar_rect.right() - 150:
            p.setFont(font_small)
            p.setPen(QColor(139, 139, 150, 220))
            p.drawText(mot_x, vy, "MOTOR DNs (GF / WALK / MDN)")

            p.setFont(font_mono)
            gf_col = QColor(239, 68, 68) if self.rate_gf > 2.0 else QColor(148, 163, 184)
            p.setPen(gf_col)
            p.drawText(mot_x, my + 9, f"GF:{self.rate_gf:.0f}Hz")

            walk_col = QColor(34, 197, 94) if self.rate_walk > 2.0 else QColor(148, 163, 184)
            p.setPen(walk_col)
            p.drawText(mot_x + 60, my + 9, f"W:{self.rate_walk:.0f}Hz")

            mdn_col = QColor(217, 70, 239) if self.rate_mdn > 2.0 else QColor(148, 163, 184)
            p.setPen(mdn_col)
            p.drawText(mot_x + 115, my + 9, f"M:{self.rate_mdn:.0f}Hz")

    # ------------------------------------------------------------------ help overlay
    def _draw_help_overlay(self, p: QPainter):
        """Draw an informative modal overlay explaining the neural circuit and mental map."""
        w, h = self.width(), self.height()

        # Dim semi-transparent background behind help modal
        p.fillRect(0, 0, w, h, QColor(0, 0, 0, 180))

        card_w = min(w - 20, 800)
        card_h = min(h - 20, 560)
        card_x = (w - card_w) // 2
        card_y = (h - card_h) // 2
        self._help_card_rect = QRect(card_x, card_y, card_w, card_h)

        # Card container with dark glassmorphism styling
        p.setBrush(QBrush(QColor(17, 18, 25, 252)))
        p.setPen(QPen(QColor(76, 201, 240, 150), 1.2))
        p.drawRoundedRect(self._help_card_rect, 10, 10)

        # Header Title & Subtitle
        p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        p.setPen(QColor(240, 240, 248))
        p.drawText(card_x + 16, card_y + 24, "MUSHROOM BODY & MENTAL MAP GUIDE")

        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(140, 145, 160))
        p.drawText(card_x + 16, card_y + 40, "FlyWire v783 full connectome 3D neural circuit & reinforcement learning guide")

        # Close button in top-right of card
        close_w, close_h = 56, 20
        close_rect = QRect(card_x + card_w - close_w - 14, card_y + 14, close_w, close_h)
        self._btn_rects["close_help"] = close_rect

        p.setBrush(QBrush(QColor(36, 36, 44, 230)))
        p.setPen(QPen(QColor(70, 70, 82, 200), 1))
        p.drawRoundedRect(close_rect, 4, 4)
        p.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
        p.setPen(QColor(220, 220, 230))
        p.drawText(close_rect, Qt.AlignCenter, "Close")

        # 2x2 Information Panels
        panel_w = (card_w - 36) // 2
        panel_h = (card_h - 66) // 2
        panel1_x = card_x + 12
        panel2_x = panel1_x + panel_w + 12
        top_y = card_y + 52
        bot_y = top_y + panel_h + 8

        panels = [
            (
                QRect(panel1_x, top_y, panel_w, panel_h),
                "1. MENTAL MAP & COLOR CODES",
                [
                    (QColor(16, 185, 129), "Green / Gold (Reward)", "PAM reinforcement -> Approach desire (Appetitive MBON)."),
                    (QColor(239, 68, 68), "Red / Ruby (Punishment)", "PPL1 depression -> Avoidance fear (Aversive MBON)."),
                    (QColor(56, 189, 248), "Cyan Blue (Naive)", "Untrained, neutral odor synapses."),
                    (QColor(255, 255, 255), "White Sparks (Activity)", "Live spiking Kenyon cells and axonal propagation."),
                ],
            ),
            (
                QRect(panel2_x, top_y, panel_w, panel_h),
                "2. BIOLOGICAL CIRCUITS (LAYERS)",
                [
                    (QColor(245, 158, 11), "Mushroom Body (MB)", "Calyx somas, pedunculus & lobes for learned memory."),
                    (QColor(56, 189, 248), "Central Complex (CX)", "Ellipsoid Body (EB) compass heading ring & PB bridge."),
                    (QColor(249, 115, 22), "Visual System (VPN)", "LC4/LPLC2 (looming threat) & LC10 (target steering)."),
                    (QColor(239, 68, 68), "Motor Hub (DN)", "Giant Fiber (GF escape), MDN (moonwalk), DNp09 (walk)."),
                ],
            ),
            (
                QRect(panel1_x, bot_y, panel_w, panel_h),
                "3. DOPAMINE & LIVE TELEMETRY",
                [
                    (QColor(245, 158, 11), "PAM Dopamine Cluster", "Emits golden waves during sugar reward to reinforce approach."),
                    (QColor(239, 68, 68), "PPL1 Dopamine Cluster", "Emits red waves during nociception/impact to teach avoidance."),
                    (QColor(236, 236, 241), "Valence [-1.0 ... +1.0]", "-1.0 full avoidance fear, +1.0 full approach desire."),
                    (QColor(56, 189, 248), "MBON & Active KC", "Net output balance and instantaneous cell count recognizing cue."),
                ],
            ),
            (
                QRect(panel2_x, bot_y, panel_w, panel_h),
                "4. MOUSE & CAMERA CONTROLS",
                [
                    (QColor(140, 145, 160), "Left Click + Drag", "3D free rotation (Orbit)."),
                    (QColor(140, 145, 160), "Right Click / Shift", "2D camera translation (Pan)."),
                    (QColor(140, 145, 160), "Mouse Wheel", "Camera zoom in / out."),
                    (QColor(140, 145, 160), "Double Click / Reset", "Return to default isometric view."),
                ],
            ),
        ]

        font_panel_title = QFont("Segoe UI", 8, QFont.Bold)
        font_item_title = QFont("Segoe UI", 8, QFont.DemiBold)
        font_item_desc = QFont("Segoe UI", 7, QFont.Normal)

        for p_rect, title, items in panels:
            p.setBrush(QBrush(QColor(24, 25, 34, 190)))
            p.setPen(QPen(QColor(50, 52, 65, 160), 1))
            p.drawRoundedRect(p_rect, 6, 6)

            p.setFont(font_panel_title)
            p.setPen(QColor(76, 201, 240))
            p.drawText(p_rect.x() + 12, p_rect.y() + 18, title)

            item_step = max(24, (p_rect.height() - 28) // 4)
            iy = p_rect.y() + 32

            for dot_col, heading, desc in items:
                p.setBrush(QBrush(dot_col))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(p_rect.x() + 14, iy - 3), 3.0, 3.0)

                p.setFont(font_item_title)
                p.setPen(QColor(230, 230, 240))
                p.drawText(p_rect.x() + 24, iy, heading)

                p.setFont(font_item_desc)
                p.setPen(QColor(145, 150, 165))
                p.drawText(p_rect.x() + 24, iy + 12, desc)

                iy += item_step

    # ------------------------------------------------------------------ events
    def mousePressEvent(self, event):
        pos = event.pos()

        # Check button clicks
        for bid, rect in self._btn_rects.items():
            if rect.contains(pos):
                if bid == "help":
                    self._show_help = not self._show_help
                    self.update()
                    return
                elif bid == "close_help":
                    self._show_help = False
                    self.update()
                    return
                elif bid == "iso":
                    self.set_preset(*PRESET_ISOMETRIC)
                elif bid == "dorsal":
                    self.set_preset(*PRESET_DORSAL)
                elif bid == "ant":
                    self.set_preset(*PRESET_ANTERIOR)
                elif bid == "rot":
                    self.toggle_auto_rotate()
                elif bid == "reset":
                    self.reset_camera()
                elif bid.startswith("mode_"):
                    self.set_view_mode(bid.replace("mode_", ""))
                elif bid.startswith("layer_"):
                    self.toggle_layer(bid.replace("layer_", ""))
                elif bid.startswith("tool_"):
                    self.set_tool_mode(bid.replace("tool_", ""))
                elif bid == "close_inspector":
                    self._inspected_neuron = None
                    self._inspected_pinned = False
                    self.update()
                return

        # If help overlay is open: dismiss if clicked outside card, ignore inside clicks
        if self._show_help:
            if not self._help_card_rect.contains(pos):
                self._show_help = False
                self.update()
            return

        # Virtual Optogenetics click stimulation
        if self.tool_mode == "opto" or event.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
            thresh = 42.0 if self.tool_mode == "opto" else 22.0
            target = self._find_opto_target_at(pos, threshold_px=thresh)
            if target is not None:
                self.trigger_opto_stimulus(target)
                return
            elif self.tool_mode == "opto":
                self._trigger_opto_at_screen_pos(pos)
                return

        # Neuron & Connectome Inspector click handling
        if self.tool_mode == "inspect" or (event.modifiers() & Qt.ShiftModifier and event.button() == Qt.LeftButton):
            neuron = self._find_neuron_at(pos, threshold_px=28.0)
            if neuron is not None:
                self._inspected_neuron = neuron
                self._inspected_pinned = True
                self.update()
                return
            else:
                if self._inspected_pinned:
                    self._inspected_neuron = None
                    self._inspected_pinned = False
                    self.update()

        self._last_mouse_pos = pos
        if event.button() == Qt.RightButton or event.modifiers() & Qt.ShiftModifier:
            self._is_panning = True
        else:
            self._is_panning = False

    def mouseMoveEvent(self, event):
        pos = event.pos()

        if self._last_mouse_pos is not None:
            dx = pos.x() - self._last_mouse_pos.x()
            dy = pos.y() - self._last_mouse_pos.y()
            self._last_mouse_pos = pos

            if self._is_panning:
                self.pan_x += dx
                self.pan_y += dy
            else:
                self.yaw += dx * 0.008
                self.pitch = max(-1.55, min(1.55, self.pitch + dy * 0.008))
                self._cam_R = None

            self.update()
        else:
            # Hover detection for tooltips
            self._check_hover(pos)

            # Live inspect hover preview
            if self.tool_mode == "inspect" and not self._inspected_pinned:
                neuron = self._find_neuron_at(pos, threshold_px=22.0)
                if neuron != self._inspected_neuron:
                    self._inspected_neuron = neuron
                    self.update()

    def mouseReleaseEvent(self, event):
        self._last_mouse_pos = None
        self._is_panning = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 0.89
        self.zoom = max(60.0, min(900.0, self.zoom * factor))
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.reset_camera()

    def _check_hover(self, pos: QPoint):
        """Detect which button or anatomical region is under the cursor."""
        found = ""

        # Check button hover first
        for bid, rect in self._btn_rects.items():
            if rect.contains(pos):
                btn_tooltips = {
                    "help": "Mental Map & Biological Guide [Toggle]",
                    "close_help": "Close Guide",
                    "iso": "3D Isometric View Angle",
                    "dorsal": "Top (Dorsal) View Angle",
                    "ant": "Front (Anterior) View Angle",
                    "rot": "Toggle Auto 3D Rotation",
                    "reset": "Reset Camera Angle & Position",
                    "tool_camera": "Camera Orbit & Pan Tool [Default]",
                    "tool_opto": "Virtual Optogenetics (ChR2 Laser): Micro-stimulate neural hubs to evoke behavioral reflexes",
                    "tool_inspect": "Neuron & Synapse Inspector: Inspect connectome cell types and neurotransmitters",
                    "mode_valence": "Mental Map Mode: Valence (Reward / Punishment)",
                    "mode_plasticity": "Synaptic Plasticity Mode: |Δw| Weight Shifts",
                    "mode_anatomy": "Anatomy Mode: Biological Lobe Compartments",
                    "mode_activity": "Live Activity Mode: Neural Spikes & Sparks",
                    "layer_mb": "Mushroom Body: Kenyon cell somas, MBONs, and dopamine memory circuit [Toggle]",
                    "layer_cx": "Central Complex: Ellipsoid Body (EB) compass heading ring and PB bridge [Toggle]",
                    "layer_visual": "Visual Projections: Optic lobes to central brain (LC4/LPLC2 looming, LC10 steering) [Toggle]",
                    "layer_olfactory": "Antennal Lobes: Food & pheromone projection neurons (ALPN) to Calyx [Toggle]",
                    "layer_motor": "Descending Motor Neurons: Giant Fiber (escape), MDN (moonwalk), DNp09 (walk), DNa02 (steer) [Toggle]",
                    "layer_hulls": "Translucent Neuropil Volumetric Envelopes (AL, MB, CX, Optic Lobes) [Toggle]",
                    "layer_wireframe": "Drosophila Brain Wireframe Envelope [Toggle]",
                }
                found = btn_tooltips.get(bid, "")
                if found:
                    break

        # Check optogenetic targets if in opto tool mode or general hover
        if not found:
            opto_target = self._find_opto_target_at(pos, threshold_px=28.0)
            if opto_target is not None:
                found = f"⚡ ChR2 Target: {opto_target['name']} — Click to evoke {opto_target['action']}"

        # Check proximity to known lobe centers if not hovering over a button
        if not found:
            for name, center_3d in self.data.lobe_centers.items():
                sx, sy, _, _ = self._project_points(np.array([center_3d], dtype=np.float32))
                dist = math.hypot(pos.x() - sx[0], pos.y() - sy[0])
                if dist < 28:
                    titles = {
                        "calyx_left": "Left Calyx (Kenyon Cell Somas & Olfactory Input)",
                        "calyx_right": "Right Calyx (Kenyon Cell Somas & Olfactory Input)",
                        "gamma_left": "Left Gamma Lobe (Horizontal Lobe: Odor Memory)",
                        "gamma_right": "Right Gamma Lobe (Horizontal Lobe: Odor Memory)",
                        "alpha_left": "Left Alpha/Beta Lobe (Vertical Lobe: Avoidance / Approach)",
                        "alpha_right": "Right Alpha/Beta Lobe (Vertical Lobe: Avoidance / Approach)",
                        "pam_reward": "PAM Dopamine Cluster (Reward / Sugar / Food)",
                        "ppl1_punish": "PPL1 Dopamine Cluster (Punishment / Nociception / Wall Shock)",
                    }
                    found = titles.get(name, name)
                    break

        if found != self._hover_text:
            self._hover_text = found
            self.update()


class FloatingMiniBrainHUD(QWidget):
    """Floating on-top desktop picture-in-picture widget for live 3D connectome visualization."""
    def __init__(self, runner=None, data: Mushroom3DData | None = None, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(300, 270)

        self._drag_pos = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        # Glass container frame
        container = QFrame(self)
        container.setObjectName("hud_container")
        container.setStyleSheet("""
            QFrame#hud_container {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(18, 20, 28, 235), stop:1 rgba(10, 12, 16, 245));
                border: 1px solid rgba(56, 189, 248, 120);
                border-radius: 12px;
            }
        """)
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(6, 6, 6, 6)
        c_layout.setSpacing(4)

        # Header title and close button
        head_row = QHBoxLayout()
        head_row.setContentsMargins(6, 2, 4, 2)
        lbl_title = QLabel("🧠 MINI-BRAIN HUD")
        lbl_title.setStyleSheet("color: rgba(220, 230, 245, 220); font-family: 'Segoe UI'; font-weight: bold; font-size: 10px; background: transparent;")
        head_row.addWidget(lbl_title)
        head_row.addStretch(1)

        btn_close = QPushButton("×")
        btn_close.setFixedSize(18, 18)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet("""
            QPushButton {
                color: #8b8b96;
                background: transparent;
                border: none;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #ff4757;
            }
        """)
        btn_close.clicked.connect(self.close)
        head_row.addWidget(btn_close)
        c_layout.addLayout(head_row)

        # 3D View embedded in PIP mode
        self.view3d = MushroomBody3DView(runner=runner, data=data, pip_mode=True, parent=container)
        c_layout.addWidget(self.view3d, 1)

        layout.addWidget(container)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
