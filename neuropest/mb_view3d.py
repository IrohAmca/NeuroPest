"""FlyWire v783 3D Mushroom Body (Mantar Gövdesi) Interactive Visualization.

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
from PySide6.QtWidgets import QToolTip, QWidget

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
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.runner = runner
        self.data = data or load_mushroom_3d()

        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)

        # 3D Camera attributes
        self.yaw, self.pitch = PRESET_ISOMETRIC
        self.zoom = 220.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.auto_rotate = False
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

        # Animation state (traveling pulses, dopamine glow ripple)
        self._pulse_phase = 0.0
        self._last_tick = time.perf_counter()

        # Render timer (adaptive: 33ms active, 200ms idle)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)

        # Interactive UI buttons on canvas (rectangles in screen coordinates)
        self._btn_rects: dict[str, QRect] = {}

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

        # Check if animation or telemetry is active
        is_active = (
            self.auto_rotate
            or self._is_panning
            or self.pam > 0.02
            or self.ppl1 > 0.02
            or self.kc_active_count > 0
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

        if (
            abs(val - self.valence) > 0.01
            or abs(pam - self.pam) > 0.02
            or abs(ppl1 - self.ppl1) > 0.02
            or abs(mbon_app - self.mbon_app) > 0.02
            or abs(mbon_av - self.mbon_av) > 0.02
            or kc_act != self.kc_active_count
        ):
            dirty = True

        self.valence = val
        self.pam = pam
        self.ppl1 = ppl1
        self.mbon_app = mbon_app
        self.mbon_av = mbon_av
        prev_kc = self.kc_active_count
        self.kc_active_count = kc_act

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
        self._draw_brain_wireframe(p)

        # 3. Draw Axon Streamlines (Calyx -> Pedunculus -> Lobes)
        self._draw_axon_tracts(p)

        # 4. Draw Kenyon Cell Somas (Calyx Cups)
        self._draw_kenyon_somas(p)

        # 5. Draw Dopaminergic Neurons (PAM & PPL1) & Burst Ripples
        self._draw_dopamine_clusters(p)

        # 6. Draw MBON Output Terminals
        self._draw_mbons(p)

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

    # ------------------------------------------------------------------ color logic
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

    # ------------------------------------------------------------------ HUD
    def _draw_hud(self, p: QPainter):
        """Draw titles, control buttons, and live telemetry overlays."""
        w, h = self.width(), self.height()
        font_title = QFont("Segoe UI", 10, QFont.Bold)
        font_small = QFont("Segoe UI", 8, QFont.Normal)
        font_mono = QFont("Cascadia Mono", 9, QFont.DemiBold)

        # 1. Top-Left: Header
        p.setFont(font_title)
        p.setPen(QColor(236, 236, 241, 230))
        p.drawText(16, 26, "3D MANTAR GÖVDESİ & ZİHİN HARİTASI")

        p.setFont(font_small)
        p.setPen(QColor(139, 139, 150, 200))
        p.drawText(16, 42, f"FlyWire v783 • {self.data.n_kc:,} Kenyon Hücresi | {self.data.n_mbon} MBON | {self.data.n_dan} DAN")

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

    def _draw_toolbar(self, p: QPainter, w: int):
        """Draw interactive buttons on top-right corner."""
        font_btn = QFont("Segoe UI", 8, QFont.DemiBold)
        p.setFont(font_btn)

        if w < 640:
            buttons = [
                ("iso", "İzo"),
                ("dorsal", "Üst"),
                ("ant", "Ön"),
                ("rot", "Döndür" if not self.auto_rotate else "Dur"),
                ("reset", "Sıfırla"),
                ("help", "?" if not self._show_help else "? Açık"),
            ]
            mode_buttons = [
                (MODE_VALENCE, "Zihin"),
                (MODE_PLASTICITY, "Plastisite"),
                (MODE_ANATOMY, "Anatomi"),
                (MODE_ACTIVITY, "Aktivite"),
            ]
        else:
            buttons = [
                ("iso", "3D İzometrik"),
                ("dorsal", "Üst (Dorsal)"),
                ("ant", "Ön (Anterior)"),
                ("rot", "Otomatik Dönüş" if not self.auto_rotate else "Dönüşü Durdur"),
                ("reset", "Sıfırla"),
                ("help", "?" if not self._show_help else "? Açık"),
            ]
            mode_buttons = [
                (MODE_VALENCE, "Zihin Haritası"),
                (MODE_PLASTICITY, "Plastisite |Δw|"),
                (MODE_ANATOMY, "Anatomi"),
                (MODE_ACTIVITY, "Canlı Aktivite"),
            ]

        bx = w - 16
        by = 14
        btn_h = 24
        spacing = 6

        self._btn_rects.clear()

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
        p.drawText(vx, vy, "ÖĞRENİLMİŞ DEĞERLİK (ZİHİN)")

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

        # 2. Dopamine Bars: PAM (Ödül) & PPL1 (Ceza)
        d_x = vx + v_width + 75
        p.setFont(font_small)
        p.setPen(QColor(139, 139, 150, 220))
        p.drawText(d_x, vy, "DOPAMİN (PAM ÖDÜL / PPL1 CEZA)")

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
            p.drawText(k_x, vy, "AKTİF KC (SEYREK KOD)")

            p.setFont(font_mono)
            p.setPen(QColor(236, 236, 241))
            sparsity = (self.kc_active_count / max(1, self.data.n_kc)) * 100
            p.drawText(k_x, my + 9, f"{self.kc_active_count:,} / {self.data.n_kc:,} (%{sparsity:.1f})")

        # 4. MBON Balance
        m_x = k_x + 160
        if m_x < bar_rect.right() - 110:
            p.setFont(font_small)
            p.setPen(QColor(139, 139, 150, 220))
            p.drawText(m_x, vy, "MBON DENGESİ")

            p.setFont(font_mono)
            p.setPen(QColor(56, 189, 248))
            p.drawText(m_x, my + 9, f"+{self.mbon_app:.2f} / -{self.mbon_av:.2f}")

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
        p.drawText(card_x + 16, card_y + 24, "MANTAR GÖVDESİ & ZİHİN HARİTASI REHBERİ")

        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(140, 145, 160))
        p.drawText(card_x + 16, card_y + 40, "FlyWire v783 tam konnektom 3D nöral devre ve pekiştirmeli öğrenme rehberi")

        # Close button in top-right of card
        close_w, close_h = 56, 20
        close_rect = QRect(card_x + card_w - close_w - 14, card_y + 14, close_w, close_h)
        self._btn_rects["close_help"] = close_rect

        p.setBrush(QBrush(QColor(36, 36, 44, 230)))
        p.setPen(QPen(QColor(70, 70, 82, 200), 1))
        p.drawRoundedRect(close_rect, 4, 4)
        p.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
        p.setPen(QColor(220, 220, 230))
        p.drawText(close_rect, Qt.AlignCenter, "Kapat")

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
                "1. ZİHİN HARİTASI & RENKLER",
                [
                    (QColor(16, 185, 129), "Yeşil / Altın (Ödül)", "PAM pekiştirmesi -> Yaklaşma arzusu (Appetitive MBON)."),
                    (QColor(239, 68, 68), "Kırmızı / Yakut (Ceza)", "PPL1 baskılaması -> Kaçınma korkusu (Aversive MBON)."),
                    (QColor(56, 189, 248), "Buz Mavisi (Naive)", "Deneyimlenmemiş nötr koku sinapsları."),
                    (QColor(255, 255, 255), "Beyaz Parlama (Aktivite)", "Anlık ateşlenen Kenyon hücreleri ve akson iletimi."),
                ],
            ),
            (
                QRect(panel2_x, top_y, panel_w, panel_h),
                "2. BİYOLOJİK BÖLGELER (ANATOMİ)",
                [
                    (QColor(76, 201, 240), "Kaliks (Calyx)", "Arka çanak; 5000+ KC soması koku girdisi toplar."),
                    (QColor(168, 85, 247), "Pedunkulus", "Kenyon aksonlarının loblara uzanan ana demeti."),
                    (QColor(245, 158, 11), "Alfa / Beta Lobları", "Uzun süreli bellek ve yaklaşma/kaçınma ayrımı."),
                    (QColor(6, 182, 212), "Gama Lobu", "Kısa süreli koku belleği ve hızlı plastisite alanı."),
                ],
            ),
            (
                QRect(panel1_x, bot_y, panel_w, panel_h),
                "3. DOPAMİN & CANLI TELEMETRİ",
                [
                    (QColor(245, 158, 11), "PAM Dopamin Kümesi", "Şeker ödülünde altın dalgalar yayarak ödüllendirir."),
                    (QColor(239, 68, 68), "PPL1 Dopamin Kümesi", "Acı/şok anında kırmızı dalgalar ile aversif öğretir."),
                    (QColor(236, 236, 241), "Değerlik [-1.0 ... +1.0]", "-1.0 tam kaçınma, +1.0 tam yaklaşma ibresi."),
                    (QColor(56, 189, 248), "MBON & Aktif KC", "Net çıkış oranı ve kokuyu tanıyan anlık hücre sayısı."),
                ],
            ),
            (
                QRect(panel2_x, bot_y, panel_w, panel_h),
                "4. FARE & KAMERA KONTROLLERİ",
                [
                    (QColor(140, 145, 160), "Sol Tık + Sürükle", "3D serbest döndürme (Orbit)."),
                    (QColor(140, 145, 160), "Sağ Tık / Shift", "Kamerayı iki boyutta kaydırma (Pan)."),
                    (QColor(140, 145, 160), "Fare Tekerleği", "Kamerayı yakınlaştır / uzaklaştır (Zoom)."),
                    (QColor(140, 145, 160), "Çift Tık / Sıfırla", "Varsayılan izometrik açıya geri dönüş."),
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
                return

        # If help overlay is open: dismiss if clicked outside card, ignore inside clicks
        if self._show_help:
            if not self._help_card_rect.contains(pos):
                self._show_help = False
                self.update()
            return

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
                    "help": "Zihin Haritası ve Biyolojik Rehber [Aç/Kapat]",
                    "close_help": "Rehberi Kapat",
                    "iso": "3D İzometrik Görünüm Açısı",
                    "dorsal": "Üst (Dorsal) Görünüm Açısı",
                    "ant": "Ön (Anterior) Görünüm Açısı",
                    "rot": "Otomatik 3D Döndürmeyi Başlat/Durdur",
                    "reset": "Kamera Açısını ve Konumunu Sıfırla",
                    "mode_valence": "Zihin Haritası Modu: Değerlik (Ödül / Ceza)",
                    "mode_plasticity": "Sinaptik Plastisite Modu: |Δw| Değişimi",
                    "mode_anatomy": "Anatomi Modu: Biyolojik Lob Kompartmanları",
                    "mode_activity": "Canlı Aktivite Modu: Nöral Ateşleme & Kıvılcımlar",
                }
                found = btn_tooltips.get(bid, "")
                if found:
                    break

        # Check proximity to known lobe centers if not hovering over a button
        if not found:
            for name, center_3d in self.data.lobe_centers.items():
                sx, sy, _, _ = self._project_points(np.array([center_3d], dtype=np.float32))
                dist = math.hypot(pos.x() - sx[0], pos.y() - sy[0])
                if dist < 28:
                    titles = {
                        "calyx_left": "Sol Kaliks (Kenyon Hücresi Soması ve Koku Girişi)",
                        "calyx_right": "Sağ Kaliks (Kenyon Hücresi Soması ve Koku Girişi)",
                        "gamma_left": "Sol Gamma Lobu (Yatay Lob: Koku Belleği)",
                        "gamma_right": "Sağ Gamma Lobu (Yatay Lob: Koku Belleği)",
                        "alpha_left": "Sol Alfa/Beta Lobu (Dikey Lob: Kaçınma / Yaklaşma)",
                        "alpha_right": "Sağ Alfa/Beta Lobu (Dikey Lob: Kaçınma / Yaklaşma)",
                        "pam_reward": "PAM Dopamin Kümesi (Ödül / Şeker / Nektar)",
                        "ppl1_punish": "PPL1 Dopamin Kümesi (Ceza / Nosisepsiyon / Duvar Şoku)",
                    }
                    found = titles.get(name, name)
                    break

        if found != self._hover_text:
            self._hover_text = found
            self.update()
