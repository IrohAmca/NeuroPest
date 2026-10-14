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
from PySide6.QtCore import QLineF, QPoint, QPointF, QRect, QSize, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import QMenu, QWidget

from .fly import TURN_GAIN
from .mb_geometry import (
    DAN_PAM,
    DAN_PPL1,
    LOBE_ALPHABETA,
    LOBE_ALPHAP_BETAP,
    LOBE_GAMMA,
    Mushroom3DData,
    load_mushroom_3d,
)

from .states import LAB_FULL_HZ

if TYPE_CHECKING:
    from .runner import Runner

# The lab's targets (Runner.stimulate / Runner.silence act on these spiking groups of the circuit): label, and what the
# circuit is known to do when the group is driven (the effect itself comes from the simulation, not from this table)
LAB_TARGETS = {
    "GF": ("Giant Fiber (DNp01)", "escape take-off"),
    "MDN": ("Moonwalker DNs (MDN)", "backward walking"),
    "WALK": ("Walking DNs (DNp09)", "forward walking"),
    "DNa02_L": ("DNa02, left", "a turn to the left"),
    "DNa02_R": ("DNa02, right", "a turn to the right"),
    "GROOM": ("Grooming DNs (aDN1/2)", "grooming"),
    "LC4": ("LC4 looming cells", "Giant Fiber, take-off"),
    "LPLC2": ("LPLC2 looming cells", "Giant Fiber, take-off"),
    "LPC1": ("LPC1 retreat input", "MDN, backing away"),
    "LC10_L": ("LC10, left", "DNa02 left, a turn"),
    "LC10_R": ("LC10, right", "DNa02 right, a turn"),
}
# Designed signals of the learning sandbox, not neurons of the spiking circuit: they teach the mushroom body
SANDBOX_TARGETS = {"reward": "PAM reward signal (designed)", "punish": "PPL1 punishment signal (designed)",
                   "food": "food odor cue (designed)"}
TOOLS = (("camera", "Explore"), ("opto", "Stimulate"), ("silence", "Silence"))
LAB_PULSE_S = 0.3           # a click's light pulse; holding the button keeps the light on (the worker ends it at 10 s)

# Visualization Modes
MODE_VALENCE = "valence"        # Learned mental map (Valence: Red=Aversive, Green=Appetitive, Slate=Naive)
MODE_PLASTICITY = "plasticity"  # Synaptic depression magnitude |Δw|
MODE_ANATOMY = "anatomy"        # FlyWire biological lobe compartments & cell classes
MODE_ACTIVITY = "activity"      # Live neural spikes & dopamine bursts

# Drawing cost in Qt's raster engine (measured offscreen, dpr 1-1.5): a small antialiased dot costs 7-10 us as its own
# drawEllipse, ~20 us inside one path that holds them all, and ~1 us as a pre-rendered sprite blitted at a whole device
# pixel; a curve stroked thicker than 1 px leaves the fast cosmetic stroker and costs ~10x more. So dots are sprites and
# curves use 1 px pens, many per path.
_SPRITES: dict[tuple, tuple[QPixmap, int]] = {}


def _dot_sprite(rgba: int, pen_rgba: int, pen_w: float, r: float, fx: float, fy: float) -> tuple[QPixmap, int]:
    """An antialiased dot of radius r (device px) centred at (pad + fx, pad + fy) of a transparent pixmap, and pad."""
    key = (rgba, pen_rgba, pen_w, r, fx, fy)
    hit = _SPRITES.get(key)
    if hit is None:
        if len(_SPRITES) > 4096:                  # colours follow the learned weights: keep the cache bounded
            _SPRITES.clear()
        pad = math.ceil(r + pen_w + 1)
        pm = QPixmap(2 * pad + 1, 2 * pad + 1)
        pm.fill(Qt.transparent)
        sp = QPainter(pm)
        sp.setRenderHint(QPainter.Antialiasing)
        sp.setBrush(QColor.fromRgba(rgba) if rgba else Qt.NoBrush)
        sp.setPen(QPen(QColor.fromRgba(pen_rgba), pen_w) if pen_w > 0 else Qt.NoPen)
        sp.drawEllipse(QPointF(pad + fx, pad + fy), r, r)
        sp.end()
        hit = _SPRITES[key] = (pm, pad)
    return hit


def _draw_dots(p: QPainter, xs, ys, rs) -> None:
    """Circles in the painter's current brush (and pen, if any) at (xs, ys) with radii rs, in logical coordinates of a
    painter without a world transform. Blitted from sprites at whole device pixels: centres snap to half a device pixel
    and radii to a quarter, which the eye does not see at these sizes."""
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        return
    dev = p.device()
    dpr = dev.devicePixelRatioF() if dev is not None else 1.0
    brush, pen = p.brush(), p.pen()
    rgba = brush.color().rgba() if brush.style() != Qt.NoBrush else 0
    pen_w = round(pen.widthF() * dpr * 4) / 4 if pen.style() != Qt.NoPen else 0.0
    pen_rgba = pen.color().rgba() if pen_w > 0 else 0
    if not rgba and not pen_rgba:
        return
    X = np.round(xs * dpr * 2) / 2
    Y = np.round(np.asarray(ys, dtype=np.float64) * dpr * 2) / 2
    ix, iy = np.floor(X), np.floor(Y)
    R = np.round(np.broadcast_to(np.asarray(rs, dtype=np.float64), xs.shape) * dpr * 4) / 4
    p.save()
    p.setWorldTransform(QTransform.fromScale(1.0 / dpr, 1.0 / dpr))     # device pixels: the blit stays 1:1
    for x, y, fx, fy, r in zip(ix.tolist(), iy.tolist(), (X - ix).tolist(), (Y - iy).tolist(), R.tolist()):
        if r >= 0.25:
            pm, pad = _dot_sprite(rgba, pen_rgba, pen_w, r, fx, fy)
            p.drawPixmap(int(x) - pad, int(y) - pad, pm)
    p.restore()


def _curve_path(x0, y0, cx, cy, x1, y1) -> QPainterPath:
    """One path with a quadratic curve (x0, y0) -> (x1, y1) through the control point (cx, cy) per element."""
    path = QPainterPath()
    cols = [np.asarray(a, dtype=np.float64).tolist() for a in (x0, y0, cx, cy, x1, y1)]
    for a, b, c, d, e, f in zip(*cols):
        path.moveTo(a, b)
        path.quadTo(c, d, e, f)
    return path


def _rgba_groups(rgba: np.ndarray) -> list[tuple[QColor, np.ndarray]]:
    """(colour, indices) for every distinct packed 0xAARRGGBB value in rgba."""
    if len(rgba) == 0:
        return []
    uniq, inv = np.unique(rgba, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.cumsum(np.bincount(inv, minlength=len(uniq)))[:-1]
    return [(QColor.fromRgba(int(c)), idx) for c, idx in zip(uniq.tolist(), np.split(order, bounds))]


def _pack_rgba(r, g, b, a) -> np.ndarray:
    """Pack float channel arrays (truncated like int()) into 0xAARRGGBB."""
    def ch(x):
        return np.clip(np.asarray(x, dtype=np.float64), 0, 255).astype(np.uint32)
    return (ch(a) << 24) | (ch(r) << 16) | (ch(g) << 8) | ch(b)


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
        self._cam_R_for: tuple[float, float] | None = None   # the (yaw, pitch) _cam_R was computed for

        # Render caches: the scene that only moves with the camera is drawn once into a pixmap (the base layer) and the
        # screen positions of the fixed point sets are kept per camera pose (the hit-test and the live layer reuse them)
        self._base_cache: QPixmap | None = None
        self._base_cache_key: tuple | None = None
        self._proj_memo: dict[tuple, tuple] = {}
        self._proj_key: tuple | None = None


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
        self.heading_angle = 0.0    # steering integral: the compass only falls back to it without a runner heading
        self._fly_heading = None    # the fly's real on-screen heading (rad) from the runner, None until it reports one
        self.fly_state = ""         # the behavior state the runner reports (the status line shows what a pulse evoked)

        # Memory buffer (per KC valence delta)
        self.kc_weights = np.zeros(self.data.n_kc, dtype=np.float32)
        self.active_kc_mask = np.zeros(self.data.n_kc, dtype=bool)
        self._kc_glow = np.zeros(self.data.n_kc, dtype=np.float32)   # fades after a Kenyon cell fires (a moment of persistence)
        self._kc_from_model = False     # the mask is the worker's real firing set (else an illustrative stand-in)
        self._kc_fired_now = np.zeros(self.data.n_kc, dtype=bool)   # the set of the last published step (no fade)
        self._kc_model_n = 0            # Kenyon cells of the running model (2,000 designed, 5,177 real); 0 = unknown

        # Pre-sliced geometry cache for fast rendering
        n_kc = self.data.n_kc
        # Somas: sample step=3 (1,725 somas for 5,177 KCs, fast and visually dense)
        self._soma_step = 3 if n_kc > 3000 else 1
        self._soma_indices = np.arange(0, n_kc, self._soma_step)
        self._soma_pts = self.data.kc_soma[self._soma_indices]
        self._soma_lobes = self.data.kc_lobe[self._soma_indices]

        # Axon tracts: sample ~160 representative streamlines
        tract_step = self._tract_step = max(1, n_kc // 160)
        self._tract_indices = np.arange(0, n_kc, tract_step)
        self._tract_soma = self.data.kc_soma[self._tract_indices]
        self._tract_ped = self.data.kc_ped[self._tract_indices]
        self._tract_target = self.data.kc_target[self._tract_indices]
        self._tract_lobes = self.data.kc_lobe[self._tract_indices]
        # lit cells that the stride samples above would miss: drawn on top, so every firing cell shows
        self._active_off_soma = np.zeros(0, dtype=np.int64)
        self._active_off_tract = np.zeros(0, dtype=np.int64)

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
        self._eb_ring = self._eb_ring_order()

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

        # Render timer (adaptive, see _on_tick: 33 ms while something moves, slower when only the telemetry changes or
        # a frame is expensive, 200 ms idle)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._paint_ms = 0.0                # smoothed cost of a frame

        # Interactive UI buttons on canvas (rectangles in screen coordinates)
        self._btn_rects: dict[str, QRect] = {}
        self._fonts_cache: dict[str, QFont] | None = None

        # Tools: "camera" (Explore: orbit, hover, click to inspect), "opto" (Stimulate: press a target, the light stays on
        # while the button is held), "silence" (click a target to silence or release it)
        self.tool_mode = "camera"
        self._press: tuple[QPoint, object, float] | None = None     # where / which button / when the mouse went down
        self._dragging = False
        self._held: tuple[str, float] | None = None   # the lab group lit by a held button, and since when
        self._hover_pos: QPoint | None = None
        self._hover_target: str | None = None         # lab group or sandbox action under the cursor
        self._hover_card: dict | None = None
        self._hover_scheduled = False
        self._ripples: list[tuple[str, float, QColor]] = []   # (group, start, colour): one ring per action
        self._status = ("", 0.0, False)               # status line: text, when, warning
        self._pending: tuple[str, str, float, float] | None = None   # (group, action, when, fly state then)
        self._lab = dict(available=(), silenced=(), stimulated=(), left_ms=0.0, gate=False, acked=True,
                         asked_silenced=())
        self._lab_rates: dict[str, float] = {}
        self._lab_members = self._build_lab_members()

        # Neuron & Connectome Inspector
        self._inspected_neuron: dict | None = None
        self._inspected_pinned: bool = False

        # Translucent Neuropil Volumetric Envelopes
        self._neuropil_volumes: list[dict] = []
        self._build_neuropil_volumes()

    def _build_neuropil_volumes(self):
        """Ellipsoid envelopes for the key neuropils. Each keeps its centre and semi-axes, which is all the renderer
        needs (an ellipsoid's outline on screen is an ellipse), and a coarse mesh for anything that wants the surface."""
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

        def add(vid, name, color, center, rx, ry, rz):
            center = np.asarray(center, dtype=np.float64)
            v, f = build_ellipsoid(center, rx, ry, rz)
            self._neuropil_volumes.append({"id": vid, "name": name, "color": color, "vertices": v, "faces": f,
                                           "center": center, "radii": np.array([rx, ry, rz], dtype=np.float64)})

        def side_mean(pts, side, which, fallback):
            sel = pts[side == which] if len(side) == len(pts) else pts[:0]
            return np.mean(sel, axis=0) if len(sel) else np.array(fallback, dtype=np.float32)

        # 1. Antennal Lobes (AL-L & AL-R)
        add("al_l", "Left Antennal Lobe", QColor(16, 185, 129),
            side_mean(self._alpn_soma, self._alpn_side, -1, (-0.20, -0.15, -0.28)), 0.11, 0.11, 0.11)
        add("al_r", "Right Antennal Lobe", QColor(16, 185, 129),
            side_mean(self._alpn_soma, self._alpn_side, 1, (0.20, -0.15, -0.28)), 0.11, 0.11, 0.11)

        # 2. Mushroom Body Calyces (MB-CA-L & MB-CA-R)
        add("ca_l", "Left MB Calyx", QColor(245, 158, 11),
            self.data.lobe_centers.get("calyx_left", (-0.25, -0.10, 0.28)), 0.13, 0.12, 0.11)
        add("ca_r", "Right MB Calyx", QColor(245, 158, 11),
            self.data.lobe_centers.get("calyx_right", (0.25, -0.10, 0.28)), 0.13, 0.12, 0.11)

        # 3. Mushroom Body Vertical & Medial Lobes
        add("mb_lobe_l", "Left MB Lobes", QColor(6, 182, 212),
            self.data.lobe_centers.get("gamma_left", (-0.10, 0.12, 0.05)), 0.11, 0.14, 0.10)
        add("mb_lobe_r", "Right MB Lobes", QColor(6, 182, 212),
            self.data.lobe_centers.get("gamma_right", (0.10, 0.12, 0.05)), 0.11, 0.14, 0.10)

        # 4. Central Complex: Ellipsoid Body (EB)
        cx_eb = self._cx_pts[self._cx_sub == 0] if len(self._cx_sub) == len(self._cx_pts) else self._cx_pts
        eb_center = np.mean(cx_eb, axis=0) if len(cx_eb) > 0 else np.array([0.0, 0.05, 0.15], dtype=np.float32)
        add("eb", "Ellipsoid Body", QColor(56, 189, 248), eb_center, 0.10, 0.05, 0.10)

        # 5. Optic Lobes (Left & Right)
        add("ol_l", "Left Optic Lobe", QColor(168, 85, 247),
            side_mean(self._vpn_soma, self._vpn_side, -1, (-0.52, 0.05, 0.0)), 0.16, 0.22, 0.17)
        add("ol_r", "Right Optic Lobe", QColor(168, 85, 247),
            side_mean(self._vpn_soma, self._vpn_side, 1, (0.52, 0.05, 0.0)), 0.16, 0.22, 0.17)

    def set_tool_mode(self, mode: str):
        """Canvas tool: "camera" (Explore), "opto" (Stimulate), "silence" ("inspect" is Explore: a click pins the card)."""
        mode = "camera" if mode == "inspect" else mode
        if mode in ("camera", "opto", "silence"):
            if self._held is not None:
                self._release_held()
            self.tool_mode = mode
            self.setCursor(Qt.ArrowCursor)
            self._hover_target = None
            self._schedule_hover()
            self.update()

    # ------------------------------------------------------------------ lab targets
    def _build_lab_members(self) -> dict[str, tuple[str, np.ndarray]]:
        """Which drawn neurons belong to each lab group: (system "dn" or "vpn", indices into that system's arrays)."""
        dn_t = np.array([str(t) for t in self._dn_type], dtype=object)
        dn_s = np.asarray(self._dn_side) if len(self._dn_side) == len(dn_t) else np.zeros(len(dn_t), np.int8)
        vp_t = np.array([str(t) for t in self._vpn_type], dtype=object)
        vp_s = np.asarray(self._vpn_side) if len(self._vpn_side) == len(vp_t) else np.zeros(len(vp_t), np.int8)
        lc10 = np.array([t.startswith("LC10") for t in vp_t], dtype=bool)
        members = {
            "GF": ("dn", dn_t == "GF"), "MDN": ("dn", dn_t == "MDN"),
            "WALK": ("dn", (dn_t == "WALK") | (dn_t == "DNp09")), "GROOM": ("dn", dn_t == "GROOM"),
            "DNa02_L": ("dn", (dn_t == "DNa02") & (dn_s == -1)), "DNa02_R": ("dn", (dn_t == "DNa02") & (dn_s == 1)),
            "LC4": ("vpn", vp_t == "LC4"), "LPLC2": ("vpn", vp_t == "LPLC2"), "LPC1": ("vpn", vp_t == "LPC1"),
            "LC10_L": ("vpn", lc10 & (vp_s == -1)), "LC10_R": ("vpn", lc10 & (vp_s == 1)),
        }
        return {g: (sys_, np.flatnonzero(m)) for g, (sys_, m) in members.items() if np.any(m)}

    def _group_of(self, system: str, i: int) -> str | None:
        for group, (sys_, idx) in self._lab_members.items():
            if sys_ == system and i in idx:
                return group
        return None

    def _members_xy(self, group: str):
        """Screen (x, y, persp) of a lab group's drawn neurons."""
        system, idx = self._lab_members[group]
        pts = self._dn_soma if system == "dn" else self._vpn_soma
        sx, sy, _, persp = self._proj(pts)
        return sx[idx], sy[idx], persp[idx]

    def _target_at(self, pos: QPoint, threshold_px: float = 18.0) -> tuple[str | None, dict | None]:
        """(lab group or sandbox action, card) of the neuron under the cursor among the ones a lab tool acts on."""
        hit = self._pick(pos, threshold_px, systems=("dn", "vpn", "dan", "alpn"))
        if hit is None:
            return None, None
        card = self._card(*hit)
        return card.get("lab") or card.get("sandbox"), card

    @staticmethod
    def target_label(target: str) -> str:
        if target in LAB_TARGETS:
            return LAB_TARGETS[target][0]
        return SANDBOX_TARGETS.get(target, target)

    def _lab_call(self, name: str):
        r = self.runner
        fn = getattr(r, name, None) if r is not None else None
        return fn if fn is not None and getattr(r, "ready", False) else None

    def stimulate_target(self, target: str, hold: bool = False) -> bool:
        """Light on a lab group (a LAB_PULSE_S pulse, or until _release_held with hold), or a sandbox signal. True if
        something was sent; nothing is ever sent for an unknown target."""
        if target in SANDBOX_TARGETS:
            return self._sandbox(target)
        if target not in LAB_TARGETS:
            return False
        label, effect = LAB_TARGETS[target]
        if self._lab["available"] and target not in self._lab["available"]:
            self._set_status(f"{label}: not in the running circuit", warn=True)
            return False
        if target in self._lab["silenced"]:
            self._set_status(f"{label} is silenced: release it first (light cannot make it spike)", warn=True)
            return False
        fn = self._lab_call("stimulate")
        if fn is None or not fn(target, 1.0, 10_000.0 if hold else LAB_PULSE_S * 1000.0):
            self._set_status(f"{label}: not sent, the engine is not running", warn=True)
            return False
        hz = LAB_FULL_HZ.get(target, 0.0)
        self._set_status(f"Light on {label}, {hz:.0f} Hz{' while held' if hold else f', {LAB_PULSE_S * 1000:.0f} ms'}"
                         f"  (known effect: {effect})")
        self._pending = (target, "stim", time.perf_counter(), 0.0)
        self._pending_state = self.fly_state
        self._ripples.append((target, time.perf_counter(), QColor(255, 120, 60)))
        self._kick_timer()
        return True

    def _release_held(self) -> None:
        """The button that holds the light came up: stop it, but never before a full pulse."""
        group, t0 = self._held
        self._held = None
        left = LAB_PULSE_S - (time.perf_counter() - t0)
        fn = self._lab_call("stop_stimulation")
        if fn is None:
            return
        if left <= 0.0:
            fn()
        else:
            QTimer.singleShot(int(left * 1000), lambda: self._held is None and fn())

    def toggle_silence(self, target: str) -> bool:
        if target not in LAB_TARGETS:
            if target in SANDBOX_TARGETS:
                self._set_status(f"{SANDBOX_TARGETS[target]} is a designed signal, not neurons: nothing to silence",
                                 warn=True)
            return False
        label = LAB_TARGETS[target][0]
        if self._lab["available"] and target not in self._lab["available"]:
            self._set_status(f"{label}: not in the running circuit", warn=True)
            return False
        fn = self._lab_call("silence")
        if fn is None:
            self._set_status(f"{label}: not sent, the engine is not running", warn=True)
            return False
        on = target not in self._asked_silenced()
        fn(target, on)
        self._set_status(f"{label} silenced: its neurons cannot spike (learning paused while the lab is on)" if on
                         else f"{label} released")
        self._pending = (target, "silence" if on else "release", time.perf_counter(), 0.0)
        self._ripples.append((target, time.perf_counter(), QColor(148, 163, 184) if on else QColor(56, 189, 248)))
        self._base_cache_key = None
        self._kick_timer()
        return True

    def release_all(self) -> None:
        fn = self._lab_call("clear_lab")
        if fn is not None:
            fn()
            self._set_status("Every silenced group released")
            self._pending = ("", "release_all", time.perf_counter(), 0.0)

    def _asked_silenced(self) -> tuple[str, ...]:
        return tuple(self._lab.get("asked_silenced", ())) or tuple(self._lab.get("silenced", ()))

    def _sandbox(self, action: str) -> bool:
        names = {"reward": "inject_reward", "punish": "inject_punish", "food": "inject_cue_food"}
        fn = self._lab_call(names[action])
        label = SANDBOX_TARGETS[action]
        if fn is None:
            self._set_status(f"{label}: not sent, the engine is not running", warn=True)
            return False
        fn(1.0)
        note = " (learning is paused while the lab is on)" if self._lab.get("gate") else ""
        what = {"reward": "the cues present now are learned as good", "punish": "the cues present now are learned as bad",
                "food": "the mushroom body sees the food odor for 0.8 s"}[action]
        self._set_status(f"{label}: {what}{note}")
        self._ripples.append((action, time.perf_counter(), QColor(250, 204, 21) if action != "punish" else QColor(239, 68, 68)))
        self._kick_timer()
        return True

    def _target_xy(self, target: str) -> tuple[float, float] | None:
        """Screen point a ripple / marker of a target is drawn at (its drawn neurons' centre)."""
        if target in self._lab_members:
            x, y, _ = self._members_xy(target)
            return float(np.mean(x)), float(np.mean(y))
        if target in ("reward", "punish"):
            key = "pam_reward" if target == "reward" else "ppl1_punish"
            if key in self.data.lobe_centers:
                sx, sy, _, _ = self._project_points(np.array([self.data.lobe_centers[key]], dtype=np.float32))
                return float(sx[0]), float(sy[0])
        if target == "food" and len(self._alpn_soma):
            sx, sy, _, _ = self._proj(self._alpn_soma)
            return float(np.mean(sx)), float(np.mean(sy))
        return None

    def _set_status(self, text: str, warn: bool = False) -> None:
        self._status = (text, time.perf_counter(), warn)
        self.update()

    def _kick_timer(self) -> None:
        if self._timer.interval() != 33:
            self._timer.setInterval(33)
        self.update()

    # ------------------------------------------------------------------ compass
    def _eb_ring_order(self) -> np.ndarray:
        """Indices (into the CX arrays) of the Ellipsoid Body nodes in ring order.

        The table lists them by cell type, not by position, so drawing them in table order gives a zig-zag and a bump
        that jumps between unrelated places. Sort them by their angle around the centroid in the plane of best fit."""
        pts, subs = self._cx_pts, self._cx_sub
        if len(subs) != len(pts):
            return np.zeros(0, dtype=np.int64)
        idx = np.flatnonzero(subs == 0)
        if len(idx) < 3:
            return idx
        p = np.asarray(pts[idx], dtype=np.float64)
        p = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(p, full_matrices=False)
        plane = p @ vt[:2].T
        return idx[np.argsort(np.arctan2(plane[:, 1], plane[:, 0]), kind="stable")]

    def _compass_phase(self) -> float:
        """Angle (rad) the E-PG bump sits at: the fly's real on-screen heading from the runner. Without a runner heading
        (standalone view, test doubles) it is the steering integral."""
        base = self._fly_heading if self._fly_heading is not None else self.heading_angle
        return base % (2 * math.pi)

    def _eb_lit(self, n: int) -> np.ndarray:
        """Which of the n ring positions the bump covers (within 0.45 rad of the compass phase)."""
        if n <= 0:
            return np.zeros(0, dtype=bool)
        ang = np.arange(n) * (2 * math.pi / n)
        return np.abs((ang - self._compass_phase() + math.pi) % (2 * math.pi) - math.pi) < 0.45

    def _cx_rate(self, i: int) -> float:
        """Illustrative rate of central-complex node i for the inspector (the central complex is not simulated as
        spiking cells here): an EB node in the E-PG bump (drawn white) 16 Hz, the rest of the ring 4 Hz; PB / FB nodes
        follow the compass phase."""
        if i < len(self._cx_sub) and self._cx_sub[i] == 0:
            ring_pos = np.flatnonzero(self._eb_ring == i)
            if len(ring_pos):
                return 16.0 if self._eb_lit(len(self._eb_ring))[ring_pos[0]] else 4.0
        return 10.0 + 6.0 * math.cos(self._compass_phase())

    # ------------------------------------------------------------------ picking and inspector cards
    def _pick_sources(self):
        """(system, points, visible) in the order picking breaks ties (the first wins)."""
        return (("dn", self._dn_soma, self.layer_motor), ("dan", self.data.dan_pos, self.layer_mb),
                ("mbon", self.data.mbon_pos, self.layer_mb), ("cx", self._cx_pts, self.layer_cx),
                ("vpn", self._vpn_soma, self.layer_visual), ("alpn", self._alpn_soma, self.layer_olfactory),
                ("kc", self._soma_pts, self.layer_mb))

    def _pick(self, pos: QPoint, threshold_px: float = 24.0, systems=None) -> tuple[str, int] | None:
        """(system, index) of the drawn neuron nearest to pos within threshold_px: one vectorised distance per system
        on the screen positions kept for this camera pose (~0.1 ms; the old per-point loop cost ~3 ms a mouse move)."""
        x, y = float(pos.x()), float(pos.y())
        thr2 = threshold_px * threshold_px
        best = None

        def consider(system, sx, sy):
            nonlocal best
            if len(sx) == 0:
                return
            d2 = (sx - x) ** 2 + (sy - y) ** 2
            i = int(np.argmin(d2))
            if d2[i] <= thr2 and (best is None or d2[i] < best[0]):
                best = (float(d2[i]), system, i)

        for system, pts, visible in self._pick_sources():
            if visible and len(pts) and (systems is None or system in systems):
                sx, sy, _, _ = self._proj(pts)
                consider(system, sx, sy)
        if self.layer_mb and len(self._active_off_soma) and (systems is None or "kc" in systems):
            sx, sy, _, _ = self._project_points(self.data.kc_soma[self._active_off_soma])
            consider("kc_lit", sx, sy)                # firing cells between the drawn samples
        return None if best is None else (best[1], best[2])

    def _find_neuron_at(self, pos: QPoint, threshold_px: float = 24.0, systems=None) -> dict | None:
        """Inspector card of the drawn neuron nearest to pos within threshold_px, or None."""
        hit = self._pick(pos, threshold_px, systems)
        return None if hit is None else self._card(*hit)

    def _card(self, system: str, i: int) -> dict:
        if system == "kc":
            return self._kc_card(int(self._soma_indices[i]))
        if system == "kc_lit":
            return self._kc_card(int(self._active_off_soma[i]))
        return {"dn": self._dn_card, "dan": self._dan_card, "mbon": self._mbon_card, "cx": self._cx_card,
                "vpn": self._vpn_card, "alpn": self._alpn_card}[system](i)

    def _dn_card(self, i: int) -> dict:
        dn_t = str(self._dn_type[i]) if i < len(self._dn_type) else "DN"
        group = self._group_of("dn", i)
        rate, kind = self._dn_rate(dn_t, group)
        nt = "ACh (Excitatory)"
        if dn_t == "GF":
            name, col = "Giant Fiber (DNp01 / GF)", QColor(0, 229, 255)
            role = "Ballistic takeoff escape reflex & jump initiation (< 5 ms)"
            part = "In: LPLC2, LC4 Looming • Out: TTM Jump Motor, PSI Wing"
        elif dn_t == "MDN":
            name, col = "Moonwalker DN (MDN)", QColor(255, 23, 68)
            role = "Evokes coordinated backward locomotion / retreat"
            part = "In: LPC1 visual • Out: Thoracic backward walking CPG"
        elif dn_t in ("DNp09", "WALK"):
            name, col = f"Walking Command ({dn_t})", QColor(118, 255, 3)
            role = "Initiates and sustains forward walking locomotion"
            part = "In: Central complex • Out: Leg central pattern generators"
        elif dn_t == "DNa02":
            name, col = f"Steering Command (DNa02, {'left' if group == 'DNa02_L' else 'right'})", QColor(255, 145, 0)
            role = "Turns the fly toward its own side (asymmetric leg motor bias)"
            part = "In: LC10 visual tracking • Out: Asymmetric leg motor bias"
        elif dn_t == "GROOM":
            name, col = "Grooming Command (aDN1/aDN2)", QColor(139, 92, 246)
            role = "Antennal grooming evoked by touch of the head bristles"
            part = "In: Head mechanosensory neurons • Out: Front-leg grooming"
        else:
            name, col = f"Descending Motor Neuron ({dn_t})", QColor(239, 68, 68)
            role = "Motor command descending to ventral nerve cord"
            part = "In: Brain pre-motor • Out: Ventral nerve cord effector"
            nt = "ACh (Cholinergic)"
        return {"name": name, "type": dn_t, "system": "Motor Descending (DN)", "nt": nt,
                "nt_color": QColor(56, 189, 248), "pos_3d": self._dn_soma[i],
                "target_3d": self._dn_target[i] if i < len(self._dn_target) else None,
                "rate_hz": rate, "rate_kind": kind, "dn_index": int(i), "lab": group,
                "role": role, "partners": part, "color": col}

    def _dn_rate(self, dn_t: str, group: str | None) -> tuple[str, str]:
        if group in ("DNa02_L", "DNa02_R") and group in self._lab_rates:
            return f"{self._lab_rates[group]:.1f} Hz", "spikes"
        live = {"GF": self.rate_gf, "MDN": self.rate_mdn, "WALK": self.rate_walk, "DNp09": self.rate_walk,
                "GROOM": self.rate_groom}
        if dn_t in live:
            return f"{live[dn_t]:.1f} Hz", "spikes"
        if dn_t == "DNa02":
            return f"{abs(self.rate_steer):.1f} Hz", "R-L"
        return "—", "not read"

    def _dan_card(self, i: int) -> dict:
        dan_t = str(self.data.dan_type[i]) if i < len(self.data.dan_type) else "DAN"
        clus = self.data.dan_cluster[i] if i < len(self.data.dan_cluster) else 2
        if clus == DAN_PAM:
            name, col, sandbox = f"PAM Dopamine Cluster ({dan_t})", QColor(255, 215, 0), "reward"
            role = "Appetitive reinforcement: sugar reward & learned attraction"
            hz, part = f"{self.pam * 40.0:.1f} Hz", "In: Gustatory sugar ORNs • Out: KC Kenyon cell synapses"
        elif clus == DAN_PPL1:
            name, col, sandbox = f"PPL1 Dopamine Cluster ({dan_t})", QColor(255, 61, 113), "punish"
            role = "Aversive reinforcement: electric shock, heat & wall collisions"
            hz, part = f"{self.ppl1 * 40.0:.1f} Hz", "In: Nociceptive threat sensors • Out: KC Kenyon cell synapses"
        else:
            name, col, sandbox = f"Dopaminergic Neuron ({dan_t})", QColor(217, 70, 239), None
            role = "Neuromodulatory arousal and behavioral state gating"
            hz, part = "—", "In: Sensory neuropils • Out: Central brain"
        return {"name": name, "type": dan_t, "system": "Mushroom Body Dopamine (DAN)", "nt": "DA (Dopaminergic)",
                "nt_color": QColor(255, 215, 0), "pos_3d": self.data.dan_pos[i], "target_3d": None,
                "rate_hz": hz, "rate_kind": "designed", "sandbox": sandbox, "role": role, "partners": part,
                "color": col}

    def _mbon_card(self, i: int) -> dict:
        mbon_t = str(self.data.mbon_type[i]) if i < len(self.data.mbon_type) else "MBON"
        sign = self.data.mbon_sign[i] if i < len(self.data.mbon_sign) else 0
        nt = str(self.data.mbon_nt[i]) if i < len(self.data.mbon_nt) else "ACh"
        if sign > 0:
            role, hz, col = ("Approach output pool: drives attraction toward learned odor",
                             f"{self.mbon_app * 18.0:.1f} Hz", QColor(16, 185, 129))
        elif sign < 0:
            role, hz, col = ("Avoidance output pool: drives escape from learned danger",
                             f"{self.mbon_av * 18.0:.1f} Hz", QColor(239, 68, 68))
        else:
            role, hz, col = "Neutral / context-dependent behavioral gating", "—", QColor(148, 163, 184)
        nt_map = {"ACh": "ACh (Excitatory)", "GABA": "GABA (Inhibitory)", "Glu": "Glu (Glutamatergic)"}
        nt_col = QColor(56, 189, 248) if nt == "ACh" else (QColor(249, 115, 22) if nt == "GABA" else QColor(234, 179, 8))
        return {"name": f"Output Neuron ({mbon_t})", "type": mbon_t, "system": "Mushroom Body Output (MBON)",
                "nt": nt_map.get(nt, nt), "nt_color": nt_col, "pos_3d": self.data.mbon_pos[i], "target_3d": None,
                "rate_hz": hz, "rate_kind": "designed", "role": role,
                "partners": "In: KC Kenyon cells • Out: Lateral horn & descending pre-motor", "color": col}

    def _cx_card(self, i: int) -> dict:
        cx_t = str(self._cx_type[i]) if i < len(self._cx_type) else "CX"
        sub = self._cx_sub[i] if i < len(self._cx_sub) else 0
        if sub == 0:
            sub_name, role = "Ellipsoid Body (EB Ring)", "Internal compass heading: E-PG bump tracking fly orientation"
            part = "In: Ring neurons, PB • Out: Steering pre-motor"
        elif sub == 1:
            sub_name, role = "Protocerebral Bridge (PB)", "Angular velocity integration & bilateral compass bridge"
            part = "In: EB heading ring • Out: Central fan-shaped body"
        else:
            sub_name, role = "Fan-shaped Body (FB)", "Navigation vector calculation and goal-directed heading"
            part = "In: Protocerebral bridge • Out: Premotor hubs"
        return {"name": f"{sub_name} ({cx_t})", "type": cx_t, "system": "Central Complex (CX Navigation)",
                "nt": "ACh / GABA", "nt_color": QColor(56, 189, 248), "pos_3d": self._cx_pts[i], "target_3d": None,
                "rate_hz": f"{self._cx_rate(i):.1f} Hz", "rate_kind": "illustrative",
                "cx_index": int(i),                      # a pinned card keeps following the bump (_on_tick)
                "role": role, "partners": part, "color": QColor(56, 189, 248)}

    def _vpn_card(self, i: int) -> dict:
        vpn_t = str(self._vpn_type[i]) if i < len(self._vpn_type) else "VPN"
        group = self._group_of("vpn", i)
        if group in self._lab_rates:
            hz, kind = f"{self._lab_rates[group]:.1f} Hz", "spikes"
        else:
            hz, kind = "—", "not read"
        if "LC4" in vpn_t or "LPLC2" in vpn_t:
            role, col = "Detects expanding looming predators (direct input to Giant Fiber)", QColor(224, 64, 251)
            part = "In: Lobula complex • Out: Giant Fiber (DNp01 escape)"
        elif "LC10" in vpn_t:
            role, col = "Tracks small high-contrast moving targets (drives DNa02 steering)", QColor(249, 115, 22)
            part = "In: Lobula plate • Out: DNa02 steering command"
        elif "LPC1" in vpn_t:
            role, col = "Retreat input of the model: drives the moonwalker DNs (MDN)", QColor(168, 85, 247)
            part = "In: Lobula plate • Out: MDN backward walking"
        else:
            role, col = "Visual feature projection from optic lobe to central brain", QColor(249, 115, 22)
            part = "In: Optic lobes • Out: Posterior lateral protocerebrum"
        return {"name": f"Visual Projection ({vpn_t})", "type": vpn_t, "system": "Optic Lobes (Visual VPN)",
                "nt": "ACh (Excitatory)", "nt_color": QColor(56, 189, 248), "pos_3d": self._vpn_soma[i],
                "target_3d": self._vpn_target[i] if i < len(self._vpn_target) else None,
                "rate_hz": hz, "rate_kind": kind, "vpn_index": int(i), "lab": group, "role": role, "partners": part,
                "color": col}

    def _alpn_card(self, i: int) -> dict:
        alpn_t = str(self._alpn_type[i]) if i < len(self._alpn_type) else "ALPN"
        return {"name": f"Antennal Glomerulus ({alpn_t})", "type": alpn_t, "system": "Antennal Lobe (Olfactory PN)",
                "nt": "ACh (Excitatory)", "nt_color": QColor(56, 189, 248), "pos_3d": self._alpn_soma[i],
                "target_3d": self._alpn_target[i] if i < len(self._alpn_target) else None,
                "rate_hz": f"{self.cue_food * 45.0:.1f} Hz", "rate_kind": "designed", "sandbox": "food",
                "role": "Conveys uniglomerular odor identity from antenna to MB Calyx & LH",
                "partners": "In: Olfactory receptor ORNs • Out: Kenyon Cells & Lateral Horn",
                "color": QColor(16, 185, 129)}

    def _refresh_card(self, card: dict) -> None:
        """Live numbers of a shown card (pinned or hovered)."""
        if "kc_index" in card:
            card["rate_hz"] = self._kc_status(card["kc_index"])
        elif "cx_index" in card:
            card["rate_hz"] = f"{self._cx_rate(card['cx_index']):.1f} Hz"
        elif "dn_index" in card:
            card["rate_hz"], card["rate_kind"] = self._dn_rate(str(card.get("type", "")), card.get("lab"))
        elif "vpn_index" in card:
            g = card.get("lab")
            if g in self._lab_rates:
                card["rate_hz"], card["rate_kind"] = f"{self._lab_rates[g]:.1f} Hz", "spikes"
        elif card.get("sandbox") == "reward":
            card["rate_hz"] = f"{self.pam * 40.0:.1f} Hz"
        elif card.get("sandbox") == "punish":
            card["rate_hz"] = f"{self.ppl1 * 40.0:.1f} Hz"
        elif card.get("sandbox") == "food":
            card["rate_hz"] = f"{self.cue_food * 45.0:.1f} Hz"

    @staticmethod
    def _same_card(a: dict | None, b: dict | None) -> bool:
        """Inspector cards hold numpy positions, so `a != b` on two dicts can raise (ambiguous truth value of an array)."""
        if a is None or b is None:
            return a is b
        return a.get("name") == b.get("name") and np.array_equal(np.asarray(a.get("pos_3d"), dtype=float),
                                                                np.asarray(b.get("pos_3d"), dtype=float))

    def _kc_status(self, real_idx: int) -> str:
        """What the inspector says about Kenyon cell real_idx. The model gives a binary sparse code per mushroom-body
        step (50 ms), not a rate in Hz; a cell keeps glowing for a moment after it fired."""
        if not self._kc_from_model:
            lit = bool(self.active_kc_mask[real_idx]) if real_idx < len(self.active_kc_mask) else False
            return "Active (illustrative)" if lit else "Silent (illustrative)"
        if real_idx < len(self._kc_fired_now) and self._kc_fired_now[real_idx]:
            return "Active (fired in the last 50 ms step)"
        if real_idx < len(self.active_kc_mask) and self.active_kc_mask[real_idx]:
            return "Fading (fired a moment ago)"
        return "Silent (sparse code)"

    def _kc_card(self, real_idx: int) -> dict:
        kc_t = str(self.data.kc_type[real_idx]) if real_idx < len(self.data.kc_type) else "KC"
        lobe = int(self.data.kc_lobe[real_idx])
        target_pt = self.data.kc_target[real_idx] if real_idx < len(self.data.kc_target) else None
        lobe_name = "Gamma Lobe" if lobe == 0 else ("Alpha/Beta Lobe" if lobe == 1 else "Alpha'/Beta' Lobe")
        return {
            "name": f"Kenyon Cell ({kc_t})",
            "type": kc_t,
            "system": f"Mushroom Body ({lobe_name})",
            "nt": "ACh (Cholinergic)",
            "nt_color": QColor(56, 189, 248),
            "pos_3d": self.data.kc_soma[real_idx],
            "target_3d": target_pt,
            "rate_hz": self._kc_status(real_idx),
            "rate_kind": "model" if self._kc_from_model else "illustrative",
            "kc_index": real_idx,                    # a pinned card keeps following the cell (_on_tick)
            "role": f"Sparse olfactory memory encoding in {lobe_name}",
            "partners": "In: Antennal PNs (Calyx) • Out: MBONs & DANs (Compartment)",
            "color": QColor(6, 182, 212) if lobe == 0 else (QColor(245, 158, 11) if lobe == 1 else QColor(217, 70, 239)),
        }

    def _kc_hud_text(self) -> str:
        """The ACTIVE KC readout: active cells of the running model over that model's own cell count."""
        n_model = self._kc_model_n or self.data.n_kc      # the running model's cells, not the 3D anatomy's
        sparsity = (self.kc_active_count / max(1, n_model)) * 100
        return f"{self.kc_active_count:,} / {n_model:,} ({sparsity:.1f}%)"

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
        if self._held is not None:      # a light held while the view goes away (popout closed, panel hidden)
            self._release_held()
        if self._kc_from_model:          # no ticks while hidden: the glow would come back stale when shown again
            self._kc_glow[:] = 0.0
            self._set_active_kc_mask(np.zeros(self.data.n_kc, dtype=bool))

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

        # Compass: the runner reports the fly's real heading; only without it integrate the steering command
        # (at the body's own turn gain) so a standalone view still reacts to steering
        if self._fly_heading is None and abs(self.rate_steer) > 0.1:
            self.heading_angle = (self.heading_angle + self.rate_steer * dt * TURN_GAIN) % (2 * math.pi)
        if self._kc_from_model and self._kc_glow.any():
            self._kc_glow *= math.exp(-dt / 0.15)
            if self._set_active_kc_mask(self._kc_glow > 0.2):
                dirty = True
        self._ripples = [r for r in self._ripples if now - r[1] < 0.7]

        # live numbers on the shown cards
        for card in (self._inspected_neuron, self._hover_card):
            if card is not None:
                self._refresh_card(card)

        # Frame pacing. The view shares the GUI thread with the pet (a 16 ms tick): it repaints at 30 FPS while
        # something moves, slower when only the telemetry changes and a frame is expensive, so it never takes more than
        # about a third of that thread (interval >= 3 x the measured paint time); 5 FPS when nothing happens.
        moving = (self.auto_rotate or self._dragging or bool(self._ripples) or self._held is not None
                  or bool(self._lab.get("stimulated")))
        live = (dirty or self.pam > 0.02 or self.ppl1 > 0.02 or self.kc_active_count > 0 or self.rate_gf > 1.5
                or self.rate_walk > 1.5 or self.rate_mdn > 1.5 or abs(self.rate_steer) > 1.5 or self.rate_groom > 1.5
                or self.cue_food > 0.02 or self.cue_near > 0.02 or self.view_mode == MODE_ACTIVITY)
        pinned = self._inspected_neuron is not None or bool(self._status[0])
        if moving or live or pinned:
            want = 33.0 if moving or (live and self._paint_ms < 11.0) else 66.0
            want = max(want, 3.0 * self._paint_ms)
            interval = next(v for v in (33, 66, 120, 250) if want <= v * 1.2 or v == 250)
            if self._timer.interval() != interval:
                self._timer.setInterval(interval)
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

        heading = st.get("heading")
        if heading is not None:
            heading = float(heading) % (2 * math.pi)
            moved = self._fly_heading is None or abs(
                (heading - self._fly_heading + math.pi) % (2 * math.pi) - math.pi) > 0.03
            self._fly_heading = heading
            dirty = dirty or moved

        # Check for updated KC weights
        new_dirty = st.get("weights_dirty", 0)
        if new_dirty != self.weights_dirty:
            self.weights_dirty = new_dirty
            w = self.runner.get_mb_weights()
            if len(w) > 0:
                n_cp = min(len(self.kc_weights), len(w))
                self.kc_weights[:n_cp] = w[:n_cp]
                dirty = True

        # Which Kenyon cells fire. A runner that publishes the model's real set (Runner.get_kc_active) is authoritative:
        # a cell stays lit for a moment after it fires, so a cue that moves between tuning bins does not shimmer. Runners
        # without it (test doubles, a standalone view) get the illustrative stand-in below.
        getter = getattr(self.runner, "get_kc_active", None)
        if callable(getter):
            self._kc_from_model = True
            fired = getter()
            self._kc_fired_now = np.zeros(self.data.n_kc, dtype=bool)
            if fired is not None:
                self._kc_model_n = len(fired)
                n = min(len(fired), len(self._kc_glow))
                self._kc_fired_now[:n] = fired[:n]
                self._kc_glow[:n] = np.maximum(self._kc_glow[:n], fired[:n])
            else:
                self._kc_model_n = 0
            if self._set_active_kc_mask(self._kc_glow > 0.2):
                dirty = True
        else:
            self._kc_from_model = False
            cue_food = float(st.get("cue_food", 0.0))
            cue_near = float(st.get("cue_near", 0.0))
            has_active = self.kc_active_count > 0 or cue_food > 0.05 or cue_near > 0.05
            if has_active:
                if dirty or prev_kc != self.kc_active_count:
                    rng = np.random.default_rng(123)         # fixed: an unseeded mask would strobe on every refresh
                    prob = max(0.04, min(0.15, self.kc_active_count / max(1, self.data.n_kc)))
                    self._set_active_kc_mask(rng.uniform(0, 1, self.data.n_kc) < prob)
                    dirty = True
            elif np.any(self.active_kc_mask):
                self._set_active_kc_mask(np.zeros(self.data.n_kc, dtype=bool))
                dirty = True

        # the lab: what the worker has really applied, the visual groups' rates, the fly's state
        lab_fn = getattr(self.runner, "lab_status", None)
        if callable(lab_fn):
            lab = lab_fn()
            if lab != self._lab:
                self._lab = lab
                dirty = True
        rates_fn = getattr(self.runner, "lab_rates", None)
        if callable(rates_fn):
            self._lab_rates = rates_fn()
        state = st.get("state")
        if state is not None and state != self.fly_state:
            self.fly_state = str(state)
            dirty = True
        self._confirm_pending()
        return dirty

    _STATE_WORDS = {"fly": "takes off", "retreat": "backs away", "walk": "walks", "groom": "grooms",
                    "freeze": "freezes", "stand": "stands"}

    def _confirm_pending(self) -> None:
        """Status line: once the worker has applied a request say so, and what the fly did then."""
        if self._pending is None:
            return
        group, action, t0, _ = self._pending
        age = time.perf_counter() - t0
        lab = self._lab
        if not lab.get("acked", True):
            if age > 2.0:
                self._set_status("No answer from the engine (is it restarting?)", warn=True)
                self._pending = None
            return
        label = self.target_label(group) if group else ""
        if action == "stim":
            before = getattr(self, "_pending_state", "")
            if self.fly_state and before and self.fly_state != before:
                word = self._STATE_WORDS.get(self.fly_state, self.fly_state)
                self._set_status(f"{label}: light applied, the fly {word}")
                self._pending = None
            elif age > 1.5:
                self._set_status(f"{label}: light applied, no change in behavior")
                self._pending = None
        elif action == "silence" and group in lab.get("silenced", ()):
            self._set_status(f"{label} silenced (applied): its neurons cannot spike; plasticity paused")
            self._pending = None
        elif action in ("release", "release_all") and group not in lab.get("silenced", ()):
            self._pending = None
        elif age > 2.0:
            self._pending = None

    def _set_active_kc_mask(self, mask: np.ndarray) -> bool:
        """Install a new firing mask; also index the lit cells that the soma / tract strides skip. True if it changed."""
        if np.array_equal(mask, self.active_kc_mask):
            return False
        self.active_kc_mask = mask
        lit = np.flatnonzero(mask)
        self._active_off_soma = lit[lit % self._soma_step != 0]
        self._active_off_tract = lit[lit % self._tract_step != 0]
        return True

    # ------------------------------------------------------------------ 3D projection
    def _camera_R(self) -> np.ndarray:
        """The camera rotation for the current yaw / pitch (recomputed only when they change)."""
        if self._cam_R is None or self._cam_R_for != (self.yaw, self.pitch):
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            cp, sp = math.cos(self.pitch), math.sin(self.pitch)
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
            self._cam_R = Ry @ Rx
            self._cam_R_for = (self.yaw, self.pitch)
        return self._cam_R

    def _project_points(self, pts_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project (N, 3) coordinates into screen (x, y) coordinates with depth."""
        cam = pts_3d @ self._camera_R()

        # Perspective divisor
        persp = 1.0 / np.maximum(0.2, 1.0 + cam[:, 2] * 0.22)

        cx = self.width() * 0.5 + self.pan_x
        cy_screen = self.height() * 0.48 + self.pan_y

        sx = cx + cam[:, 0] * self.zoom * persp
        sy_proj = cy_screen - cam[:, 1] * self.zoom * persp

        return sx, sy_proj, cam[:, 2], persp

    def _camera_key(self) -> tuple:
        return (round(self.yaw, 6), round(self.pitch, 6), round(self.zoom, 4), round(self.pan_x, 3),
                round(self.pan_y, 3), self.width(), self.height())

    def _proj(self, pts_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """_project_points of one of the fixed anatomy arrays, kept for the current camera pose: the base layer, the
        live layer and the hit-test all reuse it. Only for arrays that live on (the memo holds the array, so its id
        cannot be reused by another while the entry exists); slices made on the fly go to _project_points."""
        key = self._camera_key()
        if self._proj_key != key:
            self._proj_memo.clear()
            self._proj_key = key
        hit = self._proj_memo.get(id(pts_3d))
        if hit is None or hit[0] is not pts_3d:
            hit = self._proj_memo[id(pts_3d)] = (pts_3d, self._project_points(pts_3d))
        return hit[1]

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event):
        t0 = time.perf_counter()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._camera_R()

        # 1. Everything that only moves with the camera, cached (background, envelopes, circuits at rest)
        p.drawPixmap(0, 0, self._base_layer())

        # 2. What changes every step: firing cells and pulses, the compass bump, dopamine
        self._paint_live(p)

        # 3. The lab: stimulated groups glowing, the target under the cursor, markers, action ripples
        self._draw_lab_overlay(p)

        # 4. Inspected Neuron Reticle & Synaptic Streamline
        self._draw_inspected_reticle(p)

        # 5. HUD Overlay (toolbar, telemetry, inspector card, status line, hover label)
        self._draw_hud(p)

        # 6. Help Guide Overlay (if toggled)
        if self._show_help:
            self._draw_help_overlay(p)

        p.end()
        self._paint_ms += 0.2 * ((time.perf_counter() - t0) * 1000.0 - self._paint_ms)

    def _base_key(self) -> tuple:
        return (self._camera_key(), self.devicePixelRatioF(), self.view_mode, self.kc_weights.tobytes(),
                self.layer_wireframe, self.layer_hulls, self.layer_olfactory, self.layer_visual, self.layer_cx,
                self.layer_mb, self.layer_motor, tuple(self._lab.get("silenced", ())))

    def _base_layer(self) -> QPixmap:
        """The scene at rest for the current camera pose, view mode, learned weights and layers, rendered into a
        pixmap once and reused until one of them changes (a frame with the camera still only draws the live layer)."""
        key = self._base_key()
        if self._base_cache is not None and self._base_cache_key == key:
            return self._base_cache
        dpr = self.devicePixelRatioF()
        size = QSize(max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr)))
        pix = self._base_cache if self._base_cache is not None and self._base_cache.size() == size else QPixmap(size)
        pix.setDevicePixelRatio(dpr)
        bp = QPainter(pix)
        bp.setRenderHint(QPainter.Antialiasing)
        self._paint_base(bp)
        bp.end()
        self._base_cache, self._base_cache_key = pix, key
        return pix

    def _background(self, dpr: float) -> QPixmap:
        """The backdrop (gradient and a faint dotted grid) for the widget size: it does not move with the camera, and
        the dotted grid alone cost ~10 ms a time."""
        w, h = self.width(), self.height()
        key = (w, h, dpr)
        if getattr(self, "_bg_key", None) != key:
            pix = QPixmap(max(1, round(w * dpr)), max(1, round(h * dpr)))
            pix.setDevicePixelRatio(dpr)
            bp = QPainter(pix)
            bg_grad = QLinearGradient(0, 0, 0, h)
            bg_grad.setColorAt(0.0, QColor(9, 9, 13))
            bg_grad.setColorAt(1.0, QColor(14, 14, 20))
            bp.fillRect(0, 0, w, h, bg_grad)
            bp.setRenderHint(QPainter.Antialiasing)
            bp.setPen(QPen(QColor(255, 255, 255, 6), 1, Qt.DotLine))
            bp.drawLines([QLineF(gx, 0, gx, h) for gx in range(0, w, 60)]
                         + [QLineF(0, gy, w, gy) for gy in range(0, h, 60)])
            bp.end()
            self._bg_pix, self._bg_key = pix, key
        return self._bg_pix

    def _paint_base(self, p: QPainter) -> None:
        # Background and the subtle grid
        dev = p.device()
        p.drawPixmap(0, 0, self._background(dev.devicePixelRatioF() if dev is not None else 1.0))

        # Drosophila Brain Neuropil Wireframe and the translucent envelopes (AL, MB, CX, Optic)
        if self.layer_wireframe:
            self._draw_brain_wireframe(p)
        if self.layer_hulls:
            self._draw_neuropil_volumes(p)

        # Sensory Input Circuits (Olfactory & Visual)
        if self.layer_olfactory:
            self._draw_antennal_lobes(p)
        if self.layer_visual:
            self._draw_visual_projections(p)

        # Central Complex Navigation & Compass Ring
        if self.layer_cx:
            self._draw_central_complex(p)

        # Mushroom Body Learning Circuit
        if self.layer_mb:
            self._draw_axon_tracts(p)
            self._draw_kenyon_somas(p)
            self._draw_mbons(p)

        # Motor Output System (Descending Neurons)
        if self.layer_motor:
            self._draw_descending_neurons(p)

    def _paint_live(self, p: QPainter) -> None:
        """The part of the scene that changes every step, drawn over the base layer each frame."""
        if self.layer_olfactory:
            self._draw_odor_pulses(p)
        if self.layer_visual:
            self._draw_visual_pulses(p)
        if self.layer_cx:
            self._draw_compass_bump(p)
        if self.layer_mb:
            self._draw_firing_kenyon_cells(p)
            self._draw_dopamine_clusters(p)
        if self.layer_motor:
            self._draw_motor_pulses(p)

    # ------------------------------------------------------------------ wireframe
    def _draw_brain_wireframe(self, p: QPainter):
        if len(self._wf_p1) == 0:
            return

        x1, y1, _, _ = self._proj(self._wf_p1)
        x2, y2, _, _ = self._proj(self._wf_p2)

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(56, 189, 248, 22), 1.0))
        p.drawLines([QLineF(a, b, c, d) for a, b, c, d in zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist())])

    # ------------------------------------------------------------------ neuropil hulls
    def _draw_neuropil_volumes(self, p: QPainter):
        """Translucent glass envelopes around the major neuropils. Each is an ellipsoid, so its outline on screen is an
        ellipse; a gradient that brightens toward the rim gives the fresnel look (edge-on surface glows) in one fill per
        hull (the per-triangle version cost ~200 ms a frame)."""
        R = self._camera_R().astype(np.float64)
        ox = self.width() * 0.5 + self.pan_x
        oy = self.height() * 0.48 + self.pan_y
        hulls = []
        for nv in self._neuropil_volumes:
            c_cam = nv["center"] @ R
            k = self.zoom / max(0.2, 1.0 + float(c_cam[2]) * 0.22)      # perspective at the hull's centre
            # world ellipsoid c + diag(radii) u, |u| = 1 -> screen offsets k * F (R^T diag(radii))[:2] u, F flips y
            m = (R.T[:2] * nv["radii"][None, :]) * k
            m[1] *= -1.0
            vals, vecs = np.linalg.eigh(m @ m.T)
            a, b = math.sqrt(max(vals[1], 0.0)), math.sqrt(max(vals[0], 0.0))
            if b < 0.5:
                continue
            ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
            hulls.append((float(c_cam[2]), ox + c_cam[0] * k, oy - c_cam[1] * k, a, b, ang, nv["color"]))

        rim = QPen()
        rim.setCosmetic(True)
        rim.setWidthF(1.0)
        for _, x, y, a, b, ang, col in sorted(hulls, key=lambda t: -t[0]):     # far to near
            p.save()
            p.translate(x, y)
            p.rotate(ang)
            p.scale(a, b)                       # the unit circle becomes the hull's outline
            grad = QRadialGradient(0.0, 0.0, 1.0)
            for at, alpha in ((0.0, 4), (0.62, 8), (0.88, 18), (1.0, 38)):
                grad.setColorAt(at, QColor(col.red(), col.green(), col.blue(), alpha))
            p.setBrush(QBrush(grad))
            rim.setColor(QColor(col.red(), col.green(), col.blue(), 74))
            p.setPen(rim)
            p.drawEllipse(QPointF(0.0, 0.0), 1.0, 1.0)
            p.restore()

    # ------------------------------------------------------------------ colours
    def _resting_rgba(self, weight, lobe, alpha: int) -> np.ndarray:
        """_color_for_neuron(w, lobe, False, alpha) for whole arrays, packed 0xAARRGGBB. The ramps are cut into 64 steps
        (below what the eye resolves) so a few hundred cells share a few dozen colours and draw as one path each."""
        w = np.asarray(weight, dtype=np.float64)
        n = len(w)
        r, g, b = np.zeros(n), np.zeros(n), np.zeros(n)
        a = np.full(n, float(alpha))
        if self.view_mode == MODE_VALENCE:
            r[:], g[:], b[:] = 56, 189, 248                        # naive
            pos, neg = w > 0.005, w < -0.005
            t = np.round(np.minimum(1.0, w[pos] / 0.15) * 64) / 64
            r[pos], g[pos], b[pos] = 16 * (1 - t) + 251 * t, 185 * (1 - t) + 191 * t, 129 * (1 - t) + 36 * t
            a[pos] = np.minimum(255, np.floor(alpha * (1.0 + t * 0.4)))
            t = np.round(np.minimum(1.0, -w[neg] / 0.15) * 64) / 64
            r[neg], g[neg], b[neg] = 244 * (1 - t) + 220 * t, 63 * (1 - t) + 38 * t, 94 * (1 - t) + 38 * t
            a[neg] = np.minimum(255, np.floor(alpha * (1.0 + t * 0.4)))
        elif self.view_mode == MODE_PLASTICITY:
            mag = np.round(np.minimum(1.0, np.abs(w) / 0.12) * 64) / 64
            lo = mag < 0.5
            t = np.where(lo, mag / 0.5, (mag - 0.5) / 0.5)
            r = np.where(lo, 59 * (1 - t) + 249 * t, 249 * (1 - t) + 253 * t)
            g = np.where(lo, 130 * (1 - t) + 115 * t, 115 * (1 - t) + 224 * t)
            b = np.where(lo, 246 * (1 - t) + 22 * t, 22 * (1 - t) + 71 * t)
            a = np.minimum(255, np.floor(alpha * (0.8 + mag * 0.5)))
        elif self.view_mode == MODE_ANATOMY:
            lobe = np.asarray(lobe)
            gam, ab = lobe == LOBE_GAMMA, lobe == LOBE_ALPHABETA
            r = np.where(gam, 6, np.where(ab, 245, 217))
            g = np.where(gam, 182, np.where(ab, 158, 70))
            b = np.where(gam, 212, np.where(ab, 11, 239))
        else:                                                      # MODE_ACTIVITY: dim at rest, sparks on top
            r[:], g[:], b[:] = 71, 85, 105
        return _pack_rgba(r, g, b, a)

    # ------------------------------------------------------------------ tracts
    def _draw_axon_tracts(self, p: QPainter):
        """Curved axon bundles from the calyx through the pedunculus to the lobes, at rest (the firing ones are drawn
        bright on top every frame by _draw_firing_kenyon_cells)."""
        if len(self._tract_indices) == 0:
            return
        sx, sy, _, _ = self._proj(self._tract_soma)
        px, py, _, _ = self._proj(self._tract_ped)
        tx, ty, _, _ = self._proj(self._tract_target)
        rgba = self._resting_rgba(self.kc_weights[self._tract_indices], self._tract_lobes, alpha=45)
        p.setBrush(Qt.NoBrush)
        for col, idx in _rgba_groups(rgba):
            p.setPen(QPen(col, 1.0))
            p.drawPath(_curve_path(sx[idx], sy[idx], px[idx], py[idx], tx[idx], ty[idx]))

    # ------------------------------------------------------------------ somas
    def _draw_kenyon_somas(self, p: QPainter):
        """Kenyon cell somas in the bilateral calyx cups, at rest: the plain cells in one point pass per colour, the
        learned ones (every one in the plasticity and activity modes) as dots in their colour, strongly learned ones
        with a halo. The firing cells are drawn on top every frame."""
        if len(self._soma_indices) == 0:
            return

        sx, sy, _, persp = self._proj(self._soma_pts)
        weights = self.kc_weights[self._soma_indices]
        lobes = self._soma_lobes

        if self.view_mode == MODE_VALENCE:
            special = np.abs(weights) > 0.005
            plain = [(QColor(56, 189, 248, 140), ~special)]
        elif self.view_mode == MODE_ANATOMY:
            special = np.zeros(len(weights), dtype=bool)
            gam, ab = lobes == LOBE_GAMMA, lobes == LOBE_ALPHABETA
            plain = [(QColor(6, 182, 212, 140), gam), (QColor(245, 158, 11, 140), ab),
                     (QColor(217, 70, 239, 140), ~gam & ~ab)]
        else:  # MODE_PLASTICITY, MODE_ACTIVITY
            special = np.ones(len(weights), dtype=bool)
            plain = []

        p.setPen(Qt.NoPen)
        for col, sel in plain:
            idx = np.flatnonzero(sel)
            if len(idx):
                p.setBrush(col)
                _draw_dots(p, sx[idx], sy[idx], 1.7)

        idx = np.flatnonzero(special)
        if len(idx) == 0:
            return
        radius = 1.8 * persp[idx]
        strong = np.abs(weights[idx]) > 0.04
        for col, sub in _rgba_groups(self._resting_rgba(weights[idx], lobes[idx], alpha=140)):
            cells = idx[sub]
            p.setBrush(col)
            _draw_dots(p, sx[cells], sy[cells], radius[sub])
            halo = sub[strong[sub]]
            if len(halo):
                p.setBrush(QColor(col.red(), col.green(), col.blue(), 45))
                _draw_dots(p, sx[idx[halo]], sy[idx[halo]], radius[halo] * 2.2)

    def _draw_firing_kenyon_cells(self, p: QPainter):
        """The cells that fire now (or glow a moment after): a bright axon with a travelling action potential, a white
        soma with a halo. Includes the firing cells the soma / tract samples skip (_active_off_soma / _off_tract)."""
        mask = self.active_kc_mask

        # axons: (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2 for the pulse
        segs = []
        on = np.flatnonzero(mask[self._tract_indices]) if len(self._tract_indices) else []
        if len(on):
            segs.append([a[on] for q in (self._tract_soma, self._tract_ped, self._tract_target)
                         for a in self._proj(q)[:2]])
        lit = self._active_off_tract
        if len(lit):
            segs.append([a for q in (self.data.kc_soma[lit], self.data.kc_ped[lit], self.data.kc_target[lit])
                         for a in self._project_points(q)[:2]])
        if segs:
            x0, y0, cx, cy, x1, y1 = (np.concatenate(c) for c in zip(*segs))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 235), 1.0))
            p.drawPath(_curve_path(x0, y0, cx, cy, x1, y1))
            t = self._pulse_phase
            bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
            by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 230))
            _draw_dots(p, bx, by, 2.2)

        # somas
        xs, ys, rs = [], [], []
        on = np.flatnonzero(mask[self._soma_indices]) if len(self._soma_indices) else []
        if len(on):
            sx, sy, _, persp = self._proj(self._soma_pts)
            xs.append(sx[on])
            ys.append(sy[on])
            rs.append(np.maximum(1.8 * persp[on] * 1.8, 3.2))
        lit = self._active_off_soma
        if len(lit):
            lx, ly, _, lp = self._project_points(self.data.kc_soma[lit])
            xs.append(lx)
            ys.append(ly)
            rs.append(np.maximum(1.8 * lp * 1.8, 3.2))
        if xs:
            x, y, r = np.concatenate(xs), np.concatenate(ys), np.concatenate(rs)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 255))
            _draw_dots(p, x, y, r)
            p.setBrush(QColor(255, 255, 255, 45))
            _draw_dots(p, x, y, r * 2.2)

    # ------------------------------------------------------------------ dopamine
    def _draw_dopamine_clusters(self, p: QPainter):
        """Draw PAM (reward) and PPL1 (punishment) dopaminergic clusters (their glow follows the live dopamine)."""
        dans = self.data.dan_pos
        if len(dans) == 0:
            return
        dx, dy, _, persp = self._proj(dans)
        clusters = np.asarray(self.data.dan_cluster)[:len(dans)]
        pam_i = np.flatnonzero(clusters == DAN_PAM)
        ppl_i = np.flatnonzero(clusters == DAN_PPL1)
        other = np.flatnonzero((clusters != DAN_PAM) & (clusters != DAN_PPL1))

        # 1. Radiant shockwaves if PAM or PPL1 is actively firing
        if self.pam > 0.05 and len(pam_i) > 0:
            self._draw_dopamine_wave(p, float(np.mean(dx[pam_i])), float(np.mean(dy[pam_i])),
                                     QColor(250, 204, 21), self.pam)
        if self.ppl1 > 0.05 and len(ppl_i) > 0:
            self._draw_dopamine_wave(p, float(np.mean(dx[ppl_i])), float(np.mean(dy[ppl_i])),
                                     QColor(239, 68, 68), self.ppl1)

        # 2. The DAN nodes: radiant gold (reward) and fiery scarlet (punishment) glow with their dopamine
        def glow(level: float) -> int:
            return int(max(0.0, min(1.0, 0.4 + level * 0.6)) * 255)

        p.setPen(Qt.NoPen)
        for idx, col, r in ((other, QColor(168, 85, 247, 120), 1.8),
                            (pam_i, QColor(245, 158, 11, glow(self.pam)), 2.4 + self.pam * 2.0),
                            (ppl_i, QColor(239, 68, 68, glow(self.ppl1)), 2.4 + self.ppl1 * 2.0)):
            if len(idx):
                p.setBrush(col)
                _draw_dots(p, dx[idx], dy[idx], r * persp[idx])

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
            ring_col.setAlpha(int(fade * 140 * min(1.0, strength)))
            p.setPen(QPen(ring_col, 1.8))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r, r)

    # ------------------------------------------------------------------ mbons
    def _draw_mbons(self, p: QPainter):
        """Draw Mushroom Body Output Neuron (MBON) terminals: approach (+1) green, avoidance (-1) rose."""
        mbons = self.data.mbon_pos
        if len(mbons) == 0:
            return
        mx, my, _, persp = self._proj(mbons)
        signs = np.asarray(self.data.mbon_sign)[:len(mbons)]

        p.setPen(QPen(QColor(255, 255, 255, 100), 0.8))
        for sel, col in ((signs > 0, QColor(34, 197, 94, 210)), (signs < 0, QColor(244, 63, 94, 210)),
                         (signs == 0, QColor(148, 163, 184, 160))):
            idx = np.flatnonzero(sel)
            if len(idx):
                p.setBrush(col)
                _draw_dots(p, mx[idx], my[idx], 3.0 * persp[idx])

    # ------------------------------------------------------------------ central complex
    def _draw_central_complex(self, p: QPainter):
        """Central Complex (CX) navigation hub at rest: the Ellipsoid Body (EB) compass ring, PB bridge and FB."""
        if len(self._cx_pts) == 0:
            return

        cx, cy, _, persp = self._proj(self._cx_pts)
        subs = self._cx_sub if len(self._cx_sub) == len(self._cx_pts) else np.zeros(0, dtype=np.int8)

        def poly(idx):
            return QPolygonF([QPointF(x, y) for x, y in zip(cx[idx].tolist(), cy[idx].tolist())])

        # 1. Ellipsoid Body (EB) Ring: toroidal compass ring (nodes in ring order, see _eb_ring_order); the bump on it
        # is live (_draw_compass_bump)
        eb_idx = self._eb_ring
        if len(eb_idx) > 0:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(56, 189, 248, 150), 1.0))
            if len(eb_idx) > 2:
                p.drawPolygon(poly(eb_idx))
            else:
                p.drawPolyline(poly(eb_idx))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(56, 189, 248, 180))
            _draw_dots(p, cx[eb_idx], cy[eb_idx], 2.2 * persp[eb_idx])

        # 2. Protocerebral Bridge (PB): Horizontal arched mustache
        pb_idx = np.flatnonzero(subs == 1)
        if len(pb_idx):
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(168, 85, 247, 160), 1.0))
            p.drawPolyline(poly(pb_idx))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(168, 85, 247, 200))
            _draw_dots(p, cx[pb_idx], cy[pb_idx], 2.0 * persp[pb_idx])

        # 3. Fan-shaped Body (FB)
        fb_idx = np.flatnonzero(subs == 2)
        if len(fb_idx):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(234, 179, 8, 150))
            _draw_dots(p, cx[fb_idx], cy[fb_idx], 1.8 * persp[fb_idx])

    def _draw_compass_bump(self, p: QPainter):
        """The E-PG bump: the ring nodes at the fly's heading, white over the resting ring."""
        eb_idx = self._eb_ring
        if len(eb_idx) == 0:
            return
        lit = eb_idx[np.flatnonzero(self._eb_lit(len(eb_idx)))]
        if len(lit):
            cx, cy, _, persp = self._proj(self._cx_pts)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 250))
            _draw_dots(p, cx[lit], cy[lit], 3.6 * persp[lit])

    # ------------------------------------------------------------------ visual projections
    def _vpn_groups(self) -> list[tuple[QColor, np.ndarray]]:
        """(colour, indices) of the visual projection neurons by type: LC4 amber (looming speed), LC10 sky blue (target
        tracking), LPLC2 coral (looming size), LPC1 violet (retreat), others blue."""
        types = self._vpn_type
        if getattr(self, "_vpn_groups_for", None) is not types:
            cols = (QColor(245, 158, 11, 140), QColor(14, 165, 233, 140), QColor(249, 115, 22, 140),
                    QColor(168, 85, 247, 140), QColor(56, 189, 248, 100))
            kind = np.array([next((k for k, tag in enumerate(("LC4", "LC10", "LPLC2", "LPC1")) if tag in str(t)), 4)
                             for t in types], dtype=np.int64)
            self._vpn_groups_cache = [(cols[k], np.flatnonzero(kind == k)) for k in range(5) if np.any(kind == k)]
            self._vpn_groups_for = types
        return self._vpn_groups_cache

    def _vpn_curves(self):
        sx, sy, _, s_persp = self._proj(self._vpn_soma)
        tx, ty, _, _ = self._proj(self._vpn_target)
        return sx, sy, (sx + tx) * 0.5, np.minimum(sy, ty) - 8.0, tx, ty, s_persp

    def _draw_visual_projections(self, p: QPainter):
        """Draw Visual Projection Neurons (VPN) from Optic Lobes to central brain (LC4, LC10, LPLC2, LPC1)."""
        if len(self._vpn_soma) == 0:
            return
        sx, sy, mx, my, tx, ty, s_persp = self._vpn_curves()
        groups = self._vpn_groups()
        p.setBrush(Qt.NoBrush)
        for col, idx in groups:
            p.setPen(QPen(col, 1.0))
            p.drawPath(_curve_path(sx[idx], sy[idx], mx[idx], my[idx], tx[idx], ty[idx]))
        p.setPen(Qt.NoPen)
        for col, idx in groups:
            p.setBrush(col)
            _draw_dots(p, sx[idx], sy[idx], 2.2 * s_persp[idx])
        self._draw_silenced(p, "vpn", sx, sy, 2.6 * s_persp)

    def _draw_visual_pulses(self, p: QPainter):
        """Travelling action potentials along the visual projections while there is something to see."""
        if len(self._vpn_soma) == 0:
            return
        if not (self.cue_near > 0.02 or self.rate_gf > 1.5 or abs(self.rate_steer) > 1.5
                or self.view_mode == MODE_ACTIVITY):
            return
        sx, sy, mx, my, tx, ty, _ = self._vpn_curves()
        t = (self._pulse_phase + (np.arange(len(sx)) % 7) * 0.14) % 1.0
        bx = (1 - t) ** 2 * sx + 2 * (1 - t) * t * mx + t ** 2 * tx
        by = (1 - t) ** 2 * sy + 2 * (1 - t) * t * my + t ** 2 * ty
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 240))
        _draw_dots(p, bx, by, 2.4)

    # ------------------------------------------------------------------ antennal lobes
    def _alpn_curves(self):
        sx, sy, _, s_persp = self._proj(self._alpn_soma)
        tx, ty, _, _ = self._proj(self._alpn_target)
        return sx, sy, (sx + tx) * 0.5, (sy + ty) * 0.5 - 12.0, tx, ty, s_persp

    def _draw_antennal_lobes(self, p: QPainter):
        """Draw Antennal Lobes (AL) and olfactory projection neurons (ALPN) to Calyx & Lateral Horn."""
        if len(self._alpn_soma) == 0:
            return
        sx, sy, mx, my, tx, ty, s_persp = self._alpn_curves()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(16, 185, 129, 90), 1.0))
        p.drawPath(_curve_path(sx, sy, mx, my, tx, ty))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(16, 185, 129, 210))
        _draw_dots(p, sx, sy, 2.6 * s_persp)

    def _draw_odor_pulses(self, p: QPainter):
        """Travelling olfactory packets along the projection neurons while there is an odour (or a reward)."""
        if len(self._alpn_soma) == 0:
            return
        if not (self.cue_food > 0.02 or self.pam > 0.05 or self.view_mode == MODE_ACTIVITY):
            return
        sx, sy, mx, my, tx, ty, _ = self._alpn_curves()
        t = (self._pulse_phase + (np.arange(len(sx)) % 5) * 0.20) % 1.0
        bx = (1 - t) ** 2 * sx + 2 * (1 - t) * t * mx + t ** 2 * tx
        by = (1 - t) ** 2 * sy + 2 * (1 - t) * t * my + t ** 2 * ty
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(167, 243, 208, 240))
        _draw_dots(p, bx, by, 2.2)

    # ------------------------------------------------------------------ descending neurons
    _DN_STYLES = (("GF", QColor(239, 68, 68, 240), 2.6),      # Giant Fiber (blazing crimson)
                  ("MDN", QColor(217, 70, 239, 230), 2.0),    # Moonwalker (fuchsia)
                  ("WALK", QColor(34, 197, 94, 230), 2.0),    # Forward walk (lime green)
                  ("DNa", QColor(6, 182, 212, 230), 2.0),     # Steering (cyan); matched as a prefix
                  ("GROOM", QColor(139, 92, 246, 230), 2.0),  # Grooming (violet)
                  ("", QColor(148, 163, 184, 180), 1.4))

    def _dn_style_index(self) -> np.ndarray:
        types = self._dn_type
        if getattr(self, "_dn_style_for", None) is not types:
            def style(t: str) -> int:
                if t in ("GF", "MDN", "WALK", "GROOM"):
                    return ("GF", "MDN", "WALK", "DNa", "GROOM").index(t)
                return 3 if "DNa" in t else 5
            self._dn_style_cache = np.array([style(str(t)) for t in types], dtype=np.int64)
            self._dn_style_for = types
        return self._dn_style_cache

    def _draw_descending_neurons(self, p: QPainter):
        """Draw Descending Motor Neurons (DN Hub) plunging through cervical connective toward the VNC."""
        if len(self._dn_soma) == 0:
            return
        sx, sy, _, s_persp = self._proj(self._dn_soma)
        tx, ty, _, t_persp = self._proj(self._dn_target)
        style = self._dn_style_index()
        for k, (_, col, line_w) in enumerate(self._DN_STYLES):
            idx = np.flatnonzero(style == k)
            if len(idx) == 0:
                continue
            # descending axon trunks (straight lines stay cheap at 2-2.6 px; it is curves that a wide pen makes slow)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, line_w))
            p.drawLines([QLineF(a, b, c, d) for a, b, c, d in
                         zip(sx[idx].tolist(), sy[idx].tolist(), tx[idx].tolist(), ty[idx].tolist())])
            # soma nodes in the central brain and the cervical terminal markers
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            _draw_dots(p, sx[idx], sy[idx], 3.8 * s_persp[idx])
            _draw_dots(p, tx[idx], ty[idx], 2.0 * t_persp[idx])

        self._draw_silenced(p, "dn", sx, sy, 4.2 * s_persp)

        # labelled tags for the primary anchor neurons
        p.setFont(self._fonts()["tag"])
        for i in range(0, len(self._dn_soma), 2):
            ntype = str(self._dn_type[i])
            if ntype in ("GF", "MDN", "WALK"):
                p.setPen(self._DN_STYLES[int(style[i])][1])
                p.drawText(int(sx[i]) + 6, int(sy[i]) + 3, ntype)

    def _draw_motor_pulses(self, p: QPainter):
        """Action potentials travelling down the trunks of the descending neurons that fire."""
        if len(self._dn_soma) == 0:
            return
        types = self._dn_type
        if self.view_mode == MODE_ACTIVITY:
            firing = np.ones(len(types), dtype=bool)
        else:
            rate = {"GF": self.rate_gf, "WALK": self.rate_walk, "MDN": self.rate_mdn, "GROOM": self.rate_groom}
            firing = np.array([(rate.get(str(t), 0.0) > 1.5) or ("DNa" in str(t) and abs(self.rate_steer) > 1.5)
                               for t in types], dtype=bool)
        idx = np.flatnonzero(firing)
        if len(idx) == 0:
            return
        sx, sy, _, _ = self._proj(self._dn_soma)
        tx, ty, _, _ = self._proj(self._dn_target)
        t = (self._pulse_phase * 2.2 + (idx % 3) * 0.33) % 1.0
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 250))
        _draw_dots(p, sx[idx] + (tx[idx] - sx[idx]) * t, sy[idx] + (ty[idx] - sy[idx]) * t, 3.2)

    def _color_for_neuron(self, weight: float, lobe: int, is_active: bool, alpha: int = 180) -> QColor:
        """Map neuron attributes to color based on current visual mode."""
        if is_active:
            # Active firing spark: bright white-hot cyan
            return QColor(255, 255, 255, min(255, alpha + 50))
        return QColor.fromRgba(int(self._resting_rgba([weight], [lobe], alpha)[0]))

    def _draw_silenced(self, p: QPainter, system: str, sx, sy, r) -> None:
        """Silenced groups' neurons in grey with a dark rim (base layer: it changes only when the silencing does)."""
        idx = [m for g in self._lab.get("silenced", ()) if g in self._lab_members
               for sys_, m in [self._lab_members[g]] if sys_ == system]
        if not idx:
            return
        sel = np.concatenate(idx)
        p.setPen(QPen(QColor(15, 15, 20, 230), 1.0))
        p.setBrush(QColor(120, 128, 140, 235))
        _draw_dots(p, sx[sel], sy[sel], np.maximum(r[sel], 3.0))

    def _group_anchors(self, group: str) -> list[tuple[float, float]]:
        """Where a group's marker label goes: the centre of its drawn neurons on each side."""
        system, idx = self._lab_members[group]
        sides = self._dn_side if system == "dn" else self._vpn_side
        x, y, _ = self._members_xy(group)
        if len(sides) < 1 or group.endswith(("_L", "_R")):
            return [(float(np.mean(x)), float(np.mean(y)))]
        side = np.asarray(sides)[idx] if len(sides) > int(idx.max()) else np.zeros(len(idx))
        out = []
        for s_ in (-1, 1):
            m = side == s_
            if m.any():
                out.append((float(np.mean(x[m])), float(np.mean(y[m]))))
        return out or [(float(np.mean(x)), float(np.mean(y)))]

    def _draw_lab_overlay(self, p: QPainter) -> None:
        """Stimulated groups glow (from what the worker reports), the target under the cursor is ringed, the lab tools
        show where their targets are, and every action leaves one ripple."""
        now = time.perf_counter()
        visible = {g for g, (sys_, _) in self._lab_members.items()
                   if (self.layer_motor if sys_ == "dn" else self.layer_visual)}
        p.save()
        # 1. light on: the group's neurons pulse orange
        lit = [g for g in self._lab.get("stimulated", ()) if g in visible]
        if lit:
            glow = 0.55 + 0.45 * math.sin(now * 12.0)
            p.setPen(Qt.NoPen)
            for g in lit:
                x, y, persp = self._members_xy(g)
                p.setBrush(QColor(255, 120, 60, int(70 * glow)))
                _draw_dots(p, x, y, 9.0 * persp)
                p.setBrush(QColor(255, 196, 120, 235))
                _draw_dots(p, x, y, 3.2 * persp)
        # 2. markers of the lab tools
        p.setFont(self._fonts()["tag"])
        if self.tool_mode in ("opto", "silence"):
            avail = set(self._lab.get("available", ())) or set(self._lab_members)
            for g in sorted(visible):
                off = g not in avail
                sil = g in self._lab.get("silenced", ())
                text = "" if g in ("GF", "MDN", "WALK") else g.replace("_L", " L").replace("_R", " R").replace(
                    "GROOM", "aDN")
                for ax, ay in self._group_anchors(g):
                    col = (QColor(110, 110, 120, 150) if off else QColor(148, 163, 184, 230) if sil
                           else QColor(255, 150, 90, 230) if self.tool_mode == "opto" else QColor(186, 196, 210, 230))
                    p.setPen(QPen(col, 1.0))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(QPointF(ax, ay), 7.0, 7.0)
                    label = f"{text} ⊘".strip() if sil else text
                    if label:
                        p.drawText(QPointF(ax + 10.0, ay - 6.0), label)
        # 3. the target under the cursor: every neuron of its group ringed
        g = self._hover_target
        if g in self._lab_members and g in visible:
            x, y, persp = self._members_xy(g)
            col = QColor(255, 150, 90) if self.tool_mode == "opto" else QColor(226, 232, 240)
            p.setPen(QPen(col, 1.0))
            p.setBrush(Qt.NoBrush)
            for xx, yy, pp in zip(x.tolist(), y.tolist(), persp.tolist()):
                p.drawEllipse(QPointF(xx, yy), 6.0 * pp + 2.0, 6.0 * pp + 2.0)
        # 4. one ripple per action
        for target, t0, col in self._ripples:
            age = (now - t0) / 0.7
            xy = self._target_xy(target)
            if xy is None or age >= 1.0:
                continue
            c = QColor(col)
            c.setAlpha(int(220 * (1.0 - age)))
            p.setPen(QPen(c, 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(*xy), 8.0 + 40.0 * age, 8.0 + 40.0 * age)
        p.restore()

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

    def _card_rect(self, w: int, h: int) -> QRect:
        cw = min(380, w - 32)
        ch = 120
        return QRect(16, h - 66 - ch - 8, cw, ch)

    def _draw_inspector_card(self, p: QPainter, w: int, h: int):
        """The pinned neuron's card: what it is, its rate (and whether that is a spike rate, a designed signal or an
        illustration), and for a lab target the actions."""
        card = self._inspected_neuron
        if card is None:
            return
        f = self._fonts()
        r = self._card_rect(w, h)
        cx, cy, cw, ch = r.x(), r.y(), r.width(), r.height()
        p.save()
        bg = QLinearGradient(cx, cy, cx, cy + ch)
        bg.setColorAt(0.0, QColor(14, 16, 24, 240))
        bg.setColorAt(1.0, QColor(10, 11, 16, 248))
        p.setBrush(QBrush(bg))
        col = card.get("color", QColor(168, 85, 247))
        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 170), 1.2))
        p.drawRoundedRect(r, 6, 6)

        p.setFont(f["cat"])
        p.setPen(QColor(col.red(), col.green(), col.blue(), 230))
        p.drawText(QRect(cx + 12, cy + 6, cw - 40, 16), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(f["cat"]).elidedText(f"INSPECTOR • {card.get('system', '')}", Qt.ElideRight, cw - 40))
        close_rect = QRect(cx + cw - 22, cy + 6, 16, 16)
        self._btn_rects["close_inspector"] = close_rect
        p.setPen(QColor(150, 150, 160, 200))
        p.drawText(close_rect, Qt.AlignCenter, "×")

        p.setFont(f["name"])
        p.setPen(QColor(245, 245, 250))
        p.drawText(QRect(cx + 12, cy + 24, cw - 24, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(f["name"]).elidedText(card.get("name", "Neuron"), Qt.ElideRight, cw - 24))

        # badges: neurotransmitter, rate (and what kind of number it is)
        p.setFont(f["badge"])
        fm = QFontMetrics(f["badge"])
        bx, by = cx + 12, cy + 46
        nt_txt = card.get("nt", "ACh")
        nt_rect = QRect(bx, by, fm.horizontalAdvance(nt_txt) + 12, 18)
        nt_c = card.get("nt_color", QColor(56, 189, 248))
        p.setBrush(QColor(nt_c.red(), nt_c.green(), nt_c.blue(), 55))
        p.setPen(QPen(QColor(nt_c.red(), nt_c.green(), nt_c.blue(), 180), 1))
        p.drawRoundedRect(nt_rect, 3, 3)
        p.setPen(QColor(245, 245, 250))
        p.drawText(nt_rect, Qt.AlignCenter, nt_txt)
        bx += nt_rect.width() + 6
        kind = card.get("rate_kind", "")
        rate_txt = f"Rate: {card.get('rate_hz', '0.0 Hz')}" + (f"  ({kind})" if kind and kind != "spikes" else "")
        rate_rect = QRect(bx, by, fm.horizontalAdvance(rate_txt) + 12, 18)
        p.setBrush(QColor(30, 32, 42, 180))
        p.setPen(QPen(QColor(80, 85, 105, 140), 1))
        p.drawRoundedRect(rate_rect, 3, 3)
        p.setPen(QColor(220, 225, 235) if kind in ("spikes", "model") else QColor(170, 175, 190))
        p.drawText(rate_rect, Qt.AlignCenter, rate_txt)

        # actions of a lab target (or a sandbox signal)
        actions = []
        if card.get("lab"):
            actions = [("card_stim", "Stimulate"),
                       ("card_silence", "Release" if card["lab"] in self._asked_silenced() else "Silence")]
        elif card.get("sandbox"):
            actions = [("card_sandbox", {"reward": "Send reward", "punish": "Send punishment",
                                         "food": "Show odor"}[card["sandbox"]])]
        ax = cx + cw - 12
        for bid, text in reversed(actions):
            bw = fm.horizontalAdvance(text) + 16
            ax -= bw
            rect = QRect(ax, by, bw, 18)
            self._btn_rects[bid] = rect
            accent = QColor(255, 140, 80) if bid == "card_stim" else QColor(148, 163, 184)
            p.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 40))
            p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 190), 1))
            p.drawRoundedRect(rect, 4, 4)
            p.setPen(QColor(240, 240, 245))
            p.drawText(rect, Qt.AlignCenter, text)
            ax -= 6

        p.setFont(f["role"])
        p.setPen(QColor(180, 185, 195))
        p.drawText(QRect(cx + 12, cy + 70, cw - 24, 16), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(f["role"]).elidedText(card.get("role", ""), Qt.ElideRight, cw - 24))
        p.setFont(f["small"])
        p.setPen(QColor(130, 135, 145))
        p.drawText(QRect(cx + 12, cy + 88, cw - 24, 14), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(f["small"]).elidedText(card.get("partners", ""), Qt.ElideRight, cw - 24))
        if card.get("lab"):
            label, effect = LAB_TARGETS[card["lab"]]
            note = f"Lab group {card['lab']}: light drives it at {LAB_FULL_HZ.get(card['lab'], 0):.0f} Hz (known: {effect})"
        elif card.get("sandbox"):
            note = "Designed learning signal (not spiking neurons): teaches the cues present now"
        else:
            note = ""
        if note:
            p.setPen(QColor(150, 155, 170))
            p.drawText(QRect(cx + 12, cy + 102, cw - 24, 14), Qt.AlignLeft | Qt.AlignVCenter,
                       QFontMetrics(f["small"]).elidedText(note, Qt.ElideRight, cw - 24))
        p.restore()

    # ------------------------------------------------------------------ HUD
    def _fonts(self) -> dict[str, QFont]:
        """The HUD's fonts, made once (a QFont per text per frame cost a few ms)."""
        if self._fonts_cache is None:
            self._fonts_cache = {
                "title": QFont("Segoe UI", 9, QFont.Bold), "small": QFont("Segoe UI", 7),
                "btn": QFont("Segoe UI", 8, QFont.DemiBold), "pill": QFont("Segoe UI", 7, QFont.DemiBold),
                "mono": QFont("Cascadia Mono", 9, QFont.DemiBold), "tag": QFont("Cascadia Mono", 7, QFont.Bold),
                "cat": QFont("Segoe UI", 7, QFont.Bold), "name": QFont("Segoe UI", 10, QFont.Bold),
                "badge": QFont("Segoe UI", 7, QFont.DemiBold), "role": QFont("Segoe UI", 8),
                "status": QFont("Segoe UI", 8, QFont.DemiBold), "bar": QFont("Segoe UI", 8),
            }
        return self._fonts_cache

    def _hud_items(self) -> dict[str, list[tuple[str, str, str]]]:
        """The HUD's controls as (id, text, kind) per group; short labels on a narrow view."""
        compact = self.width() < 760
        camera = [("iso", "Iso" if compact else "3D Iso"), ("dorsal", "Top"), ("ant", "Front"),
                  ("rot", ("Spin" if compact else "Auto Rotate") if not self.auto_rotate else "Stop"),
                  ("reset", "Reset"), ("help", "?")]
        modes = [(f"mode_{MODE_VALENCE}", "Valence" if compact else "Mental Map"),
                 (f"mode_{MODE_PLASTICITY}", "Plasticity"), (f"mode_{MODE_ANATOMY}", "Anatomy"),
                 (f"mode_{MODE_ACTIVITY}", "Activity" if compact else "Live Activity")]
        layers = [("layer_mb", "Mushroom" if compact else "Mushroom Body"), ("layer_cx", "CX"),
                  ("layer_visual", "Vision"), ("layer_olfactory", "Odor"), ("layer_motor", "Motor"),
                  ("layer_hulls", "Hulls" if compact else "Neuropils"),
                  ("layer_wireframe", "Cage" if compact else "Wireframe")]
        return {"camera": [(i, t, "btn") for i, t in camera], "modes": [(i, t, "btn") for i, t in modes],
                "layers": [(i, t, "pill") for i, t in layers]}

    def _layout_hud(self) -> dict[str, QRect]:
        """Rectangles of every HUD control for the widget size. Rows of a left and a right group; when the two do not
        fit side by side the right group gets a row of its own, so no control ever covers another at any width. The
        tools stand in a column at the left edge, under the rows."""
        f = self._fonts()
        fm = {"btn": QFontMetrics(f["btn"]), "pill": QFontMetrics(f["pill"])}
        items = self._hud_items()
        w = self.width()
        margin, gap, row_h = 12, 5, 22

        def widths(group):
            return [(i, fm[k].horizontalAdvance(t) + (16 if k == "btn" else 12)) for i, t, k in group]

        def span(ws):
            return sum(bw for _, bw in ws) + gap * max(0, len(ws) - 1)

        rects: dict[str, QRect] = {}

        def place(ws, x, y):
            for i, bw in ws:
                rects[i] = QRect(int(x), y, bw, row_h)
                x += bw + gap

        y = 8
        title_w = QFontMetrics(f["title"]).horizontalAdvance("NEURAL MAP") + 8
        for left, right in ((None, widths(items["camera"])), (widths(items["layers"]), widths(items["modes"]))):
            lw = title_w if left is None else span(left)
            rw = span(right)
            if left is not None and lw > w - 2 * margin:      # the pills alone do not fit: two rows of them
                half = (len(left) + 1) // 2
                place(left[:half], margin, y)
                y += row_h + 5
                left, lw = left[half:], span(left[half:])
            if margin + lw + 16 + rw + margin > w:
                if left is not None:
                    place(left, margin, y)
                else:
                    rects["title"] = QRect(margin, y, title_w, row_h)
                y += row_h + 5
                place(right, max(margin, w - margin - rw), y)
            else:
                if left is not None:
                    place(left, margin, y)
                else:
                    rects["title"] = QRect(margin, y, title_w, row_h)
                place(right, w - margin - rw, y)
            y += row_h + 5
        # the tools: a column at the left edge
        tool_w = max(fm["btn"].horizontalAdvance(t) for _, t in TOOLS) + 18
        ty = y + 4
        for tid, _ in TOOLS:
            rects[f"tool_{tid}"] = QRect(margin, ty, tool_w, row_h + 2)
            ty += row_h + 6
        self._hud_bottom = ty
        return rects

    def _draw_hud(self, p: QPainter):
        """Draw the controls, the telemetry bar, the inspector card, the status line and the hover label."""
        self._btn_rects = {}
        w, h = self.width(), self.height()
        f = self._fonts()
        rects = self._layout_hud()
        items = {i: (t, k) for group in self._hud_items().values() for i, t, k in group}
        p.save()

        # title
        if "title" in rects:
            p.setFont(f["title"])
            p.setPen(QColor(236, 236, 241, 220))
            p.drawText(rects["title"], Qt.AlignLeft | Qt.AlignVCenter, "NEURAL MAP")

        layer_cols = {"layer_mb": QColor(245, 158, 11), "layer_cx": QColor(56, 189, 248),
                      "layer_visual": QColor(249, 115, 22), "layer_olfactory": QColor(16, 185, 129),
                      "layer_motor": QColor(239, 68, 68), "layer_hulls": QColor(168, 85, 247),
                      "layer_wireframe": QColor(148, 163, 184)}
        for bid, (text, kind) in items.items():
            rect = rects.get(bid)
            if rect is None:
                continue
            self._btn_rects[bid] = rect
            if bid.startswith("layer_"):
                on = getattr(self, bid)
                acc = layer_cols[bid]
                bg = QColor(acc.red(), acc.green(), acc.blue(), 55) if on else QColor(24, 24, 30, 160)
                border = QColor(acc.red(), acc.green(), acc.blue(), 180) if on else QColor(55, 55, 65, 130)
                fg = QColor(255, 255, 255, 240) if on else QColor(120, 120, 130, 180)
                radius, font = 4, f["pill"]
            else:
                on = ((bid == "rot" and self.auto_rotate) or (bid == "help" and self._show_help)
                      or bid == f"mode_{self.view_mode}")
                bg = QColor(76, 201, 240, 80) if on else QColor(28, 28, 33, 200)
                border = QColor(76, 201, 240, 190) if on else QColor(58, 58, 66, 160)
                fg = QColor(255, 255, 255, 245) if on else QColor(200, 200, 210, 220)
                radius, font = 5, f["btn"]
            p.setBrush(bg)
            p.setPen(QPen(border, 1))
            p.drawRoundedRect(rect, radius, radius)
            p.setFont(font)
            p.setPen(fg)
            p.drawText(rect, Qt.AlignCenter, text)

        # tools
        tool_cols = {"camera": QColor(148, 163, 184), "opto": QColor(255, 120, 60), "silence": QColor(203, 213, 225)}
        p.setFont(f["btn"])
        for tid, text in TOOLS:
            rect = rects[f"tool_{tid}"]
            self._btn_rects[f"tool_{tid}"] = rect
            on = self.tool_mode == tid
            acc = tool_cols[tid]
            p.setBrush(QColor(acc.red(), acc.green(), acc.blue(), 70) if on else QColor(24, 24, 28, 190))
            p.setPen(QPen(QColor(acc.red(), acc.green(), acc.blue(), 220) if on else QColor(50, 50, 58, 160), 1))
            p.drawRoundedRect(rect, 5, 5)
            p.setPen(QColor(255, 255, 255, 250) if on else QColor(150, 150, 160, 210))
            p.drawText(rect, Qt.AlignCenter, text)

        # silenced groups: one chip each (up to three) and "release all"
        silenced = list(self._asked_silenced())
        if silenced:
            cy = self._hud_bottom + 4
            p.setFont(f["pill"])
            fmp = QFontMetrics(f["pill"])
            shown = silenced[:3]
            chips = [(f"chip_{g}", f"{g} ⊘  ×") for g in shown]
            if len(silenced) > 3:
                chips.append(("chip_more", f"+{len(silenced) - 3}"))
            chips.append(("release_all", "Release all"))
            for cid, text in chips:
                rect = QRect(12, cy, fmp.horizontalAdvance(text) + 14, 19)
                if cid != "chip_more":
                    self._btn_rects[cid] = rect
                p.setBrush(QColor(71, 85, 105, 120))
                p.setPen(QPen(QColor(148, 163, 184, 200), 1))
                p.drawRoundedRect(rect, 4, 4)
                p.setPen(QColor(226, 232, 240))
                p.drawText(rect, Qt.AlignCenter, text)
                cy += 23

        # bottom: telemetry bar, then the card above it on the left and the status line on the right
        self._draw_bottom_telemetry(p, w, h, f["bar"], f["mono"])
        if self._inspected_neuron is not None:
            self._draw_inspector_card(p, w, h)
        self._draw_status_line(p, w, h)
        self._draw_hover_label(p, w, h)
        p.restore()

    def _draw_status_line(self, p: QPainter, w: int, h: int) -> None:
        """What the last lab action did: sent, applied by the worker, and what the fly did then (fades after 8 s)."""
        text, t0, warn = self._status
        if not text:
            return
        age = time.perf_counter() - t0
        if age > 8.0:
            self._status = ("", 0.0, False)
            return
        fade = 1.0 if age < 6.0 else max(0.0, (8.0 - age) / 2.0)
        f = self._fonts()["status"]
        fm = QFontMetrics(f)
        left = 16 + (self._card_rect(w, h).width() + 12 if self._inspected_neuron is not None else 0)
        avail = max(120, w - 16 - left)
        shown = fm.elidedText(text, Qt.ElideRight, avail - 20)
        tw = fm.horizontalAdvance(shown) + 20
        rect = QRect(w - 16 - tw, h - 66 - 8 - 24, tw, 24)
        acc = QColor(245, 158, 11) if warn else QColor(76, 201, 240)
        p.setBrush(QColor(12, 14, 20, int(225 * fade)))
        p.setPen(QPen(QColor(acc.red(), acc.green(), acc.blue(), int(200 * fade)), 1))
        p.drawRoundedRect(rect, 6, 6)
        p.setFont(f)
        p.setPen(QColor(235, 240, 248, int(255 * fade)))
        p.drawText(rect, Qt.AlignCenter, shown)

    def _draw_hover_label(self, p: QPainter, w: int, h: int) -> None:
        """A short label next to the cursor: a HUD button's meaning, or the neuron / target under it."""
        text = self._hover_text
        pos = self._hover_pos
        if not text or pos is None or self._press is not None:
            return
        f = self._fonts()["small"]
        fm = QFontMetrics(f)
        shown = fm.elidedText(text, Qt.ElideRight, max(80, w - 32))
        tw, th = fm.horizontalAdvance(shown) + 14, 20
        x = min(max(4, pos.x() + 14), w - tw - 4)
        y = pos.y() + 18 if pos.y() + 18 + th < h - 4 else pos.y() - 18 - th
        rect = QRect(x, y, tw, th)
        p.setBrush(QColor(20, 20, 26, 235))
        p.setPen(QPen(QColor(90, 95, 110, 200), 1))
        p.drawRoundedRect(rect, 4, 4)
        p.setFont(f)
        p.setPen(QColor(235, 235, 242))
        p.drawText(rect, Qt.AlignCenter, shown)

    def _draw_bottom_telemetry(self, p: QPainter, w: int, h: int, font_small: QFont, font_mono: QFont):
        """The bottom telemetry card: sections placed by their measured widths, left to right, as many as fit."""
        bar_h = 52
        bar_rect = QRect(16, h - bar_h - 14, w - 32, bar_h)
        p.setBrush(QBrush(QColor(18, 18, 23, 220)))
        p.setPen(QPen(QColor(45, 45, 53, 180), 1))
        p.drawRoundedRect(bar_rect, 10, 10)

        fm_l, fm_m = QFontMetrics(font_small), QFontMetrics(font_mono)
        vy = bar_rect.y() + 18                          # label baseline
        my = vy + 6                                     # meter top
        val = max(-1.0, min(1.0, self.valence))
        grey = QColor(148, 163, 184)
        state = {"fly": "fly", "retreat": "back", "walk": "walk", "groom": "groom", "freeze": "freeze",
                 "stand": "stand"}.get(self.fly_state, "")

        def meter(x, width, frac, col):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(28, 28, 33))
            p.drawRoundedRect(x, my, width, 10, 4, 4)
            fill = int(max(0.0, min(1.0, frac)) * width)
            if fill > 0:
                p.setBrush(col)
                p.drawRoundedRect(x, my, fill, 10, 4, 4)

        def valence(x):
            mid = x + 85
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(28, 28, 33))
            p.drawRoundedRect(x, my, 170, 10, 5, 5)
            p.setPen(QPen(QColor(100, 100, 110, 120), 1))
            p.drawLine(mid, my, mid, my + 10)
            fill = int(abs(val) * 85)
            if fill:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(16, 185, 129, 230) if val > 0 else QColor(239, 68, 68, 230))
                p.drawRoundedRect(mid if val > 0 else mid - fill, my, fill, 10, 3, 3)
            p.setFont(font_mono)
            p.setPen(QColor(16, 185, 129) if val > 0.02 else QColor(239, 68, 68) if val < -0.02 else QColor(220, 220, 230))
            p.drawText(x + 178, my + 9, f"{val:+.2f}")

        def dopamine(x):
            meter(x, 70, self.pam, QColor(245, 158, 11, 230))
            meter(x + 78, 70, self.ppl1, QColor(239, 68, 68, 230))

        def text(x, t, col):
            p.setFont(font_mono)
            p.setPen(col)
            p.drawText(x, my + 9, t)

        def motor(x):
            p.setFont(font_mono)
            for t, on, col in ((f"GF {self.rate_gf:.0f}", self.rate_gf > 2.0, QColor(239, 68, 68)),
                               (f"W {self.rate_walk:.0f}", self.rate_walk > 2.0, QColor(34, 197, 94)),
                               (f"M {self.rate_mdn:.0f}", self.rate_mdn > 2.0, QColor(217, 70, 239))):
                p.setPen(col if on else grey)
                p.drawText(x, my + 9, t)
                x += fm_m.horizontalAdvance(t) + 10

        mbon = f"+{self.mbon_app:.2f} / -{self.mbon_av:.2f}"
        motor_w = sum(fm_m.horizontalAdvance(t) + 10 for t in (f"GF {self.rate_gf:.0f}", f"W {self.rate_walk:.0f}",
                                                               f"M {self.rate_mdn:.0f}"))
        sections = [
            ("LEARNED VALENCE", 178 + fm_m.horizontalAdvance("+0.00"), valence),
            ("DOPAMINE  PAM / PPL1", 148, dopamine),
            ("ACTIVE KC", fm_m.horizontalAdvance(self._kc_hud_text()), lambda x: text(x, self._kc_hud_text(),
                                                                                      QColor(236, 236, 241))),
            ("MOTOR DNs (Hz)" + (f"  · {state}" if state else ""), motor_w, motor),
            ("MBON BALANCE", fm_m.horizontalAdvance(mbon), lambda x: text(x, mbon, QColor(56, 189, 248))),
        ]
        x = bar_rect.x() + 16
        right = bar_rect.right() - 12
        for label, content_w, draw in sections:
            sw = max(fm_l.horizontalAdvance(label), content_w)
            if x + sw > right:
                break
            p.setFont(font_small)
            p.setPen(QColor(139, 139, 150, 220))
            p.drawText(x, vy, label)
            draw(x)
            x += sw + 28

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
                "4. MOUSE, CAMERA & LAB",
                [
                    (QColor(140, 145, 160), "Drag: left orbits, right pans", "Wheel zooms; Home or double-click resets."),
                    (QColor(148, 163, 184), "Explore: click a neuron", "Pins its card (real rate, actions)."),
                    (QColor(255, 120, 60), "Stimulate: press a target", "Light on its neurons while held (0.3 s click)."),
                    (QColor(203, 213, 225), "Silence: click a target", "Its neurons cannot spike; click again to release."),
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
    _BTN_TIPS = {
        "help": "Guide to the map, the circuits and the lab", "iso": "3D isometric view", "dorsal": "Top (dorsal) view",
        "ant": "Front (anterior) view", "rot": "Rotate the brain slowly", "reset": "Reset the camera (Home)",
        "tool_camera": "Explore (1): drag to orbit, click a neuron to inspect it",
        "tool_opto": "Stimulate (2): press a target, its neurons fire while the button is held",
        "tool_silence": "Silence (3): click a target, its neurons cannot spike until released",
        "mode_valence": "Learned valence: green = appetitive, red = aversive",
        "mode_plasticity": "How much each Kenyon cell's output synapses changed |Δw|",
        "mode_anatomy": "Mushroom-body lobes: γ, α/β, α'/β'", "mode_activity": "Live spikes only",
        "layer_mb": "Mushroom body: Kenyon cells, output and dopamine neurons", "layer_cx": "Central complex compass",
        "layer_visual": "Visual projection neurons (LC4, LPLC2, LPC1, LC10)", "layer_olfactory": "Antennal lobe PNs",
        "layer_motor": "Descending neurons (GF, MDN, DNp09, DNa02, aDN)", "layer_hulls": "Neuropil envelopes",
        "layer_wireframe": "Brain outline", "release_all": "Release every silenced group",
        "close_inspector": "Close the card", "card_stim": "A light pulse on this group",
        "card_silence": "Silence or release this group", "card_sandbox": "Send this designed learning signal",
    }

    def _hud_hit(self, pos: QPoint) -> str | None:
        """The HUD control under pos, topmost (drawn last) first."""
        for bid, rect in reversed(list(self._btn_rects.items())):
            if rect.contains(pos):
                return bid
        return None

    def _on_hud(self, bid: str) -> None:
        if bid == "help":
            self._show_help = not self._show_help
        elif bid == "close_help":
            self._show_help = False
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
            self.set_view_mode(bid[5:])
        elif bid.startswith("layer_"):
            self.toggle_layer(bid[6:])
        elif bid.startswith("tool_"):
            self.set_tool_mode(bid[5:])
        elif bid == "close_inspector":
            self._inspected_neuron = None
            self._inspected_pinned = False
        elif bid == "release_all":
            self.release_all()
        elif bid.startswith("chip_"):
            self.toggle_silence(bid[5:])
        elif bid == "card_stim" and self._inspected_neuron is not None:
            self.stimulate_target(self._inspected_neuron["lab"])
        elif bid == "card_silence" and self._inspected_neuron is not None:
            self.toggle_silence(self._inspected_neuron["lab"])
        elif bid == "card_sandbox" and self._inspected_neuron is not None:
            self._sandbox(self._inspected_neuron["sandbox"])
        self.update()

    def mousePressEvent(self, event):
        pos = event.pos()
        bid = self._hud_hit(pos)
        if bid is not None:
            self._on_hud(bid)
            return
        if self._show_help:                     # a click outside the guide closes it, a click inside does nothing
            if not self._help_card_rect.contains(pos):
                self._show_help = False
                self.update()
            return
        self._press = (pos, event.button(), time.perf_counter())
        self._dragging = False
        self._last_mouse_pos = pos
        self._is_panning = event.button() == Qt.RightButton or bool(event.modifiers() & Qt.ShiftModifier)
        if self.tool_mode == "opto" and event.button() == Qt.LeftButton:
            target, _ = self._target_at(pos)
            if target in LAB_TARGETS and self.stimulate_target(target, hold=True):
                self._held = (target, time.perf_counter())       # light on while the button is down

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self._press is not None and self._held is None:
            if not self._dragging and (pos - self._press[0]).manhattanLength() > 4:
                self._dragging = True
            if self._dragging and self._last_mouse_pos is not None:
                dx = pos.x() - self._last_mouse_pos.x()
                dy = pos.y() - self._last_mouse_pos.y()
                if self._is_panning:
                    self.pan_x += dx
                    self.pan_y += dy
                else:
                    self.yaw += dx * 0.008
                    self.pitch = max(-1.55, min(1.55, self.pitch + dy * 0.008))
                    self._cam_R = None
                self._last_mouse_pos = pos
                self.update()
            return
        if self._press is None:
            self._hover_pos = pos
            self._schedule_hover()

    def mouseReleaseEvent(self, event):
        press, dragged = self._press, self._dragging
        self._press = None
        self._dragging = False
        self._last_mouse_pos = None
        self._is_panning = False
        if self._held is not None:
            self._release_held()
            return
        if press is None or dragged:
            return
        if press[1] == Qt.LeftButton:
            self._on_click(press[0])
        elif press[1] == Qt.RightButton:
            self._context_menu(press[0])

    def _on_click(self, pos: QPoint) -> None:
        """A click (no drag). Nothing is ever stimulated on a miss."""
        if self.tool_mode == "camera":
            card = self._find_neuron_at(pos, threshold_px=22.0)
            if card is not None:
                self._inspected_neuron, self._inspected_pinned = card, True
            elif self._inspected_pinned:
                self._inspected_neuron, self._inspected_pinned = None, False
        else:
            target, _ = self._target_at(pos)
            if target is None:
                self._set_status("Nothing here: the lab acts on the marked targets (descending and visual neurons)")
            elif self.tool_mode == "silence":
                self.toggle_silence(target)
            elif target in SANDBOX_TARGETS:
                self._sandbox(target)
        self.update()

    def _context_menu(self, pos: QPoint) -> None:
        """Right click without a drag: the actions for the neuron under the cursor."""
        target, card = self._target_at(pos)
        if card is None:
            card = self._find_neuron_at(pos, threshold_px=22.0)
        menu = QMenu(self)
        if target in LAB_TARGETS:
            label = LAB_TARGETS[target][0]
            menu.addAction(f"Stimulate {label} (0.3 s)", lambda: self.stimulate_target(target))
            silenced = target in self._asked_silenced()
            menu.addAction(f"{'Release' if silenced else 'Silence'} {label}", lambda: self.toggle_silence(target))
        elif target in SANDBOX_TARGETS:
            menu.addAction(f"Send: {SANDBOX_TARGETS[target]}", lambda: self._sandbox(target))
        if card is not None:
            menu.addAction("Inspect", lambda: self._pin(card))
        if self._asked_silenced():
            menu.addSeparator()
            menu.addAction("Release every silenced group", self.release_all)
        menu.addSeparator()
        menu.addAction("Reset camera", self.reset_camera)
        menu.exec(self.mapToGlobal(pos))

    def _pin(self, card: dict) -> None:
        self._inspected_neuron, self._inspected_pinned = card, True
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 0.89
        self.zoom = max(60.0, min(900.0, self.zoom * factor))
        self.update()

    def mouseDoubleClickEvent(self, event):
        """Reset the camera, in Explore only (quick clicks on a lab target must not move the view)."""
        if self.tool_mode == "camera" and self._hud_hit(event.pos()) is None:
            self.reset_camera()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Home:
            self.reset_camera()
        elif key in (Qt.Key_1, Qt.Key_2, Qt.Key_3):
            self.set_tool_mode(TOOLS[key - Qt.Key_1][0])
        elif key == Qt.Key_Escape:
            self._show_help = False
            self._inspected_neuron, self._inspected_pinned = None, False
            self.update()
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._hover_pos = None
        self._hover_text, self._hover_target, self._hover_card = "", None, None
        self.update()

    def _schedule_hover(self) -> None:
        """Hover work runs once per event-loop pass, on the last mouse position (moves arrive faster than frames)."""
        if not self._hover_scheduled:
            self._hover_scheduled = True
            QTimer.singleShot(0, self._do_hover)

    def _do_hover(self) -> None:
        self._hover_scheduled = False
        if self._hover_pos is not None:
            self._check_hover(self._hover_pos)

    def _check_hover(self, pos: QPoint):
        """What is under the cursor: a HUD control's meaning, else the target / neuron (with its group ringed)."""
        self._hover_pos = pos
        found, target, card = "", None, None
        bid = self._hud_hit(pos)
        if bid is not None:
            found = self._BTN_TIPS.get(bid, "")
            if bid.startswith("chip_"):
                found = f"Release {bid[5:]}"
        elif not self._show_help:
            if self.tool_mode in ("opto", "silence"):
                target, card = self._target_at(pos)
                if target in LAB_TARGETS:
                    label = LAB_TARGETS[target][0]
                    if target in self._lab.get("available", ()) or not self._lab.get("available"):
                        verb = ("press: light on" if self.tool_mode == "opto"
                                else "click: release" if target in self._asked_silenced() else "click: silence")
                        found = f"{label} · {card.get('rate_hz', '')} · {verb}"
                    else:
                        found = f"{label}: not in the running circuit"
                elif target in SANDBOX_TARGETS:
                    found = (f"{SANDBOX_TARGETS[target]} · click: send" if self.tool_mode == "opto"
                             else f"{SANDBOX_TARGETS[target]}: a designed signal, nothing to silence")
            else:
                card = self._find_neuron_at(pos, threshold_px=18.0)
                if card is not None:
                    found = f"{card['name']} · {card.get('rate_hz', '')}"
        changed = found != self._hover_text or target != self._hover_target
        self._hover_text, self._hover_target, self._hover_card = found, target, card
        if self.tool_mode != "camera":
            self.setCursor(Qt.PointingHandCursor if target else Qt.ArrowCursor)
        if changed or found:
            self.update()
