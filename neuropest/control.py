"""Control window: live readouts and settings, grouped into categories with a top navigation bar (design_handoff)."""
from __future__ import annotations

import importlib.util
import json
import threading

import numpy as np

from PySide6.QtCore import QRect, QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter

from . import eyeview
from .mb_view3d import MushroomBody3DView
from .mushroom import wiring_usable
from .paths import CACHE, EYE, FIELD, MUSHROOM, SKIN_THUMBNAILS, TIERS
from .preferences import Preferences
from .render import AVAILABLE_SKINS, SKIN_METADATA
from .runner import GPU_AUTO_MIN_NEURONS, EngineConfig, list_gpus, mb_wiring_from_env, pick_gpu

from .strips import CircuitView, EthoStrip, History, StripRow, legend
from .theme import (
    ACCENT, BORDER, BORDER_BTN, BORDER_HI, ERROR, FAINT, MUTED, MUTED_2, SEL_BG, STATE_STYLE, TEXT, TEXT_STRONG, fs,
)

from .visionparams import VisionParams

# circuit sizes offered in the UI (neurons). FlyWire sizes are tiers measured by tools/fidelity.py.
FLYWIRE_SIZES = [2_000, 5_000, 10_000, 15_000, 20_000, 50_000, 138_639]
FLYWIRE_DEFAULT = 15_000
TOY_SIZES = [146, 500, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 139_000]
SCAN_ITEM = "Scan for GPUs…"
DTS = [("Precise (0.1 ms)", 0.1), ("Balanced (0.5 ms)", 0.5), ("Fast (1 ms)", 1.0)]


def load_tiers() -> dict[int, dict]:
    try:
        return {t["n"]: t for t in json.loads(TIERS.read_text())["tiers"]}
    except (OSError, ValueError, KeyError):
        return {}


def load_tier_machine() -> str:
    """Machine tools/fidelity.py measured the tier speeds on (its `machine` field), for the tooltip."""
    try:
        return str(json.loads(TIERS.read_text()).get("machine") or "unknown machine")
    except (OSError, ValueError):
        return "unknown machine"


def _label(text: str = "", name: str | None = None, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    if name in ("Section", "Axis"):                 # mono caps with 1 px of letter spacing (the stylesheet has no such property)
        f = lab.font()
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
        lab.setFont(f)
    lab.setWordWrap(wrap)
    return lab


class Card(QFrame):
    """Outlined panel with a small mono upper-case heading; add rows to `.body`."""

    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(8)
        lay.addWidget(_label(title.upper(), "Section"))
        self.body = lay


class Stat(QWidget):
    """Big number with a small caption under it."""

    def __init__(self, name: str, tip: str = ""):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.value = _label("–", "StatValue")
        lay.addWidget(self.value)
        lay.addWidget(_label(name, "StatName"))
        if tip:
            self.setToolTip(tip)

    def set(self, text: str):
        self.value.setText(text)


def slider_row(target: Card | QLayout | QWidget, name: str, hint: str, slider: QSlider, fmt) -> QLabel:
    """Name and live value on one line, the slider bar under it, an optional muted hint below."""
    head = QHBoxLayout()
    head.addWidget(_label(name))
    head.addStretch(1)
    value = _label("", "Value")
    head.addWidget(value)
    if isinstance(target, Card):
        body = target.body
    elif isinstance(target, QLayout):
        body = target
    elif hasattr(target, "layout") and target.layout() is not None:
        body = target.layout()
    else:
        body = target
    body.addLayout(head)
    body.addWidget(slider)
    if hint:
        body.addWidget(_label(hint, "Faint", wrap=True))
    slider.valueChanged.connect(lambda v: value.setText(fmt(v)))
    value.setText(fmt(slider.value()))
    return value


def labeled(card: Card, name: str, widget: QWidget) -> None:
    card.body.addWidget(_label(name, "Muted"))
    card.body.addWidget(widget)


class EyePreview(QWidget):
    """Picture of what the visual front end feeds the circuit (eyeview.py). Holds the last rendered frame."""

    W, H = 440, 200

    def __init__(self):
        super().__init__()
        self.mode = 0
        self.snap = None
        self.layout_ = None
        self.message = ""
        self._img: QImage | None = None
        self._buf = None
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self.setMinimumSize(self.W // 2, self.H // 2)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return int(w * self.H / self.W)

    def sizeHint(self) -> QSize:
        return QSize(self.W, self.H)

    def set_message(self, text: str) -> None:
        if text == self.message and self._img is None:
            return
        self.snap, self._img, self.message = None, None, text
        self.update()

    def set_frame(self, snap, mode: int) -> None:
        """Render `snap` (eyeview.Snapshot) in the tab `mode`."""
        if self.layout_ is None:
            self.layout_ = eyeview.EyeLayout.load()
        if self.layout_ is None:
            self.set_message("No eye data (data/circuits/field.npz).")
            return
        self.snap, self.mode, self.message = snap, mode, ""
        self._buf = np.ascontiguousarray(eyeview.RENDERERS[mode](snap, self.layout_, self.W, self.H))
        self._img = QImage(self._buf.data, self.W, self.H, self.W * 3, QImage.Format_RGB888)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        r = self.rect()
        p.fillRect(r, QColor(14, 14, 17))
        if self._img is None:
            p.setPen(QColor(MUTED))
            p.drawText(r.adjusted(12, 8, -12, -8), Qt.AlignCenter | Qt.TextWordWrap, self.message)
            return
        p.drawImage(r, self._img)
        scale = r.width() / self.W
        f = QFont(p.font())
        f.setPixelSize(11)
        p.setFont(f)
        if self.mode == 2 and self.snap is not None:
            for i, name in enumerate(eyeview.GROUPS):
                x0, y0, tw, th = eyeview.tile_rect(i, self.W, self.H)
                p.setPen(QColor(*eyeview.GROUP_COLOURS[i]))
                p.drawText(QRect(int((x0 + 6) * scale), int(y0 * scale), int(tw * scale), int(15 * scale)),
                           Qt.AlignVCenter | Qt.AlignLeft, f"{name}  {eyeview.group_summary(self.snap, i)}")
        else:
            p.setPen(QColor(MUTED))
            note = ("left + right eye on one panorama - dashed cyan: horizon" if self.mode == 0
                    else "orange: expansion (looming) - cyan: small moving object")
            p.drawText(QRect(6, r.height() - 18, r.width() - 12, 16), Qt.AlignVCenter | Qt.AlignLeft, note)


class _ContentScrollArea(QScrollArea):
    """Vertical-only scroll area that constrains its wrapped content width to the viewport."""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.widget()
        if w is not None:
            w.setMaximumWidth(self.viewport().width())


class _CurrentPageStack(QStackedWidget):
    """A stack that is as tall as the page on show, not as the tallest page, so a short page does not scroll."""

    def sizeHint(self) -> QSize:
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()


class Control(QWidget):
    def __init__(self, overlay, runner, prefs: Preferences | None = None, auto_scan_gpus: bool = True):
        super().__init__()
        self.runner, self.overlay = runner, overlay
        self.prefs = prefs or getattr(overlay, "prefs", None) or Preferences.load()
        self.auto_scan_gpus = auto_scan_gpus
        self.tiers = load_tiers()
        self.tier_machine = load_tier_machine()
        self.setObjectName("Control")
        self.setWindowTitle("NeuroPest")
        self.resize(760, 560)
        self.setMinimumSize(620, 440)

        # Root layout: the navigation bar on top, the pages under it
        self.hist = History()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._setup_topbar(root)
        self._setup_content(root)

        self._load_sizes(runner.cfg.n)
        self._debounce = QTimer(self, singleShot=True, interval=500, timeout=self._apply)
        self.circ.currentIndexChanged.connect(self._circuit_changed)
        self.size.valueChanged.connect(self._size_moved)
        self.dt.currentIndexChanged.connect(self._on_dt_changed)
        self.hw.currentIndexChanged.connect(self._on_hw_changed)
        self._t = QTimer(self, timeout=self._on_refresh_timer, interval=250)
        self._t.start()
        self._refresh()

    def _on_refresh_timer(self):
        if self.isVisible():
            self._refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh()

    # ------------------------------------------------------------------ navigation
    def _setup_topbar(self, parent_layout: QVBoxLayout):
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(46)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(24, 0, 24, 0)
        lay.setSpacing(2)

        lay.addWidget(_label("NeuroPest", "Brand"))
        lay.addSpacing(20)

        # Short tab names; the pages keep their content and headings
        self.nav_buttons: list[QPushButton] = []
        nav_items = ["Circuit", "Behaviour", "Senses", "Odor", "Hardware", "Look", "Memory"]
        for idx, title in enumerate(nav_items):
            btn = QPushButton(title)
            btn.setObjectName("NavBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("active", "true" if idx == 0 else "false")
            btn.clicked.connect(lambda _, i=idx: self._switch_tab(i))
            self.nav_buttons.append(btn)
            lay.addWidget(btn)

        lay.addStretch(1)
        # the current behaviour state, on the right
        self.pill = QLabel()
        self.pill.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.pill)

        parent_layout.addWidget(bar)

    def _switch_tab(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self.stack.updateGeometry()
        for i, btn in enumerate(self.nav_buttons):
            active_str = "true" if i == idx else "false"
            btn.setProperty("active", active_str)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ------------------------------------------------------------------ content
    def _setup_content(self, parent_layout: QVBoxLayout):
        scroll = _ContentScrollArea(widgetResizable=True, frameShape=QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_wrap = QWidget()
        scroll.setWidget(content_wrap)

        content_lay = QVBoxLayout(content_wrap)
        content_lay.setContentsMargins(24, 12, 24, 12)
        content_lay.setSpacing(10)

        self.stack = _CurrentPageStack()

        # Page 0: Live Telemetry
        p0 = QWidget()
        p0_lay = QVBoxLayout(p0)
        p0_lay.setContentsMargins(0, 0, 0, 0)
        p0_lay.setSpacing(8)
        self._build_live_page(p0_lay)
        p0_lay.addStretch(1)
        self.stack.addWidget(p0)

        # Page 1: Behaviour
        p1 = QWidget()
        p1_lay = QVBoxLayout(p1)
        p1_lay.setContentsMargins(0, 0, 0, 0)
        p1_lay.setSpacing(10)
        self._build_behaviour_page(p1_lay)
        p1_lay.addStretch(1)
        self.stack.addWidget(p1)

        # Page 2: Visual Input
        p2 = QWidget()
        p2_lay = QVBoxLayout(p2)
        p2_lay.setContentsMargins(0, 0, 0, 0)
        p2_lay.setSpacing(10)
        self._build_vision_page(p2_lay)
        p2_lay.addStretch(1)
        self.stack.addWidget(p2)

        # Page 3: Pheromones & Odor
        p3 = QWidget()
        p3_lay = QVBoxLayout(p3)
        p3_lay.setContentsMargins(0, 0, 0, 0)
        p3_lay.setSpacing(10)
        self._build_pheromone_page(p3_lay)
        p3_lay.addStretch(1)
        self.stack.addWidget(p3)

        # Page 4: Circuit & Hardware
        p4 = QWidget()
        p4_lay = QVBoxLayout(p4)
        p4_lay.setContentsMargins(0, 0, 0, 0)
        p4_lay.setSpacing(10)
        self._build_circuit_page(p4_lay)
        p4_lay.addStretch(1)
        self.stack.addWidget(p4)

        # Page 5: Appearance
        p5 = QWidget()
        p5_lay = QVBoxLayout(p5)
        p5_lay.setContentsMargins(0, 0, 0, 0)
        p5_lay.setSpacing(10)
        self._build_view_page(p5_lay)
        p5_lay.addStretch(1)
        self.stack.addWidget(p5)

        # Page 6: Learning & Memory
        p6 = QWidget()
        p6_lay = QVBoxLayout(p6)
        p6_lay.setContentsMargins(0, 0, 0, 0)
        p6_lay.setSpacing(10)
        self._build_learning_page(p6_lay)
        p6_lay.addStretch(1)
        self.stack.addWidget(p6)

        content_lay.addWidget(self.stack)
        parent_layout.addWidget(scroll, 1)

    # ------------------------------------------------------------------ pages
    def _build_live_page(self, lay: QVBoxLayout):
        # The Circuit tab: 8 sensory -> 7 descending -> 6 behaviour nodes, each a kymograph strip of its last 22 s
        self.circuit = CircuitView()
        lay.addWidget(self.circuit, 0, Qt.AlignHCenter)

        # the numbers the nodes show (kept as readouts; the first four are the nodes' values, not drawn again)
        self.s_gf = Stat("Giant Fiber", "Firing rate of escape/takeoff command neuron")
        self.s_mdn = Stat("MDN", "Firing rate of backward walking command neurons")
        self.s_steer = Stat("Steering", "Right minus left DNa02 rate (positive: turns right)")
        self.s_rt = Stat("Real-time", "Ratio of simulation speed to real time; below 1× means slow motion")
        self.s_cpu = Stat("CPU Usage", "Single-core CPU utilization of the engine process")
        self.s_spk = Stat("Spikes / s")
        self.s_mn9 = Stat("MN9 (proboscis)", "Firing rate of the proboscis motor neuron: food under the fly drives the "
                          "sugar receptor neurons, they drive MN9, and the fly feeds while it fires (hunger raises it)")
        for s_ in (self.s_gf, self.s_mdn, self.s_steer, self.s_mn9):
            s_.setParent(self)
            s_.hide()
        # the same numbers in the footer line (the stats above stay for the readouts and tooltips)
        for s_ in (self.s_rt, self.s_cpu, self.s_spk):
            s_.setParent(self)
            s_.hide()
        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        foot.addWidget(_label("EVERY STRIP SHARES ONE TIME AXIS", "Axis"))
        foot.addStretch(1)
        self.footer = _label("", "Axis")
        foot.addWidget(self.footer)
        lay.addLayout(foot)
        self.active = _label("", "Muted")
        self.active.hide()
        self.telemetry = _label("", "Faint")
        lay.addWidget(self.telemetry)
        self.warn = _label("", "Warn", wrap=True)
        self.warn.hide()
        lay.addWidget(self.warn)

    def _build_behaviour_page(self, outer: QVBoxLayout):
        # two columns: hunger and the ethogram on the left, touch and the advanced settings on the right
        cols = QHBoxLayout()
        cols.setSpacing(10)
        left, right = QVBoxLayout(), QVBoxLayout()
        left.setSpacing(10)
        right.setSpacing(10)
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)
        outer.addLayout(cols)
        left.addStretch(1)
        right.addStretch(1)
        # 1. Hunger & Metabolism (Primary Biological Drive)
        hunger_card = Card("Hunger & Metabolism")
        self.hunger_enable_box = QCheckBox("Hunger and Metabolism Simulation")
        self.hunger_enable_box.setToolTip(
            "When enabled, the fly expends metabolic energy and seeks food as hunger rises. "
            "When disabled, operates in constant hunger mode."
        )
        is_hunger_on = getattr(getattr(self.overlay, "metabolism", None), "enabled", True)
        self.hunger_enable_box.setChecked(is_hunger_on)
        self.hunger_enable_box.toggled.connect(self._on_hunger_enable_toggled)
        hunger_card.body.addWidget(self.hunger_enable_box)

        cur_mult = getattr(getattr(getattr(self.overlay, "metabolism", None), "cfg", None), "rate_mult", 1.0)
        h_slider = QSlider(Qt.Horizontal, minimum=20, maximum=300, value=int(cur_mult * 100))
        h_slider.valueChanged.connect(self._on_metabolic_rate_changed)
        slider_row(hunger_card, "Metabolic Rate",
                   "Controls how quickly hunger develops (Slow: ~5 min, Balanced: ~90 s, Fast: ~30 s).",
                   h_slider, lambda v: f"×{v/100:.1f}")

        status_row = QHBoxLayout()
        status_row.addWidget(_label("Current Status:"))
        status_row.addStretch(1)
        self.hunger_label = _label("–", "Value")
        status_row.addWidget(self.hunger_label)
        hunger_card.body.addLayout(status_row)
        self.hunger_strip = StripRow("Hunger", TEXT, unit="")
        hunger_card.body.addWidget(self.hunger_strip)

        btn_row = QHBoxLayout()
        self.btn_starve = QPushButton("Starve Fly")
        self.btn_starve.setToolTip("Instantly depletes the fly's energy, triggering food-seeking behaviour (foraging).")
        self.btn_starve.clicked.connect(self._on_starve_clicked)
        btn_row.addWidget(self.btn_starve)

        self.btn_feed = QPushButton("Feed Fly")
        self.btn_feed.setToolTip("Instantly satiates the fly; suppresses food tracking and transitions to resting/grooming.")
        self.btn_feed.clicked.connect(self._on_feed_clicked)
        btn_row.addWidget(self.btn_feed)

        hunger_card.body.addLayout(btn_row)
        left.insertWidget(left.count() - 1, hunger_card)

        etho_card = Card("Behaviour, last 22 s")
        self.etho = EthoStrip()
        etho_card.body.addWidget(self.etho)
        etho_card.body.addWidget(legend())
        left.insertWidget(left.count() - 1, etho_card)

        # 2. Touch & Grooming
        touch_card = Card("Touch & Grooming")
        self.touch_groom_box = QCheckBox("Groom on Cursor Touch (Grooming)")
        self.touch_groom_box.setToolTip(
            "When the cursor touches the fly, stimulates mechanosensory neurons (aDN1/aDN2) causing "
            "the fly to pause and groom its head. When disabled (default), cursor contact does not force grooming."
        )
        is_touch_on = getattr(self.overlay, "touch_groom_enabled", False) if self.overlay is not None else False
        self.touch_groom_box.setChecked(is_touch_on)
        self.touch_groom_box.toggled.connect(self._on_touch_groom_toggled)
        touch_card.body.addWidget(self.touch_groom_box)
        right.insertWidget(right.count() - 1, touch_card)

        # 3. Advanced Behaviour Settings (Collapsible Accordion Card)
        adv_card = Card("Advanced Behaviour Settings")
        self.adv_btn = QPushButton("Show Advanced Settings")
        self.adv_btn.setFlat(True)
        self.adv_btn.setCursor(Qt.PointingHandCursor)
        self.adv_btn.setStyleSheet(f"text-align: left; font-size: {fs(12.5)}; font-weight: 600; padding: 4px 0; "
                                   f"color: {TEXT_STRONG};")
        self.adv_btn.clicked.connect(self._toggle_adv_settings)
        adv_card.body.addWidget(self.adv_btn)

        self.adv_content = QWidget()
        adv_lay = QVBoxLayout(self.adv_content)
        adv_lay.setContentsMargins(0, 8, 0, 0)
        adv_lay.setSpacing(14)

        self.w_slider = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=int(self.runner.bias * 100))
        self.w_slider.valueChanged.connect(self._on_bias_changed)
        slider_row(adv_lay, "Activity (Spontaneous Walking)",
                   "How often the fly starts a walking bout on its own (drive on the DNp09/P9 walking command neurons); "
                   "hunger makes bouts longer and rests shorter, and a food odor sets a hungry fly walking whatever "
                   "this is. 0% = it walks only for odor, threats and searches. Default: 65%.",
                   self.w_slider, lambda v: f"{v}%")

        cur_skittish = getattr(self.runner, "skittish", 1.0)
        import math
        k_val = int(round(50.0 + 25.0 * math.log2(max(0.1, cur_skittish))))
        self.k_slider = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=max(0, min(100, k_val)))
        self.k_slider.valueChanged.connect(self._on_skittish_changed)
        slider_row(adv_lay, "Skittishness (Escape Sensitivity)",
                   "Sensitivity to approaching objects: multiplies retreat and escape thresholds. Default: ×1.00.",
                   self.k_slider, lambda v: f"×{2.0 ** ((v - 50) / 25.0):.2f}")

        cur_pain = getattr(self.runner, "wall_pain", 1.0)
        self.pain_slider = QSlider(Qt.Horizontal, minimum=0, maximum=200, value=int(cur_pain * 100))
        self.pain_slider.valueChanged.connect(self._on_wall_pain_changed)
        slider_row(adv_lay, "Edge Aversion (Nociceptive Wall Pain)",
                   "Pain/punishment signal sent to Mushroom Body (PPL1 dopamine) upon colliding with screen borders. Trains fly to avoid edges. Default: ×1.00.",
                   self.pain_slider, lambda v: f"×{v / 100.0:.2f}")

        self.btn_reset_adv = QPushButton("Reset to Default Parameters")
        self.btn_reset_adv.setToolTip("Resets locomotion drive (65%), skittishness (×1.00), and edge aversion (×1.00) to defaults.")
        self.btn_reset_adv.clicked.connect(self._reset_adv_settings)
        adv_lay.addWidget(self.btn_reset_adv)

        self.adv_content.setVisible(False)
        adv_card.body.addWidget(self.adv_content)
        right.insertWidget(right.count() - 1, adv_card)

    def _toggle_adv_settings(self):
        visible = not self.adv_content.isVisible()
        self.adv_content.setVisible(visible)
        self.adv_btn.setText(f"{'Hide' if visible else 'Show'} Advanced Settings")

    def _reset_adv_settings(self):
        self.w_slider.setValue(65)
        self.k_slider.setValue(50)
        self.pain_slider.setValue(100)

    def _on_bias_changed(self, v: int):
        val = v / 100.0
        setattr(self.runner, "bias", val)
        self.prefs.bias = val
        self.prefs.save()

    def _on_skittish_changed(self, v: int):
        val = 2.0 ** ((v - 50) / 25.0)
        setattr(self.runner, "skittish", val)
        self.prefs.skittish = val
        self.prefs.save()

    def _on_wall_pain_changed(self, v: int):
        val = v / 100.0
        setattr(self.runner, "wall_pain", val)
        self.prefs.wall_pain = val
        self.prefs.save()

    def _on_touch_groom_toggled(self, checked: bool):
        if self.overlay is not None:
            self.overlay.touch_groom_enabled = checked
        self.prefs.touch_groom_enabled = checked
        self.prefs.save()

    def _on_hunger_enable_toggled(self, checked: bool):
        if self.overlay and hasattr(self.overlay, "metabolism"):
            self.overlay.metabolism.enabled = checked
        self.prefs.hunger_enabled = checked
        self.prefs.save()

    def _on_metabolic_rate_changed(self, value: int):
        mult = value / 100.0
        if self.overlay and hasattr(self.overlay, "metabolism"):
            self.overlay.metabolism.cfg.rate_mult = mult
        self.prefs.metabolic_rate = mult
        self.prefs.save()

    def _on_starve_clicked(self):
        if self.overlay and hasattr(self.overlay, "metabolism"):
            self.overlay.metabolism.starve()

    def _on_feed_clicked(self):
        if self.overlay and hasattr(self.overlay, "metabolism"):
            self.overlay.metabolism.satiate()

    def _build_vision_page(self, lay: QVBoxLayout):
        neurons = Card("Visual neurons, last 22 s")
        self.vis_strips = {}
        for name, key in (("LC4", "lc4"), ("LPLC2", "lplc2"), ("LC10 L", "lc10L"), ("LC10 R", "lc10R")):
            self.vis_strips[key] = StripRow(name)
            neurons.body.addWidget(self.vis_strips[key])
        lay.addWidget(neurons)
        card = Card("Visual Input & Screen Perception")
        self.has_eye = EYE.exists() and FIELD.exists()
        self.vision = QCheckBox("Screen Capture Vision (Compound Eye Perception)", enabled=self.has_eye)
        self.vision.setToolTip("Feeds real desktop screen pixels into the fly's retinotopic visual detectors (480 px visual cone) instead of synthetic cursor coordinates.")
        init_on = bool(self.has_eye and (getattr(self.runner, "vision", False) or getattr(self.prefs, "vision_enabled", False)))
        self.vision.setChecked(init_on)
        self.runner.vision = init_on
        self.vision.toggled.connect(self._on_vision_toggled)
        card.body.addWidget(self.vision)

        hgt = QSlider(Qt.Horizontal, minimum=40, maximum=300,
                      value=int(getattr(self.prefs, "eye_height", self.runner.eye_height)),
                      enabled=self.has_eye)
        hgt.valueChanged.connect(self._on_eye_height_changed)
        slider_row(card, "Eye Height (Viewing Elevation)",
                   "Simulated altitude above screen plane: higher = steeper downward cone angle.",
                   hgt, lambda v: f"{v} px")

        self.vision_info = _label("", "Faint", wrap=True)
        card.body.addWidget(self.vision_info)

        # fly's-eye preview: what the visual front end hands to the circuit (model input, not a real optic lobe)
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER};")
        card.body.addWidget(sep)
        self.preview_check = QCheckBox("Fly's-eye preview", enabled=self.has_eye)
        self.preview_check.setToolTip("Shows what the visual front end feeds into the circuit: the luminance each eye "
                                      "column sees, the motion detectors and the drive of the looming, retreat and "
                                      "small-object neurons. Preview only: it does not change behaviour.")
        self.preview_check.setChecked(bool(self.has_eye and getattr(self.prefs, "eye_preview", False)))
        card.body.addWidget(self.preview_check)
        self.preview_tabs = QWidget()
        self.preview_tabs.setStyleSheet(f"QPushButton:checked {{ border: 1px solid {ACCENT}; color: {ACCENT}; background: {BORDER}; }}")
        tabs = QHBoxLayout(self.preview_tabs)
        tabs.setContentsMargins(0, 0, 0, 0)
        tabs.setSpacing(6)
        self._preview_buttons: list[QPushButton] = []
        tab = int(getattr(self.prefs, "eye_preview_tab", 0))
        for i, name in enumerate(eyeview.TABS):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(i == tab)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=i: self._on_preview_tab(i))
            self._preview_buttons.append(b)
            tabs.addWidget(b)
        tabs.addStretch(1)
        card.body.addWidget(self.preview_tabs)
        self.preview = EyePreview()
        card.body.addWidget(self.preview)
        self.preview_note = _label("The model's input, not a recording of a real optic lobe: the detectors are "
                                   "hand-built (vision.py). Redrawn about 10 times a second while this page is open.",
                                   "Faint", wrap=True)
        card.body.addWidget(self.preview_note)
        self.preview_check.toggled.connect(self._on_preview_toggled)
        self._preview_timer = QTimer(self, interval=100, timeout=self._update_preview)
        lay.addWidget(card)
        self._describe_vision(1.0 if init_on else 0.0)
        self._on_preview_toggled(self.preview_check.isChecked(), save=False)

    def _describe_vision(self, state: float):
        if not self.has_eye:
            text = "No eye data available: run 'uv run python tools/build_eye.py' (requires raw data, see README)."
        elif state > 0:
            text = ("Real screen capture active: Fly perceives the desktop through a 480 px visual cone; "
                    "escape (flight), backward walking, freezing, and steering decisions are driven by actual screen pixels.")
        elif state < 0 and self.vision.isChecked():
            text = "No visual input supported in this circuit tier (requires LPLC2/LC4 with receptive fields): using synthetic cursor coordinates."
        else:
            text = ""
        self.vision_info.setText(text)
        self.vision_info.setVisible(bool(text))

    def _on_vision_toggled(self, on: bool):
        setattr(self.runner, "vision", on)
        self.prefs.vision_enabled = on
        self.prefs.save()
        self._update_preview()

    def _on_preview_toggled(self, on: bool, save: bool = True):
        for w in (self.preview_tabs, self.preview, self.preview_note):
            w.setVisible(on)
        if on:
            self._preview_timer.start()
        else:
            self._preview_timer.stop()
            setattr(self.runner, "preview", False)
        if save:
            self.prefs.eye_preview = on
            self.prefs.save()
        self._update_preview()

    def _on_preview_tab(self, i: int):
        for j, b in enumerate(self._preview_buttons):
            b.setChecked(j == i)
        self.prefs.eye_preview_tab = i
        self.prefs.save()
        self._update_preview()

    def _preview_mode(self) -> int:
        return next((i for i, b in enumerate(self._preview_buttons) if b.isChecked()), 0)

    def _update_preview(self):
        """Ask the worker for snapshots only while the preview is on AND on screen, and redraw from the latest one."""
        on = self.preview_check.isChecked() and self.has_eye
        shown = on and self.isVisible() and self.stack.currentIndex() == 2
        setattr(self.runner, "preview", bool(shown and self.vision.isChecked()))
        if not shown:
            return
        snap_of = getattr(self.runner, "preview_snapshot", None)
        if not self.vision.isChecked():
            self.preview.set_message("Turn on Screen Capture Vision to see what the fly sees.")
        elif self.runner.stats().get("vision", 0.0) < 0:
            self.preview.set_message("This circuit has no looming neurons with receptive fields: no visual input.")
        elif snap_of is None:
            self.preview.set_message("Waiting for the first frame...")
        else:
            snap = snap_of()
            if snap is None:
                if self.preview.snap is None:
                    self.preview.set_message("Waiting for the first frame...")
            elif self.preview.snap is None or snap.seq != self.preview.snap.seq or self._preview_mode() != self.preview.mode:
                self.preview.set_frame(snap, self._preview_mode())

    def _on_eye_height_changed(self, v: int):
        h = float(v)
        setattr(self.runner, "eye_height", h)
        self.prefs.eye_height = h
        self.prefs.save()

    def _build_pheromone_page(self, lay: QVBoxLayout):
        card = Card("Pheromone Field & Olfaction")

        self.phero_enable = QCheckBox("Enable Pheromones and Olfaction")
        self.phero_enable.setToolTip("Enables the fly to detect pheromone gradients with dual antennae (tropotaxis) and navigate toward/away from odors.")
        is_on = getattr(self.overlay, "pheromone_enabled", True) if self.overlay is not None else True
        self.phero_enable.setChecked(is_on)
        self.phero_enable.toggled.connect(self._on_phero_toggled)
        card.body.addWidget(self.phero_enable)

        self.cursor_phero_combo = QComboBox()
        self.cursor_phero_modes = [
            ("attract", "Attractive (Food / Sugar scent — fly approaches)"),
            ("repel", "Repellent (Threat / Alarm scent — fly flees)"),
            ("none", "Disabled (Cursor emits no pheromone)"),
        ]
        self.cursor_phero_combo.addItems([label for _, label in self.cursor_phero_modes])
        cur_mode = getattr(self.overlay, "cursor_phero_mode", "attract") if self.overlay is not None else "attract"
        cur_idx = next((i for i, (m, _) in enumerate(self.cursor_phero_modes) if m == cur_mode), 0)
        self.cursor_phero_combo.setCurrentIndex(cur_idx)
        self.cursor_phero_combo.currentIndexChanged.connect(self._on_cursor_phero_changed)
        labeled(card, "Cursor Pheromone Mode", self.cursor_phero_combo)
        lay.addWidget(card)

        desc_card = Card("What is Pheromone & Tropotaxis?")
        info_text = (
            "• Dual-Antenna Olfactory Steering (Tropotaxis): The fly compares odor concentration between "
            "its left and right antennae, steering toward attractive scents or away from repellent scents "
            "via DNa02 steering neurons.\n\n"
            "• Invisible Food Sources: three invisible nectar/food sources of different sizes lie on the screen; a "
            "bigger one smells stronger and further. On food the fly's sugar receptor neurons fire, they drive the "
            "proboscis motor neuron MN9, and while MN9 fires the fly stands and eats (about ten seconds for a "
            "hungry fly; a sated fly's MN9 stays low and it does not eat). A food eaten up appears again elsewhere.\n\n"
            "• Local Search: after a meal the fly circles the place for a while (longer when still hungry), as real "
            "flies do (a designed steering drive on the DNa02 neurons).\n\n"
            "• Boundary Repulsion: A subtle invisible repellent field at the outer 18 px screen margin "
            "prevents the fly from leaving or getting lost outside the display bounds.\n\n"
            "• Cursor Attraction/Repulsion: Set to attractive to make the fly chase and land on your cursor; "
            "set to repellent to make it run away from the cursor.\n\n"
            "• Smell and Startle: a hungry fly that smells food reacts less to something approaching fast, in "
            "proportion to the odor strength. While it approaches an attractive cursor or sits in the middle of a "
            "plume it chases instead of fleeing; once it has eaten (right after a bite) it startles normally. A disabled "
            "or repellent cursor carries no food scent, so only the food plumes still soften a hungry fly's startle."
        )
        desc_card.body.addWidget(_label(info_text, "Faint", wrap=True))
        lay.addWidget(desc_card)

    def _on_phero_toggled(self, checked: bool):
        if self.overlay is not None:
            self.overlay.pheromone_enabled = checked
        self.prefs.pheromone_enabled = checked
        self.prefs.save()

    def _on_cursor_phero_changed(self, idx: int):
        if 0 <= idx < len(self.cursor_phero_modes):
            mode = self.cursor_phero_modes[idx][0]
            if self.overlay is not None:
                self.overlay.cursor_phero_mode = mode
            self.prefs.cursor_phero_mode = mode
            self.prefs.save()

    def _build_learning_page(self, lay: QVBoxLayout):
        # 1. 3D Neural Circuit & Mushroom Body Module
        card_3d = Card("3D Neural Circuit & Mushroom Body")
        self.neural_desc_label = _label(
            "FlyWire v783 full connectome 3D Mushroom Body neural circuit and mental map. "
            "Models 1,700+ Kenyon cell somas, axon tracts (pedunculus and lobes), dopaminergic clusters "
            "(PAM reward & PPL1 punishment) with real-time synaptic firing and learned valence in 3D space.",
            "Muted", wrap=True)
        card_3d.body.addWidget(self.neural_desc_label)

        # Visualization options
        popout_row = QHBoxLayout()
        self.btn_toggle_3d = QPushButton("View 3D Neural Circuit")
        self.btn_toggle_3d.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_3d.setCheckable(True)
        self.btn_toggle_3d.setChecked(False)
        self.btn_toggle_3d.setToolTip("Toggle 3D neural circuit visualization inside this panel.")
        self.btn_toggle_3d.clicked.connect(lambda: self._toggle_3d_view())
        popout_row.addWidget(self.btn_toggle_3d)

        self.popout_btn = QPushButton("Open in Dedicated Window")
        self.popout_btn.setCursor(Qt.PointingHandCursor)
        self.popout_btn.setToolTip("Opens 3D Neural Circuit in an expanded standalone window.")
        self.popout_btn.clicked.connect(self._open_3d_popout)
        popout_row.addWidget(self.popout_btn)

        popout_row.addStretch(1)
        card_3d.body.addLayout(popout_row)

        # 3D Visualizer Container (optional: hidden by default to save resources)
        self.mb_view3d_container = QWidget()
        v3d_lay = QVBoxLayout(self.mb_view3d_container)
        v3d_lay.setContentsMargins(0, 4, 0, 0)
        v3d_lay.setSpacing(6)

        self.mb_view3d = MushroomBody3DView(runner=self.runner)
        self.mb_view3d.setFixedHeight(410)
        v3d_lay.addWidget(self.mb_view3d)

        v3d_hint = _label(
            "Drag: left orbits, right pans • Scroll zooms • Home resets • Tools: Explore (click a neuron), "
            "Stimulate (press a target: its neurons fire while held), Silence (click a target)",
            "Faint", wrap=True
        )
        v3d_lay.addWidget(v3d_hint)

        self.mb_view3d_container.setVisible(False)
        card_3d.body.addWidget(self.mb_view3d_container)

        # System Load Warning Note
        self.lbl_system_load_warn = _label(
            "Note: 3D neural visualization calculates and renders thousands of neurons and axon projections "
            "in real time, which adds extra system load (CPU/GPU).",
            "Warn", wrap=True
        )
        card_3d.body.addWidget(self.lbl_system_load_warn)

        lay.addWidget(card_3d)

        card_da = Card("Dopamine neurons, last 22 s")
        self.da_strips = {"pam": StripRow("PAM (reward)", TEXT, unit=""), "ppl1": StripRow("PPL1 (punishment)", TEXT, unit="")}
        for w_ in self.da_strips.values():
            card_da.body.addWidget(w_)
        lay.addWidget(card_da)

        # 2. Experiment & Learning Sandbox Console
        card_sim = Card("Learning & Experiment Sandbox")
        card_sim.body.addWidget(_label(
            "Inject real-time olfactory/visual cues and reward/punishment to test dynamic changes in the "
            "3D mental map, dopamine neurons, and synaptic weights:",
            "Muted", wrap=True))

        sim_btn_grid = QGridLayout()
        sim_btn_grid.setHorizontalSpacing(10)
        sim_btn_grid.setVerticalSpacing(8)

        btn_reward = QPushButton("Inject Reward (PAM)")
        btn_reward.setToolTip("Fires PAM dopamine neurons to establish positive valence (desire) toward the active cue.")
        btn_reward.clicked.connect(lambda: self._inject_test_action("reward"))

        btn_punish = QPushButton("Inject Punishment (PPL1)")
        btn_punish.setToolTip("Fires PPL1 dopamine neurons to establish negative valence (avoidance/fear).")
        btn_punish.clicked.connect(lambda: self._inject_test_action("punish"))

        btn_cue_food = QPushButton("Food Odor Cue")
        btn_cue_food.setToolTip("Shows the mushroom body the food odor for 0.8 s (a designed cue; the fly smells nothing).")
        btn_cue_food.clicked.connect(lambda: self._inject_test_action("food"))

        btn_cue_near = QPushButton("Cursor Visual Cue")
        btn_cue_near.setToolTip("Shows the mushroom body the cursor-near cue for 0.8 s (a designed cue; nothing is seen).")
        btn_cue_near.clicked.connect(lambda: self._inject_test_action("cursor"))

        sim_btn_grid.addWidget(btn_reward, 0, 0)
        sim_btn_grid.addWidget(btn_punish, 0, 1)
        sim_btn_grid.addWidget(btn_cue_food, 1, 0)
        sim_btn_grid.addWidget(btn_cue_near, 1, 1)

        card_sim.body.addLayout(sim_btn_grid)
        # while the 3D view's lab silences or stimulates neurons the mushroom body does not learn (Brain.lab_active)
        self.lbl_lab_gate = _label("", "Warn", wrap=True)
        self.lbl_lab_gate.setVisible(False)
        card_sim.body.addWidget(self.lbl_lab_gate)

        # Telemetry stats grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        self.s_val = Stat("Valence", "Learned affective response to current cue: + desire, − fear")
        self.s_vm = Stat("V_motor", "Combined drive: threat + desire + frustration")
        self.s_gear = Stat("Locomotion Mode", "Stand, walk, escape flight (Giant Fiber) or foraging flight to a food "
                                              "source (no Giant Fiber)")
        for i, s in enumerate((self.s_val, self.s_vm, self.s_gear)):
            grid.addWidget(s, 0, i)
        card_sim.body.addLayout(grid)

        ctrl_row = QHBoxLayout()
        self.learn_enable = QCheckBox("Enable Learning")
        self.learn_enable.setToolTip("When disabled, the fly ceases updating synaptic weights; engine reloads.")
        self.learn_enable.setChecked(getattr(self.runner.cfg, "learning", True))
        ctrl_row.addWidget(self.learn_enable)

        ctrl_row.addStretch(1)

        self.forget_btn = QPushButton("Reset Memory (Amnesia)")
        self.forget_btn.setToolTip("Clears all learned weights immediately, restoring the 3D mental map to its naive state.")
        self.forget_btn.clicked.connect(self._forget)
        ctrl_row.addWidget(self.forget_btn)

        card_sim.body.addLayout(ctrl_row)

        # Which mushroom body learns: the designed 2,000-cell layer or the connectome's real cells
        self.mb_wiring_options = [
            ("random", "Designed model (2,000 Kenyon cells)"),
            ("flywire", "Real FlyWire cells (5,177 Kenyon cells)"),
        ]
        self.mb_wiring_combo = QComboBox()
        self.mb_wiring_combo.addItems([label for _, label in self.mb_wiring_options])
        cur_wiring = getattr(self.runner.cfg, "mb_wiring", "random")
        self.mb_wiring_combo.setCurrentIndex(1 if cur_wiring == "flywire" else 0)
        self._sync_mb_wiring_enabled()
        self.mb_wiring_combo.currentIndexChanged.connect(self._on_mb_wiring_changed)
        self.learn_enable.toggled.connect(self._on_learning_toggled)
        labeled(card_sim, "Mushroom Body Model", self.mb_wiring_combo)
        card_sim.body.addWidget(_label(
            "The engine reloads when this changes and each model keeps its own memory file. The real cells learn the "
            "cursor cue more weakly; the cue-to-projection-neuron mapping is assumed (experimental).",
            "Faint", wrap=True))
        lay.addWidget(card_sim)

        # 3. Scientific Anatomy & Function Card
        desc_card = Card("Mushroom Body & Biological Memory")
        bio_text = (
            "• Color Codes (Mental Map):\n"
            "   - Emerald Green: Rewarded appetitive memory (approach drive via PAM dopamine)\n"
            "   - Ruby Red: Punished aversive memory (avoidance/fear drive via PPL1 nociception)\n"
            "   - Cyan Blue: Naive, untrained neural state\n"
            "   - Bright White Sparks: Live spiking Kenyon cells (~5% sparse coding)\n\n"
            "• Calyx: Posterior cup containing Kenyon cell somas and dendritic claws. "
            "Olfactory projection neurons (ALPN) and visual projections synapse here.\n\n"
            "• Pedunculus: Dense axon cable extending anteriorly from the calyx to the heel, "
            "bifurcating into vertical and horizontal lobes.\n\n"
            "• Lobes: Vertical (α, α') and horizontal/medial (β, β', γ) compartments. Each division "
            "is innervated by specific MBON output neurons and dopaminergic modulators (PAM/PPL1).\n\n"
            "• Three-Factor Plasticity Rule: Active Kenyon Cell (1) + Dopamine Release (2) "
            "→ Long-Term Depression (LTD) at corresponding MBON synapse (3). "
            "This converts momentary experiences into persistent behavioural adaptations."
        )
        desc_card.body.addWidget(_label(bio_text, "Faint", wrap=True))
        lay.addWidget(desc_card)

    def _mb_wiring(self) -> str:
        i = self.mb_wiring_combo.currentIndex()
        return self.mb_wiring_options[i][0] if 0 <= i < len(self.mb_wiring_options) else "random"

    def _sync_mb_wiring_enabled(self) -> None:
        """The model can be chosen only while learning is on, the real cells are built, and no env var decides."""
        forced = mb_wiring_from_env()
        usable = wiring_usable(MUSHROOM)
        state = "missing" if not MUSHROOM.exists() else "damaged"
        if forced == "flywire" and not usable:
            ok, why = False, (f"NEUROPEST_MB asks for the real cells, but data/circuits/mushroom_flywire.npz is {state}: "
                              "the designed model runs (build it with tools/build_mushroom.py).")
        elif forced is not None:
            ok, why = False, f"Set by the NEUROPEST_MB environment variable ({forced})."
        elif not usable:
            ok, why = False, (f"The real cells need data/circuits/mushroom_flywire.npz, which is {state} "
                              "(run tools/build_mushroom.py).")
        elif not self.learn_enable.isChecked():
            ok, why = False, "Learning is off."
        else:
            ok, why = True, ""
        self.mb_wiring_combo.setEnabled(ok)
        tip = ("Designed: 2,000 Kenyon cells with random calyx wiring (default). Real FlyWire cells: the Kenyon cells, "
               "output, dopamine and projection neurons of the connectome (experimental).")
        self.mb_wiring_combo.setToolTip(f"{tip}\n{why}" if why else tip)

    def _on_learning_toggled(self, _checked: bool) -> None:
        self._sync_mb_wiring_enabled()
        self._apply()

    def _on_mb_wiring_changed(self, _idx: int) -> None:
        # saved here only: the user picked it. _apply() runs for every other change too and must not overwrite a saved
        # choice with what happens to run now (the designed model while the real cells are missing, or an env override)
        self.prefs.mb_wiring = self._mb_wiring()
        self._apply()

    def _toggle_3d_view(self, visible: bool | None = None):
        """Toggle 3D neural network visualization on/off on demand to save resources."""
        if visible is None:
            visible = not self.mb_view3d_container.isVisible()
        self.mb_view3d_container.setVisible(visible)
        self.btn_toggle_3d.setChecked(visible)
        if visible:
            self.btn_toggle_3d.setText("Hide 3D Neural Circuit (Save Power)")
            self.btn_toggle_3d.setStyleSheet(
                f"background: {BORDER}; border: 1px solid {TEXT_STRONG}; color: {TEXT_STRONG}; font-weight: 600;"
            )
        else:
            self.btn_toggle_3d.setText("View 3D Neural Circuit")
            self.btn_toggle_3d.setStyleSheet("")

    def _open_3d_popout(self):
        """Open a large dedicated 3D Mushroom Body viewer window."""
        dialog = QDialog(self)
        dialog.setWindowTitle("NeuroPest - 3D Neural Circuit & Mushroom Body (FlyWire v783)")
        dialog.resize(920, 680)
        d_lay = QVBoxLayout(dialog)
        d_lay.setContentsMargins(14, 14, 14, 14)
        d_lay.setSpacing(10)

        # Dialog header info
        d_head = QVBoxLayout()
        d_head.setSpacing(4)
        d_head.addWidget(_label(
            "FlyWire v783 full connectome 3D neural model (Kenyon cell somas, axon tracts, "
            "lobes, and dopaminergic modulation).",
            "Muted", wrap=True))
        d_head.addWidget(_label(
            "Note: 3D neural visualization may add extra system load (CPU/GPU).",
            "Warn", wrap=True))
        d_lay.addLayout(d_head)

        pop_view = MushroomBody3DView(runner=self.runner, data=self.mb_view3d.data, parent=dialog)
        d_lay.addWidget(pop_view, 1)
        # one live view at a time: the embedded one would keep painting behind the dialog on the same GUI thread
        embedded = self.mb_view3d_container.isVisible()
        if embedded:
            self.mb_view3d.setVisible(False)
        try:
            dialog.exec()
        finally:
            if embedded:
                self.mb_view3d.setVisible(True)

    def _inject_test_action(self, action: str):
        r = self.runner
        if r is None or not getattr(r, "ready", False):
            return
        if action == "reward":
            if hasattr(r, "inject_cue_food"):
                r.inject_cue_food(1.0)
            if hasattr(r, "inject_reward"):
                r.inject_reward(1.0)
        elif action == "punish":
            if hasattr(r, "inject_cue_near"):
                r.inject_cue_near(1.0)
            if hasattr(r, "inject_punish"):
                r.inject_punish(1.0)
        elif action == "food":
            if hasattr(r, "inject_cue_food"):
                r.inject_cue_food(1.0)
        elif action == "cursor":
            if hasattr(r, "inject_cue_near"):
                r.inject_cue_near(1.0)

    def _forget(self):
        forget = getattr(self.runner, "forget", None)
        if forget:
            forget()

    def _build_circuit_page(self, lay: QVBoxLayout):
        card = Card("Neural Circuit")
        self.circuits = ([("flywire", "FlyWire v783 (real connectome)")] if CACHE.exists() else []) \
            + [("toy", "Toy circuit (synthetic benchmark)")]
        self.circ = QComboBox()
        self.circ.addItems([name for _, name in self.circuits])
        self.circ.setCurrentIndex([c for c, _ in self.circuits].index(self.runner.cfg.circuit))
        labeled(card, "Circuit Type", self.circ)

        if not CACHE.exists():
            download_btn = QPushButton("Download FlyWire Connectome (~135 MB)")
            download_btn.setObjectName("Primary")
            download_btn.setToolTip("Download FlyWire v783 and initialize real biological brain circuits.")

            def _trigger_download():
                from .data_manager import prompt_startup_data
                if prompt_startup_data(self):
                    self.circuits = [("flywire", "FlyWire v783 (real connectome)"), ("toy", "Toy circuit (synthetic benchmark)")]
                    self.circ.blockSignals(True)
                    self.circ.clear()
                    self.circ.addItems([name for _, name in self.circuits])
                    self.circ.setCurrentIndex(0)
                    self.circ.blockSignals(False)
                    self._circuit_changed(0)
                    self._sync_mb_wiring_enabled()          # the download also built the real mushroom-body cells
                    download_btn.setVisible(False)

            download_btn.clicked.connect(_trigger_download)
            card.body.addWidget(download_btn)

        head = QHBoxLayout()
        head.addWidget(_label("Circuit Size (Neuron Count)"))
        head.addStretch(1)
        self.size_label = _label("", "Value")
        head.addWidget(self.size_label)
        card.body.addLayout(head)
        self.size = QSlider(Qt.Horizontal)
        card.body.addWidget(self.size)

        self.tier_info = _label("", "Faint", wrap=True)
        card.body.addWidget(self.tier_info)

        self.dt = QComboBox()
        self.dt.addItems([n for n, _ in DTS])
        self.dt.setCurrentIndex([d for _, d in DTS].index(self.runner.cfg.dt))
        labeled(card, "Time step dt (smaller: higher accuracy, heavier compute)", self.dt)
        lay.addWidget(card)

        # Advanced Circuit Options
        adv_card = Card("Advanced Circuit Options")
        self.sym_options = [
            ("individual", "Individual (FlyWire original)"),
            ("symmetric", "Symmetric (Left-right balanced)"),
        ]
        self.sym_combo = QComboBox()
        self.sym_combo.addItems([label for _, label in self.sym_options])
        cur_sym = getattr(self.runner.cfg, "symmetry", "individual")
        self.sym_combo.setCurrentIndex(0 if cur_sym == "individual" else 1)
        self.sym_combo.setEnabled(self._kind() == "flywire")
        self.sym_combo.currentIndexChanged.connect(self._on_sym_changed)
        labeled(adv_card, "Hemispheric Symmetry", self.sym_combo)
        self.sym_hint = _label(
            "Individual: FlyWire's single fly brain connectome (biological asymmetry preserved). This brain turns left more readily "
            "(left turn gain is ~2.8× right); odor and steering filters usually compensate.\n"
            "Symmetric: Left and right hemisphere connection weights are averaged together (balances left-right turning dynamics).",
            "Faint", wrap=True)
        adv_card.body.addWidget(self.sym_hint)
        lay.addWidget(adv_card)

        # Compute sub-card
        comp_card = Card("Compute Hardware")
        self.hw = QComboBox()
        self.hw.addItems(["Auto", "CPU"])
        self.hw.setToolTip(f"Auto: Uses GPU for {GPU_AUTO_MIN_NEURONS:,} neurons and above, CPU below")
        labeled(comp_card, "Compute Processor / GPU", self.hw)

        self.gpus: list[dict] = []
        self._gpu_result: list | None = None
        self._gpu_scanned = False
        self._gpu_scanning = False
        self._apply_pending = False
        self.has_wgpu = importlib.util.find_spec("wgpu") is not None

        # Pre-select based on saved preference
        if self.prefs.hardware == "cpu":
            self.hw.setCurrentIndex(1)
        elif self.prefs.hardware == "gpu":
            pref_name = self.prefs.gpu_name or "GPU"
            pref_backend = f" ({self.prefs.gpu_backend})" if self.prefs.gpu_backend else ""
            self.hw.addItem(f"GPU: {pref_name}{pref_backend}")
            self.hw.setCurrentIndex(2)
        else:
            self.hw.setCurrentIndex(0)

        if self.has_wgpu:
            if self.auto_scan_gpus:
                self.gpu_note = _label("Scanning for GPUs…", "Faint")
                comp_card.body.addWidget(self.gpu_note)
                self._gpu_poll = QTimer(self, timeout=self._gpus_found, interval=200)
                self._scan_gpus()
            else:
                self.hw.addItem(SCAN_ITEM)
                self.hw.activated.connect(self._hw_activated)
                self.gpu_note = _label("GPUs discovered on demand (~1 s, ~100 MB)", "Faint")
                comp_card.body.addWidget(self.gpu_note)
                self._gpu_poll = QTimer(self, timeout=self._gpus_found, interval=250)
        else:
            comp_card.body.addWidget(_label("GPU acceleration is not supported on this platform.", "Faint", wrap=True))
        lay.addWidget(comp_card)

    def _build_view_page(self, lay: QVBoxLayout):
        card = Card("Appearance & Display")
        s = QSlider(Qt.Horizontal, minimum=5, maximum=40, value=int(self.overlay.scale * 10))
        s.valueChanged.connect(self._on_scale_changed)
        slider_row(card, "Fly Size", "Visible size on desktop", s, lambda v: f"×{v / 10:.1f}")

        # Fly Skin Selector
        card.body.addWidget(_label("Fly Skin (Costume)", "Muted"))

        skin_container = QWidget()
        skin_lay = QHBoxLayout(skin_container)
        skin_lay.setContentsMargins(0, 4, 0, 4)
        skin_lay.setSpacing(6)

        self.skin_group = QButtonGroup(self)
        self.skin_group.setExclusive(True)
        self.skin_buttons: dict[str, QToolButton] = {}

        cur_skin = getattr(self.overlay, "skin", "classic")

        for key, meta in SKIN_METADATA.items():
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setChecked(key == cur_skin)
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedSize(86, 76)

            thumb_path = SKIN_THUMBNAILS / f"{key}.png"
            if thumb_path.exists():
                btn.setIcon(QIcon(str(thumb_path)))
            btn.setIconSize(QSize(40, 40))
            btn.setText(meta["title"])
            btn.setToolTip(f"{meta['full_name']}\n{meta['desc']}")

            btn.setStyleSheet(f"""
                QToolButton {{
                    background: transparent;
                    border: 1px solid {BORDER_BTN};
                    border-radius: 6px;
                    padding: 6px 2px 4px 2px;
                    color: {MUTED_2};
                    font-size: {fs(11)};
                    font-weight: 500;
                }}
                QToolButton:hover {{
                    background: {SEL_BG};
                    border: 1px solid {BORDER_HI};
                    color: {TEXT_STRONG};
                }}
                QToolButton:checked {{
                    background: {SEL_BG};
                    border: 2px solid {TEXT_STRONG};
                    color: {TEXT_STRONG};
                    font-weight: 600;
                }}
            """)

            self.skin_buttons[key] = btn
            self.skin_group.addButton(btn)
            btn.clicked.connect(lambda _, k=key: self._on_skin_selected(k))
            skin_lay.addWidget(btn)

        skin_lay.addStretch(1)
        card.body.addWidget(skin_container)

        active_meta = SKIN_METADATA.get(cur_skin, SKIN_METADATA["classic"])
        self.skin_desc_label = _label(f"{active_meta['full_name']}: {active_meta['desc']}", "Faint", wrap=True)
        card.body.addWidget(self.skin_desc_label)

        screens = QApplication.screens()
        if len(screens) > 1:
            box = QComboBox()
            items = ["All Screens (Span Virtual Desktop)"] + [f"{i + 1}: {s.name()}" for i, s in enumerate(screens)]
            box.addItems(items)
            current_idx = 0 if self.overlay.home is None else (screens.index(self.overlay.home) + 1 if self.overlay.home in screens else 0)
            box.setCurrentIndex(current_idx)
            box.currentIndexChanged.connect(lambda i: self._on_screen_changed(i, screens))
            labeled(card, "Virtual Screen / Monitor", box)
        lay.addWidget(card)

    def _on_scale_changed(self, v: int):
        sc = v / 10.0
        if self.overlay is not None:
            self.overlay.scale = sc
        self.prefs.scale = sc
        self.prefs.save()

    def _on_screen_changed(self, i: int, screens: list):
        screen = None if i == 0 else screens[i - 1]
        if self.overlay is not None:
            self.overlay.set_home(screen)
        self.prefs.monitor_index = i
        self.prefs.monitor_name = screen.name() if screen is not None else ""
        self.prefs.save()

    def _on_skin_selected(self, skin_key: str):
        if self.overlay is not None:
            self.overlay.skin = skin_key
            self.overlay.update()
        self.prefs.skin = skin_key
        self.prefs.save()
        if skin_key in SKIN_METADATA and hasattr(self, "skin_desc_label"):
            meta = SKIN_METADATA[skin_key]
            self.skin_desc_label.setText(f"{meta['full_name']}: {meta['desc']}")
        if skin_key in self.skin_buttons and not self.skin_buttons[skin_key].isChecked():
            self.skin_buttons[skin_key].setChecked(True)



    # ------------------------------------------------------------ circuit choice
    def _kind(self) -> str:
        return self.circuits[self.circ.currentIndex()][0]

    def _sizes(self) -> list[int]:
        return FLYWIRE_SIZES if self._kind() == "flywire" else TOY_SIZES

    def _load_sizes(self, want: int):
        sizes = self._sizes()
        self.size.blockSignals(True)
        self.size.setRange(0, len(sizes) - 1)
        self.size.setValue(min(range(len(sizes)), key=lambda i: abs(sizes[i] - want)))
        self.size.blockSignals(False)
        self._describe()

    def _circuit_changed(self, _):
        self._load_sizes(FLYWIRE_DEFAULT if self._kind() == "flywire" else TOY_SIZES[0])
        self.sym_combo.setEnabled(self._kind() == "flywire")
        self.prefs.circuit = self._kind()
        self.prefs.save()
        self._debounce.start()

    def _size_moved(self, _):
        self._describe()
        n = self._sizes()[self.size.value()]
        self.prefs.neurons = n
        self.prefs.save()
        self._debounce.start()

    def _on_sym_changed(self, idx: int):
        if 0 <= idx < len(self.sym_options):
            self.prefs.symmetry = self.sym_options[idx][0]
            self.prefs.save()
        self._debounce.start()

    def _on_dt_changed(self, idx: int):
        if 0 <= idx < len(DTS):
            self.prefs.dt = DTS[idx][1]
            self.prefs.save()
        self._debounce.start()

    def _on_hw_changed(self, i: int):
        if not self._gpu_scanned and self.hw.itemText(i) == SCAN_ITEM:
            return
        if i == 0:
            self.prefs.hardware = "auto"
            self.prefs.gpu_name = ""
            self.prefs.gpu_backend = ""
            self.prefs.gpu_index = None
        elif i == 1:
            self.prefs.hardware = "cpu"
            self.prefs.gpu_name = ""
            self.prefs.gpu_backend = ""
            self.prefs.gpu_index = None
        elif i >= 2 and self.gpus and (i - 2) < len(self.gpus):
            g = self.gpus[i - 2]
            self.prefs.hardware = "gpu"
            self.prefs.gpu_name = g["name"]
            self.prefs.gpu_backend = g["backend"]
            self.prefs.gpu_index = g["index"]
        self.prefs.save()
        self._debounce.start()

    def _describe(self):
        n = self._sizes()[self.size.value()]
        self.size_label.setText(f"{n:,} neurons")
        if self._kind() == "flywire":
            t = self.tiers.get(n)
            if t:
                self.tier_info.setText(
                    f"Deviation from full brain: takeoff {t['gf_err']:.0f}% · backward walking {t['mdn_err']:.0f}% · "
                    f"steering {t['steer_err']:.0f}%\nDescending correlation {t['dn_corr']:.3f} · benchmarked speed "
                    f"×{t['realtime']:.1f} (worst ×{t['rt_min']:.1f})")
                pinned = (f"The first {t['pinned']:,} neurons are fixed input and output neurons across all tiers; "
                          f"{t['free']:,} selected by firing order. " if t.get("pinned") else "")
                self.tier_info.setToolTip("Nested tier selected according to FlyWire firing order. " + pinned +
                                          "Deviation benchmarked for looming, retreat, and steering inputs "
                                          "(tools/fidelity.py). Speed measured on benchmark machine "
                                          f"({self.tier_machine}).")
            else:
                self.tier_info.setText("No benchmark data for this size.")
        else:
            self.tier_info.setText("" if n == TOY_SIZES[0] else "Synthetic benchmark: does not alter behaviour, "
                                   "only tests computational load.")
        self.tier_info.setVisible(bool(self.tier_info.text()))

    def _hw_activated(self, i: int):
        if not self._gpu_scanned and self.hw.itemText(i) == SCAN_ITEM:
            self.hw.setCurrentIndex(0)
            self._scan_gpus()

    def _scan_gpus(self):
        if self._gpu_scanned or self._gpu_scanning or not self.has_wgpu:
            return
        self._gpu_scanning = True
        self.gpu_note.setText("Scanning for GPUs…")
        threading.Thread(target=lambda: setattr(self, "_gpu_result", list_gpus()), daemon=True).start()
        self._gpu_poll.start()

    def _gpus_found(self):
        if self._gpu_result is None:
            return
        self._gpu_poll.stop()
        self._gpu_scanning, self._gpu_scanned = False, True
        self.gpus = self._gpu_result
        self.hw.blockSignals(True)
        while self.hw.count() > 2:
            self.hw.removeItem(2)
        self.hw.addItems([f"GPU: {g['name']} ({g['backend']})" for g in self.gpus])

        # Restore selection according to self.prefs
        if self.prefs.hardware == "cpu":
            target_idx = 1
        elif self.prefs.hardware == "gpu":
            found_i = None
            if self.prefs.gpu_name:
                for idx, g in enumerate(self.gpus):
                    if g.get("name") == self.prefs.gpu_name and g.get("backend") == self.prefs.gpu_backend:
                        found_i = idx
                        break
                if found_i is None:
                    for idx, g in enumerate(self.gpus):
                        if g.get("name") == self.prefs.gpu_name:
                            found_i = idx
                            break
            if found_i is None and self.prefs.gpu_index is not None:
                for idx, g in enumerate(self.gpus):
                    if g.get("index") == self.prefs.gpu_index:
                        found_i = idx
                        break
            if found_i is None and self.gpus:
                found_i = 0

            if found_i is not None:
                target_idx = 2 + found_i
                g = self.gpus[found_i]
                self.prefs.gpu_name = g["name"]
                self.prefs.gpu_backend = g["backend"]
                self.prefs.gpu_index = g["index"]
                self.prefs.save()
            else:
                target_idx = 1
        else:
            target_idx = 0

        self.hw.setCurrentIndex(target_idx)
        self.hw.blockSignals(False)

        self.gpu_note.setText(f"{len(self.gpus)} GPU(s) found" if self.gpus else "No compatible GPU found")

        n = self._sizes()[self.size.value()]
        if self.prefs.hardware == "gpu" and self.gpus:
            self._apply()
        elif target_idx == 0 and n >= GPU_AUTO_MIN_NEURONS and self.gpus:
            self._apply()
        elif self._apply_pending:
            self._apply_pending = False
            self._apply()

    def _hardware(self, n: int) -> tuple[str, int | None]:
        i = self.hw.currentIndex()
        if i >= 2 and self._gpu_scanned and (i - 2) < len(self.gpus):
            return "gpu", self.gpus[i - 2]["index"]
        if i >= 2 and not self._gpu_scanned and self.prefs.hardware == "gpu" and self.prefs.gpu_index is not None:
            return "gpu", self.prefs.gpu_index
        if i == 0 and self.gpus and n >= GPU_AUTO_MIN_NEURONS:
            return "gpu", pick_gpu(self.gpus)
        return "cpu", None

    def _apply(self):
        n = self._sizes()[self.size.value()]
        if self.hw.currentIndex() == 0 and n >= GPU_AUTO_MIN_NEURONS and self.has_wgpu and not self._gpu_scanned:
            self._apply_pending = True
            self._scan_gpus()
            return
        backend, adapter = self._hardware(n)
        sym = self.sym_options[self.sym_combo.currentIndex()][0]
        cfg = EngineConfig(self._kind(), n, DTS[self.dt.currentIndex()][1], backend=backend, adapter=adapter,
                           symmetry=sym, memory_path=self.runner.cfg.memory_path,
                           learning=self.learn_enable.isChecked(), mb_wiring=self._mb_wiring())
        self.prefs.circuit = cfg.circuit
        self.prefs.neurons = cfg.n
        self.prefs.dt = cfg.dt
        self.prefs.symmetry = cfg.symmetry
        self.prefs.learning = cfg.learning
        self.prefs.save()
        if cfg != self.runner.cfg:
            self.runner.start(cfg)

    # ---------------------------------------------------------------- telemetry
    def _set_pill(self, text: str, color: str):
        self.pill.setText(f"●  {text}")
        self.pill.setStyleSheet(f"color: {color}; background: transparent; border: none; font-size: {fs(14)}; "
                                "font-weight: 600;")

    def _show_warn(self, text: str, error: bool = False):
        self.warn.setObjectName("Error" if error else "Warn")
        self.warn.style().unpolish(self.warn)
        self.warn.style().polish(self.warn)
        self.warn.setText(text)
        self.warn.setVisible(bool(text))

    def _clear_stats(self):
        for s in (self.s_gf, self.s_mdn, self.s_steer, self.s_mn9, self.s_rt, self.s_cpu, self.s_spk,
                  self.s_val, self.s_vm, self.s_gear):
            s.set("–")
        self.footer.setText("")
        self.active.setText("")
        if hasattr(self, "lbl_lab_gate"):       # starting / failed: no lab state to report (a restart drops the light)
            self.lbl_lab_gate.setVisible(False)

    def _update_strips(self, r, st: dict) -> None:
        """One 4 Hz sample into the history, then every strip."""
        lab = r.lab_rates() if hasattr(r, "lab_rates") else {}
        hunger = 0.0
        m = getattr(self.overlay, "metabolism", None)
        if m is not None:
            hunger = (m.hunger_pct / 100.0) if getattr(m, "enabled", True) else 1.0
        v = {"gf": st.get("gf", 0.0), "mdn": st.get("mdn", 0.0), "dnaL": lab.get("DNa02_L", 0.0),
             "dnaR": lab.get("DNa02_R", 0.0), "dnp09": st.get("walk", 0.0), "mn9": st.get("mn9", 0.0),
             "adn1": st.get("groom", 0.0), "adn2": st.get("groom", 0.0), "lc4": lab.get("LC4", 0.0),
             "lplc2": lab.get("LPLC2", 0.0), "lpc1": lab.get("LPC1", 0.0), "lc10L": lab.get("LC10_L", 0.0),
             "lc10R": lab.get("LC10_R", 0.0), "p9": st.get("walk", 0.0), "sugar": st.get("sugar_hz", 0.0),
             "bristle": st.get("touch_hz", 0.0), "pam": st.get("pam", 0.0), "ppl1": st.get("ppl1", 0.0),
             "hunger": hunger}
        self.hist.push(v, r.state)
        h = self.hist
        page = self.stack.currentIndex()
        if page == 0:
            self.circuit.update_view(h, v, r.state)
        elif page == 1:
            self.hunger_strip.set(h.cells("hunger"), text=f"{hunger * 100:.0f} %")
            self.etho.set_history(h)
        elif page == 2:
            for key, row in self.vis_strips.items():
                row.set(h.cells(key), v[key])
        elif page == 6:
            for key, row in self.da_strips.items():
                row.set(h.cells(key), text=f"{v[key]:.2f}")

    def _refresh(self):
        r = self.runner
        if r.failed or not r.alive:
            self._set_pill("Error", ERROR)
            self.circuit.set_message("The engine stopped.")
            self._clear_stats()
            self.telemetry.setText("")
            self._show_warn("Engine encountered an error or stopped (see console for details). Try selecting a smaller circuit size.",
                            error=True)
            return
        st = r.stats()
        if not st["ready"]:
            self._set_pill("Starting", MUTED)
            self.circuit.set_message("Starting the engine…")
            self._clear_stats()
            self.telemetry.setText("Engine initializing (initial compilation takes a few seconds)…")
            self._show_warn("")
            return
        color, name = STATE_STYLE.get(r.state, (FAINT, r.state))
        self._set_pill(name, color)
        self._update_strips(r, st)
        self.s_gf.set(f"{st['gf']:.0f} Hz")
        self.s_mdn.set(f"{st['mdn']:.0f} Hz")
        self.s_steer.set(f"{st['steer']:+.0f}")
        self.s_mn9.set(f"{st.get('mn9', 0.0):.0f} Hz")
        self.s_rt.set(f"×{st['rt']:.1f}")
        self.s_cpu.set(f"%{100 * st['cpu']:.0f}")
        self.s_spk.set(f"{st['spikes']:,.0f}")
        act = f"{st['active']:,.0f} ACTIVE" if st["active"] >= 0 else f"{st['n']:,} NEURONS"
        self.footer.setText(f"{st['rt']:.1f}× REAL-TIME · CPU {100 * st['cpu']:.0f}% · {act} · {st['spikes']:,.0f} SPIKES/S")
        v = st.get("valence", 0.0)
        self.s_val.set(f"{v:+.2f} " + ("desire" if v > 0.05 else "fear" if v < -0.05 else "neutral"))
        self.s_vm.set(f"{st.get('v_motor', 0.0):.2f}")
        self.s_gear.set({"stand": "stand", "walk": "walk", "fly_short": "escape flight (GF)",
                         "fly_long": "foraging flight (no GF)"}.get(st.get("gear", "stand"), "–"))
        if hasattr(self, "lbl_lab_gate"):
            lab = r.lab_status() if hasattr(r, "lab_status") else {}
            parts = []
            if lab.get("silenced"):
                parts.append("silenced: " + ", ".join(lab["silenced"]))
            if lab.get("stimulated"):
                parts.append("stimulated: " + ", ".join(lab["stimulated"]))
            gate = bool(lab.get("gate"))
            self.lbl_lab_gate.setVisible(gate)
            if gate:
                self.lbl_lab_gate.setText("Lab is on (" + ("; ".join(parts) or "ending") + "): the mushroom body does "
                                          "not learn meanwhile, so these buttons teach nothing until it is released.")
        self.active.setText(f"Active neurons: {st['active']:,.0f} / {st['n']:,}" if st["active"] >= 0
                            else f"Neurons: {st['n']:,} (GPU updates all every step)")
        where = "CPU"
        if r.cfg.backend == "gpu":
            g = next((g for g in self.gpus if g["index"] == r.cfg.adapter), None)
            where = f"GPU {g['name']} ({g['backend']})" if g else "GPU"
        self.telemetry.setText(f"Backend: {where} · time step {r.cfg.dt:g} ms")
        self._describe_vision(st.get("vision", 0.0))
        if hasattr(self, "hunger_label") and self.overlay and hasattr(self.overlay, "metabolism"):
            m = self.overlay.metabolism
            if not m.enabled:
                self.hunger_label.setText("Disabled (Constant Hunger)")
            else:
                pct = m.hunger_pct
                status_desc = "Starving" if pct >= 75 else "Hungry" if pct >= 40 else "Satiated"
                self.hunger_label.setText(f"Hunger: {pct}% ({status_desc})")
        slow = st["rt"] < 1.0 or st["lag_ms"] > 100
        self._show_warn("Simulation is heavy for this machine: running in slow motion. "
                        "Reduce size or increase time step." if slow else "")
