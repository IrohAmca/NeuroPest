"""Control window: live readouts and settings, grouped into categories with a sidebar navigation."""
from __future__ import annotations

import importlib.util
import json
import threading

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
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

from .paths import CACHE, EYE, FIELD, SKIN_THUMBNAILS, TIERS
from .render import AVAILABLE_SKINS, SKIN_METADATA
from .runner import GPU_AUTO_MIN_NEURONS, EngineConfig, list_gpus, pick_gpu

from .theme import ERROR, FAINT, MUTED, STATE_STYLE, fly_icon

from .visionparams import VisionParams

# circuit sizes offered in the UI (neurons). FlyWire sizes are tiers measured by tools/fidelity.py.
FLYWIRE_SIZES = [2_000, 5_000, 10_000, 15_000, 20_000, 50_000, 138_639]
FLYWIRE_DEFAULT = 15_000
TOY_SIZES = [146, 500, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 139_000]
SCAN_ITEM = "GPU'ları ara…"
DTS = [("Hassas (0.1 ms)", 0.1), ("Dengeli (0.5 ms)", 0.5), ("Hızlı (1 ms)", 1.0)]


def load_tiers() -> dict[int, dict]:
    try:
        return {t["n"]: t for t in json.loads(TIERS.read_text())["tiers"]}
    except (OSError, ValueError, KeyError):
        return {}


def load_tier_machine() -> str:
    """Machine tools/fidelity.py measured the tier speeds on (its `machine` field), for the tooltip."""
    try:
        return str(json.loads(TIERS.read_text()).get("machine") or "bilinmeyen makine")
    except (OSError, ValueError):
        return "bilinmeyen makine"


def _label(text: str = "", name: str | None = None, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    return lab


class Card(QFrame):
    """Rounded panel with a small upper-case heading; add rows to `.body`."""

    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 18)
        lay.setSpacing(12)
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


class Control(QWidget):
    def __init__(self, overlay, runner):
        super().__init__()
        self.runner, self.overlay = runner, overlay
        self.tiers = load_tiers()
        self.tier_machine = load_tier_machine()
        self.setObjectName("Control")
        self.setWindowTitle("NeuroPest")
        self.setWindowIcon(fly_icon())
        self.resize(760, 560)
        self.setMinimumSize(620, 440)

        # Root two-column layout: Sidebar on the left, Content pages on the right
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._setup_sidebar(root)
        self._setup_content(root)

        self._load_sizes(runner.cfg.n)
        self._debounce = QTimer(self, singleShot=True, interval=500, timeout=self._apply)
        self.circ.currentIndexChanged.connect(self._circuit_changed)
        self.size.valueChanged.connect(self._size_moved)
        self.dt.currentIndexChanged.connect(lambda _: self._debounce.start())
        self.hw.currentIndexChanged.connect(lambda _: self._debounce.start())
        self._t = QTimer(self, timeout=self._on_refresh_timer, interval=250)
        self._t.start()
        self._refresh()

    def _on_refresh_timer(self):
        if self.isVisible():
            self._refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh()

    # ------------------------------------------------------------------ sidebar
    def _setup_sidebar(self, parent_layout: QHBoxLayout):
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)
        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(16, 20, 16, 18)
        lay.setSpacing(10)

        # App Brand Header
        brand = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(fly_icon().pixmap(32, 32))
        brand.addWidget(icon)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        titles.addWidget(_label("NeuroPest", "Title"))
        titles.addWidget(_label("FlyWire Pet", "Subtitle"))
        brand.addLayout(titles)
        brand.addStretch(1)
        lay.addLayout(brand)

        # State Pill right under header
        self.pill = QLabel()
        self.pill.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.pill)

        lay.addSpacing(14)
        lay.addWidget(_label("KATEGORİLER", "Section"))

        # Navigation buttons
        self.nav_buttons: list[QPushButton] = []
        nav_items = [
            ("Canlı İzleme", 0),
            ("Davranış", 1),
            ("Görsel Girdi", 2),
            ("Feromon && Koku", 3),
            ("Devre && Donanım", 4),
            ("Görünüm", 5),
            ("Öğrenme && Hafıza", 6),
        ]
        for title, idx in nav_items:
            btn = QPushButton(title)
            btn.setObjectName("NavBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("active", "true" if idx == 0 else "false")
            btn.clicked.connect(lambda _, i=idx: self._switch_tab(i))
            self.nav_buttons.append(btn)
            lay.addWidget(btn)

        lay.addStretch(1)
        lay.addWidget(_label("FlyWire v783 connectome\nShiu et al. (Nature 2024)", "Faint", wrap=True))

        parent_layout.addWidget(sidebar)

    def _switch_tab(self, idx: int):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_buttons):
            active_str = "true" if i == idx else "false"
            btn.setProperty("active", active_str)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ------------------------------------------------------------------ content
    def _setup_content(self, parent_layout: QHBoxLayout):
        scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_wrap = QWidget()
        scroll.setWidget(content_wrap)

        content_lay = QVBoxLayout(content_wrap)
        content_lay.setContentsMargins(24, 20, 24, 20)
        content_lay.setSpacing(16)

        self.stack = QStackedWidget()

        # Page 0: Canlı İzleme
        p0 = QWidget()
        p0_lay = QVBoxLayout(p0)
        p0_lay.setContentsMargins(0, 0, 0, 0)
        p0_lay.setSpacing(14)
        self._build_live_page(p0_lay)
        p0_lay.addStretch(1)
        self.stack.addWidget(p0)

        # Page 1: Davranış
        p1 = QWidget()
        p1_lay = QVBoxLayout(p1)
        p1_lay.setContentsMargins(0, 0, 0, 0)
        p1_lay.setSpacing(14)
        self._build_behaviour_page(p1_lay)
        p1_lay.addStretch(1)
        self.stack.addWidget(p1)

        # Page 2: Görsel Girdi
        p2 = QWidget()
        p2_lay = QVBoxLayout(p2)
        p2_lay.setContentsMargins(0, 0, 0, 0)
        p2_lay.setSpacing(14)
        self._build_vision_page(p2_lay)
        p2_lay.addStretch(1)
        self.stack.addWidget(p2)

        # Page 3: Feromon & Koku
        p3 = QWidget()
        p3_lay = QVBoxLayout(p3)
        p3_lay.setContentsMargins(0, 0, 0, 0)
        p3_lay.setSpacing(14)
        self._build_pheromone_page(p3_lay)
        p3_lay.addStretch(1)
        self.stack.addWidget(p3)

        # Page 4: Devre & Donanım
        p4 = QWidget()
        p4_lay = QVBoxLayout(p4)
        p4_lay.setContentsMargins(0, 0, 0, 0)
        p4_lay.setSpacing(14)
        self._build_circuit_page(p4_lay)
        p4_lay.addStretch(1)
        self.stack.addWidget(p4)

        # Page 5: Görünüm
        p5 = QWidget()
        p5_lay = QVBoxLayout(p5)
        p5_lay.setContentsMargins(0, 0, 0, 0)
        p5_lay.setSpacing(14)
        self._build_view_page(p5_lay)
        p5_lay.addStretch(1)
        self.stack.addWidget(p5)

        # Page 6: Öğrenme & Hafıza
        p6 = QWidget()
        p6_lay = QVBoxLayout(p6)
        p6_lay.setContentsMargins(0, 0, 0, 0)
        p6_lay.setSpacing(14)
        self._build_learning_page(p6_lay)
        p6_lay.addStretch(1)
        self.stack.addWidget(p6)

        content_lay.addWidget(self.stack)
        parent_layout.addWidget(scroll, 1)

    # ------------------------------------------------------------------ pages
    def _build_live_page(self, lay: QVBoxLayout):
        card = Card("Canlı Ölçümler")
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(14)
        self.s_gf = Stat("Giant Fiber", "Kaçış (uçuş) komut nöronunun ateşleme hızı")
        self.s_mdn = Stat("MDN", "Geri yürüme komut nöronlarının ateşleme hızı")
        self.s_steer = Stat("Yön", "Sağ eksi sol DNa02 hızı (pozitif: sağa döner)")
        self.s_rt = Stat("Gerçek zaman", "Simülasyonun gerçek zamana oranı; 1'in altı yavaş çekim demek")
        self.s_cpu = Stat("İşlemci", "Motor sürecinin tek çekirdek kullanımı")
        self.s_spk = Stat("Spike / sn")
        for i, s in enumerate([self.s_gf, self.s_mdn, self.s_steer, self.s_rt, self.s_cpu, self.s_spk]):
            grid.addWidget(s, i // 3, i % 3)
        card.body.addLayout(grid)
        self.active = _label("", "Muted")
        card.body.addWidget(self.active)
        self.telemetry = _label("", "Faint", wrap=True)
        card.body.addWidget(self.telemetry)
        self.warn = _label("", "Warn", wrap=True)
        self.warn.hide()
        card.body.addWidget(self.warn)
        lay.addWidget(card)

    def _build_behaviour_page(self, lay: QVBoxLayout):
        # 1. Açlık & Metabolizma (Temel Biyolojik Dürtü)
        hunger_card = Card("Açlık & Metabolizma")
        self.hunger_enable_box = QCheckBox("Açlık ve Metabolizma Simülasyonu")
        self.hunger_enable_box.setToolTip(
            "Açıkken sinek metabolik enerji harcar ve acıktıkça besin arar. "
            "Kapalıyken sürekli aç (v0.6) modunda çalışır."
        )
        is_hunger_on = getattr(getattr(self.overlay, "metabolism", None), "enabled", True)
        self.hunger_enable_box.setChecked(is_hunger_on)
        self.hunger_enable_box.toggled.connect(self._on_hunger_enable_toggled)
        hunger_card.body.addWidget(self.hunger_enable_box)

        cur_mult = getattr(getattr(getattr(self.overlay, "metabolism", None), "cfg", None), "rate_mult", 1.0)
        h_slider = QSlider(Qt.Horizontal, minimum=20, maximum=300, value=int(cur_mult * 100))
        h_slider.valueChanged.connect(self._on_metabolic_rate_changed)
        slider_row(hunger_card, "Metabolizma Hızı",
                   "Açlığın ne kadar hızlı geliştiğini belirler (Yavaş: ~5 dk, Dengeli: ~90 sn, Hızlı: ~30 sn).",
                   h_slider, lambda v: f"×{v/100:.1f}")

        status_row = QHBoxLayout()
        status_row.addWidget(_label("Mevcut Durum:"))
        status_row.addStretch(1)
        self.hunger_label = _label("–", "Value")
        status_row.addWidget(self.hunger_label)
        hunger_card.body.addLayout(status_row)

        btn_row = QHBoxLayout()
        self.btn_starve = QPushButton("⚡ Sineği Acıktır")
        self.btn_starve.setToolTip("Sineğin enerjisini anında tüketerek besin arama dürtüsünü (foraging) tetikler.")
        self.btn_starve.clicked.connect(self._on_starve_clicked)
        btn_row.addWidget(self.btn_starve)

        self.btn_feed = QPushButton("🍯 Karnını Doyur")
        self.btn_feed.setToolTip("Sineği anında doyurur; koku ilgisini kapatır ve dinlenme/temizlenme durumuna geçirir.")
        self.btn_feed.clicked.connect(self._on_feed_clicked)
        btn_row.addWidget(self.btn_feed)

        hunger_card.body.addLayout(btn_row)
        lay.addWidget(hunger_card)

        # 2. Temas & Tımar
        touch_card = Card("Temas & Tımar")
        self.touch_groom_box = QCheckBox("İmleç Dokunduğunda Kaşınma / Tımar (Grooming)")
        self.touch_groom_box.setToolTip(
            "İmleç sineğin üzerine geldiğinde mekanik dokunma nöronlarını (aDN1/aDN2) uyararak "
            "sineğin durup başını kaşımasını sağlar. Kapalıyken (varsayılan) imleç teması kaşınmayı zorlamaz."
        )
        is_touch_on = getattr(self.overlay, "touch_groom_enabled", False) if self.overlay is not None else False
        self.touch_groom_box.setChecked(is_touch_on)
        self.touch_groom_box.toggled.connect(self._on_touch_groom_toggled)
        touch_card.body.addWidget(self.touch_groom_box)
        lay.addWidget(touch_card)

        # 3. Gelişmiş Davranış Ayarları (Katlanabilir Akordeon Kart)
        adv_card = Card("Gelişmiş Davranış Ayarları")
        self.adv_btn = QPushButton("▸ Gelişmiş Parametreleri Göster")
        self.adv_btn.setFlat(True)
        self.adv_btn.setCursor(Qt.PointingHandCursor)
        self.adv_btn.setStyleSheet("text-align: left; font-size: 13px; font-weight: 600; padding: 4px 0;")
        self.adv_btn.clicked.connect(self._toggle_adv_settings)
        adv_card.body.addWidget(self.adv_btn)

        self.adv_content = QWidget()
        adv_lay = QVBoxLayout(self.adv_content)
        adv_lay.setContentsMargins(0, 8, 0, 0)
        adv_lay.setSpacing(14)

        self.w_slider = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=int(self.runner.bias * 100))
        self.w_slider.valueChanged.connect(lambda v: setattr(self.runner, "bias", v / 100))
        slider_row(adv_lay, "Hareketlilik (Baz Yürüme Sürücüsü)",
                   "DNp09/P9 yürüme komut nöronlarına verilen tonik akım: açlık ve koku yokken taban istek. Varsayılan: %65.",
                   self.w_slider, lambda v: f"%{v}")

        cur_skittish = getattr(self.runner, "skittish", 1.0)
        import math
        k_val = int(round(50.0 + 25.0 * math.log2(max(0.1, cur_skittish))))
        self.k_slider = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=max(0, min(100, k_val)))
        self.k_slider.valueChanged.connect(lambda v: setattr(self.runner, "skittish", 2.0 ** ((v - 50) / 25.0)))
        slider_row(adv_lay, "Ürkeklik (Kaçış Duyarlılığı)",
                   "Yaklaşan nesnelere karşı hassasiyet: geri çekilme ve uçuş eşiklerini çarpar. Varsayılan: ×1.00.",
                   self.k_slider, lambda v: f"×{2.0 ** ((v - 50) / 25.0):.2f}")

        cur_pain = getattr(self.runner, "wall_pain", 1.0)
        self.pain_slider = QSlider(Qt.Horizontal, minimum=0, maximum=200, value=int(cur_pain * 100))
        self.pain_slider.valueChanged.connect(lambda v: setattr(self.runner, "wall_pain", v / 100.0))
        slider_row(adv_lay, "Kenar Acısı (Nosiseptif Darbe Cezası)",
                   "Sinek ekran sınırına tosladığında Mantar Cismi'ne iletilen acı/ceza (PPL1 dopamin). Sinek kenarlardan sakınmayı öğrenir. Varsayılan: ×1.00.",
                   self.pain_slider, lambda v: f"×{v / 100.0:.2f}")

        self.btn_reset_adv = QPushButton("↺ Varsayılan Parametrelere Sıfırla")
        self.btn_reset_adv.setToolTip("Hareketlilik (%65), ürkeklik (×1.00) ve kenar acısı (×1.00) değerlerini varsayılana döndürür.")
        self.btn_reset_adv.clicked.connect(self._reset_adv_settings)
        adv_lay.addWidget(self.btn_reset_adv)

        self.adv_content.setVisible(False)
        adv_card.body.addWidget(self.adv_content)
        lay.addWidget(adv_card)

    def _toggle_adv_settings(self):
        visible = not self.adv_content.isVisible()
        self.adv_content.setVisible(visible)
        arrow = "▾" if visible else "▸"
        self.adv_btn.setText(f"{arrow} Gelişmiş Parametreleri {'Gizle' if visible else 'Göster'}")

    def _reset_adv_settings(self):
        self.w_slider.setValue(65)
        self.k_slider.setValue(50)
        self.pain_slider.setValue(100)

    def _on_touch_groom_toggled(self, checked: bool):
        if self.overlay is not None:
            self.overlay.touch_groom_enabled = checked

    def _on_hunger_enable_toggled(self, checked: bool):
        if self.overlay and hasattr(self.overlay, "metabolism"):
            self.overlay.metabolism.enabled = checked

    def _on_metabolic_rate_changed(self, value: int):
        if self.overlay and hasattr(self.overlay, "metabolism"):
            self.overlay.metabolism.cfg.rate_mult = value / 100.0

    def _on_starve_clicked(self):
        if self.overlay and hasattr(self.overlay, "metabolism"):
            self.overlay.metabolism.starve()

    def _on_feed_clicked(self):
        if self.overlay and hasattr(self.overlay, "metabolism"):
            self.overlay.metabolism.satiate()

    def _build_vision_page(self, lay: QVBoxLayout):
        card = Card("Görsel Girdi & Ekran Yakalama")
        self.has_eye = CACHE.exists() and EYE.exists() and FIELD.exists()
        self.vision = QCheckBox("Ekranı sineğin gözüyle gör (Gerçek Ekran Yakalama)", enabled=self.has_eye)
        self.vision.setToolTip("İmleç sayıları yerine gerçek masaüstü görüntüsü: 480 px huni görüşü, retinotopik dedektörler")
        self.vision.toggled.connect(lambda on: setattr(self.runner, "vision", on))
        card.body.addWidget(self.vision)

        hgt = QSlider(Qt.Horizontal, minimum=40, maximum=300, value=int(self.runner.eye_height),
                      enabled=self.has_eye)
        hgt.valueChanged.connect(lambda v: setattr(self.runner, "eye_height", float(v)))
        slider_row(card, "Göz Yüksekliği (Bakış Eğimi)",
                   "Ekran düzleminin kaç px üstünden bakıyor: büyük = daha dikey (tepeden) huni açısı.",
                   hgt, lambda v: f"{v} px")

        self.vision_info = _label("", "Faint", wrap=True)
        card.body.addWidget(self.vision_info)
        lay.addWidget(card)
        self._describe_vision(0.0)

    def _describe_vision(self, state: float):
        if not self.has_eye:
            text = "Göz verisi yok: uv run python tools/build_eye.py (ham veri gerekir, README'ye bak)."
        elif state > 0:
            text = ("Gerçek ekran yakalama aktif: Sinek masaüstünü 480 px huni görüşüyle görüyor; "
                    "kaçış (uçuş), geri yürüme, donma ve yönelme kararları ekrandaki gerçek piksellerden geliyor.")
        elif state < 0 and self.vision.isChecked():
            text = "Bu devrede görsel girdi yok (alıcı alanı olan LPLC2/LC4 gerekir): imleç sayılarıyla çalışılıyor."
        else:
            text = ""
        self.vision_info.setText(text)
        self.vision_info.setVisible(bool(text))

    def _build_pheromone_page(self, lay: QVBoxLayout):
        card = Card("Feromon Alanı & Koku Duyusu")

        self.phero_enable = QCheckBox("Feromon ve Koku Duyusunu Etkinleştir")
        self.phero_enable.setToolTip("Sineğin iki anteniyle (tropotaksis) feromon gradyanını koklayıp yönelmesini sağlar.")
        is_on = getattr(self.overlay, "pheromone_enabled", True) if self.overlay is not None else True
        self.phero_enable.setChecked(is_on)
        self.phero_enable.toggled.connect(self._on_phero_toggled)
        card.body.addWidget(self.phero_enable)

        self.cursor_phero_combo = QComboBox()
        self.cursor_phero_modes = [
            ("attract", "Olumlu / Çekici (Besin / Nektar kokusu — sinek yaklaşır)"),
            ("repel", "Olumsuz / İtici (Tehdit kokusu — sinek uzaklaşır)"),
            ("none", "Kapalı (İmleç feromon yaymaz)"),
        ]
        self.cursor_phero_combo.addItems([label for _, label in self.cursor_phero_modes])
        cur_mode = getattr(self.overlay, "cursor_phero_mode", "attract") if self.overlay is not None else "attract"
        cur_idx = next((i for i, (m, _) in enumerate(self.cursor_phero_modes) if m == cur_mode), 0)
        self.cursor_phero_combo.setCurrentIndex(cur_idx)
        self.cursor_phero_combo.currentIndexChanged.connect(self._on_cursor_phero_changed)
        labeled(card, "İmleç Feromon Modu", self.cursor_phero_combo)
        lay.addWidget(card)

        desc_card = Card("Feromon & Tropotaksis Nedir?")
        info_text = (
            "• Çift Antenle Koku Yönelimi (Tropotaksis): Sinek, sağ ve sol antenlerindeki koku "
            "reseptörleri arasındaki yoğunluk farkını karşılaştırarak kokunun yoğun olduğu tarafa "
            "doğru yönelir veya itici kokudan kaçar (DNa02 dönüş nöronları üzerinden).\n\n"
            "• Görünmez Besin Kaynakları: Ekranda rastgele noktalarda görünmez nektar/besin kaynakları bulunur. "
            "Sinek kokuyu takip ederek besine ulaşır, hedefe vardığında besini tüketir ve duraklar.\n\n"
            "• Kenar İticiliği: Ekranın en dış sınırlarında (18 px) sineğin ekran arkasına/dışına kaçmasını "
            "önleyen ince ve görünmez bir itici alan bulunur.\n\n"
            "• İmleç Etkisi: İmleci olumlu seçerseniz sinek imlecin peşinden koşar ve üstüne konar; "
            "olumsuz seçerseniz imleçten kaçar."
        )
        desc_card.body.addWidget(_label(info_text, "Faint", wrap=True))
        lay.addWidget(desc_card)

    def _on_phero_toggled(self, checked: bool):
        if self.overlay is not None:
            self.overlay.pheromone_enabled = checked

    def _on_cursor_phero_changed(self, idx: int):
        if self.overlay is not None and 0 <= idx < len(self.cursor_phero_modes):
            self.overlay.cursor_phero_mode = self.cursor_phero_modes[idx][0]

    def _build_learning_page(self, lay: QVBoxLayout):
        card = Card("Öğrenme & Hafıza")
        card.body.addWidget(_label(
            "Sinek yaşadıklarından öğrenir. Bir kokuyla ya da imleçle birlikte beslenirse onu sever ve ona yönelir; "
            "o sırada korkarsa ondan kaçınır. Öğrendiği, uygulama kapanınca da saklanır.\n\n"
            "Unutma üç şekilde olur: anılar zamanla yavaşça silinir, ödül ya da korku gelmeden tekrarlanan "
            "ipucu onu çabuk unutturur, \"Hafızayı sil\" ile de hepsi hemen silinir.", "Muted", wrap=True))
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        self.s_val = Stat("Değerlik", "Şu an algıladığı şeye karşı öğrenilmiş his: + arzu, − korku")
        self.s_vm = Stat("V_motor", "Tehdit + arzu + ulaşamama gerilimi")
        self.s_gear = Stat("Vites", "Durma, yürüme, kaçış uçuşu ya da kovalama uçuşu")
        for i, s in enumerate((self.s_val, self.s_vm, self.s_gear)):
            grid.addWidget(s, 0, i)
        card.body.addLayout(grid)
        self.learn_enable = QCheckBox("Öğrenme açık")
        self.learn_enable.setToolTip("Kapalıysa sinek hiçbir şey öğrenmez; motor yeniden başlar.")
        self.learn_enable.setChecked(getattr(self.runner.cfg, "learning", True))
        self.learn_enable.toggled.connect(lambda _: self._apply())
        card.body.addWidget(self.learn_enable)
        self.forget_btn = QPushButton("Hafızayı sil")
        self.forget_btn.setToolTip("Sinek öğrendiği her şeyi hemen unutur.")
        self.forget_btn.clicked.connect(self._forget)
        card.body.addWidget(self.forget_btn)
        lay.addWidget(card)

    def _forget(self):
        forget = getattr(self.runner, "forget", None)
        if forget:
            forget()

    def _build_circuit_page(self, lay: QVBoxLayout):
        card = Card("Sinir Devresi")
        self.circuits = ([("flywire", "FlyWire v783 (gerçek bağlantı)")] if CACHE.exists() else []) \
            + [("toy", "Oyuncak devre (sentetik yük)")]
        self.circ = QComboBox()
        self.circ.addItems([name for _, name in self.circuits])
        self.circ.setCurrentIndex([c for c, _ in self.circuits].index(self.runner.cfg.circuit))
        labeled(card, "Devre Tipi", self.circ)

        if not CACHE.exists():
            card.body.addWidget(_label("Gerçek devre için: uv run python tools/build_flywire.py "
                                       "(README'ye bak)", "Hint", wrap=True))

        head = QHBoxLayout()
        head.addWidget(_label("Devre Boyutu (Nöron Sayısı)"))
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
        labeled(card, "Zaman adımı dt (küçük: daha doğru, daha ağır)", self.dt)
        lay.addWidget(card)

        # Gelişmiş Seçenekler
        adv_card = Card("Gelişmiş Seçenekler")
        self.sym_options = [
            ("individual", "Bireysel (FlyWire orijinal)"),
            ("symmetric", "Simetrik (Sağ-sol dengeli)"),
        ]
        self.sym_combo = QComboBox()
        self.sym_combo.addItems([label for _, label in self.sym_options])
        cur_sym = getattr(self.runner.cfg, "symmetry", "individual")
        self.sym_combo.setCurrentIndex(0 if cur_sym == "individual" else 1)
        self.sym_combo.setEnabled(self._kind() == "flywire")
        self.sym_combo.currentIndexChanged.connect(lambda _: self._debounce.start())
        labeled(adv_card, "Bağlantı Simetrisi", self.sym_combo)
        self.sym_hint = _label(
            "Bireysel: FlyWire'ın tek sinek beyni (orijinal biyolojik asimetri korunur).\n"
            "Simetrik: Sağ ve sol yarımküre bağlantı ağırlıkları ortalamaya eşitlenir (sağ-sol dönüş döngüsünü dengeler).",
            "Faint", wrap=True)
        adv_card.body.addWidget(self.sym_hint)
        lay.addWidget(adv_card)

        # Compute sub-card
        comp_card = Card("Hesaplama Donanımı")
        self.hw = QComboBox()
        self.hw.addItems(["Otomatik", "CPU"])
        self.hw.setToolTip(f"Otomatik: {GPU_AUTO_MIN_NEURONS:,} nöron ve üstünde GPU, altında CPU")
        labeled(comp_card, "İşlemci / GPU Tercihi", self.hw)

        self.gpus: list[dict] = []
        self._gpu_result: list | None = None
        self._gpu_scanned = False
        self._gpu_scanning = False
        self._apply_pending = False
        self.has_wgpu = importlib.util.find_spec("wgpu") is not None
        if self.has_wgpu:
            self.hw.addItem(SCAN_ITEM)
            self.hw.activated.connect(self._hw_activated)
            self.gpu_note = _label("GPU'lar arama isteğiyle bulunur (~1 s, ~100 MB)", "Faint")
            comp_card.body.addWidget(self.gpu_note)
            self._gpu_poll = QTimer(self, timeout=self._gpus_found, interval=250)
        else:
            comp_card.body.addWidget(_label("GPU desteği için: uv sync --extra gpu", "Hint", wrap=True))
        lay.addWidget(comp_card)

    def _build_view_page(self, lay: QVBoxLayout):
        card = Card("Görünüm & Monitör")
        s = QSlider(Qt.Horizontal, minimum=5, maximum=40, value=int(self.overlay.scale * 10))
        s.valueChanged.connect(lambda v: setattr(self.overlay, "scale", v / 10))
        slider_row(card, "Sinek Boyutu", "Masaüstündeki görünür büyüklük", s, lambda v: f"×{v / 10:.1f}")

        # Sinek Görünümü (Visual Cards Selector)
        card.body.addWidget(_label("Sinek Görünümü (Kostüm)", "Muted"))

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
            btn.setFixedSize(84, 76)

            thumb_path = SKIN_THUMBNAILS / f"{key}.png"
            if thumb_path.exists():
                btn.setIcon(QIcon(str(thumb_path)))
            btn.setIconSize(QSize(38, 38))
            btn.setText(meta["title"])
            btn.setToolTip(f"{meta['full_name']}\n{meta['desc']}")

            btn.setStyleSheet("""
                QToolButton {
                    background: #11151c;
                    border: 1px solid #27303f;
                    border-radius: 8px;
                    padding: 6px 2px 4px 2px;
                    color: #94a3b8;
                    font-size: 11px;
                    font-weight: 500;
                }
                QToolButton:hover {
                    background: #18202c;
                    border: 1px solid #475569;
                    color: #f1f5f9;
                }
                QToolButton:checked {
                    background: rgba(52, 211, 153, 0.14);
                    border: 2px solid #34d399;
                    color: #34d399;
                    font-weight: bold;
                }
            """)

            self.skin_buttons[key] = btn
            self.skin_group.addButton(btn)
            btn.clicked.connect(lambda _, k=key: self._on_skin_selected(k))
            skin_lay.addWidget(btn)

        skin_lay.addStretch(1)
        card.body.addWidget(skin_container)


        active_meta = SKIN_METADATA.get(cur_skin, SKIN_METADATA["classic"])
        self.skin_desc_label = _label(f"✓ {active_meta['full_name']}: {active_meta['desc']}", "Faint", wrap=True)
        card.body.addWidget(self.skin_desc_label)

        screens = QApplication.screens()
        if len(screens) > 1:
            box = QComboBox()
            items = ["Tüm Ekranlar (Bağımsız Gezinme)"] + [f"{i + 1}: {s.name()}" for i, s in enumerate(screens)]
            box.addItems(items)
            current_idx = 0 if self.overlay.home is None else (screens.index(self.overlay.home) + 1 if self.overlay.home in screens else 0)
            box.setCurrentIndex(current_idx)
            box.currentIndexChanged.connect(lambda i: self.overlay.set_home(None if i == 0 else screens[i - 1]))
            labeled(card, "Sanal Alan / Monitör", box)
        lay.addWidget(card)

    def _on_skin_selected(self, skin_key: str):
        if self.overlay is not None:
            self.overlay.skin = skin_key
            self.overlay.update()
        if skin_key in SKIN_METADATA and hasattr(self, "skin_desc_label"):
            meta = SKIN_METADATA[skin_key]
            self.skin_desc_label.setText(f"✓ {meta['full_name']}: {meta['desc']}")
        if skin_key in self.skin_buttons and not self.skin_buttons[skin_key].isChecked():
            self.skin_buttons[skin_key].setChecked(True)

    def _on_skin_changed(self, idx: int):
        keys = list(SKIN_METADATA.keys())
        if 0 <= idx < len(keys):
            self._on_skin_selected(keys[idx])



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
        self._debounce.start()

    def _size_moved(self, _):
        self._describe()
        self._debounce.start()

    def _describe(self):
        n = self._sizes()[self.size.value()]
        self.size_label.setText(f"{n:,} nöron")
        if self._kind() == "flywire":
            t = self.tiers.get(n)
            if t:
                self.tier_info.setText(
                    f"Tam beyne göre sapma: kalkış %{t['gf_err']:.0f} · geri yürüme %{t['mdn_err']:.0f} · "
                    f"yön %{t['steer_err']:.0f}\nDescending korelasyonu {t['dn_corr']:.3f} · ölçülen hız "
                    f"×{t['realtime']:.1f} (en kötü ×{t['rt_min']:.1f})")
                pinned = (f"Sayıdaki ilk {t['pinned']:,} nöron her katmanda sabit olan girdi ve çıktı nöronlarıdır; "
                          f"sıralamayla seçilen {t['free']:,}. " if t.get("pinned") else "")
                self.tier_info.setToolTip("FlyWire'daki ateşleme sırasına göre seçilen iç içe katman. " + pinned +
                                          "Sapma yalnız yaklaşan nesne, geri çekilme ve yön girdileri "
                                          "için ölçüldü (tools/fidelity.py). Hız, ölçümün yapıldığı "
                                          f"makineye özeldir ({self.tier_machine}).")
            else:
                self.tier_info.setText("Bu boyut için ölçüm yok.")
        else:
            self.tier_info.setText("" if n == TOY_SIZES[0] else "Sentetik yük: davranışı değiştirmez, "
                                   "yalnız hesaplama maliyetini dener.")
        self.tier_info.setVisible(bool(self.tier_info.text()))

    def _hw_activated(self, i: int):
        if not self._gpu_scanned and self.hw.itemText(i) == SCAN_ITEM:
            self.hw.setCurrentIndex(0)
            self._scan_gpus()

    def _scan_gpus(self):
        if self._gpu_scanned or self._gpu_scanning or not self.has_wgpu:
            return
        self._gpu_scanning = True
        self.gpu_note.setText("GPU'lar aranıyor…")
        threading.Thread(target=lambda: setattr(self, "_gpu_result", list_gpus()), daemon=True).start()
        self._gpu_poll.start()

    def _gpus_found(self):
        if self._gpu_result is None:
            return
        self._gpu_poll.stop()
        self._gpu_scanning, self._gpu_scanned = False, True
        self.gpus = self._gpu_result
        self.hw.blockSignals(True)
        if self.hw.itemText(2) == SCAN_ITEM:
            self.hw.removeItem(2)
        self.hw.addItems([f"GPU: {g['name']} ({g['backend']})" for g in self.gpus])
        self.hw.blockSignals(False)
        self.gpu_note.setText(f"{len(self.gpus)} GPU bulundu" if self.gpus else "Kullanılabilir GPU bulunamadı")
        if self._apply_pending:
            self._apply_pending = False
            self._apply()

    def _hardware(self, n: int) -> tuple[str, int | None]:
        i = self.hw.currentIndex()
        if i >= 2 and self._gpu_scanned:
            return "gpu", self.gpus[i - 2]["index"]
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
                           learning=self.learn_enable.isChecked(), mb_wiring=self.runner.cfg.mb_wiring)
        if cfg != self.runner.cfg:
            self.runner.start(cfg)

    # ---------------------------------------------------------------- telemetry
    def _set_pill(self, text: str, color: str):
        self.pill.setText(f"●  {text}")
        self.pill.setStyleSheet(
            f"color: {color}; background: rgba(255,255,255,0.04); border: 1px solid {color}55;"
            "border-radius: 13px; padding: 4px 12px; font-size: 12px; font-weight: 600;")

    def _show_warn(self, text: str, error: bool = False):
        self.warn.setObjectName("Error" if error else "Warn")
        self.warn.style().unpolish(self.warn)
        self.warn.style().polish(self.warn)
        self.warn.setText(text)
        self.warn.setVisible(bool(text))

    def _clear_stats(self):
        for s in (self.s_gf, self.s_mdn, self.s_steer, self.s_rt, self.s_cpu, self.s_spk,
                  self.s_val, self.s_vm, self.s_gear):
            s.set("–")
        self.active.setText("")

    def _refresh(self):
        r = self.runner
        if r.failed or not r.alive:
            self._set_pill("Hata", ERROR)
            self._clear_stats()
            self.telemetry.setText("")
            self._show_warn("Motor hata verdi ya da durdu (ayrıntı konsolda). Daha küçük bir boyut seçmeyi dene.",
                            error=True)
            return
        st = r.stats()
        if not st["ready"]:
            self._set_pill("Başlıyor", MUTED)
            self._clear_stats()
            self.telemetry.setText("Motor başlıyor (ilk açılışta derleme birkaç saniye sürer)…")
            self._show_warn("")
            return
        color, name = STATE_STYLE.get(r.state, (FAINT, r.state))
        self._set_pill(name, color)
        self.s_gf.set(f"{st['gf']:.0f} Hz")
        self.s_mdn.set(f"{st['mdn']:.0f} Hz")
        self.s_steer.set(f"{st['steer']:+.0f}")
        self.s_rt.set(f"×{st['rt']:.1f}")
        self.s_cpu.set(f"%{100 * st['cpu']:.0f}")
        self.s_spk.set(f"{st['spikes']:,.0f}")
        v = st.get("valence", 0.0)
        self.s_val.set(f"{v:+.2f} " + ("arzu" if v > 0.05 else "korku" if v < -0.05 else "nötr"))
        self.s_vm.set(f"{st.get('v_motor', 0.0):.2f}")
        self.s_gear.set({"stand": "durma", "walk": "yürüme", "fly_short": "kaçış uçuşu",
                         "fly_long": "kovalama uçuşu"}.get(st.get("gear", "stand"), "–"))
        self.active.setText(f"Aktif nöron: {st['active']:,.0f} / {st['n']:,}" if st["active"] >= 0
                            else f"Nöron: {st['n']:,} (GPU hepsini her adımda günceller)")
        where = "CPU"
        if r.cfg.backend == "gpu":
            g = next((g for g in self.gpus if g["index"] == r.cfg.adapter), None)
            where = f"GPU {g['name']} ({g['backend']})" if g else "GPU"
        self.telemetry.setText(f"Çalışan: {where} · zaman adımı {r.cfg.dt:g} ms")
        self._describe_vision(st.get("vision", 0.0))
        if hasattr(self, "hunger_label") and self.overlay and hasattr(self.overlay, "metabolism"):
            m = self.overlay.metabolism
            if not m.enabled:
                self.hunger_label.setText("Devre Dışı (Sürekli Aç)")
            else:
                pct = m.hunger_pct
                status_desc = "Çok Aç" if pct >= 75 else "İştahlı" if pct >= 40 else "Tok"
                self.hunger_label.setText(f"Açlık: %{pct} ({status_desc})")
        slow = st["rt"] < 1.0 or st["lag_ms"] > 100
        self._show_warn("Bu ayar bu bilgisayar için ağır: sinek yavaş çekimde. "
                        "Boyutu küçült ya da zaman adımını büyüt." if slow else "")
