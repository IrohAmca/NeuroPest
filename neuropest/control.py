"""Control window: live readouts and settings, grouped into categories with a sidebar navigation."""
from __future__ import annotations

import importlib.util
import json
import threading

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .paths import CACHE, EYE, FIELD, TIERS
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


def slider_row(card: Card, name: str, hint: str, slider: QSlider, fmt) -> QLabel:
    """Name and live value on one line, the slider bar under it, an optional muted hint below."""
    head = QHBoxLayout()
    head.addWidget(_label(name))
    head.addStretch(1)
    value = _label("", "Value")
    head.addWidget(value)
    card.body.addLayout(head)
    card.body.addWidget(slider)
    if hint:
        card.body.addWidget(_label(hint, "Faint", wrap=True))
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
        self._t = QTimer(self, timeout=self._refresh, interval=250)
        self._t.start()
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
            ("Devre & Donanım", 3),
            ("Görünüm", 4),
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

        # Page 3: Devre & Donanım
        p3 = QWidget()
        p3_lay = QVBoxLayout(p3)
        p3_lay.setContentsMargins(0, 0, 0, 0)
        p3_lay.setSpacing(14)
        self._build_circuit_page(p3_lay)
        p3_lay.addStretch(1)
        self.stack.addWidget(p3)

        # Page 4: Görünüm
        p4 = QWidget()
        p4_lay = QVBoxLayout(p4)
        p4_lay.setContentsMargins(0, 0, 0, 0)
        p4_lay.setSpacing(14)
        self._build_view_page(p4_lay)
        p4_lay.addStretch(1)
        self.stack.addWidget(p4)

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
        card = Card("Davranış Ayarları")
        w = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=int(self.runner.bias * 100))
        w.valueChanged.connect(lambda v: setattr(self.runner, "bias", v / 100))
        slider_row(card, "Hareketlilik (İleri Yürüme Sürücüsü)",
                   "Yürüme komut nöronlarına (DNp09/P9) verilen tonik akım: düşükse durur, yüksekse gezer.",
                   w, lambda v: f"%{v}")

        k = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=50)
        k.valueChanged.connect(lambda v: setattr(self.runner, "skittish", 2.0 ** ((v - 50) / 25.0)))
        slider_row(card, "Ürkeklik (Kaçış Duyarlılığı)",
                   "Yaklaşan nesnelere karşı hassasiyet: geri çekilme ve uçuş eşiklerini çarpar.",
                   k, lambda v: f"×{2.0 ** ((v - 50) / 25.0):.2f}")
        lay.addWidget(card)

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

        self.visible = QCheckBox("Sinek görünür", checked=True)
        self.visible.toggled.connect(self.overlay.setVisible)
        card.body.addWidget(self.visible)

        screens = QApplication.screens()
        if len(screens) > 1:
            box = QComboBox()
            box.addItems([f"{i + 1}: {s.name()}" for i, s in enumerate(screens)])
            box.setCurrentIndex(screens.index(self.overlay.home))
            box.currentIndexChanged.connect(lambda i: self.overlay.set_home(screens[i]))
            labeled(card, "Sanal Alan / Monitör", box)
        lay.addWidget(card)

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
        cfg = EngineConfig(self._kind(), n, DTS[self.dt.currentIndex()][1], backend=backend, adapter=adapter)
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
        for s in (self.s_gf, self.s_mdn, self.s_steer, self.s_rt, self.s_cpu, self.s_spk):
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
        self.active.setText(f"Aktif nöron: {st['active']:,.0f} / {st['n']:,}" if st["active"] >= 0
                            else f"Nöron: {st['n']:,} (GPU hepsini her adımda günceller)")
        where = "CPU"
        if r.cfg.backend == "gpu":
            g = next((g for g in self.gpus if g["index"] == r.cfg.adapter), None)
            where = f"GPU {g['name']} ({g['backend']})" if g else "GPU"
        self.telemetry.setText(f"Çalışan: {where} · zaman adımı {r.cfg.dt:g} ms")
        self._describe_vision(st.get("vision", 0.0))
        slow = st["rt"] < 1.0 or st["lag_ms"] > 100
        self._show_warn("Bu ayar bu bilgisayar için ağır: sinek yavaş çekimde. "
                        "Boyutu küçült ya da zaman adımını büyüt." if slow else "")
