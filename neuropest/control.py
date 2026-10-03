"""Control window: live readouts and settings, grouped into cards on the dark theme."""
from __future__ import annotations

import importlib.util
import json
import threading

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget)

from .paths import CACHE, EYE, FIELD, TIERS
from .runner import GPU_AUTO_MIN_NEURONS, EngineConfig, list_gpus, pick_gpu
from .theme import ERROR, FAINT, MUTED, STATE_STYLE, fly_icon

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
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(10)
        lay.addWidget(_label(title.upper(), "Section"))
        self.body = lay


class Stat(QWidget):
    """Big number with a small caption under it."""

    def __init__(self, name: str, tip: str = ""):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        self.value = _label("–", "StatValue")
        lay.addWidget(self.value)
        lay.addWidget(_label(name, "StatName"))
        if tip:
            self.setToolTip(tip)

    def set(self, text: str):
        self.value.setText(text)


def slider_row(card: Card, name: str, hint: str, slider: QSlider, fmt) -> QLabel:
    """Name and live value on one line, the slider under it, an optional muted hint below."""
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
        self.setObjectName("Control")
        self.setWindowTitle("NeuroPest")
        self.setWindowIcon(fly_icon())
        self.resize(420, 780)
        self.setMinimumWidth(380)

        scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page = QWidget()
        scroll.setWidget(page)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.page = QVBoxLayout(page)
        self.page.setContentsMargins(18, 18, 18, 18)
        self.page.setSpacing(12)

        self._header()
        self._live_card()
        self._behaviour_card()
        self._vision_card()
        self._circuit_card()
        self._compute_card()
        self._view_card()
        self.page.addStretch(1)
        self.page.addWidget(_label("Bağlantı verisi: FlyWire v783 (Dorkenwald ve ark.; Schlegel ve ark., "
                                   "Nature 2024), CC-BY 4.0. Nöron modeli: Shiu ve ark. 2024.", "Faint", wrap=True))

        self._load_sizes(runner.cfg.n)
        self._debounce = QTimer(self, singleShot=True, interval=500, timeout=self._apply)
        self.circ.currentIndexChanged.connect(self._circuit_changed)
        self.size.valueChanged.connect(self._size_moved)
        self.dt.currentIndexChanged.connect(lambda _: self._debounce.start())
        self.hw.currentIndexChanged.connect(lambda _: self._debounce.start())
        self._t = QTimer(self, timeout=self._refresh, interval=250)
        self._t.start()
        self._refresh()

    # ------------------------------------------------------------------ layout
    def _header(self):
        row = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(fly_icon().pixmap(36, 36))
        row.addWidget(icon)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(_label("NeuroPest", "Title"))
        titles.addWidget(_label("FlyWire beyniyle yaşayan sinek", "Subtitle"))
        row.addLayout(titles)
        row.addStretch(1)
        self.pill = QLabel()
        self.pill.setAlignment(Qt.AlignCenter)
        row.addWidget(self.pill, 0, Qt.AlignVCenter)
        self.page.addLayout(row)

    def _live_card(self):
        card = Card("Canlı")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
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
        self.page.addWidget(card)

    def _behaviour_card(self):
        card = Card("Davranış")
        w = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=int(self.runner.bias * 100))
        w.valueChanged.connect(lambda v: setattr(self.runner, "bias", v / 100))
        slider_row(card, "Hareketlilik", "Yürüme komut nöronlarına verilen sürücü: düşükse durur, yüksekse gezer.",
                   w, lambda v: f"%{v}")
        k = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=50)
        k.valueChanged.connect(lambda v: setattr(self.runner, "skittish", 2.0 ** ((v - 50) / 25.0)))
        slider_row(card, "Ürkeklik", "Yaklaşan imlece duyarlılık: geri yürüme ve kaçış eşiklerini ölçekler.",
                   k, lambda v: f"×{2.0 ** ((v - 50) / 25.0):.2f}")
        self.page.addWidget(card)

    def _vision_card(self):
        card = Card("Görsel girdi")
        self.has_eye = CACHE.exists() and EYE.exists() and FIELD.exists()
        self.vision = QCheckBox("Ekranı sineğin gözüyle gör", enabled=self.has_eye)
        self.vision.setToolTip("İmleç sayıları yerine görüntü: huni görüşü, retinotopik dedektörler ve projeksiyon "
                               "nöronlarının bağlantıdan çıkarılan alıcı alanları (neuropest/vision.py)")
        self.vision.toggled.connect(lambda on: setattr(self.runner, "vision", on))
        card.body.addWidget(self.vision)
        hgt = QSlider(Qt.Horizontal, minimum=40, maximum=300, value=int(self.runner.eye_height), enabled=self.has_eye)
        hgt.valueChanged.connect(lambda v: setattr(self.runner, "eye_height", float(v)))
        slider_row(card, "Göz yüksekliği", "Ekran düzleminin kaç px üstünden bakıyor: büyük = daha dikey (tepeden) bakış.",
                   hgt, lambda v: f"{v} px")
        self.vision_info = _label("", "Faint", wrap=True)
        card.body.addWidget(self.vision_info)
        self.page.addWidget(card)
        self._describe_vision(0.0)

    def _describe_vision(self, state: float):
        """`state` is the worker's vision flag: 1 on, 0 off, -1 wanted but unusable."""
        if not self.has_eye:
            text = "Göz verisi yok: uv run python tools/build_eye.py (ham veri gerekir, README'ye bak)."
        elif state > 0:
            text = ("Sinek imleci kendi gözünden, yere yakın koyu bir disk olarak görüyor; yaklaşma ve yön bilgisi "
                    "retinotopik dedektörlerden geliyor.")
        elif state < 0 and self.vision.isChecked():
            text = "Bu devrede görsel girdi yok (alıcı alanı olan LPLC2/LC4 gerekir): imleç sayılarıyla çalışılıyor."
        else:
            text = ""
        self.vision_info.setText(text)
        self.vision_info.setVisible(bool(text))

    def _circuit_card(self):
        card = Card("Devre")
        self.circuits = ([("flywire", "FlyWire v783 (gerçek bağlantı)")] if CACHE.exists() else []) \
            + [("toy", "Oyuncak devre (sentetik yük)")]
        self.circ = QComboBox()
        self.circ.addItems([name for _, name in self.circuits])
        self.circ.setCurrentIndex([c for c, _ in self.circuits].index(self.runner.cfg.circuit))
        card.body.addWidget(self.circ)
        if not CACHE.exists():
            card.body.addWidget(_label("Gerçek devre için: uv run python tools/build_flywire.py "
                                       "(README'ye bak)", "Hint", wrap=True))

        head = QHBoxLayout()
        head.addWidget(_label("Devre boyutu"))
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
        labeled(card, "Zaman adımı (küçük: daha doğru, daha ağır)", self.dt)
        self.page.addWidget(card)

    def _compute_card(self):
        card = Card("Hesaplama")
        self.hw = QComboBox()
        self.hw.addItems(["Otomatik", "CPU"])
        self.hw.setToolTip(f"Otomatik: {GPU_AUTO_MIN_NEURONS:,} nöron ve üstünde GPU, altında CPU")
        card.body.addWidget(self.hw)
        self.gpus: list[dict] = []
        self._gpu_result: list | None = None
        self._gpu_scanned = False
        self._gpu_scanning = False
        self._apply_pending = False                                 # a size that needs the GPU list waits for the scan
        self.has_wgpu = importlib.util.find_spec("wgpu") is not None
        if self.has_wgpu:
            # Listing adapters costs ~1 s and ~100 MB in a helper process, so it happens only when asked for:
            # the entry below, or an automatic choice big enough to want a GPU.
            self.hw.addItem(SCAN_ITEM)
            self.hw.activated.connect(self._hw_activated)
            self.gpu_note = _label("GPU'lar arama isteğiyle bulunur (~1 s, ~100 MB)", "Faint")
            card.body.addWidget(self.gpu_note)
            self._gpu_poll = QTimer(self, timeout=self._gpus_found, interval=250)
        else:
            card.body.addWidget(_label("GPU desteği için: uv sync --extra gpu", "Hint", wrap=True))
        self.page.addWidget(card)

    def _view_card(self):
        card = Card("Görünüm")
        s = QSlider(Qt.Horizontal, minimum=5, maximum=40, value=int(self.overlay.scale * 10))
        s.valueChanged.connect(lambda v: setattr(self.overlay, "scale", v / 10))
        slider_row(card, "Sinek boyutu", "", s, lambda v: f"×{v / 10:.1f}")
        self.visible = QCheckBox("Sinek görünür", checked=True)
        self.visible.toggled.connect(self.overlay.setVisible)
        card.body.addWidget(self.visible)
        screens = QApplication.screens()
        if len(screens) > 1:
            box = QComboBox()
            box.addItems([f"{i + 1}: {s.name()}" for i, s in enumerate(screens)])
            box.setCurrentIndex(screens.index(self.overlay.home))
            box.currentIndexChanged.connect(lambda i: self.overlay.set_home(screens[i]))
            labeled(card, "Ekran", box)
        self.page.addWidget(card)

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
                self.tier_info.setToolTip("FlyWire'daki ateşleme sırasına göre seçilen iç içe katman. "
                                          "Sapma yalnız yaklaşan nesne, geri çekilme ve yön girdileri "
                                          "için ölçüldü (tools/fidelity.py).")
            else:
                self.tier_info.setText("Bu boyut için ölçüm yok.")
        else:
            self.tier_info.setText("" if n == TOY_SIZES[0] else "Sentetik yük: davranışı değiştirmez, "
                                   "yalnız hesaplama maliyetini dener.")
        self.tier_info.setVisible(bool(self.tier_info.text()))

    def _hw_activated(self, i: int):
        if not self._gpu_scanned and self.hw.itemText(i) == SCAN_ITEM:
            self.hw.setCurrentIndex(0)                              # back to "Otomatik" while the list is built
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
        """(backend, adapter) for the chosen 'Hesaplama' entry; automatic = GPU for big tiers."""
        i = self.hw.currentIndex()
        if i >= 2 and self._gpu_scanned:
            return "gpu", self.gpus[i - 2]["index"]
        if i == 0 and self.gpus and n >= GPU_AUTO_MIN_NEURONS:
            return "gpu", pick_gpu(self.gpus)
        return "cpu", None

    def _apply(self):
        n = self._sizes()[self.size.value()]
        if self.hw.currentIndex() == 0 and n >= GPU_AUTO_MIN_NEURONS and self.has_wgpu and not self._gpu_scanned:
            self._apply_pending = True                              # "Otomatik" needs to know the GPUs first
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
        self.warn.style().unpolish(self.warn)          # re-apply the stylesheet for the new name
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
