from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QCursor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QLabel, QMenu, QSlider,
                               QSystemTrayIcon, QVBoxLayout, QWidget)

from .fly import Fly
from .paths import CACHE, TIERS
from .render import draw_fly
from .runner import GPU_AUTO_MIN_NEURONS, EngineConfig, Runner, list_gpus, pick_gpu

# circuit sizes offered in the UI (neurons). FlyWire sizes are tiers measured by tools/fidelity.py.
FLYWIRE_SIZES = [2_000, 5_000, 10_000, 15_000, 20_000, 50_000, 138_639]
FLYWIRE_DEFAULT = 15_000
TOY_SIZES = [146, 500, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 139_000]
DTS = [("Hassas (0.1 ms)", 0.1), ("Dengeli (0.5 ms)", 0.5), ("Hızlı (1 ms)", 1.0)]


def load_tiers() -> dict[int, dict]:
    try:
        return {t["n"]: t for t in json.loads(TIERS.read_text())["tiers"]}
    except (OSError, ValueError, KeyError):
        return {}


class Overlay(QWidget):
    """Transparent, frameless, click-through, always-on-top canvas over all screens."""

    def __init__(self, runner: Runner):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(QApplication.primaryScreen().virtualGeometry())
        self.runner = runner
        self.scale = 1.4
        self.home = QApplication.primaryScreen()
        c = self.home.availableGeometry().center()
        self.fly = Fly(c.x(), c.y())
        self.last = time.perf_counter()
        self.prev_dist = None
        self._watch_home()
        self.timer = QTimer(self, timeout=self.tick, interval=16)
        self.timer.start()

    def _watch_home(self):
        self.home.availableGeometryChanged.connect(lambda *_: self.fly.clamp(self.play_rect()))

    def set_home(self, screen):
        """Move the fly's playground to another monitor."""
        try:
            self.home.availableGeometryChanged.disconnect()
        except RuntimeError:
            pass
        self.home = screen
        self._watch_home()
        c = screen.availableGeometry().center()
        self.fly.x, self.fly.y = c.x(), c.y()

    def play_rect(self):
        """Where the fly's center may be: the monitor's available area (above the taskbar)."""
        a = self.home.availableGeometry()
        m = 18 * self.scale
        return (a.left() + m, a.top() + m, a.right() - m, a.bottom() - m)

    def tick(self):
        now = time.perf_counter()
        dt = min(now - self.last, 0.05)
        self.last = now
        cur = QCursor.pos()
        dist = ((cur.x() - self.fly.x) ** 2 + (cur.y() - self.fly.y) ** 2) ** 0.5
        closing = 0.0 if self.prev_dist is None else (self.prev_dist - dist) / max(dt, 1e-3)
        self.prev_dist = dist
        self.runner.send(dist, closing, self.fly.bearing_of((cur.x(), cur.y())))
        self.fly.update(dt, self.runner.state, (cur.x(), cur.y()), self.play_rect(), self.runner.steer)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        g = self.geometry()
        draw_fly(p, self.fly.x - g.left(), self.fly.y - g.top(), self.fly.heading,
                 self.runner.state, self.fly.phase, self.scale)


class Control(QWidget):
    def __init__(self, overlay: Overlay, runner: Runner):
        super().__init__()
        self.runner, self.overlay = runner, overlay
        self.tiers = load_tiers()
        self.setWindowTitle("NeuroPest")
        lay = QVBoxLayout(self)
        self.status = QLabel()
        lay.addWidget(self.status)

        lay.addWidget(QLabel("Boyut"))
        s = QSlider(Qt.Horizontal, minimum=5, maximum=40, value=int(overlay.scale * 10))
        s.valueChanged.connect(lambda v: setattr(overlay, "scale", v / 10))
        lay.addWidget(s)

        lay.addWidget(QLabel("Hareketlilik (yürüme sürücüsü)"))
        w = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=int(runner.bias * 100))
        w.valueChanged.connect(lambda v: setattr(runner, "bias", v / 100))
        lay.addWidget(w)

        lay.addWidget(QLabel("Ürkeklik (yaklaşan imlece tepki)"))
        k = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=50)
        k.valueChanged.connect(lambda v: setattr(runner, "skittish", 2.0 ** ((v - 50) / 25.0)))
        lay.addWidget(k)

        lay.addWidget(QLabel("Devre"))
        self.circuits = ([("flywire", "FlyWire v783 (gerçek bağlantı)")] if CACHE.exists() else []) \
            + [("toy", "Oyuncak devre (sentetik yük)")]
        self.circ = QComboBox()
        self.circ.addItems([name for _, name in self.circuits])
        self.circ.setCurrentIndex([c for c, _ in self.circuits].index(runner.cfg.circuit))
        lay.addWidget(self.circ)
        if not CACHE.exists():
            hint = QLabel("Gerçek devre için: uv run python tools/build_flywire.py (README'ye bak)")
            hint.setWordWrap(True)
            lay.addWidget(hint)

        self.size_label = QLabel()
        lay.addWidget(self.size_label)
        self.size = QSlider(Qt.Horizontal)
        lay.addWidget(self.size)
        self.tier_info = QLabel()
        self.tier_info.setWordWrap(True)
        lay.addWidget(self.tier_info)

        lay.addWidget(QLabel("Zaman adımı (küçük = daha doğru, daha ağır)"))
        self.dt = QComboBox()
        self.dt.addItems([n for n, _ in DTS])
        self.dt.setCurrentIndex([d for _, d in DTS].index(runner.cfg.dt))
        lay.addWidget(self.dt)

        lay.addWidget(QLabel("Hesaplama"))
        self.hw = QComboBox()
        self.hw.addItems(["Otomatik", "CPU"])
        lay.addWidget(self.hw)
        self.gpus: list[dict] = []
        self._gpu_result: list | None = None
        if importlib.util.find_spec("wgpu") is not None:            # GPUs are found in a helper process
            threading.Thread(target=lambda: setattr(self, "_gpu_result", list_gpus()), daemon=True).start()
            self._gpu_poll = QTimer(self, timeout=self._gpus_found, interval=500)
            self._gpu_poll.start()
        else:
            lay.addWidget(QLabel("GPU desteği için: uv sync --extra gpu"))

        self.telemetry = QLabel()
        lay.addWidget(self.telemetry)
        self.warn = QLabel()
        self.warn.setStyleSheet("color: #c0392b")
        self.warn.setWordWrap(True)
        lay.addWidget(self.warn)

        top = QCheckBox("Sinek görünür", checked=True)
        top.toggled.connect(overlay.setVisible)
        lay.addWidget(top)
        screens = QApplication.screens()
        if len(screens) > 1:
            lay.addWidget(QLabel("Ekran"))
            box = QComboBox()
            box.addItems([f"{i + 1}: {s.name()}" for i, s in enumerate(screens)])
            box.setCurrentIndex(screens.index(overlay.home))
            box.currentIndexChanged.connect(lambda i: overlay.set_home(screens[i]))
            lay.addWidget(box)

        self._load_sizes(runner.cfg.n)
        self._debounce = QTimer(self, singleShot=True, interval=500, timeout=self._apply)
        self.circ.currentIndexChanged.connect(self._circuit_changed)
        self.size.valueChanged.connect(self._size_moved)
        self.dt.currentIndexChanged.connect(lambda _: self._debounce.start())
        self.hw.currentIndexChanged.connect(lambda _: self._debounce.start())
        self._t = QTimer(self, timeout=self._refresh, interval=250)
        self._t.start()

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
        if self._kind() == "flywire":
            self.size_label.setText(f"Devre boyutu: {n:,} nöron (FlyWire'dan, ölçülen katman)")
            t = self.tiers.get(n)
            if t:
                self.tier_info.setText(
                    f"Tam beyne göre sapma: kalkış %{t['gf_err']:.0f}, geri yürüme %{t['mdn_err']:.0f}, "
                    f"yön %{t['steer_err']:.0f}; descending nöron korelasyonu {t['dn_corr']:.3f}. "
                    f"Ölçülen hız ×{t['realtime']:.1f} (en kötü ×{t['rt_min']:.1f}). "
                    "Yalnız yaklaşan nesne, geri çekilme ve yön girdileri için ölçüldü.")
            else:
                self.tier_info.setText("")
        else:
            extra = "" if n == TOY_SIZES[0] else " (sentetik yük, davranışı değiştirmez)"
            self.size_label.setText(f"Devre boyutu: {n:,} nöron{extra}")
            self.tier_info.setText("")

    def _gpus_found(self):
        if self._gpu_result is None:
            return
        self._gpu_poll.stop()
        self.gpus = self._gpu_result
        self.hw.blockSignals(True)
        self.hw.addItems([f"GPU: {g['name']} ({g['backend']})" for g in self.gpus])
        self.hw.blockSignals(False)

    def _hardware(self, n: int) -> tuple[str, int | None]:
        """(backend, adapter) for the chosen 'Hesaplama' entry; automatic = GPU for big tiers."""
        i = self.hw.currentIndex()
        if i >= 2:
            return "gpu", self.gpus[i - 2]["index"]
        if i == 0 and self.gpus and n >= GPU_AUTO_MIN_NEURONS:
            return "gpu", pick_gpu(self.gpus)
        return "cpu", None

    def _apply(self):
        n = self._sizes()[self.size.value()]
        backend, adapter = self._hardware(n)
        cfg = EngineConfig(self._kind(), n, DTS[self.dt.currentIndex()][1], backend=backend, adapter=adapter)
        if cfg != self.runner.cfg:
            self.runner.start(cfg)

    # ---------------------------------------------------------------- telemetry
    def _refresh(self):
        r = self.runner
        st = r.stats()
        if r.failed or not r.alive:
            self.telemetry.setText("Motor hata verdi ya da durdu (ayrıntı konsolda). "
                                   "Daha küçük bir boyut seçmeyi dene.")
            return
        if not st["ready"]:
            self.telemetry.setText("Motor başlıyor (ilk açılışta derleme birkaç sn sürer)…")
            self.warn.setText("")
            return
        where = "CPU"
        if r.cfg.backend == "gpu":
            g = next((g for g in self.gpus if g["index"] == r.cfg.adapter), None)
            where = f"GPU {g['name']} ({g['backend']})" if g else "GPU"
        active = f"Aktif nöron: {st['active']:,.0f} / {st['n']:,}" if st["active"] >= 0 \
            else f"Nöron: {st['n']:,} (GPU hepsini her adımda günceller)"
        self.telemetry.setText(
            f"Durum: {r.state}   GF {st['gf']:.0f} Hz   MDN {st['mdn']:.0f} Hz   yön {st['steer']:+.0f}\n"
            f"{where}: gerçek zaman çarpanı ×{st['rt']:.1f}, işlemci %{100 * st['cpu']:.0f} (tek çekirdek)\n"
            f"{active}   Spike/sn: {st['spikes']:,.0f}")
        slow = st["rt"] < 1.0 or st["lag_ms"] > 100
        self.warn.setText("Bu ayar bu bilgisayar için ağır: sinek yavaş çekimde. "
                          "Boyutu küçült ya da zaman adımını büyüt." if slow else "")


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    runner = Runner()
    app.aboutToQuit.connect(runner.stop)
    overlay = Overlay(runner)
    overlay.show()
    ctrl = Control(overlay, runner)
    ctrl.show()
    pm = QPixmap(16, 16)
    pm.fill(QColor(30, 30, 30))
    tray = QSystemTrayIcon(QIcon(pm), app)
    menu = QMenu()
    show = QAction("Kontrol penceresi", menu)
    show.triggered.connect(ctrl.show)
    quit_ = QAction("Çıkış", menu)
    quit_.triggered.connect(app.quit)
    menu.addAction(show)
    menu.addAction(quit_)
    tray.setContextMenu(menu)
    tray.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
