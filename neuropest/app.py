from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QCursor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QLabel, QMenu, QSlider,
                               QSystemTrayIcon, QVBoxLayout, QWidget)

from .fly import Fly
from .render import draw_fly
from .runner import EngineConfig, Runner

# circuit size choices offered in the UI (neurons); the real FlyWire tiers will replace these
SIZES = [146, 500, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 139_000]
DTS = [("Hassas (0.1 ms)", 0.1), ("Dengeli (0.5 ms)", 0.5), ("Hızlı (1 ms)", 1.0)]


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
        self.runner.send(dist, closing)
        self.fly.update(dt, self.runner.state, (cur.x(), cur.y()), self.play_rect())
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
        self.setWindowTitle("NeuroPest")
        lay = QVBoxLayout(self)
        self.status = QLabel()
        lay.addWidget(self.status)

        lay.addWidget(QLabel("Boyut"))
        s = QSlider(Qt.Horizontal, minimum=5, maximum=40, value=int(overlay.scale * 10))
        s.valueChanged.connect(lambda v: setattr(overlay, "scale", v / 10))
        lay.addWidget(s)

        lay.addWidget(QLabel("Hareketlilik"))
        w = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=int(runner.bias * 100))
        w.valueChanged.connect(lambda v: setattr(runner, "bias", v / 100))
        lay.addWidget(w)

        self.size_label = QLabel()
        lay.addWidget(self.size_label)
        self.size = QSlider(Qt.Horizontal, minimum=0, maximum=len(SIZES) - 1,
                            value=SIZES.index(runner.cfg.n) if runner.cfg.n in SIZES else 0)
        self.size.valueChanged.connect(self._size_moved)
        lay.addWidget(self.size)

        lay.addWidget(QLabel("Zaman adımı (küçük = daha doğru, daha ağır)"))
        self.dt = QComboBox()
        self.dt.addItems([n for n, _ in DTS])
        self.dt.setCurrentIndex([d for _, d in DTS].index(runner.cfg.dt))
        self.dt.currentIndexChanged.connect(lambda _: self._apply())
        lay.addWidget(self.dt)

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

        self._debounce = QTimer(self, singleShot=True, interval=500, timeout=self._apply)
        self._size_moved(self.size.value(), restart=False)
        self._t = QTimer(self, timeout=self._refresh, interval=250)
        self._t.start()

    def _size_moved(self, i: int, restart: bool = True):
        n = SIZES[i]
        extra = "" if n == SIZES[0] else " (sentetik yük; gerçek FlyWire alt devresi henüz bağlı değil)"
        self.size_label.setText(f"Devre boyutu: {n:,} nöron{extra}")
        if restart:
            self._debounce.start()

    def _apply(self):
        cfg = EngineConfig(n=SIZES[self.size.value()], dt=DTS[self.dt.currentIndex()][1])
        if cfg != self.runner.cfg:
            self.runner.start(cfg)

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
        self.telemetry.setText(
            f"Durum: {r.state}   GF {st['gf']:.0f} Hz\n"
            f"Gerçek zaman çarpanı: ×{st['rt']:.1f}   CPU: %{100 * st['cpu']:.0f} (tek çekirdek)\n"
            f"Aktif nöron: {st['active']:,.0f} / {st['n']:,}")
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
