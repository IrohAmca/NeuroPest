from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QCursor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QLabel, QMenu, QSlider,
                               QSystemTrayIcon, QVBoxLayout, QWidget)

from .brain import Brain
from .fly import Fly
from .render import draw_fly


class Overlay(QWidget):
    """Transparent, frameless, click-through, always-on-top canvas over all screens."""

    def __init__(self, brain: Brain):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(QApplication.primaryScreen().virtualGeometry())
        self.brain = brain
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
        state = self.brain.step(dt * 1000, dist, closing)
        self.fly.update(dt, state, (cur.x(), cur.y()), self.play_rect())
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        g = self.geometry()
        draw_fly(p, self.fly.x - g.left(), self.fly.y - g.top(), self.fly.heading,
                 self.brain.state, self.fly.phase, self.scale)


class Control(QWidget):
    def __init__(self, overlay: Overlay, brain: Brain):
        super().__init__()
        self.setWindowTitle("NeuroPest")
        lay = QVBoxLayout(self)
        self.status = QLabel()
        lay.addWidget(self.status)
        lay.addWidget(QLabel("Boyut"))
        s = QSlider(Qt.Horizontal, minimum=5, maximum=40, value=int(overlay.scale * 10))
        s.valueChanged.connect(lambda v: setattr(overlay, "scale", v / 10))
        lay.addWidget(s)
        lay.addWidget(QLabel("Hareketlilik (yürüme sürücüsü)"))
        w = QSlider(Qt.Horizontal, minimum=0, maximum=300, value=0)
        w.valueChanged.connect(lambda v: setattr(brain, "walk_bias", float(v)))
        lay.addWidget(w)
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
        lay.addWidget(QLabel(f"Devre: {brain.c.n} nöron (oyuncak alt devre)"))
        t = QTimer(self, timeout=lambda: self.status.setText(
            f"Durum: {brain.state}   GF: {brain.c.rate[brain.g['GF']].mean():.0f} Hz"))
        t.start(200)
        self._t = t


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    brain = Brain()
    overlay = Overlay(brain)
    overlay.show()
    ctrl = Control(overlay, brain)
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
