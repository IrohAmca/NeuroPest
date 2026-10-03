from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from .control import Control
from .fly import Fly
from .render import draw_fly
from .runner import Runner
from .theme import apply_theme
from .tray import Tray

TOUCH_RADIUS_PX = 14.0     # cursor this close to the fly center (times its scale) touches it


class Overlay(QWidget):
    """Transparent, frameless, click-through, always-on-top canvas over all screens."""

    def __init__(self, runner: Runner):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(QApplication.primaryScreen().virtualGeometry())
        self.runner = runner
        self.scale = 1.0
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
        elapsed = now - self.last
        dt = min(elapsed, 0.05)                 # the fly's physics never takes a step longer than this
        self.last = now
        cur = QCursor.pos()
        dist = ((cur.x() - self.fly.x) ** 2 + (cur.y() - self.fly.y) ** 2) ** 0.5
        closing = 0.0 if self.prev_dist is None else (self.prev_dist - dist) / max(elapsed, 1e-3)     # real frame time
        self.prev_dist = dist
        touch = 1.0 if dist < TOUCH_RADIUS_PX * self.scale else 0.0      # click-through overlay: hover = cursor over the fly
        # pose and cursor in screen px feed the visual input (runner.vision); the numbers feed the cursor drive
        self.runner.send(dist, closing, self.fly.bearing_of((cur.x(), cur.y())), touch,
                         (self.fly.x, self.fly.y, self.fly.heading), (cur.x(), cur.y()), stamp=now)
        self.fly.update(dt, self.runner.state, (cur.x(), cur.y()), self.play_rect(), self.runner.steer)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        g = self.geometry()
        draw_fly(p, self.fly.x - g.left(), self.fly.y - g.top(), self.fly.heading,
                 self.runner.state, self.fly.phase, self.scale)


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    apply_theme(app)
    runner = Runner()
    app.aboutToQuit.connect(runner.stop)
    overlay = Overlay(runner)
    overlay.show()
    ctrl = Control(overlay, runner)
    ctrl.show()
    tray = Tray(app, ctrl, runner)
    tray.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
