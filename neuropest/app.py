from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from .control import Control
from .fly import Fly, PlayArea, _wrap
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

        # Exclude this overlay window from screen capture (GDI/DWM BitBlt) so the fly doesn't see itself
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x11)
            except Exception:
                pass

        self.runner = runner
        self.scale = 1.0
        self.home = None  # None = roam all screens, or QScreen = pinned to that monitor
        c = QApplication.primaryScreen().availableGeometry().center()
        self.fly = Fly(float(c.x()), float(c.y()))
        self.last = time.perf_counter()
        self.prev_dist = None
        self._watch_screens()
        self.timer = QTimer(self, timeout=self.tick, interval=16)
        self.timer.start()

    def _watch_screens(self):
        for s in QApplication.screens():
            try:
                s.availableGeometryChanged.connect(self._on_screens_changed)
            except Exception:
                pass
        app = QApplication.instance()
        if app:
            try:
                app.screenAdded.connect(self._on_screens_changed)
                app.screenRemoved.connect(self._on_screens_changed)
            except Exception:
                pass

    def _on_screens_changed(self, *args):
        self.setGeometry(QApplication.primaryScreen().virtualGeometry())
        self.fly.clamp(self.play_area())

    def set_home(self, screen):
        """Set a specific monitor to constrain the fly, or None to roam freely across all monitors."""
        self.home = screen
        if screen is not None:
            c = screen.availableGeometry().center()
            self.fly.x, self.fly.y = float(c.x()), float(c.y())
        self.fly.clamp(self.play_area())

    def play_area(self) -> PlayArea:
        """Playable domain: all screens (multi-monitor roaming) or the chosen monitor."""
        m = 18 * self.scale
        if self.home is not None:
            return PlayArea.from_screens([self.home], margin=m)
        return PlayArea.from_screens(QApplication.screens(), margin=m)

    def play_rect(self):
        """Compatibility helper returning the active play area."""
        return self.play_area()

    def to_physical(self, x: float, y: float) -> tuple[float, float]:
        """Maps Qt logical coordinates to physical screen coordinates for multi-monitor GDI capture."""
        from PySide6.QtCore import QPoint
        s = QApplication.screenAt(QPoint(int(x), int(y)))
        if s is None:
            screens = QApplication.screens()
            if screens:
                def dist_to(sc):
                    g = sc.geometry()
                    cx = min(max(int(x), g.left()), g.right())
                    cy = min(max(int(y), g.top()), g.bottom())
                    return (int(x) - cx) ** 2 + (int(y) - cy) ** 2
                s = min(screens, key=dist_to)
            else:
                return x, y
        dpr = s.devicePixelRatio()
        if dpr == 1.0:
            return x, y
        g = s.geometry()
        return g.x() + (x - g.x()) * dpr, g.y() + (y - g.y()) * dpr

    def tick(self):
        now = time.perf_counter()
        elapsed = now - self.last
        dt = min(elapsed, 0.05)                 # the fly's physics never takes a step longer than this
        self.last = now
        cur = QCursor.pos()

        cap_pos = self.to_physical(self.fly.x, self.fly.y)
        cur_pos = self.to_physical(cur.x(), cur.y())

        # Physical Euclidean distance across monitor boundaries
        pdx = cur_pos[0] - cap_pos[0]
        pdy = cur_pos[1] - cap_pos[1]
        dist = (pdx ** 2 + pdy ** 2) ** 0.5
        closing = 0.0 if self.prev_dist is None else (self.prev_dist - dist) / max(elapsed, 1e-3)
        self.prev_dist = dist
        touch = 1.0 if dist < TOUCH_RADIUS_PX * self.scale else 0.0

        # Physical bearing of cursor relative to fly heading
        import math
        bearing = _wrap(math.atan2(pdy, pdx) - self.fly.heading)

        # pose and cursor in screen px feed the visual input (runner.vision); the numbers feed the cursor drive
        self.runner.send(dist, closing, bearing, touch,
                         (self.fly.x, self.fly.y, self.fly.heading), (cur.x(), cur.y()), stamp=now,
                         capture_pos=cap_pos)

        # In fly coordinate space, represent cursor vector accurately for fleeing
        from PySide6.QtCore import QPoint
        s_fly = QApplication.screenAt(QPoint(int(self.fly.x), int(self.fly.y)))
        dpr_fly = s_fly.devicePixelRatio() if s_fly is not None else 1.0
        effective_cursor = (self.fly.x + pdx / dpr_fly, self.fly.y + pdy / dpr_fly)

        area = self.play_area()
        self.fly.update(dt, self.runner.state, effective_cursor, area, self.runner.steer)
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
