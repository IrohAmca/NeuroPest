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
        self._update_geometry()

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
        ps = QApplication.primaryScreen()
        pg = ps.geometry()
        pdpr = float(ps.devicePixelRatio())
        cx = float(pg.x()) + float(pg.width()) * pdpr / 2.0
        cy = float(pg.y()) + float(pg.height()) * pdpr / 2.0
        self.fly = Fly(cx, cy)
        self.last = time.perf_counter()
        self.prev_dist = None
        self._watch_screens()
        self.timer = QTimer(self, timeout=self.tick, interval=16)
        self.timer.start()

    def _update_geometry(self):
        if sys.platform == "win32":
            import ctypes
            u = ctypes.windll.user32
            vx = u.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
            vy = u.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
            vw = u.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
            vh = u.GetSystemMetrics(79)   # SM_CYVIRTUALSCREEN
            self.setGeometry(vx, vy, vw, vh)
        else:
            self.setGeometry(QApplication.primaryScreen().virtualGeometry())

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
        self._update_geometry()
        self.fly.clamp(self.play_area())

    def set_home(self, screen):
        """Set a specific monitor to constrain the fly, or None to roam freely across all monitors."""
        self.home = screen
        if screen is not None:
            g = screen.geometry()
            dpr = float(screen.devicePixelRatio())
            cx = float(g.x()) + float(g.width()) * dpr / 2.0
            cy = float(g.y()) + float(g.height()) * dpr / 2.0
            self.fly.x, self.fly.y = cx, cy
        self.fly.clamp(self.play_area())

    def play_area(self) -> PlayArea:
        """Playable domain: all screens (multi-monitor roaming) or the chosen monitor."""
        m = 2.0 * self.scale
        if self.home is not None:
            return PlayArea.from_screens([self.home], margin=m)
        return PlayArea.from_screens(QApplication.screens(), margin=m)

    def play_rect(self):
        """Compatibility helper returning the active play area."""
        return self.play_area()

    def to_physical(self, x: float, y: float) -> tuple[float, float]:
        """Coordinates are in physical screen space."""
        return x, y

    def tick(self):
        now = time.perf_counter()
        elapsed = now - self.last
        dt = min(elapsed, 0.05)                 # the fly's physics never takes a step longer than this
        self.last = now

        if sys.platform == "win32":
            import ctypes, ctypes.wintypes
            pt = ctypes.wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            cur_x, cur_y = float(pt.x), float(pt.y)
        else:
            cur = QCursor.pos()
            cur_x, cur_y = float(cur.x()), float(cur.y())

        # Physical Euclidean distance across monitor boundaries
        pdx = cur_x - self.fly.x
        pdy = cur_y - self.fly.y
        dist = (pdx ** 2 + pdy ** 2) ** 0.5
        closing = 0.0 if self.prev_dist is None else (self.prev_dist - dist) / max(elapsed, 1e-3)
        self.prev_dist = dist
        touch = 1.0 if dist < TOUCH_RADIUS_PX * self.scale else 0.0

        # Physical bearing of cursor relative to fly heading
        import math
        bearing = _wrap(math.atan2(pdy, pdx) - self.fly.heading)

        # pose and cursor in screen px feed the visual input (runner.vision); the numbers feed the cursor drive
        self.runner.send(dist, closing, bearing, touch,
                         (self.fly.x, self.fly.y, self.fly.heading), (cur_x, cur_y), stamp=now,
                         capture_pos=(self.fly.x, self.fly.y))

        effective_cursor = (cur_x, cur_y)

        area = self.play_area()
        self.fly.update(dt, self.runner.state, effective_cursor, area, self.runner.steer)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        g = self.geometry()
        draw_fly(p, self.fly.x - g.left(), self.fly.y - g.top(), self.fly.heading,
                 self.runner.state, self.fly.phase, self.scale)


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            u = ctypes.windll.user32
            u.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
            u.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            pass

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
