from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from .control import Control
from .fly import Fly, PlayArea, _wrap
from .metabolism import MetabolicState
from .pheromone import PheromoneField
from .render import RewardEffect, draw_fly, draw_reward_plus
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
        self.pheromone = PheromoneField(border_margin=60.0 * self.scale, border_strength=0.7)
        self.pheromone.spawn_random_sources(self.play_area().screens, count=7)
        self.pheromone_enabled = True
        self.cursor_phero_mode = "attract"  # "attract", "repel", or "none"
        self.feed_effects: list[RewardEffect] = []
        self.feed_glow: float = 0.0
        self._last_cursor_reward: float = 0.0
        self.touch_groom_enabled = False  # cursor touch grooming toggle (default: False/off)
        self.metabolism = MetabolicState()
        self._cached_play_area: PlayArea | None = None
        self.last = time.perf_counter()
        self.prev_dist = None
        self._watch_screens()
        self.timer = QTimer(self, timeout=self.tick, interval=16)
        self.timer.start()

    def _update_geometry(self):
        self._cached_play_area = None
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
        self._cached_play_area = None
        self._update_geometry()
        self.fly.clamp(self.play_area())
        if not self.pheromone.sources:
            self.pheromone.spawn_random_sources(self.play_area().screens, count=7, attract_ratio=0.75)

    def set_home(self, screen):
        """Set a specific monitor to constrain the fly, or None to roam freely across all monitors."""
        self._cached_play_area = None
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
        if self._cached_play_area is None:
            from .fly import FLY_MARGIN_PX
            m = FLY_MARGIN_PX * self.scale
            if self.home is not None:
                self._cached_play_area = PlayArea.from_screens([self.home], margin=m)
            else:
                self._cached_play_area = PlayArea.from_screens(QApplication.screens(), margin=m)
        return self._cached_play_area

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
        touch = 1.0 if (self.touch_groom_enabled and dist < TOUCH_RADIUS_PX * self.scale) else 0.0

        # Physical bearing of cursor relative to fly heading
        import math
        bearing = _wrap(math.atan2(pdy, pdx) - self.fly.heading)

        # Update pheromone field, cursor attractant and sample bilateral antennae
        area = self.play_area()
        if self.pheromone_enabled:
            self.pheromone.update(dt)
            cursor_enabled = (self.cursor_phero_mode != "none")
            cursor_kind = self.cursor_phero_mode if cursor_enabled else "attract"
            self.pheromone.update_cursor(cur_x, cur_y, kind=cursor_kind, enabled=cursor_enabled)
            ant = self.pheromone.sample_antennae(self.fly.x, self.fly.y, self.fly.heading, area,
                                                 antenna_dist=12.0 * self.scale)

            # Tropotaxis steering bias (Hz): attract delta (+) turns right, repel delta (+) turns left
            phero_steer = ant["delta_attr"] * 180.0 - ant["delta_rep"] * 220.0
            phero_drive = ant["total_attr"]
            phero_repel = ant["total_rep"]

            # Check if arrived at any attractive source/nectar
            at_target = False
            target_radius = 28.0 * self.scale
            for src in self.pheromone.sources:
                if math.hypot(src.x - self.fly.x, src.y - self.fly.y) < target_radius:
                    at_target = True
                    break
            at_cursor = (self.pheromone.cursor_active and cursor_kind == "attract" and dist < target_radius)
            if not at_target and at_cursor:
                at_target = True

            # When fly lands or arrives at food source, consume it so it disappears!
            if at_target:
                consumed = self.pheromone.consume_at(self.fly.x, self.fly.y, consume_radius=target_radius,
                                                     screen_boxes=area.screens if hasattr(area, "screens") else None)
                if consumed:
                    self.metabolism.feed()
                    self.feed_effects.append(RewardEffect(self.fly.x, self.fly.y))
                    self.feed_glow = 1.0
                elif at_cursor and (now - self._last_cursor_reward > 3.0):
                    self._last_cursor_reward = now
                    self.metabolism.feed()
                    self.feed_effects.append(RewardEffect(self.fly.x, self.fly.y))
                    self.feed_glow = 1.0
        else:
            phero_steer = 0.0
            phero_drive = 0.0
            phero_repel = 0.0
            at_target = False

        # Update metabolic energy depletion based on active state (fly burns ~16x rest)
        self.metabolism.update(dt, self.runner.state)

        # pose and cursor in screen px feed visual input; pheromone feeds antennae
        cursor_is_attract = self.pheromone_enabled and (self.cursor_phero_mode == "attract")
        effective_closing = 0.0 if cursor_is_attract else closing

        effective_cursor = (cur_x, cur_y)
        self.fly.update(dt, self.runner.state, effective_cursor, area, self.runner.steer)

        self.runner.send(dist, effective_closing, bearing, touch,
                         (self.fly.x, self.fly.y, self.fly.heading), (cur_x, cur_y), stamp=now,
                         capture_pos=(self.fly.x, self.fly.y),
                         phero_steer=phero_steer, phero_drive=phero_drive,
                         phero_repel=phero_repel, at_target=at_target,
                         hunger=self.metabolism.hunger,
                         wall_bump=self.fly.hit_wall)

        # Update feeding animation and floating reward effects
        if self.feed_glow > 0.0:
            self.feed_glow = max(0.0, self.feed_glow - dt / 0.8)
        if self.feed_effects:
            self.feed_effects = [e for e in self.feed_effects if e.update(dt)]

        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        g = self.geometry()
        # Draw fly with feeding animation and crop glow
        draw_fly(p, self.fly.x - g.left(), self.fly.y - g.top(), self.fly.heading,
                 self.runner.state, self.fly.phase, self.scale, feed_glow=self.feed_glow)
        # Draw floating reward plus effects
        for eff in self.feed_effects:
            draw_reward_plus(p, eff.x - g.left(), eff.y - g.top(), eff.progress, scale=self.scale)


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
