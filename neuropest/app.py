from dataclasses import replace
import math
import sys
import time

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    _u32 = ctypes.windll.user32
    _u32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    _u32.GetCursorPos.restype = wintypes.BOOL
else:
    ctypes = None
    wintypes = None
    _u32 = None

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QCursor, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from .control import Control
from .fly import FLY_MARGIN_PX, Fly, PlayArea, _wrap
from .states import FEED, FLY
from .metabolism import MetabolicState
from .pheromone import FOOD_COUNT, GoalTracker, PheromoneField
from .preferences import Preferences
from .render import RewardEffect, draw_fly, draw_reward_plus
from .runner import EngineConfig, Runner, default_config, effective_mb_wiring
from .theme import apply_theme
from .tray import Tray

TOUCH_RADIUS_PX = 14.0     # cursor this close to the fly center (times its scale) touches it
OVERLAY_INSET_PX = 1       # keep the overlay strictly inside the desktop, see overlay_rect()
EAT_RATE = 0.28            # food eaten per second with the proboscis fully out: a food of size 1 in ~5 s for a hungry fly,
                           # whose proboscis is out ~0.7 on average while it eats (MN9 falls as it fills up; pheromone.FOOD_*)
PROBOSCIS_TAU_S = 0.15     # the drawn proboscis follows the brain's feed level (MN9) this fast


def overlay_rect(x: int, y: int, w: int, h: int, inset: int = OVERLAY_INSET_PX) -> QRect:
    """Overlay window rectangle for a desktop of the given bounds.

    A borderless window that covers a monitor edge to edge is treated by the Windows shell as a
    fullscreen app: it drops the taskbar's always-on-top state, so maximized windows slide under the
    taskbar. Insetting by a pixel keeps the overlay from ever matching a monitor rectangle. The fly
    stays at least FLY_MARGIN_PX away from every edge, so nothing visible is cut off.
    """
    return QRect(x + inset, y + inset, max(1, w - 2 * inset), max(1, h - 2 * inset))


class Overlay(QWidget):
    """Transparent, frameless, click-through, always-on-top canvas over all screens."""

    def __init__(self, runner: Runner, prefs: Preferences | None = None):
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
        self.prefs = prefs or Preferences.load()
        self.scale = float(self.prefs.scale)
        self.skin = self.prefs.skin
        self.home = None  # None = roam all screens, or QScreen = pinned to that monitor
        ps = QApplication.primaryScreen()
        pg = ps.availableGeometry() if hasattr(ps, "availableGeometry") else ps.geometry()
        pdpr = float(ps.devicePixelRatio())
        cx = float(pg.x()) + float(pg.width()) * pdpr / 2.0
        cy = float(pg.y()) + float(pg.height()) * pdpr / 2.0
        self.fly = Fly(cx, cy)
        self.pheromone = PheromoneField(border_margin=30.0 * self.scale, border_strength=0.4)
        self.pheromone.spawn_random_sources(self.play_area().screens, count=FOOD_COUNT)
        self.pheromone_enabled = bool(self.prefs.pheromone_enabled)
        self.cursor_phero_mode = self.prefs.cursor_phero_mode
        self.goal = GoalTracker()  # the food source the fly would fly to (fixed during the flight)
        self.feed_effects: list[RewardEffect] = []
        self.feed_glow: float = 0.0       # how far the drawn proboscis is out (follows the brain's MN9)
        self._was_feeding = False
        self._meal_spot: tuple[float, float] | None = None   # where the fly last ate: its local search circles it
        self.touch_groom_enabled = bool(self.prefs.touch_groom_enabled)
        self.metabolism = MetabolicState(enabled=bool(self.prefs.hunger_enabled))
        self.metabolism.cfg.rate_mult = float(self.prefs.metabolic_rate)

        # Restore pinned monitor if specified in preferences
        screens = QApplication.screens()
        if self.prefs.monitor_index > 0 and screens:
            matched = None
            if self.prefs.monitor_name:
                matched = next((s for s in screens if s.name() == self.prefs.monitor_name), None)
            if matched is None and 1 <= self.prefs.monitor_index <= len(screens):
                matched = screens[self.prefs.monitor_index - 1]
            if matched is not None:
                self.set_home(matched)

        self._cached_play_area: PlayArea | None = None
        self._cursor_pt = wintypes.POINT() if sys.platform == "win32" else None
        self._prev_fly_rect: QRect | None = None
        self._prev_effect_rects: list[QRect] = []
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
            self.setGeometry(overlay_rect(vx, vy, vw, vh))
        else:
            g = QApplication.primaryScreen().virtualGeometry()
            self.setGeometry(overlay_rect(g.x(), g.y(), g.width(), g.height()))

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
        self._prev_fly_rect = None
        self._prev_effect_rects.clear()
        self._update_geometry()
        self.fly.clamp(self.play_area())
        if not self.pheromone.sources:
            self.pheromone.spawn_random_sources(self.play_area().screens, count=FOOD_COUNT)

    def _get_fly_rect(self, g) -> QRect:
        lx = int(self.fly.x - g.left())
        ly = int(self.fly.y - g.top())
        pad = int(math.ceil(70.0 * self.scale))
        return QRect(lx - pad, ly - pad, pad * 2, pad * 2)

    def _get_effect_rect(self, eff: RewardEffect, g) -> QRect:
        lx = int(eff.x - g.left())
        ly = int(eff.y - g.top() - 30.0 * self.scale)
        pw = int(math.ceil(35.0 * self.scale))
        ph = int(math.ceil(50.0 * self.scale))
        return QRect(lx - pw, ly - ph, pw * 2, ph * 2)

    def set_home(self, screen):
        """Set a specific monitor to constrain the fly, or None to roam freely across all monitors."""
        self._cached_play_area = None
        self.home = screen
        if screen is not None:
            g = screen.availableGeometry() if hasattr(screen, "availableGeometry") else screen.geometry()
            dpr = float(screen.devicePixelRatio())
            cx = float(g.x()) + float(g.width()) * dpr / 2.0
            cy = float(g.y()) + float(g.height()) * dpr / 2.0
            self.fly.x, self.fly.y = cx, cy
        self.fly.clamp(self.play_area())
        if hasattr(self, "pheromone"):
            self.pheromone.spawn_random_sources(self.play_area().screens, count=FOOD_COUNT)

    def play_area(self) -> PlayArea:
        """Playable domain: all screens (multi-monitor roaming) or the chosen monitor."""
        if self._cached_play_area is None:
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
            _u32.GetCursorPos(ctypes.byref(self._cursor_pt))
            cur_x, cur_y = float(self._cursor_pt.x), float(self._cursor_pt.y)
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
        bearing = _wrap(math.atan2(pdy, pdx) - self.fly.heading)

        # state and gear of the same worker step, from one shared slot (two separate reads can straddle a landing)
        state_gear = getattr(self.runner, "state_gear", None)
        state, gear = state_gear() if state_gear is not None else (self.runner.state, self.runner.gear)
        long_flight = state == FLY and gear == "fly_long"

        # Update pheromone field, cursor attractant and sample bilateral antennae
        area = self.play_area()
        target_radius = 28.0 * self.scale
        if self.pheromone_enabled:
            self.pheromone.update(dt)
            cursor_enabled = (self.cursor_phero_mode != "none")
            cursor_kind = self.cursor_phero_mode if cursor_enabled else "attract"
            self.pheromone.update_cursor(cur_x, cur_y, kind=cursor_kind, enabled=cursor_enabled)
            # bilateral antennae: tropotaxis steering bias (Hz), attractant and repellent intensity, the attractant's
            # right-minus-left difference (the two antennae's ORN input)
            phero_steer, phero_drive, phero_repel, odor_delta = self.pheromone.smell(
                self.fly.x, self.fly.y, self.fly.heading, area, self.scale)

            # Food under the fly is its taste input (the sugar receptor neurons). Whether it feeds is the brain's: MN9
            # fires and it stands in FEED; meanwhile it eats in proportion to how far the proboscis is out, a food bit by
            # bit (gone once eaten up), the attractive cursor without end. A sated fly's MN9 stays low: it does not eat.
            src = self.pheromone.food_at(self.fly.x, self.fly.y, target_radius)
            at_cursor = (self.pheromone.cursor_active and cursor_kind == "attract" and dist < target_radius)
            at_target = src is not None or at_cursor
            feeding = state == FEED and at_target
            if feeding:
                amount = EAT_RATE * self.runner.feed_level * dt
                if src is not None:
                    amount = self.pheromone.eat_at(self.fly.x, self.fly.y, amount, radius=target_radius,
                                                   screen_boxes=area.screens if hasattr(area, "screens") else None)
                self.metabolism.eat(amount)
                self._meal_spot = (self.fly.x, self.fly.y)
                if not self._was_feeding:               # a meal starts: the counter lands a foraging flight on it
                    self.runner.note_fed()
                    self.feed_effects.append(RewardEffect(self.fly.x, self.fly.y))
            self._was_feeding = feeding

            # the food the fly would fly to: the brain decides whether to; during the flight it stays the same source
            self.goal.update(self.pheromone, self.fly.x, self.fly.y, area, flying_to_it=long_flight)
        else:
            phero_steer = 0.0
            phero_drive = 0.0
            phero_repel = 0.0
            odor_delta = 0.0
            at_target = False
            self._was_feeding = False
            self.goal.source = None

        # Update metabolic energy depletion based on active state (fly burns ~16x rest)
        self.metabolism.update(dt, state)

        # pose and cursor in screen px feed visual input; pheromone feeds antennae. The worker measures the cursor's
        # closing speed itself from the stamped samples (`closing` is only its fallback), and a food odor weakens the
        # startle reflexes there in proportion to its strength (brain.odor_gate): the attractive cursor is a lure only
        # while its own scent is strong at the fly, not at every distance.
        effective_cursor = (cur_x, cur_y)
        self.fly.update(dt, state, effective_cursor, area, self.runner.steer,
                        mode="long" if long_flight else "", goal=self.goal.target(target_radius))
        # the goal as the fly has it after this step; where it last ate (the local search's path-integrator stand-in)
        goal_gap, goal_bearing, goal_odor = self.goal.reading(self.fly.x, self.fly.y, self.fly.heading, target_radius)
        if self._meal_spot is not None:
            meal_dist = math.hypot(self._meal_spot[0] - self.fly.x, self._meal_spot[1] - self.fly.y)
            meal_bearing = self.fly.bearing_of(self._meal_spot)
        else:
            meal_dist, meal_bearing = 1e6, 0.0

        self.runner.send(dist, closing, bearing, touch,
                         (self.fly.x, self.fly.y, self.fly.heading), (cur_x, cur_y), stamp=now,
                         capture_pos=(self.fly.x, self.fly.y),
                         phero_steer=phero_steer, phero_drive=phero_drive,
                         phero_repel=phero_repel, at_target=at_target,
                         hunger=self.metabolism.hunger,
                         wall_bump=self.fly.hit_wall,
                         goal_gap=goal_gap, goal_bearing=goal_bearing, goal_odor=goal_odor,
                         sugar=1.0 if at_target else 0.0, meal_dist=meal_dist, meal_bearing=meal_bearing,
                         odor_delta=odor_delta)

        # The drawn proboscis follows MN9 (the brain's feed level) while the fly feeds; floating reward effects
        want = self.runner.feed_level if state == FEED else 0.0
        self.feed_glow += (1.0 - math.exp(-dt / PROBOSCIS_TAU_S)) * (want - self.feed_glow)
        if self.feed_glow < 0.01:
            self.feed_glow = 0.0
        if self.feed_effects:
            self.feed_effects = [e for e in self.feed_effects if e.update(dt)]

        # Partial dirty-region invalidation to avoid full-screen / multi-monitor recomposition at 60 FPS
        g = self.geometry()
        fly_rect = self._get_fly_rect(g)
        curr_effect_rects = [self._get_effect_rect(e, g) for e in self.feed_effects]

        if self._prev_fly_rect is not None:
            self.update(self._prev_fly_rect)
        self.update(fly_rect)

        for r in self._prev_effect_rects:
            self.update(r)
        for r in curr_effect_rects:
            self.update(r)

        self._prev_fly_rect = fly_rect
        self._prev_effect_rects = curr_effect_rects

    def paintEvent(self, _):
        p = QPainter(self)
        g = self.geometry()
        # Draw fly with feeding animation and crop glow
        draw_fly(p, self.fly.x - g.left(), self.fly.y - g.top(), self.fly.heading,
                 self.runner.state, self.fly.phase, self.scale, feed_glow=self.feed_glow,
                 skin=self.skin)
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

    if "--download-data" in sys.argv or "--download" in sys.argv:
        from .data_manager import main as download_main
        download_main()
        sys.exit(0)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    apply_theme(app)

    from .theme import fly_icon
    app.setWindowIcon(fly_icon())

    # Prompt user to download FlyWire connectome if missing on startup, or to rebuild one built by an older version
    from .data_manager import is_connectome_ready
    if not is_connectome_ready() and "--toy" not in sys.argv:
        from .data_manager import prompt_startup_data
        prompt_startup_data()

    prefs = Preferences.load()

    # Initial engine config respecting saved preferences
    def_cfg = default_config()
    backend = "cpu"
    adapter = None
    if prefs.hardware == "cpu":
        backend = "cpu"
        adapter = None
    elif prefs.hardware == "gpu" and prefs.gpu_index is not None:
        backend = "gpu"
        adapter = prefs.gpu_index

    cfg = replace(
        def_cfg,
        circuit=prefs.circuit,
        n=prefs.neurons,
        dt=prefs.dt,
        symmetry=prefs.symmetry,
        learning=prefs.learning,
        mb_wiring=effective_mb_wiring(prefs.mb_wiring),
        backend=backend,
        adapter=adapter,
    )

    runner = Runner(cfg)
    runner.bias = prefs.bias
    runner.skittish = prefs.skittish
    runner.wall_pain = prefs.wall_pain
    runner.eye_height = prefs.eye_height
    runner.vision = prefs.vision_enabled

    app.aboutToQuit.connect(runner.stop)
    overlay = Overlay(runner, prefs=prefs)
    overlay.show()
    ctrl = Control(overlay, runner, prefs=prefs, auto_scan_gpus=True)
    ctrl.show()
    tray = Tray(app, ctrl, runner)
    tray.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    main()
