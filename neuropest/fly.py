"""Fly body: position/heading physics driven by the behavior state."""
from __future__ import annotations

import math
import random

from .states import FLY, FREEZE, GROOM, RETREAT, STAND, WALK

Rect = tuple[float, float, float, float]   # left, top, right, bottom of the area the fly's center may use
TURN_GAIN = 0.02                           # rad/s of turning per Hz of right-minus-left DNa02 rate
FLY_RADIUS_PX: float = 22.0                # physical extent of body, folded wings (-22px) and legs
FLY_MARGIN_PX: float = 24.0                # radius + 2px safety margin to never poke past the bezel



class ScreenBox:
    """Descriptor for a single monitor's coordinate bounds and DPI mapping."""

    def __init__(self, id: int, name: str,
                 raw_l: float, raw_t: float, raw_r: float, raw_b: float,
                 phys_l: float, phys_t: float, phys_r: float, phys_b: float,
                 dpr: float = 1.0, margin: float = 2.0):
        self.id = id
        self.name = name
        self.raw_l = raw_l
        self.raw_t = raw_t
        self.raw_r = raw_r
        self.raw_b = raw_b
        self.phys_l = phys_l
        self.phys_t = phys_t
        self.phys_r = phys_r
        self.phys_b = phys_b
        self.dpr = dpr
        self.margin = margin

    @property
    def usable_l(self) -> float:
        return self.raw_l + self.margin

    @property
    def usable_t(self) -> float:
        return self.raw_t + self.margin

    @property
    def usable_r(self) -> float:
        return self.raw_r - self.margin

    @property
    def usable_b(self) -> float:
        return self.raw_b - self.margin

    def contains_point(self, x: float, y: float) -> bool:
        return self.raw_l <= x <= self.raw_r and self.raw_t <= y <= self.raw_b

    def to_phys(self, x: float, y: float) -> tuple[float, float]:
        px = self.phys_l + (x - self.raw_l) * self.dpr
        py = self.phys_t + (y - self.raw_t) * self.dpr
        return px, py

    def from_phys(self, px: float, py: float) -> tuple[float, float]:
        x = self.raw_l + (px - self.phys_l) / self.dpr
        y = self.raw_t + (py - self.phys_t) / self.dpr
        return x, y


class PlayArea:
    """Unified playable domain across one or multiple monitors."""

    def __init__(self, screens: list[ScreenBox]):
        if not screens:
            raise ValueError("PlayArea requires at least one screen")
        self.screens = screens
        self.last_hit_wall = False

    def __iter__(self):
        s = self.screens[0]
        return iter((s.usable_l, s.usable_t, s.usable_r, s.usable_b))

    def __getitem__(self, idx: int) -> float:
        s = self.screens[0]
        return (s.usable_l, s.usable_t, s.usable_r, s.usable_b)[idx]

    def __len__(self) -> int:
        return 4

    @classmethod
    def from_rect(cls, rect: Rect, margin: float = 0.0) -> PlayArea:
        l, t, r, b = rect
        s = ScreenBox(
            id=0, name="rect",
            raw_l=l - margin, raw_t=t - margin, raw_r=r + margin, raw_b=b + margin,
            phys_l=l - margin, phys_t=t - margin, phys_r=r + margin, phys_b=b + margin,
            dpr=1.0, margin=margin,
        )
        return cls([s])

    @classmethod
    def from_screens(cls, screens, margin: float = 2.0) -> PlayArea:
        boxes = []
        for i, s in enumerate(screens):
            g = s.geometry()
            dpr = float(s.devicePixelRatio())
            # In physical desktop coordinates (matching GDI capture and transparent Overlay HWND)
            phys_l = float(g.x())
            phys_t = float(g.y())
            phys_r = float(g.x()) + float(g.width()) * dpr
            phys_b = float(g.y()) + float(g.height()) * dpr
            boxes.append(ScreenBox(
                id=i, name=s.name(),
                raw_l=phys_l, raw_t=phys_t, raw_r=phys_r, raw_b=phys_b,
                phys_l=phys_l, phys_t=phys_t, phys_r=phys_r, phys_b=phys_b,
                dpr=1.0, margin=margin,
            ))
        return cls(boxes)

    def get_screen(self, x: float, y: float) -> ScreenBox:
        for s in self.screens:
            if s.contains_point(x, y):
                return s
        best, bd = self.screens[0], float("inf")
        for s in self.screens:
            cx = min(max(x, s.raw_l), s.raw_r)
            cy = min(max(y, s.raw_t), s.raw_b)
            d = (x - cx) ** 2 + (y - cy) ** 2
            if d < bd:
                bd, best = d, s
        return best

    def has_portal(self, s: ScreenBox, side: str, x: float, y: float) -> ScreenBox | None:
        if len(self.screens) <= 1:
            return None
        px, py = s.to_phys(x, y)
        eps = 2.0
        for other in self.screens:
            if other.id == s.id:
                continue
            if side == "right" and abs(s.phys_r - other.phys_l) <= eps:
                if max(s.phys_t, other.phys_t) <= py <= min(s.phys_b, other.phys_b):
                    return other
            elif side == "left" and abs(s.phys_l - other.phys_r) <= eps:
                if max(s.phys_t, other.phys_t) <= py <= min(s.phys_b, other.phys_b):
                    return other
            elif side == "bottom" and abs(s.phys_b - other.phys_t) <= eps:
                if max(s.phys_l, other.phys_l) <= px <= min(s.phys_r, other.phys_r):
                    return other
            elif side == "top" and abs(s.phys_t - other.phys_b) <= eps:
                if max(s.phys_l, other.phys_l) <= px <= min(s.phys_r, other.phys_r):
                    return other
        return None

    def wall_push(self, x: float, y: float, margin: float) -> tuple[float, float]:
        if len(self.screens) == 1:
            s = self.screens[0]
            wx = max(0.0, (margin - (x - s.usable_l)) / margin) - max(0.0, (margin - (s.usable_r - x)) / margin)
            wy = max(0.0, (margin - (y - s.usable_t)) / margin) - max(0.0, (margin - (s.usable_b - y)) / margin)
            return wx, wy

        s = self.get_screen(x, y)
        wx, wy = 0.0, 0.0
        if not self.has_portal(s, "left", x, y):
            lim = s.raw_l + s.margin
            wx += max(0.0, (margin - (x - lim)) / margin)
        if not self.has_portal(s, "right", x, y):
            lim = s.raw_r - s.margin
            wx -= max(0.0, (margin - (lim - x)) / margin)
        if not self.has_portal(s, "top", x, y):
            lim = s.raw_t + s.margin
            wy += max(0.0, (margin - (y - lim)) / margin)
        if not self.has_portal(s, "bottom", x, y):
            lim = s.raw_b - s.margin
            wy -= max(0.0, (margin - (lim - y)) / margin)
        return wx, wy

    def step(self, x: float, y: float, heading: float, speed: float, dt: float) -> tuple[float, float, float]:
        nx = x + math.cos(heading) * speed * dt
        ny = y + math.sin(heading) * speed * dt
        nh = heading
        hit_wall = False
        s = self.get_screen(x, y)

        portal = self.has_portal(s, "right", x, y)
        if portal:
            if nx > s.raw_r:
                over = nx - s.raw_r
                px, py = s.to_phys(s.raw_r, ny)
                nx, ny = portal.from_phys(portal.phys_l + over * (portal.dpr / s.dpr), py)
                s = portal
        else:
            if nx > s.usable_r:
                nx = s.usable_r
                nh = math.pi - nh
                if math.cos(nh) > -0.25:
                    nh = math.pi - 0.4 if math.sin(nh) >= 0 else -math.pi + 0.4
                hit_wall = True

        portal = self.has_portal(s, "left", x, y)
        if portal:
            if nx < s.raw_l:
                over = s.raw_l - nx
                px, py = s.to_phys(s.raw_l, ny)
                nx, ny = portal.from_phys(portal.phys_r - over * (portal.dpr / s.dpr), py)
                s = portal
        else:
            if nx < s.usable_l:
                nx = s.usable_l
                nh = math.pi - nh
                if math.cos(nh) < 0.25:
                    nh = 0.4 if math.sin(nh) >= 0 else -0.4
                hit_wall = True

        portal = self.has_portal(s, "bottom", nx, y)
        if portal:
            if ny > s.raw_b:
                over = ny - s.raw_b
                px, py = s.to_phys(nx, s.raw_b)
                nx, ny = portal.from_phys(px, portal.phys_t + over * (portal.dpr / s.dpr))
                s = portal
        else:
            if ny > s.usable_b:
                ny = s.usable_b
                nh = -nh
                if math.sin(nh) > -0.25:
                    nh = -0.4 if math.cos(nh) >= 0 else -math.pi + 0.4
                hit_wall = True

        portal = self.has_portal(s, "top", nx, y)
        if portal:
            if ny < s.raw_t:
                over = s.raw_t - ny
                px, py = s.to_phys(nx, s.raw_t)
                nx, ny = portal.from_phys(px, portal.phys_b - over * (portal.dpr / s.dpr))
                s = portal
        else:
            if ny < s.usable_t:
                ny = s.usable_t
                nh = -nh
                if math.sin(nh) < 0.25:
                    nh = 0.4 if math.cos(nh) >= 0 else math.pi - 0.4
                hit_wall = True

        min_x = s.raw_l if self.has_portal(s, "left", nx, ny) else s.usable_l
        max_x = s.raw_r if self.has_portal(s, "right", nx, ny) else s.usable_r
        min_y = s.raw_t if self.has_portal(s, "top", nx, ny) else s.usable_t
        max_y = s.raw_b if self.has_portal(s, "bottom", nx, ny) else s.usable_b
        nx = min(max(nx, min_x), max_x)
        ny = min(max(ny, min_y), max_y)
        self.last_hit_wall = hit_wall
        return nx, ny, _wrap(nh)

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        s = self.get_screen(x, y)
        min_x = s.raw_l if self.has_portal(s, "left", x, y) else s.usable_l
        max_x = s.raw_r if self.has_portal(s, "right", x, y) else s.usable_r
        min_y = s.raw_t if self.has_portal(s, "top", x, y) else s.usable_t
        max_y = s.raw_b if self.has_portal(s, "bottom", x, y) else s.usable_b
        return min(max(x, min_x), max_x), min(max(y, min_y), max_y)


class Fly:
    def __init__(self, x: float, y: float, alpha_steer: float = 0.4):
        self.x, self.y, self.heading = x, y, random.uniform(0, math.tau)
        self.phase = 0.0
        self.turn_t = 0.0
        self.turn = 0.0
        self.steer_ema = 0.0
        self.alpha_steer = alpha_steer
        self.hit_wall = False

    def bearing_of(self, cursor: tuple[float, float]) -> float:
        """Angle of `cursor` relative to the heading, radians, positive when it is to the fly's right."""
        return _wrap(math.atan2(cursor[1] - self.y, cursor[0] - self.x) - self.heading)

    def update(self, dt: float, state: str, cursor: tuple[float, float], rect: Rect | PlayArea, steer: float = 0.0):
        """steer: right-minus-left DNa02 rate (Hz) from the brain; turns a walking or retreating fly."""
        speed = {STAND: 0.0, WALK: 70.0, FLY: 420.0, RETREAT: -45.0, GROOM: 0.0, FREEZE: 0.0}[state]     # retreat = backward walking
        margin = 12.0 if state == FLY else 4.0

        # Transient filter on DNa02 steer (Rayshubskiy et al. 2025 biphasic filter: persistent input adapts)
        a_steer = 1.0 - math.exp(-dt / 0.4)
        self.steer_ema += a_steer * (steer - self.steer_ema)
        steer_eff = steer - self.alpha_steer * self.steer_ema

        if state == FLY:
            # Flight turning is driven purely by descending steering command neurons (DNa02 tropotaxis/pursuit)
            self.heading += (steer_eff * TURN_GAIN * 1.5) * dt
            self.heading = _wrap(self.heading)
        elif state == WALK:
            self.turn_t -= dt
            if self.turn_t <= 0:
                self.turn, self.turn_t = random.uniform(-1.5, 1.5), random.uniform(0.3, 1.2)
            self.heading += (self.turn * 0.5 + steer_eff * TURN_GAIN) * dt
            self.heading = _wrap(self.heading)
        elif state == RETREAT:
            self.heading += steer_eff * TURN_GAIN * dt          # keeps facing the stimulus while backing away
            self.heading = _wrap(self.heading)

        self.hit_wall = False
        if isinstance(rect, PlayArea):
            self.x, self.y, self.heading = rect.step(self.x, self.y, self.heading, speed, dt)
            self.hit_wall = getattr(rect, "last_hit_wall", False)
        else:
            l, t, r, b = rect
            self.x += math.cos(self.heading) * speed * dt
            self.y += math.sin(self.heading) * speed * dt
            # hard limits: stay inside the usable area (e.g. above the taskbar), bounce off
            if self.x <= l:
                self.x = l
                self.heading = math.pi - self.heading
                self.hit_wall = True
            elif self.x >= r:
                self.x = r
                self.heading = math.pi - self.heading
                self.hit_wall = True
            if self.y <= t:
                self.y = t
                self.heading = -self.heading
                self.hit_wall = True
            elif self.y >= b:
                self.y = b
                self.heading = -self.heading
                self.hit_wall = True
            self.heading = _wrap(self.heading)

        if self.hit_wall:
            self.turn_t = 0.6
            self.turn = 0.0

        rate = {STAND: 0.0, WALK: 14.0, FLY: 120.0, RETREAT: -10.0, GROOM: 22.0, FREEZE: 0.0}[state]
        self.phase = (self.phase + rate * dt) % math.tau

    def clamp(self, rect: Rect | PlayArea):
        """Pull the fly back inside `rect` (screen or taskbar changed under it)."""
        if isinstance(rect, PlayArea):
            self.x, self.y = rect.clamp(self.x, self.y)
        else:
            l, t, r, b = rect
            self.x = min(max(self.x, l), r)
            self.y = min(max(self.y, t), b)


def _wall_push(x: float, y: float, rect: Rect, margin: float) -> tuple[float, float]:
    """Inward unit-ish vector, growing from 0 to 1 per axis within `margin` px of a wall."""
    l, t, r, b = rect
    wx = max(0.0, (margin - (x - l)) / margin) - max(0.0, (margin - (r - x)) / margin)
    wy = max(0.0, (margin - (y - t)) / margin) - max(0.0, (margin - (b - y)) / margin)
    return wx, wy


def _wrap(a: float) -> float:
    return (a + math.pi) % math.tau - math.pi
