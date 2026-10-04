"""Fly body: position/heading physics driven by the behavior state."""
from __future__ import annotations

import math
import random

from .states import FLY, FREEZE, GROOM, RETREAT, STAND, WALK

Rect = tuple[float, float, float, float]   # left, top, right, bottom of the area the fly's center may use
TURN_GAIN = 0.02                           # rad/s of turning per Hz of right-minus-left DNa02 rate


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
        s = self.get_screen(x, y)

        if nx > s.raw_r:
            portal = self.has_portal(s, "right", x, y)
            if portal:
                over = nx - s.raw_r
                px, py = s.to_phys(s.raw_r, ny)
                nx, ny = portal.from_phys(portal.phys_l + over * (portal.dpr / s.dpr), py)
                s = portal
            else:
                nx = s.raw_r - s.margin
                nh = math.pi - nh
        elif nx < s.raw_l:
            portal = self.has_portal(s, "left", x, y)
            if portal:
                over = s.raw_l - nx
                px, py = s.to_phys(s.raw_l, ny)
                nx, ny = portal.from_phys(portal.phys_r - over * (portal.dpr / s.dpr), py)
                s = portal
            else:
                nx = s.raw_l + s.margin
                nh = math.pi - nh

        if ny > s.raw_b:
            portal = self.has_portal(s, "bottom", nx, y)
            if portal:
                over = ny - s.raw_b
                px, py = s.to_phys(nx, s.raw_b)
                nx, ny = portal.from_phys(px, portal.phys_t + over * (portal.dpr / s.dpr))
                s = portal
            else:
                ny = s.raw_b - s.margin
                nh = -nh
        elif ny < s.raw_t:
            portal = self.has_portal(s, "top", nx, y)
            if portal:
                over = s.raw_t - ny
                px, py = s.to_phys(nx, s.raw_t)
                nx, ny = portal.from_phys(px, portal.phys_b - over * (portal.dpr / s.dpr))
                s = portal
            else:
                ny = s.raw_t + s.margin
                nh = -nh

        min_x = s.raw_l if self.has_portal(s, "left", nx, ny) else s.raw_l + s.margin
        max_x = s.raw_r if self.has_portal(s, "right", nx, ny) else s.raw_r - s.margin
        min_y = s.raw_t if self.has_portal(s, "top", nx, ny) else s.raw_t + s.margin
        max_y = s.raw_b if self.has_portal(s, "bottom", nx, ny) else s.raw_b - s.margin
        nx = min(max(nx, min_x), max_x)
        ny = min(max(ny, min_y), max_y)
        return nx, ny, _wrap(nh)

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        s = self.get_screen(x, y)
        min_x = s.raw_l if self.has_portal(s, "left", x, y) else s.raw_l + s.margin
        max_x = s.raw_r if self.has_portal(s, "right", x, y) else s.raw_r - s.margin
        min_y = s.raw_t if self.has_portal(s, "top", x, y) else s.raw_t + s.margin
        max_y = s.raw_b if self.has_portal(s, "bottom", x, y) else s.raw_b - s.margin
        return min(max(x, min_x), max_x), min(max(y, min_y), max_y)


class Fly:
    def __init__(self, x: float, y: float):
        self.x, self.y, self.heading = x, y, random.uniform(0, math.tau)
        self.phase = 0.0
        self.turn_t = 0.0
        self.turn = 0.0

    def bearing_of(self, cursor: tuple[float, float]) -> float:
        """Angle of `cursor` relative to the heading, radians, positive when it is to the fly's right."""
        return _wrap(math.atan2(cursor[1] - self.y, cursor[0] - self.x) - self.heading)

    def update(self, dt: float, state: str, cursor: tuple[float, float], rect: Rect | PlayArea, steer: float = 0.0):
        """steer: right-minus-left DNa02 rate (Hz) from the brain; turns a walking or retreating fly."""
        speed = {STAND: 0.0, WALK: 70.0, FLY: 420.0, RETREAT: -45.0, GROOM: 0.0, FREEZE: 0.0}[state]     # retreat = backward walking
        margin = 24.0 if state == FLY else 14.0

        if isinstance(rect, PlayArea):
            wx, wy = rect.wall_push(self.x, self.y, margin)
        else:
            wx, wy = _wall_push(self.x, self.y, rect, margin)

        if state == FLY:
            # flee from the cursor, bending away from walls so it never pins itself to an edge
            ax, ay = self.x - cursor[0], self.y - cursor[1]
            n = math.hypot(ax, ay) or 1.0
            desired = math.atan2(ay / n + 1.6 * wy, ax / n + 1.6 * wx)
            self.heading += _wrap(desired - self.heading) * min(1.0, 8 * dt)
        elif state == WALK:
            self.turn_t -= dt
            if self.turn_t <= 0:
                self.turn, self.turn_t = random.uniform(-1.5, 1.5), random.uniform(0.3, 1.2)
            self.heading += (self.turn * 0.5 + steer * TURN_GAIN) * dt
            w = math.hypot(wx, wy)
            if w > 0:
                self.heading += _wrap(math.atan2(wy, wx) - self.heading) * min(1.0, 5 * w * dt)
        elif state == RETREAT:
            self.heading += steer * TURN_GAIN * dt          # keeps facing the cursor while backing away

        if isinstance(rect, PlayArea):
            self.x, self.y, self.heading = rect.step(self.x, self.y, self.heading, speed, dt)
        else:
            l, t, r, b = rect
            self.x += math.cos(self.heading) * speed * dt
            self.y += math.sin(self.heading) * speed * dt
            # hard limits: stay inside the usable area (e.g. above the taskbar), bounce off
            if self.x < l or self.x > r:
                self.x = min(max(self.x, l), r)
                self.heading = math.pi - self.heading
            if self.y < t or self.y > b:
                self.y = min(max(self.y, t), b)
                self.heading = -self.heading
            self.heading = _wrap(self.heading)

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
