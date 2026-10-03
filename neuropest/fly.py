"""Fly body: position/heading physics driven by the behavior state."""
from __future__ import annotations

import math
import random

from .states import FLY, STAND, WALK

Rect = tuple[float, float, float, float]   # left, top, right, bottom of the area the fly's center may use


class Fly:
    def __init__(self, x: float, y: float):
        self.x, self.y, self.heading = x, y, random.uniform(0, math.tau)
        self.phase = 0.0
        self.turn_t = 0.0
        self.turn = 0.0

    def update(self, dt: float, state: str, cursor: tuple[float, float], rect: Rect):
        l, t, r, b = rect
        speed = {STAND: 0.0, WALK: 70.0, FLY: 420.0}[state]
        wx, wy = _wall_push(self.x, self.y, rect, 160.0 if state == FLY else 110.0)
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
            self.heading += self.turn * dt
            w = math.hypot(wx, wy)
            if w > 0:
                self.heading += _wrap(math.atan2(wy, wx) - self.heading) * min(1.0, 5 * w * dt)
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
        rate = {STAND: 0.0, WALK: 14.0, FLY: 120.0}[state]
        self.phase = (self.phase + rate * dt) % math.tau

    def clamp(self, rect: Rect):
        """Pull the fly back inside `rect` (screen or taskbar changed under it)."""
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
