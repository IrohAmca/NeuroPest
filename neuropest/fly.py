"""Fly body: position/heading physics driven by the behavior state."""
from __future__ import annotations

import math
import random

from .brain import FLY, STAND, WALK


class Fly:
    def __init__(self, x: float, y: float):
        self.x, self.y, self.heading = x, y, random.uniform(0, math.tau)
        self.phase = 0.0
        self.turn_t = 0.0
        self.turn = 0.0

    def update(self, dt: float, state: str, cursor: tuple[float, float],
               bounds: tuple[float, float, float, float]):
        speed = {STAND: 0.0, WALK: 70.0, FLY: 420.0}[state]
        if state == FLY:
            # flee: head directly away from the cursor
            ang = math.atan2(self.y - cursor[1], self.x - cursor[0])
            self.heading += _wrap(ang - self.heading) * min(1.0, 6 * dt)
        elif state == WALK:
            self.turn_t -= dt
            if self.turn_t <= 0:
                self.turn, self.turn_t = random.uniform(-1.5, 1.5), random.uniform(0.3, 1.2)
            self.heading += self.turn * dt
        self.x += math.cos(self.heading) * speed * dt
        self.y += math.sin(self.heading) * speed * dt
        l, t, r, b = bounds
        if not l + 20 < self.x < r - 20 or not t + 20 < self.y < b - 20:
            self.heading = math.atan2((t + b) / 2 - self.y, (l + r) / 2 - self.x)
            self.x = min(max(self.x, l + 20), r - 20)
            self.y = min(max(self.y, t + 20), b - 20)
        rate = {STAND: 0.0, WALK: 14.0, FLY: 120.0}[state]
        self.phase = (self.phase + rate * dt) % math.tau


def _wrap(a: float) -> float:
    return (a + math.pi) % math.tau - math.pi
