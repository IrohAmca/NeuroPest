"""Pheromone field and tropotaxis (bilateral chemosensation).

Manages invisible positive chemical (nectar/food) sources across the desktop
and an invisible slim border repulsion keeping the fly on-screen.
Pheromone sources are invisible to the user and disappear (get consumed)
when the fly reaches and feeds on them.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PheromoneSource:
    x: float
    y: float
    kind: str = "attract"  # all free sources are purely attractive (food/nectar)
    radius: float = 220.0  # spatial diffusion scale (px)
    strength: float = 1.0

    def concentration_at(self, px: float, py: float) -> float:
        d2 = (self.x - px) ** 2 + (self.y - py) ** 2
        sig2 = self.radius ** 2
        # Smooth Gaussian falloff with a finite outer cutoff at 2.5 * radius
        if d2 > sig2 * 6.25:
            return 0.0
        return self.strength * math.exp(-d2 / (2.0 * (sig2 * 0.4)))


class PheromoneField:
    """Manages active invisible pheromone sources, border repulsion, and antenna sampling."""

    def __init__(self, border_margin: float = 0.0, border_strength: float = 0.0):
        self.sources: List[PheromoneSource] = []
        self.border_margin = border_margin  # 0.0 = no artificial border repulsion zone
        self.border_strength = border_strength
        self.cursor_source: PheromoneSource | None = None
        self.cursor_active = False

    def clear(self) -> None:
        self.sources.clear()

    def add_source(self, x: float, y: float, kind: str = "attract",
                   radius: float = 220.0, strength: float = 1.0) -> PheromoneSource:
        src = PheromoneSource(x=x, y=y, kind=kind, radius=radius, strength=strength)
        self.sources.append(src)
        return src

    def spawn_random_sources(self, screen_boxes, count: int = 6) -> None:
        """Seed desktop with purely positive/attractive food sources."""
        self.sources.clear()
        if not screen_boxes:
            return
        for _ in range(count):
            self._spawn_one(screen_boxes)

    def _spawn_one(self, screen_boxes) -> None:
        if not screen_boxes:
            return
        s = random.choice(screen_boxes)
        margin = 80.0
        w = max(100.0, (s.usable_r - s.usable_l) - 2 * margin)
        h = max(100.0, (s.usable_b - s.usable_t) - 2 * margin)
        x = s.usable_l + margin + random.random() * w
        y = s.usable_t + margin + random.random() * h
        self.add_source(x, y, radius=random.uniform(190.0, 250.0), strength=1.0)

    def consume_at(self, fly_x: float, fly_y: float, consume_radius: float = 30.0, screen_boxes=None) -> bool:
        """When fly reaches a positive source, it consumes it: source disappears, new one spawns elsewhere."""
        consumed = False
        remaining = []
        for s in self.sources:
            if math.hypot(s.x - fly_x, s.y - fly_y) <= consume_radius:
                consumed = True  # source is eaten and disappears
            else:
                remaining.append(s)
        self.sources = remaining
        if consumed and screen_boxes and len(self.sources) < 6:
            self._spawn_one(screen_boxes)
        return consumed

    def update_cursor(self, cx: float, cy: float, kind: str = "attract", enabled: bool = True) -> None:
        if not enabled:
            self.cursor_active = False
            return
        self.cursor_active = True
        if self.cursor_source is None:
            self.cursor_source = PheromoneSource(cx, cy, kind=kind, radius=200.0, strength=1.0)
        else:
            self.cursor_source.x = cx
            self.cursor_source.y = cy
            self.cursor_source.kind = kind

    def update(self, dt: float) -> None:
        pass  # pheromones are static spatial diffusions

    def sample_point(self, x: float, y: float, play_area=None) -> Tuple[float, float]:
        """Returns (c_attract, c_repel) at given coordinate (x, y)."""
        c_attr = 0.0
        c_rep = 0.0

        for s in self.sources:
            c_attr += s.concentration_at(x, y)

        if self.cursor_source and self.cursor_active:
            c = self.cursor_source.concentration_at(x, y)
            if self.cursor_source.kind == "attract":
                c_attr += c
            else:
                c_rep += c

        # Border repulsion: only calculated if border_margin and border_strength are positive
        if self.border_margin > 0.0 and self.border_strength > 0.0 and play_area is not None:
            sbox = play_area.get_screen(x, y) if hasattr(play_area, "get_screen") else None
            if sbox:
                m = self.border_margin
                wall_dists = []
                if not (hasattr(play_area, "has_portal") and play_area.has_portal(sbox, "left", x, y)):
                    wall_dists.append(x - sbox.usable_l)
                if not (hasattr(play_area, "has_portal") and play_area.has_portal(sbox, "right", x, y)):
                    wall_dists.append(sbox.usable_r - x)
                if not (hasattr(play_area, "has_portal") and play_area.has_portal(sbox, "top", x, y)):
                    wall_dists.append(y - sbox.usable_t)
                if not (hasattr(play_area, "has_portal") and play_area.has_portal(sbox, "bottom", x, y)):
                    wall_dists.append(sbox.usable_b - y)
                if wall_dists:
                    dist_edge = min(wall_dists)
                    if dist_edge < m:
                        edge_factor = max(0.0, (m - max(0.0, dist_edge)) / max(m, 1e-4)) ** 2
                        c_rep += self.border_strength * edge_factor

        return c_attr, c_rep

    def sample_antennae(self, fly_x: float, fly_y: float, heading: float,
                        play_area=None, antenna_dist: float = 12.0,
                        antenna_angle_rad: float = 0.55) -> dict:
        """Sample bilateral olfactory concentration at left and right antenna positions."""
        hl = heading - antenna_angle_rad
        ax_l = fly_x + math.cos(hl) * antenna_dist
        ay_l = fly_y + math.sin(hl) * antenna_dist

        hr = heading + antenna_angle_rad
        ax_r = fly_x + math.cos(hr) * antenna_dist
        ay_r = fly_y + math.sin(hr) * antenna_dist

        al_attr, al_rep = self.sample_point(ax_l, ay_l, play_area)
        ar_attr, ar_rep = self.sample_point(ax_r, ay_r, play_area)

        delta_attr = ar_attr - al_attr
        delta_rep = ar_rep - al_rep

        total_attr = (al_attr + ar_attr) * 0.5
        total_rep = (al_rep + ar_rep) * 0.5

        return {
            "left_pos": (ax_l, ay_l),
            "right_pos": (ax_r, ay_r),
            "al_attr": al_attr,
            "ar_attr": ar_attr,
            "al_rep": al_rep,
            "ar_rep": ar_rep,
            "delta_attr": delta_attr,  # + = right stronger
            "delta_rep": delta_rep,    # + = right stronger
            "total_attr": total_attr,
            "total_rep": total_rep,
        }

    def render(self, p, play_area=None, scale: float = 1.0) -> None:
        """Pheromones are invisible to the user: nothing is rendered."""
        pass
