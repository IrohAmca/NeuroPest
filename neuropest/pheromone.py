"""Pheromone field and tropotaxis (bilateral chemosensation).

Manages invisible positive chemical (nectar/food) sources across the desktop
and an invisible slim border repulsion keeping the fly on-screen.
Pheromone sources are invisible to the user. A food has a size: its odor spreads further and smells stronger the
bigger it is, the fly eats it bit by bit while it feeds (brain FEED state, driven by MN9), and it is gone (a new one
appears elsewhere) once eaten up.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

# Food on the desktop [DESIGNED]. Few and sparse (was 7): their odors do not merge into one field and the fly has to
# search (motor-asamalari-inceleme 4b). A food of size 1 spreads its odor over FOOD_RADIUS_PX; radius and strength grow
# with the square root of the size, so a big food is smelled from further and more strongly while the steepest left-right
# difference (the steering input) stays the same. Eating a whole food of size 1 takes ~10 s at a fully extended proboscis
# (app.EAT_RATE).
FOOD_COUNT = 3
FOOD_RADIUS_PX = 300.0
FOOD_SIZE_RANGE = (0.6, 1.4)
FOOD_MIN_SIZE = 0.05            # below this the food is eaten up


@dataclass
class PheromoneSource:
    x: float
    y: float
    kind: str = "attract"  # all free sources are purely attractive (food/nectar)
    radius: float = 220.0  # spatial diffusion scale (px) at size 1
    strength: float = 1.0  # peak concentration at size 1
    size: float = 1.0      # how much food is left (radius and strength scale with sqrt(size))

    def __post_init__(self):
        self.base_radius, self.base_strength = self.radius, self.strength
        self.set_size(self.size)

    def set_size(self, size: float) -> None:
        """Food left; the odor shrinks with it."""
        self.size = max(0.0, float(size))
        f = math.sqrt(self.size)
        self.radius = max(1e-3, self.base_radius * f)
        self.strength = self.base_strength * f
        sig2 = self.radius ** 2
        self._cutoff_d2 = sig2 * 6.25
        self._denom = 2.0 * (sig2 * 0.4)

    def concentration_at(self, px: float, py: float) -> float:
        d2 = (self.x - px) ** 2 + (self.y - py) ** 2
        # Smooth Gaussian falloff with a finite outer cutoff at 2.5 * radius
        if d2 > self._cutoff_d2:
            return 0.0
        return self.strength * math.exp(-d2 / self._denom)


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
                   radius: float = 220.0, strength: float = 1.0, size: float = 1.0) -> PheromoneSource:
        src = PheromoneSource(x=x, y=y, kind=kind, radius=radius, strength=strength, size=size)
        self.sources.append(src)
        return src

    def spawn_random_sources(self, screen_boxes, count: int = FOOD_COUNT) -> None:
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
        self.add_source(x, y, radius=FOOD_RADIUS_PX, strength=1.0, size=random.uniform(*FOOD_SIZE_RANGE))

    def food_at(self, x: float, y: float, radius: float) -> PheromoneSource | None:
        """The attractive source the fly at (x, y) stands on (the nearest within `radius`), or None."""
        best, best_d = None, radius
        for s in self.sources:
            d = math.hypot(s.x - x, s.y - y)
            if s.kind == "attract" and d <= best_d:
                best, best_d = s, d
        return best

    def eat_at(self, x: float, y: float, amount: float, radius: float = 30.0, screen_boxes=None) -> float:
        """The fly at (x, y) eats up to `amount` of the food it stands on; returns how much it ate. A food eaten up
        disappears and a new one appears elsewhere (keeping FOOD_COUNT)."""
        src = self.food_at(x, y, radius)
        if src is None or amount <= 0.0:
            return 0.0
        eaten = min(float(amount), src.size)
        src.set_size(src.size - eaten)
        if src.size < FOOD_MIN_SIZE:
            eaten += src.size
            self.sources = [s for s in self.sources if s is not src]
            if screen_boxes and len(self.sources) < FOOD_COUNT:
                self._spawn_one(screen_boxes)
        return eaten

    def consume_at(self, fly_x: float, fly_y: float, consume_radius: float = 30.0, screen_boxes=None) -> bool:
        """The whole food under the fly at once: it disappears and a new one spawns elsewhere (eat_at eats it bit by bit)."""
        consumed = False
        remaining = []
        for s in self.sources:
            if math.hypot(s.x - fly_x, s.y - fly_y) <= consume_radius:
                consumed = True  # source is eaten and disappears
            else:
                remaining.append(s)
        self.sources = remaining
        if consumed and screen_boxes and len(self.sources) < FOOD_COUNT:
            self._spawn_one(screen_boxes)
        return consumed

    def goal_for(self, x: float, y: float, screen=None) -> tuple[PheromoneSource | None, float]:
        """The food source a fly at (x, y) would fly to, and its odor there: the one it smells most, on `screen` (a
        ScreenBox) when given, so a flight never crosses to another monitor. The cursor is never a goal: in attract mode
        the pet can feed at it and would keep hopping to it. (None, 0.0) when it smells none."""
        best, best_c = None, 0.0
        for s in self.sources:
            if s.kind != "attract" or (screen is not None and not screen.contains_point(s.x, s.y)):
                continue
            c = s.concentration_at(x, y)
            if c > best_c:
                best, best_c = s, c
        return best, best_c

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

    def sample_point(self, x: float, y: float, play_area=None, sbox=None) -> Tuple[float, float]:
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
            if sbox is None:
                sbox = play_area.get_screen(x, y) if hasattr(play_area, "get_screen") else None
            elif hasattr(sbox, "contains_point") and not sbox.contains_point(x, y):
                sbox = play_area.get_screen(x, y) if hasattr(play_area, "get_screen") else None
            if sbox:
                m = self.border_margin
                wall_dists = []
                has_portal = getattr(play_area, "has_portal", None)
                if not (has_portal and has_portal(sbox, "left", x, y)):
                    wall_dists.append(x - sbox.usable_l)
                if not (has_portal and has_portal(sbox, "right", x, y)):
                    wall_dists.append(sbox.usable_r - x)
                if not (has_portal and has_portal(sbox, "top", x, y)):
                    wall_dists.append(y - sbox.usable_t)
                if not (has_portal and has_portal(sbox, "bottom", x, y)):
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

        fly_sbox = play_area.get_screen(fly_x, fly_y) if (play_area is not None and hasattr(play_area, "get_screen")) else None

        al_attr, al_rep = self.sample_point(ax_l, ay_l, play_area, sbox=fly_sbox)
        ar_attr, ar_rep = self.sample_point(ax_r, ay_r, play_area, sbox=fly_sbox)

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

    def smell(self, x: float, y: float, heading: float, play_area=None, scale: float = 1.0) -> tuple[float, float, float]:
        """What the fly's antennae give the brain: (steer, drive, repel) = the right-minus-left tropotaxis bias (Hz; an
        attractant on the right turns it right, a repellent turns it away), the attractant and the repellent intensity."""
        ant = self.sample_antennae(x, y, heading, play_area, antenna_dist=12.0 * scale)
        return ant["delta_attr"] * 180.0 - ant["delta_rep"] * 220.0, ant["total_attr"], ant["total_rep"]

    def render(self, p, play_area=None, scale: float = 1.0) -> None:
        """Pheromones are invisible to the user: nothing is rendered."""
        pass


class GoalTracker:
    """The food source a foraging flight goes to (brain.GOAL_*): chosen again every tick while the fly is not flying to
    it (PheromoneField.goal_for, on the fly's own screen), kept during the flight, dropped once it is eaten."""

    def __init__(self):
        self.source: PheromoneSource | None = None

    def update(self, field: PheromoneField, x: float, y: float, play_area=None, flying_to_it: bool = False):
        if flying_to_it:
            if self.source is not None and not any(s is self.source for s in field.sources):
                self.source = None                  # eaten: the brain lands the fly on the meal counter
        else:
            screen = play_area.get_screen(x, y) if play_area is not None and hasattr(play_area, "get_screen") else None
            self.source, _ = field.goal_for(x, y, screen)
        return self.source

    def target(self, radius: float) -> tuple[float, float, float] | None:
        """(x, y, eating radius) for fly.update, or None."""
        return None if self.source is None else (self.source.x, self.source.y, radius)

    def reading(self, x: float, y: float, heading: float, radius: float) -> tuple[float, float, float]:
        """(gap, bearing, odor) for the brain: px past the eating radius (<= 0: at it), rad (+ to the right), the
        source's own odor at (x, y). Without a goal: (1e6, 0, 0)."""
        s = self.source
        if s is None:
            return 1e6, 0.0, 0.0
        dx, dy = s.x - x, s.y - y
        bearing = (math.atan2(dy, dx) - heading + math.pi) % math.tau - math.pi
        return math.hypot(dx, dy) - radius, bearing, s.concentration_at(x, y)
