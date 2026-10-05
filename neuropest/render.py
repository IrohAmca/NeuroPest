"""Procedural skeleton-style fly sprites (placeholder art)."""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap

from .paths import SKINS_DIR
from .states import FLY, GROOM, RETREAT, WALK


INK = QColor(235, 235, 235, 235)


class RewardEffect:
    """Floating and fading reward marker spawned on pheromone food capture."""

    def __init__(self, x: float, y: float, duration: float = 0.9):
        self.x = x
        self.y = y
        self.duration = duration
        self.age = 0.0

    def update(self, dt: float) -> bool:
        self.age += dt
        return self.age < self.duration

    @property
    def progress(self) -> float:
        return min(1.0, self.age / max(self.duration, 1e-4))


def draw_reward_plus(p: QPainter, x: float, y: float, progress: float, scale: float = 1.0):
    """Draw a floating, fading emerald green plus sign (+) above (x, y) when pheromone/food is captured."""
    if progress < 0.0 or progress >= 1.0:
        return
    p.save()
    p.setRenderHint(QPainter.Antialiasing)

    # Float upwards over time
    rise = (26.0 * (progress ** 0.75)) * scale
    cy = y - (18.0 * scale) - rise
    cx = x

    # Fade curve: smooth in (first 15%), smooth out (last 75%)
    if progress < 0.15:
        alpha_factor = progress / 0.15
    else:
        alpha_factor = max(0.0, 1.0 - (progress - 0.15) / 0.85)

    alpha = int(245 * alpha_factor)
    if alpha <= 0:
        p.restore()
        return

    # Elastic slight pop/scale: starts at 0.85, reaches ~1.2, settles
    s = scale * (0.85 + 0.35 * math.sin(progress * math.pi * 0.9))

    # 1. Subtle soft halo / glow behind the plus
    halo_alpha = int(alpha * 0.22)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(52, 211, 153, halo_alpha))
    p.drawEllipse(QPointF(cx, cy), 9.0 * s, 9.0 * s)

    # 2. Main green plus arms
    arm = 5.2 * s
    thick = 2.4 * s
    pen_glow = QPen(QColor(46, 204, 113, alpha), thick, Qt.SolidLine, Qt.RoundCap)
    p.setPen(pen_glow)
    p.drawLine(QPointF(cx - arm, cy), QPointF(cx + arm, cy))
    p.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy + arm))

    # 3. Bright inner core for a crisp, luminous look
    core_thick = max(1.0, 1.2 * s)
    pen_core = QPen(QColor(230, 255, 235, alpha), core_thick, Qt.SolidLine, Qt.RoundCap)
    p.setPen(pen_core)
    core_arm = arm * 0.65
    p.drawLine(QPointF(cx - core_arm, cy), QPointF(cx + core_arm, cy))
    p.drawLine(QPointF(cx, cy - core_arm), QPointF(cx, cy + core_arm))

    p.restore()


SKIN_METADATA: dict[str, dict[str, str]] = {
    "classic": {
        "title": "Classic",
        "full_name": "Classic (Vector)",
        "desc": "Scientific skeleton; proboscis, head, thorax, and 6-legged tripod gait geometry.",
    },
    "chubby": {
        "title": "Chubby",
        "full_name": "Chubby Fly (Chibi)",
        "desc": "Fluffy body, cute wings, and 16-frame dorsal walking & flight animation.",
    },
    "cyborg": {
        "title": "Cyborg",
        "full_name": "Cyborg Fly (Mecha)",
        "desc": "Mecha robotic body, glowing blue energy wings, and cybernetic joints.",
    },
    "candy": {
        "title": "Candy",
        "full_name": "Candy Fly (Fairy)",
        "desc": "Colorful fairy/candy wings, soft pastel tones, and delicate footsteps.",
    },
    "cartoon": {
        "title": "Cartoon",
        "full_name": "Cartoon Fly (Retro)",
        "desc": "Characteristic big eyes, patterned back, and classic cartoon style.",
    },
}

AVAILABLE_SKINS: dict[str, str] = {k: v["full_name"] for k, v in SKIN_METADATA.items()}



class SkinManager:
    """Loads and caches sprite frames for top-down fly skins with transparent backgrounds."""

    def __init__(self, skins_dir: Path | None = None):
        if skins_dir is None:
            skins_dir = SKINS_DIR
        self.skins_dir = Path(skins_dir)
        self._skins: dict[str, dict[str, list[QPixmap] | QPixmap | None]] = {}

    def get_frame(self, skin_key: str, state: str, phase: float) -> QPixmap | None:
        if skin_key not in self._skins:
            self._load_skin(skin_key)
        data = self._skins.get(skin_key)
        if not data:
            return None

        idle = data.get("idle")
        walk = data.get("walk") or []
        fly = data.get("fly") or []

        # Normalized phase [0.0, 1.0)
        norm = (phase % math.tau) / math.tau if math.tau > 0 else 0.0

        if state == FLY:
            if fly:
                idx = int(norm * len(fly)) % len(fly)
                return fly[idx]
        elif state in (WALK, RETREAT):
            if walk:
                idx = int(norm * len(walk)) % len(walk)
                return walk[idx]
        elif state == GROOM:
            if walk:
                idx = int((norm * 2.0) * len(walk)) % len(walk)
                return walk[idx]
            return idle

        # STAND, FREEZE or fallback
        if idle is not None:
            return idle
        return walk[0] if walk else (fly[0] if fly else None)

    def _load_skin(self, skin_key: str):
        sp = self.skins_dir / skin_key
        if not sp.is_dir():
            self._skins[skin_key] = {}
            return

        idle_path = sp / "idle.png"
        idle_pm = QPixmap(str(idle_path)) if idle_path.exists() else None

        walk_files = sorted(sp.glob("walk_*.png"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0)
        walk_pms = [QPixmap(str(p)) for p in walk_files if not QPixmap(str(p)).isNull()]

        fly_files = sorted(sp.glob("fly_*.png"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0)
        fly_pms = [QPixmap(str(p)) for p in fly_files if not QPixmap(str(p)).isNull()]

        self._skins[skin_key] = {
            "idle": idle_pm,
            "walk": walk_pms,
            "fly": fly_pms,
        }


_SKIN_MANAGER = SkinManager()


def _draw_fly_sprite(p: QPainter, x: float, y: float, heading: float, state: str,
                     phase: float, scale: float, feed_glow: float, pixmap: QPixmap):
    p.save()
    p.translate(x, y)
    p.rotate(math.degrees(heading))   # +x axis = head direction
    p.scale(scale, scale)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)

    # Base display size: 46.0 px (matches procedural vector fly proportions)
    s_size = 46.0
    half = s_size / 2.0
    target_rect = QRectF(-half, -half, s_size, s_size)
    source_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
    p.drawPixmap(target_rect, pixmap, source_rect)

    # Feeding effects: proboscis extension and emerald nectar glow at head
    if feed_glow > 0.0:
        glow_a = int(190 * feed_glow)
        p.setBrush(QColor(46, 204, 113, glow_a))
        p.setPen(Qt.NoPen)
        ext = 2.0 + 1.2 * math.sin(phase * 4.0)
        p.drawEllipse(QPointF(half * 0.72 + ext, 0.0), 3.6, 3.6)

    p.restore()


def _draw_fly_procedural(p: QPainter, x: float, y: float, heading: float, state: str,
                         phase: float, scale: float = 1.0, feed_glow: float = 0.0):
    p.save()
    p.translate(x, y)
    p.rotate(math.degrees(heading))   # +x axis = head direction
    p.scale(scale, scale)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(INK, 1.6, Qt.SolidLine, Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)

    # Antennae
    ant_wiggle = math.sin(phase * 3.0) * 0.8 if feed_glow > 0 else 0.0
    p.drawLine(QPointF(14, -1.5), QPointF(17.5, -3.2 + ant_wiggle))
    p.drawLine(QPointF(14, 1.5), QPointF(17.5, 3.2 - ant_wiggle))

    # Proboscis extension reflex (PER) during feeding
    if feed_glow > 0.0:
        prob_pen = QPen(QColor(52, 211, 153, int(220 * feed_glow)), 1.4, Qt.SolidLine, Qt.RoundCap)
        p.setPen(prob_pen)
        ext = 2.4 + 1.2 * math.sin(phase * 4.0)
        p.drawLine(QPointF(15, 0), QPointF(15 + ext, 0))
        p.setPen(pen)

    p.drawEllipse(QPointF(11, 0), 4.5, 4)       # head
    p.drawEllipse(QPointF(0, 0), 6.5, 5)        # thorax
    p.drawEllipse(QPointF(-12, 0), 8, 5.2)      # abdomen

    # Soft emerald nectar glow in the abdomen when feeding
    if feed_glow > 0.0:
        glow_a = int(180 * feed_glow)
        p.setBrush(QColor(46, 204, 113, glow_a))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(-12, 0), 6.8, 4.4)
        p.setBrush(Qt.NoBrush)
        p.setPen(pen)

    p.setBrush(QColor(210, 40, 40, 220))        # eyes
    p.setPen(Qt.NoPen)
    p.drawEllipse(QPointF(13, -2.2), 1.6, 1.6)
    p.drawEllipse(QPointF(13, 2.2), 1.6, 1.6)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    if state == FLY:
        beat = math.sin(phase)
        for s in (-1, 1):
            ang = math.radians(60 + 25 * beat)
            tip = QPointF(-2 + 10 * math.cos(ang), s * (4 + 20 * math.sin(ang)))
            p.drawLine(QPointF(0, s * 4), tip)
            p.save()
            p.translate(tip)
            p.rotate(s * (90 - math.degrees(ang)) - 90 * s + 90 * s)
            p.drawEllipse(QPointF(0, s * 1.0), 4.5, 11)
            p.restore()
        _legs(p, 0.0, tucked=True)
    else:
        for s in (-1, 1):  # folded wings
            p.drawLine(QPointF(-1, s * 3), QPointF(-22, s * 5))
        _legs(p, phase if state in (WALK, RETREAT) else 0.0, tucked=False, groom=phase if state == GROOM else None)
    p.restore()


def draw_fly(p: QPainter, x: float, y: float, heading: float, state: str,
             phase: float, scale: float = 1.0, feed_glow: float = 0.0,
             skin: str = "classic"):
    """Draw fly using either procedural vector graphics or sprite-based skins."""
    if skin and skin != "classic":
        pixmap = _SKIN_MANAGER.get_frame(skin, state, phase)
        if pixmap is not None and not pixmap.isNull():
            _draw_fly_sprite(p, x, y, heading, state, phase, scale, feed_glow, pixmap)
            return

    _draw_fly_procedural(p, x, y, heading, state, phase, scale, feed_glow)



def _legs(p: QPainter, phase: float, tucked: bool, groom: float | None = None):
    # three leg pairs; alternating tripod gait while walking; while grooming the front legs rub the head
    for i, ax in enumerate((6, 0, -6)):
        for s in (-1, 1):
            if groom is not None and i == 0:
                rub = math.sin(groom + (math.pi if s > 0 else 0))
                p.drawLine(QPointF(ax, s * 4), QPointF(9, s * 8))
                p.drawLine(QPointF(9, s * 8), QPointF(13 + 1.5 * rub, s * (3 + 1.5 * rub)))
                continue
            sw = math.sin(phase + (math.pi if (i + (s > 0)) % 2 else 0)) * 5
            if tucked:
                end = QPointF(ax - 5, s * 8)
            else:
                end = QPointF(ax + sw + (-3 if i == 2 else 3 if i == 0 else 0), s * 15)
            knee = QPointF(ax + (end.x() - ax) * 0.5, s * 9)
            p.drawLine(QPointF(ax, s * 4), knee)
            p.drawLine(knee, end)
