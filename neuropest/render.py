"""Procedural skeleton-style fly sprites (placeholder art)."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from .states import FLY, RETREAT, WALK

INK = QColor(235, 235, 235, 235)


def draw_fly(p: QPainter, x: float, y: float, heading: float, state: str,
             phase: float, scale: float = 1.0):
    p.save()
    p.translate(x, y)
    p.rotate(math.degrees(heading))   # +x axis = head direction
    p.scale(scale, scale)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(INK, 1.6, Qt.SolidLine, Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(11, 0), 4.5, 4)       # head
    p.drawEllipse(QPointF(0, 0), 6.5, 5)        # thorax
    p.drawEllipse(QPointF(-12, 0), 8, 5.2)      # abdomen
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
        _legs(p, phase if state in (WALK, RETREAT) else 0.0, tucked=False)
    p.restore()


def _legs(p: QPainter, phase: float, tucked: bool):
    # three leg pairs; alternating tripod gait while walking
    for i, ax in enumerate((6, 0, -6)):
        for s in (-1, 1):
            sw = math.sin(phase + (math.pi if (i + (s > 0)) % 2 else 0)) * 5
            if tucked:
                end = QPointF(ax - 5, s * 8)
            else:
                end = QPointF(ax + sw + (-3 if i == 2 else 3 if i == 0 else 0), s * 15)
            knee = QPointF(ax + (end.x() - ax) * 0.5, s * 9)
            p.drawLine(QPointF(ax, s * 4), knee)
            p.drawLine(knee, end)
