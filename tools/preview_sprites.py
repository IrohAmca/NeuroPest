"""Render the placeholder sprites offscreen to sprites_preview.png and check walk drive."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from neuropest.brain import Brain
from neuropest.render import draw_fly
from neuropest.toy_circuit import build

b = Brain(build(146))
b.walk_bias = 1.0

app = QApplication(sys.argv)
img = QImage(480, 120, QImage.Format_ARGB32)
img.fill(QColor(40, 40, 40))
p = QPainter(img)
for i, (s, ph, fg) in enumerate([("stand", 0, 0.0), ("walk", 1.0, 0.0), ("fly", 0.5, 0.0), ("stand", 0.8, 1.0)]):
    draw_fly(p, 60 + 120 * i, 60, -1.57, s, ph, 2.0, feed_glow=fg)
p.end()
img.save("sprites_preview.png")
