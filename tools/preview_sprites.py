"""Render the placeholder sprites offscreen to sprites_preview.png and check walk drive."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from neuropest.brain import Brain
from neuropest.render import draw_fly

b = Brain()
b.walk_bias = 100
seen = set(b.step(16, 600, 0) for _ in range(150))
print("walk drive 100 Hz:", seen, float(b.c.rate[b.g["WALK"]].mean()),
      float(b.c.rate[b.g["REST"]].mean()))

app = QApplication(sys.argv)
img = QImage(360, 120, QImage.Format_ARGB32)
img.fill(QColor(40, 40, 40))
p = QPainter(img)
for i, (s, ph) in enumerate([("stand", 0), ("walk", 1.0), ("fly", 0.5)]):
    draw_fly(p, 60 + 120 * i, 60, -1.57, s, ph, 2.0)
p.end()
img.save("sprites_preview.png")
