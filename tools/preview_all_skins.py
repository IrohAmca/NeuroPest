"""Render all available skins offscreen across all animation states to preview_all_skins.png."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication

from neuropest.render import AVAILABLE_SKINS, draw_fly

app = QApplication(sys.argv)

skins = list(AVAILABLE_SKINS.keys())
states = [("stand", 0.0, 0.0, "Duruş"),
          ("walk", 1.2, 0.0, "Yürüyüş"),
          ("fly", 0.6, 0.0, "Uçuş"),
          ("groom", 2.0, 0.0, "Temizlenme"),
          ("walk", 2.5, 0.8, "Beslenme (Nektar)")]

cols = len(states)
rows = len(skins)
cell_w, cell_h = 130, 130
header_h = 50
label_w = 160

width = label_w + cols * cell_w
height = header_h + rows * cell_h

img = QImage(width, height, QImage.Format_ARGB32)
img.fill(QColor(24, 26, 32))  # Dark sleek background to check transparency and edges

p = QPainter(img)
p.setRenderHint(QPainter.Antialiasing)
p.setRenderHint(QPainter.TextAntialiasing)

font = QFont("Segoe UI", 10, QFont.Bold)
p.setFont(font)
p.setPen(QColor(200, 210, 225))

# Draw column headers
for c, (_, _, _, state_title) in enumerate(states):
    x = label_w + c * cell_w
    p.drawText(x, 10, cell_w, 35, Qt.AlignCenter, state_title)

# Draw rows
for r, skin_key in enumerate(skins):
    y = header_h + r * cell_h
    skin_title = AVAILABLE_SKINS[skin_key]
    
    # Skin name label
    p.setPen(QColor(160, 180, 205))
    p.drawText(15, y, label_w - 20, cell_h, Qt.AlignVCenter | Qt.AlignLeft, skin_title)
    
    # Draw fly for each state
    for c, (state, phase, fg, _) in enumerate(states):
        cx = label_w + c * cell_w + cell_w / 2
        cy = y + cell_h / 2
        
        # Draw a faint target circle behind
        p.setPen(QColor(40, 45, 55))
        p.setBrush(QColor(30, 34, 42))
        p.drawEllipse(cx - 35, cy - 35, 70, 70)
        
        # Heading facing slightly to the right (0.0 rad = East / right)
        heading = 0.0 if c != 2 else -0.3
        draw_fly(p, cx, cy, heading, state, phase, scale=1.5, feed_glow=fg, skin=skin_key)

p.end()

out_path = "preview_all_skins.png"
img.save(out_path)
print(f"Successfully generated preview image: {out_path} ({width}x{height})")
