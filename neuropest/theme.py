"""Dark theme for the control window and tray menu: colour tokens, Qt stylesheet, tray icon."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication

from .paths import ROOT
from .states import FLY, RETREAT, STAND, WALK

ASSETS = (ROOT / "neuropest" / "assets").as_posix()

BG = "#0b0b0d"          # window
SURFACE = "#141417"     # cards
RAISED = "#1c1c21"      # inputs, menus
BORDER = "#26262c"
BORDER_HI = "#3a3a42"
TEXT = "#ececf1"
MUTED = "#8b8b96"
FAINT = "#5c5c66"
ACCENT = "#4cc9f0"
WARN = "#f5a524"
ERROR = "#ff5d5d"

# colour and Turkish label per behaviour state, used by the state pill and the tray icon
STATE_STYLE = {
    STAND: ("#a1a1aa", "Duruyor"),
    WALK: ("#3ddc97", "Yürüyor"),
    RETREAT: (WARN, "Geri çekiliyor"),
    FLY: (ERROR, "Uçarak kaçıyor"),
}

QSS = f"""
* {{
    font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans", sans-serif;
    font-size: 13px;
    color: {TEXT};
}}
QWidget#Control {{ background: {BG}; }}
QLabel {{ background: transparent; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER_HI}; padding: 6px; }}

QFrame#Card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QLabel#Title {{ font-size: 18px; font-weight: 600; }}
QLabel#Subtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#Section {{
    color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1.2px;
}}
QLabel#Muted {{ color: {MUTED}; font-size: 12px; }}
QLabel#Faint {{ color: {FAINT}; font-size: 11px; }}
QLabel#Value {{ color: {TEXT}; font-size: 12px; font-family: "Cascadia Mono", "Consolas", monospace; }}
QLabel#StatValue {{ font-size: 17px; font-weight: 600; font-family: "Cascadia Mono", "Consolas", monospace; }}
QLabel#StatName {{ color: {MUTED}; font-size: 11px; }}
QLabel#Warn {{
    color: {WARN}; background: rgba(245, 165, 36, 0.08);
    border: 1px solid rgba(245, 165, 36, 0.35); border-radius: 8px; padding: 8px 10px;
}}
QLabel#Error {{
    color: {ERROR}; background: rgba(255, 93, 93, 0.08);
    border: 1px solid rgba(255, 93, 93, 0.35); border-radius: 8px; padding: 8px 10px;
}}
QLabel#Hint {{
    color: {MUTED}; background: {RAISED}; border-radius: 8px; padding: 8px 10px; font-size: 12px;
}}

QSlider {{ min-height: 22px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER_HI}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -6px 0; border-radius: 8px;
    background: {TEXT}; border: 1px solid {BG};
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; border: 2px solid {ACCENT}; }}
QSlider::sub-page:horizontal:disabled {{ background: {FAINT}; }}

QComboBox {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 6px 10px; min-height: 18px;
}}
QComboBox:hover {{ border-color: {BORDER_HI}; }}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url({ASSETS}/chevron.svg); width: 10px; height: 6px; }}
QComboBox QAbstractItemView {{
    background: {RAISED}; border: 1px solid {BORDER_HI}; border-radius: 8px; padding: 4px;
    selection-background-color: #26343b; selection-color: {TEXT}; outline: none;
}}

QCheckBox {{ spacing: 10px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 5px;
    border: 1px solid {BORDER_HI}; background: {RAISED};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT}; image: url({ASSETS}/check.svg);
}}

QPushButton {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 14px;
}}
QPushButton:hover {{ border-color: {BORDER_HI}; background: #232329; }}
QPushButton:pressed {{ background: {SURFACE}; }}

QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER_HI}; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}

QMenu {{
    background: {RAISED}; border: 1px solid {BORDER_HI}; border-radius: 10px; padding: 6px;
}}
QMenu::item {{ padding: 7px 22px 7px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: #26343b; }}
QMenu::item:disabled {{ color: {MUTED}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 6px; }}
QMenu::indicator {{ width: 14px; height: 14px; margin-left: 6px; }}
QMenu::indicator:checked {{ image: url({ASSETS}/check_accent.svg); }}
"""


def apply_theme(app: QApplication) -> None:
    """Fusion style, dark palette (for anything the stylesheet misses) and the stylesheet."""
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):          # Qt 6.8+: dark native title bar on Windows 10/11
        hints.setColorScheme(Qt.ColorScheme.Dark)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in [(QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, RAISED),
                      (QPalette.AlternateBase, SURFACE), (QPalette.Text, TEXT), (QPalette.Button, RAISED),
                      (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT),
                      (QPalette.HighlightedText, BG), (QPalette.ToolTipBase, RAISED),
                      (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, FAINT)]:
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(QSS)


def fly_icon(state: str = STAND, size: int = 64) -> QIcon:
    """Tray/window icon: a small fly silhouette on a dark disc, eyes tinted by behaviour state."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 64
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(BG))
    p.drawEllipse(QPointF(32 * s, 32 * s), 31 * s, 31 * s)
    p.translate(32 * s, 34 * s)
    p.rotate(-90)                       # head up
    p.scale(s, s)
    wing = QColor(TEXT)
    wing.setAlpha(70)
    p.setBrush(wing)
    for side in (-1, 1):
        p.save()
        p.rotate(side * 155)
        p.drawEllipse(QPointF(14, 0), 14, 6)
        p.restore()
    p.setBrush(QColor(TEXT))
    p.drawEllipse(QPointF(-10, 0), 11, 8)       # abdomen
    p.drawEllipse(QPointF(4, 0), 8, 7.5)        # thorax
    p.drawEllipse(QPointF(15, 0), 6, 6.5)       # head
    p.setBrush(QColor(STATE_STYLE.get(state, STATE_STYLE[STAND])[0] if state != STAND else "#e5484d"))
    for side in (-1, 1):
        p.drawEllipse(QPointF(17, side * 4), 3.2, 3.2)
    p.setPen(QPen(QColor(TEXT), 2.0, Qt.SolidLine, Qt.RoundCap))
    for side in (-1, 1):
        for a in (-40, 0, 40):              # three legs per side, splayed fore and aft
            r = math.radians(a)
            p.drawLine(QPointF(4, side * 6), QPointF(4 + 12 * math.sin(r), side * (6 + 9 * math.cos(r))))
    p.end()
    return QIcon(pm)
