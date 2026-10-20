"""Dark theme for the control window and tray menu: colour tokens, Qt stylesheet, tray icon."""
from __future__ import annotations

import math
import re

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication

from pathlib import Path

from .states import FEED, FLY, FREEZE, GROOM, RETREAT, STAND, WALK

ASSETS = (Path(__file__).resolve().parent / "assets").as_posix()

# "Fluorescence microscope" (design_handoff, final decisions): near-black ground, pale green-white text, thin dark-green
# hairlines, no shadows, blur, gradients or glow. The behaviour colours are the only saturated colour.
BG = "#05070b"          # window
STRIP_BG = "#0b120d"    # kymograph strip ground
SURFACE = STRIP_BG
BORDER = "#16261b"      # panel border / hairline
BORDER_BTN = "#24382a"  # button and select border
BORDER_HI = "#2f4a37"   # hover border
BTN_BG = "#08100b"
RAISED = BTN_BG         # inputs, buttons, menus
SEL_BG = "#0d1a11"      # hover / selected background
EDGE_OFF = "#1f3326"    # inactive circuit edge
NODE_OFF = "#2a3f30"    # inactive circuit node
TEXT = "#cfe9cf"
TEXT_STRONG = "#e6fbe2"  # values, active
MUTED = "#6f8a74"       # hints, inactive labels
MUTED_2 = "#9fbfa4"     # inactive names on the secondary pages
FAINT = "#4a6450"       # section labels
ACCENT = TEXT_STRONG
WARN = "#f5a524"
WARN_BORDER = "#4a3a16"
ERROR = "#ff5d5d"

def pt(px: float) -> float:
    """Point size of a CSS pixel size at the 96 dpi the design was drawn at. Qt rounds a fractional `px` in a stylesheet
    up (12.5px -> 13, 9.5px -> 10), while a fractional `pt` is kept, so every size goes through here."""
    return px * 0.75


def fs(px: float) -> str:
    """`font-size` value for a stylesheet, from CSS pixels (see pt)."""
    return f"{pt(px):g}pt"


def _fractional_pt(qss: str) -> str:
    """Every `font-size: Npx` of a stylesheet as the equal point size."""
    return re.sub(r"font-size:\s*([\d.]+)px", lambda m: f"font-size: {fs(float(m.group(1)))}", qss)


UI_FONT = '"Geist", "Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans", sans-serif'
MONO_FONT = '"Geist Mono", "Cascadia Mono", "Consolas", monospace'

# colour and English label per behaviour state, used by the state pill, the circuit and the tray icon
STATE_STYLE = {
    STAND: ("#a1a1aa", "Standing"),
    WALK: ("#3ddc97", "Walking"),
    RETREAT: (WARN, "Retreating"),
    FLY: (ERROR, "Escaping"),
    GROOM: ("#7dd3fc", "Grooming"),
    FREEZE: ("#c4b5fd", "Freezing"),
    FEED: ("#fbbf24", "Feeding"),
}

_QSS = f"""
* {{
    font-family: {UI_FONT};
    font-size: 12.5px;
    color: {TEXT};
}}
QWidget#Control {{ background: {BG}; }}
QLabel {{ background: transparent; }}
QToolTip {{ background: {BTN_BG}; color: {TEXT}; border: 1px solid {BORDER_HI}; padding: 5px; }}

QFrame#Card {{
    background: transparent;
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QLabel#Brand {{ font-size: 15px; font-weight: 600; color: {TEXT_STRONG}; }}
QLabel#Title {{ font-size: 16px; font-weight: 600; color: {TEXT_STRONG}; }}
QLabel#Subtitle {{ color: {MUTED}; font-size: 11px; }}
QLabel#Section {{
    color: {FAINT}; font-family: {MONO_FONT}; font-size: 9.5px; letter-spacing: 1px;
}}
QLabel#Axis {{ color: {FAINT}; font-family: {MONO_FONT}; font-size: 9.5px; letter-spacing: 1px; }}
QLabel#Muted {{ color: {MUTED}; font-size: 11.5px; }}
QLabel#Faint {{ color: {MUTED}; font-size: 11px; }}
QLabel#Value {{ color: {TEXT_STRONG}; font-size: 11.5px; font-family: {MONO_FONT}; }}
QLabel#StatValue {{ font-size: 15px; font-weight: 600; color: {TEXT_STRONG}; font-family: {MONO_FONT}; }}
QLabel#StatName {{ color: {MUTED}; font-size: 10.5px; }}
QLabel#StripName {{ color: {MUTED_2}; font-size: 11px; }}
QLabel#StripValue {{ color: {TEXT_STRONG}; font-size: 10.5px; font-family: {MONO_FONT}; }}
QLabel#Warn {{
    color: {WARN}; background: transparent; border: 1px solid {WARN_BORDER}; border-radius: 4px; padding: 6px 9px;
}}
QLabel#Error {{
    color: {ERROR}; background: transparent; border: 1px solid #4a1a1a; border-radius: 4px; padding: 6px 9px;
}}
QLabel#Hint {{
    color: {MUTED}; background: transparent; border: 1px solid {BORDER}; border-radius: 4px; padding: 6px 9px;
    font-size: 11.5px;
}}

QSlider {{ min-height: 20px; }}
QSlider::groove:horizontal {{ height: 3px; background: {EDGE_OFF}; border-radius: 1px; }}
QSlider::sub-page:horizontal {{ background: {MUTED}; border-radius: 1px; }}
QSlider::handle:horizontal {{
    width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
    background: {TEXT_STRONG}; border: 1px solid {BG};
}}
QSlider::handle:horizontal:hover {{ border: 2px solid {BORDER_HI}; }}
QSlider::sub-page:horizontal:disabled {{ background: {NODE_OFF}; }}

QComboBox {{
    background: {BTN_BG}; border: 1px solid {BORDER_BTN}; border-radius: 4px;
    padding: 4px 9px; min-height: 18px;
}}
QComboBox:hover {{ border-color: {BORDER_HI}; background: {SEL_BG}; }}
QComboBox:focus {{ border-color: {TEXT_STRONG}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url({ASSETS}/chevron.svg); width: 10px; height: 6px; }}
QComboBox QAbstractItemView {{
    background: {BTN_BG}; border: 1px solid {BORDER_HI}; border-radius: 4px; padding: 3px;
    selection-background-color: {SEL_BG}; selection-color: {TEXT_STRONG}; outline: none;
}}

QCheckBox {{ spacing: 9px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid {FAINT}; background: transparent;
}}
QCheckBox::indicator:hover {{ border-color: {TEXT}; }}
QCheckBox::indicator:checked {{
    background: {TEXT}; border-color: {TEXT}; image: url({ASSETS}/check.svg);
}}

QPushButton {{
    background: {BTN_BG}; border: 1px solid {BORDER_BTN}; border-radius: 4px; padding: 5px 12px;
}}
QPushButton:hover {{ border-color: {BORDER_HI}; background: {SEL_BG}; }}
QPushButton:pressed {{ background: {SEL_BG}; border-color: {MUTED}; }}
QPushButton:disabled {{ color: {FAINT}; }}

QFrame#TopBar {{
    background: {STRIP_BG};
    border-bottom: 1px solid {BORDER_HI};
}}
QPushButton#NavBtn {{
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    border-radius: 0;
    padding: 12px 11px 9px 11px;
    font-size: 13.5px;
    font-weight: 500;
    color: {MUTED_2};
}}
QPushButton#NavBtn:hover {{ color: {TEXT_STRONG}; background: {SEL_BG}; }}
QPushButton#NavBtn[active="true"] {{
    color: {TEXT_STRONG};
    font-weight: 600;
    background: {SEL_BG};
    border-bottom: 3px solid {TEXT_STRONG};
}}

QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 7px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER_BTN}; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}

QMenu {{
    background: {BG}; border: 1px solid {BORDER_BTN}; border-radius: 6px; padding: 4px;
}}
QMenu::item {{ padding: 6px 20px 6px 11px; border-radius: 4px; }}
QMenu::item:selected {{ background: {SEL_BG}; color: {TEXT_STRONG}; }}
QMenu::item:disabled {{ color: {MUTED}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QMenu::indicator {{ width: 14px; height: 14px; margin-left: 6px; }}
QMenu::indicator:checked {{ image: url({ASSETS}/check_accent.svg); }}
"""


QSS = _fractional_pt(_QSS)


def apply_theme(app: QApplication) -> None:
    """Fusion style, dark palette (for anything the stylesheet misses) and the stylesheet."""
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):          # Qt 6.8+: dark native title bar on Windows 10/11
        hints.setColorScheme(Qt.ColorScheme.Dark)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in [(QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, RAISED),
                      (QPalette.AlternateBase, SURFACE), (QPalette.Text, TEXT), (QPalette.Button, RAISED),
                      (QPalette.ButtonText, TEXT), (QPalette.Highlight, SEL_BG),
                      (QPalette.HighlightedText, TEXT_STRONG), (QPalette.ToolTipBase, RAISED),
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
