"""Kymograph strips: the theme's one recurring visual element (design_handoff).

A strip shows the last 22 s of one neuron group as a row of 44 cells, oldest on the left, each cell a mix of the
behaviour colour and the strip ground by rate / max. `History` keeps the samples the control window takes (4 Hz),
`KymoStrip` / `StripRow` draw them, and `CircuitView` is the Circuit tab: 8 sensory -> 7 descending -> 6 behaviour nodes
joined by the real FlyWire connections, every node a strip on one shared time axis.
"""
from __future__ import annotations

from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .states import FEED, FLY, GROOM, RETREAT, WALK
from .theme import (
    BORDER, EDGE_OFF, FAINT, MONO_FONT, MUTED, MUTED_2, STATE_STYLE, STRIP_BG, TEXT, TEXT_STRONG, pt,
)

CELLS = 44                      # cells in a strip
SAMPLES = 2 * CELLS             # samples averaged in pairs into the cells (88 at ~4 Hz = 22 s)
AXIS = "−22 s → now"
SHORT_NAME = {"stand": "Stand", "walk": "Walk", "retreat": "Retreat", "fly": "Escape", "groom": "Groom", "freeze": "Freeze", "feed": "Feed"}

# the rate that fills a strip's cells completely (design_handoff, "Max rates used for normalisation")
MAX_HZ = {"gf": 200.0, "mdn": 100.0, "dnaL": 40.0, "dnaR": 40.0, "dnp09": 60.0, "mn9": 70.0, "adn1": 80.0, "adn2": 80.0,
          "lc4": 150.0, "lplc2": 150.0, "lpc1": 80.0, "lc10L": 40.0, "lc10R": 40.0, "p9": 50.0, "sugar": 110.0,
          "bristle": 70.0, "pam": 1.0, "ppl1": 1.0, "hunger": 1.0}

# Circuit tab: (name, key) per node and the lane (behaviour pathway) each belongs to. Lanes: escape, retreat, steer, walk,
# feed, groom. Edges are the connections the circuit carries (sensory index -> descending index, descending -> behaviour).
LANES = (("Escape", FLY), ("Retreat", RETREAT), ("Steer", WALK), ("Walk", WALK), ("Feed", FEED), ("Groom", GROOM))
SENSORY = (("LC4", "lc4", 0), ("LPLC2", "lplc2", 0), ("LPC1", "lpc1", 1), ("LC10 L", "lc10L", 2), ("LC10 R", "lc10R", 2),
           ("Tonic drive", "p9", 3), ("Sugar LB3c", "sugar", 4), ("Bristles · JO", "bristle", 5))
DESCENDING = (("Giant Fiber", "gf", 0, 60), ("MDN", "mdn", 1, 128), ("DNa02 L", "dnaL", 2, 186), ("DNa02 R", "dnaR", 2, 236),
              ("DNp09", "dnp09", 3, 298), ("MN9", "mn9", 4, 350), ("aDN1", "adn1", 5, 400))
BEHAVIOUR = (("Escape", 0, 60), ("Retreat", 1, 128), ("Steer", 2, 211), ("Walk", 3, 298), ("Feed", 4, 350), ("Groom", 5, 400))
EDGES_SD = ((0, 0), (1, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6))
EDGES_DB = ((0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (5, 4), (6, 5))
CANVAS_W, CANVAS_H = 712, 450
NODE_H = 34.0                   # name row 14 + gap 4 + strip 16
STRIP_H = 16                    # 12 px of cells, 1 px padding, 1 px border


def mix(c: str | QColor, ground: str | QColor, t: float) -> QColor:
    """`t` of colour `c` over `ground` (t clamped to 0..1)."""
    t = max(0.0, min(1.0, t))
    a, b = QColor(c), QColor(ground)
    return QColor(round(b.red() + (a.red() - b.red()) * t), round(b.green() + (a.green() - b.green()) * t),
                  round(b.blue() + (a.blue() - b.blue()) * t))


class History:
    """The last SAMPLES values of each key, plus the behaviour state per sample."""

    def __init__(self):
        self.series: dict[str, deque] = {}
        self.states: deque = deque(maxlen=SAMPLES)

    def push(self, values: dict[str, float], state: str | None) -> None:
        for k, v in values.items():
            self.series.setdefault(k, deque(maxlen=SAMPLES)).append(float(v))
        self.states.append(state)

    @staticmethod
    def _pairs(a: list[float]) -> list[float]:
        """44 cells from the newest 88 samples: pairs averaged, oldest left; missing history reads 0."""
        out = []
        n = len(a)
        for j in range(CELLS):
            i = n - 2 * (CELLS - j)
            out.append(0.0 if i < 0 else (a[i] + (a[i + 1] if i + 1 < n else a[i])) / 2.0)
        return out

    def cells(self, key: str, vmax: float | None = None) -> list[float]:
        """Cell intensities 0..1 of `key`'s rate."""
        vmax = vmax or MAX_HZ.get(key, 1.0)
        return [min(1.0, v / vmax) for v in self._pairs(list(self.series.get(key, ())))]

    def occupancy(self, state: str) -> list[float]:
        """Cell intensities 0..1: the share of each pair of samples the fly spent in `state`."""
        return self._pairs([1.0 if s == state else 0.0 for s in self.states])

    def last(self, key: str) -> float:
        a = self.series.get(key)
        return a[-1] if a else 0.0


def paint_strip(p: QPainter, x: float, y: float, w: float, h: float, cells: list[float], colour: str,
                border: str = BORDER) -> None:
    """One strip at (x, y): 1 px border, 3 px radius, 1 px padding, 1 px gap between the cells."""
    p.setPen(QPen(QColor(border), 1))
    p.setBrush(QColor(STRIP_BG))
    p.drawRoundedRect(QRectF(x + 0.5, y + 0.5, w - 1, h - 1), 3, 3)
    p.setPen(Qt.NoPen)
    n = max(1, len(cells))
    inner_w, inner_h = w - 4.0, h - 4.0
    cw = (inner_w - (n - 1)) / n
    for j, v in enumerate(cells):
        p.setBrush(mix(colour, STRIP_BG, v))
        p.drawRect(QRectF(x + 2 + j * (cw + 1), y + 2, max(0.5, cw), inner_h))


class KymoStrip(QWidget):
    """A strip as a widget (10-12 px tall, as wide as it is given)."""

    def __init__(self, colour: str = TEXT, height: int = STRIP_H):
        super().__init__()
        self.colour, self.border, self.cells = colour, BORDER, [0.0] * CELLS
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, cells: list[float], colour: str | None = None, active: bool = False) -> None:
        colour = colour or self.colour
        border = colour if active else BORDER
        if cells != self.cells or colour != self.colour or border != self.border:
            self.cells, self.colour, self.border = list(cells), colour, border
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        paint_strip(p, 0, 0, self.width(), self.height(), self.cells, self.colour, self.border)


class StripRow(QWidget):
    """Name on the left, rate right-aligned in mono, the strip under them."""

    def __init__(self, name: str, colour: str = TEXT, unit: str = "Hz"):
        super().__init__()
        self.unit = unit
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.name = QLabel(name)
        self.name.setObjectName("StripName")
        self.value = QLabel("")
        self.value.setObjectName("StripValue")
        head.addWidget(self.name)
        head.addStretch(1)
        head.addWidget(self.value)
        lay.addLayout(head)
        self.strip = KymoStrip(colour)
        lay.addWidget(self.strip)

    def set(self, cells: list[float], rate: float | None = None, colour: str | None = None, active: bool = False,
            text: str | None = None) -> None:
        self.strip.set_data(cells, colour, active)
        if text is None and rate is not None:
            text = f"{rate:.0f} {self.unit}" if self.unit else f"{rate:.2f}"
        self.value.setText(text or "")
        self.name.setStyleSheet(f"color: {TEXT_STRONG};" if active else "")


class CircuitView(QWidget):
    """The Circuit tab's kymograph network (712 x 440 canvas)."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(CANVAS_W, CANVAS_H)
        self.hist = History()
        self.rates: dict[str, float] = {}
        self.state: str | None = None
        self.tick = 0
        self.message = ""

    def set_message(self, text: str) -> None:
        if text != self.message:
            self.message = text
            self.update()

    def update_view(self, hist: History, rates: dict[str, float], state: str | None) -> None:
        self.hist, self.rates, self.state = hist, rates, state
        self.tick += 1
        self.message = ""
        self.update()

    # -------------------------------------------------------------- painting
    def _lane_on(self, i: int) -> bool:
        return self.state is not None and self.state == LANES[i][1]

    def _lane_colour(self, i: int) -> str:
        return STATE_STYLE[LANES[i][1]][0]

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        mono = QFont([f.strip().strip('"') for f in MONO_FONT.split(",")])
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSizeF(pt(10))
        small = QFont(self.font())
        small.setPointSizeF(pt(11.5))
        small.setWeight(QFont.DemiBold)
        head = QFont(mono)
        head.setPointSizeF(pt(9.5))
        head.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
        hist, rates = self.hist, self.rates

        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, CANVAS_W - 1, CANVAS_H - 1), 6, 6)
        for x, w, title in ((14, 162, "SENSORY"), (272, 170, "DESCENDING"), (540, 156, "BEHAVIOUR")):
            p.setFont(head)
            p.setPen(QColor(FAINT))
            p.drawText(QRectF(x, 12, w, 14), Qt.AlignLeft | Qt.AlignVCenter, title)
            p.drawText(QRectF(x, 12, w, 14), Qt.AlignRight | Qt.AlignVCenter, AXIS)

        sens = [(14.0, 40.0 + i * 49.0, 162.0) for i in range(len(SENSORY))]
        desc = [(272.0, float(d[3]), 170.0) for d in DESCENDING]
        beh = [(540.0, float(b[2]), 156.0) for b in BEHAVIOUR]

        def cy(n):
            return n[1] + 26.0

        # edges first, so the cards sit on them
        def edge(a, b, key, lane, k):
            on = self._lane_on(lane)
            c = QColor(self._lane_colour(lane) if on else EDGE_OFF)
            norm = min(1.0, rates.get(key, 0.0) / MAX_HZ.get(key, 1.0))
            x1, y1, x2, y2 = a[0] + a[2], cy(a), b[0], cy(b)
            m = (x1 + x2) / 2
            path = QPainterPath(QPointF(x1, y1))
            path.cubicTo(QPointF(m, y1), QPointF(m, y2), QPointF(x2, y2))
            pen = QPen(c, 1.0 + 3.5 * norm)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            if on:
                t = ((self.tick * 9 + k * 23) % 100) / 100.0
                p.setPen(Qt.NoPen)
                p.setBrush(c)
                pt = path.pointAtPercent(t)
                p.drawEllipse(pt, 3, 3)

        for k, (si, di) in enumerate(EDGES_SD):
            edge(sens[si], desc[di], SENSORY[si][1], DESCENDING[di][2], k)
        for k, (di, bi) in enumerate(EDGES_DB):
            edge(desc[di], beh[bi], DESCENDING[di][1], BEHAVIOUR[bi][1], k + 3)

        def card(n, name, hz_text, lane, cells):
            x, y, w = n
            on = self._lane_on(lane)
            col = self._lane_colour(lane)
            p.setFont(small)
            p.setPen(QColor(TEXT_STRONG if on else MUTED))
            p.drawText(QRectF(x, y, w - 56, 14), Qt.AlignLeft | Qt.AlignVCenter, name)
            p.setFont(mono)
            p.setPen(QColor(TEXT_STRONG if on else MUTED))
            p.drawText(QRectF(x + w - 56, y, 56, 14), Qt.AlignRight | Qt.AlignVCenter, hz_text)
            paint_strip(p, x, y + 18, w, STRIP_H, cells, col, col if on else BORDER)

        for n, (name, key, lane) in zip(sens, SENSORY):
            card(n, name, f"{rates.get(key, 0.0):.0f} Hz", lane, hist.cells(key))
        for n, (name, key, lane, _y) in zip(desc, DESCENDING):
            card(n, name, f"{rates.get(key, 0.0):.0f} Hz", lane, hist.cells(key))
        dl, dr = hist.cells("dnaL"), hist.cells("dnaR")
        walk_occ = hist.occupancy(WALK)
        for n, (name, lane, _y) in zip(beh, BEHAVIOUR):
            key = LANES[lane][1]
            if name == "Steer":              # walking, in proportion to how far the two DNa02 differ
                cells = [o * min(1.0, abs(r - l) * 40.0 / 14.0) for o, l, r in zip(walk_occ, dl, dr)]
            else:
                cells = hist.occupancy(key)
            card(n, name, "now" if self._lane_on(lane) else "", lane, cells)

        if self.message:
            p.setFont(small)
            p.setPen(QColor(MUTED))
            p.fillRect(QRectF(0, 0, CANVAS_W, CANVAS_H), QColor(5, 7, 11, 200))
            p.drawText(QRectF(0, 0, CANVAS_W, CANVAS_H), Qt.AlignCenter | Qt.TextWordWrap, self.message)


class EthoStrip(QWidget):
    """The ethogram: one cell per pair of samples, in the colour of the behaviour state the fly was in (the cell ground
    where there is no history yet)."""

    def __init__(self, height: int = STRIP_H):
        super().__init__()
        self.cells: list[str] = [STRIP_BG] * CELLS
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_history(self, hist: History) -> None:
        st = list(hist.states)
        n = len(st)
        cells = []
        for j in range(CELLS):
            i = n - 2 * (CELLS - j) + 1               # the newer sample of the pair
            s = st[i] if 0 <= i < n else None
            cells.append(STATE_STYLE[s][0] if s in STATE_STYLE else STRIP_BG)
        if cells != self.cells:
            self.cells = cells
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(QColor(STRIP_BG))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 3, 3)
        p.setPen(Qt.NoPen)
        cw = (w - 4.0 - (CELLS - 1)) / CELLS
        for j, c in enumerate(self.cells):
            p.setBrush(QColor(c))
            p.drawRect(QRectF(2 + j * (cw + 1), 2, max(0.5, cw), h - 4))


def legend(states=None) -> QWidget:
    """Colour key of the behaviour states, one small square and a name each."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(10)
    for s, (c, name) in STATE_STYLE.items():
        if states is not None and s not in states:
            continue
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background: {c}; border-radius: 2px;")
        lab = QLabel(SHORT_NAME.get(s, name))
        lab.setObjectName("Faint")
        lay.addWidget(dot)
        lay.addWidget(lab)
        lay.addSpacing(2)
    lay.addStretch(1)
    return w
