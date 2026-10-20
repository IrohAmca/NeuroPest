"""Kymograph strips and the Circuit tab (design_handoff)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from neuropest.control import Control
from neuropest.runner import EngineConfig
from neuropest.strips import (
    CELLS, SAMPLES, BEHAVIOUR, DESCENDING, EDGES_DB, EDGES_SD, SENSORY, CircuitView, EthoStrip, History, KymoStrip, mix,
)
from neuropest.theme import STATE_STYLE, STRIP_BG, apply_theme


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    apply_theme(a)
    return a


def test_cells_average_samples_in_pairs_oldest_left_and_missing_history_reads_zero():
    h = History()
    for i in range(SAMPLES):
        h.push({"gf": 0.0 if i < SAMPLES - 2 else 200.0}, "walk")
    cells = h.cells("gf")
    assert len(cells) == CELLS and cells[:-1] == [0.0] * (CELLS - 1) and cells[-1] == 1.0
    short = History()
    short.push({"mdn": 100.0}, "stand")
    short.push({"mdn": 100.0}, "stand")
    c = short.cells("mdn")
    assert c[-1] == 1.0 and c[0] == 0.0                      # two samples fill the newest cell only


def test_cells_are_clamped_and_follow_the_max_rate_table():
    h = History()
    for _ in range(4):
        h.push({"dnaL": 80.0, "gf": 100.0}, None)
    assert h.cells("dnaL")[-1] == 1.0                        # 80 Hz over a 40 Hz max
    assert h.cells("gf")[-1] == 0.5                          # 100 Hz of a 200 Hz max


def test_occupancy_is_the_share_of_each_pair_spent_in_the_state():
    h = History()
    for s in ("walk", "walk", "walk", "fly"):
        h.push({"gf": 0.0}, s)
    occ_walk, occ_fly = h.occupancy("walk"), h.occupancy("fly")
    assert occ_walk[-2] == 1.0 and occ_walk[-1] == 0.5
    assert occ_fly[-1] == 0.5 and occ_fly[-2] == 0.0


def test_mix_blends_the_colour_over_the_ground_and_clamps():
    assert mix("#ffffff", "#000000", 0.5).red() == 128
    assert mix("#ffffff", "#000000", 7).name() == "#ffffff" and mix("#ffffff", "#000000", -1).name() == "#000000"


def _pixel(img: QImage, x: int, y: int) -> str:
    return QColor(img.pixel(x, y)).name()


def test_a_strip_cell_is_painted_in_the_mixed_colour(app):
    s = KymoStrip("#3ddc97", 12)
    s.resize(200, 12)
    s.set_data([0.0] * (CELLS - 1) + [1.0])
    img = s.grab().toImage()
    assert _pixel(img, 198 - 2, 6) == "#3ddc97"                 # the newest cell, full colour
    assert _pixel(img, 4, 6) == STRIP_BG                        # the oldest, the ground


def test_the_ethogram_paints_the_state_colour_of_each_cell(app):
    h = History()
    for s in ("walk",) * 4 + ("fly",) * 4:
        h.push({"gf": 0.0}, s)
    e = EthoStrip()
    e.resize(200, 12)
    e.set_history(h)
    assert e.cells[-1] == STATE_STYLE["fly"][0] and e.cells[-3] == STATE_STYLE["walk"][0] and e.cells[0] == STRIP_BG


def test_the_circuit_has_the_designed_nodes_and_edges_and_every_edge_points_at_a_node():
    assert (len(SENSORY), len(DESCENDING), len(BEHAVIOUR)) == (8, 7, 6)
    assert all(a < len(SENSORY) and b < len(DESCENDING) for a, b in EDGES_SD)
    assert all(a < len(DESCENDING) and b < len(BEHAVIOUR) for a, b in EDGES_DB)
    assert {lane for *_, lane in SENSORY} == set(range(6))


def test_the_circuit_paints_active_edges_in_the_behaviour_colour_and_idle_ones_dark(app):
    v = CircuitView()
    h = History()
    for _ in range(SAMPLES):
        h.push({"gf": 150.0, "lc4": 100.0, "dnaL": 0.0, "dnaR": 0.0}, "fly")
    v.update_view(h, {"gf": 150.0, "lc4": 100.0}, "fly")
    img = v.grab().toImage()
    colours = {_pixel(img, x, y) for x in range(182, 268, 2) for y in range(40, 150, 2)}
    assert any(c.startswith("#f") and c != "#ffffff" and QColor(c).red() > 120 for c in colours)   # the escape red
    v.update_view(h, {}, "stand")
    img = v.grab().toImage()
    assert not any(QColor(_pixel(img, x, y)).red() > 150 for x in range(182, 268, 2) for y in range(40, 150, 2))


def _runner(state="walk"):
    class R:
        cfg = EngineConfig("toy", 146, 0.5)
        bias, skittish, vision, eye_height = 0.65, 1.0, False, 100.0
        ready, failed, alive = True, False, True

        def __init__(self):
            self.state = state

        def stats(self):
            return dict(ready=True, n=146, rt=5.0, active=40, cpu=0.1, lag_ms=5, gf=10.0, walk=30, rest=0, mdn=2,
                        steer=4, sim_s=1.0, spikes=900, vision=0.0, mn9=20.0, groom=5.0, pam=0.5, ppl1=0.2,
                        sugar_hz=60.0, touch_hz=0.0)

        def lab_rates(self):
            return {"LC4": 50.0, "DNa02_L": 20.0, "DNa02_R": 5.0}

        def start(self, cfg):
            pass

    return R()


class _Overlay:
    scale, home, touch_groom_enabled = 1.4, None, False

    def setVisible(self, v):
        pass


def test_the_control_window_keeps_a_history_and_feeds_the_strips_of_the_page_on_show(app):
    c = Control(_Overlay(), _runner())
    n0 = len(c.hist.states)
    for _ in range(6):
        c._refresh()
    assert len(c.hist.states) == n0 + 6
    assert c.hist.last("dnaL") == 20.0 and c.hist.last("lc4") == 50.0 and c.hist.last("sugar") == 60.0
    c._switch_tab(1)
    c._refresh()
    assert c.etho.cells[-1] == STATE_STYLE["walk"][0] and c.hunger_strip.value.text()
    c._switch_tab(2)
    c._refresh()
    assert c.vis_strips["lc4"].value.text() == "50 Hz"
    c._switch_tab(6)
    c._refresh()
    assert c.da_strips["pam"].value.text() == "0.50"


def test_the_navigation_is_a_top_bar_of_short_tab_names_and_the_state_on_the_right(app):
    c = Control(_Overlay(), _runner("feed"))
    assert [b.text() for b in c.nav_buttons] == ["Circuit", "Behaviour", "Senses", "Odor", "Hardware", "Look", "Memory"]
    c._refresh()
    assert "Feeding" in c.pill.text()
    c._switch_tab(3)
    assert c.stack.currentIndex() == 3 and c.nav_buttons[3].property("active") == "true"


def test_a_short_page_is_not_as_tall_as_the_tallest_page(app):
    c = Control(_Overlay(), _runner())
    assert c.stack.sizeHint().height() == c.stack.currentWidget().sizeHint().height()
    c._switch_tab(4)
    assert c.stack.sizeHint().height() == c.stack.currentWidget().sizeHint().height()


def test_the_circuit_says_so_while_the_engine_starts_or_has_stopped(app):
    r = _runner()
    c = Control(_Overlay(), r)
    r.ready = False
    r.stats = lambda: dict(ready=False)
    c._refresh()
    assert c.circuit.message
    r.failed = True
    c._refresh()
    assert "stopped" in c.circuit.message
