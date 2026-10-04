"""Render the control window and tray menu offscreen to PNGs, with a fake engine (no simulation).

Run: uv run python tools/ui_preview.py [out_dir]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QScrollArea

from neuropest import control
from neuropest.control import Control
from neuropest.runner import EngineConfig
from neuropest.theme import BG, STATE_STYLE, apply_theme, fly_icon
from neuropest.tray import Tray


class FakeRunner:
    def __init__(self, state="walk", rt=7.2):
        self.cfg = EngineConfig("flywire", 15_000, 0.5)
        self.bias, self.skittish = 0.65, 1.0
        self.vision, self.eye_height = False, 100.0
        self.state, self.ready, self.failed, self.alive = state, True, False, True
        self._rt = rt

    def stats(self):
        return dict(ready=True, n=15_000, rt=self._rt, active=3_412, cpu=0.38, lag_ms=12, gf=0, walk=41,
                    rest=0, mdn=3, steer=-7, sim_s=12.0, spikes=48_230, vision=0.0, valence=0.42, v_motor=1.3,
                    gear="walk")

    def start(self, cfg):
        self.cfg = cfg


class FakeOverlay:
    scale = 1.4
    home = None

    def setVisible(self, _):
        pass


def main():
    # pretend the FlyWire cache exists so the real-circuit layout is shown; all numbers here are made up
    control.CACHE = Path(__file__)
    control.load_tiers = lambda: {15_000: dict(n=15_000, gf_err=1.2, mdn_err=1.8, steer_err=0.9, dn_corr=0.991,
                                               realtime=3.1, rt_min=1.6)}
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "ui_preview")
    out.mkdir(exist_ok=True)
    app = QApplication(sys.argv)
    apply_theme(app)
    shots = [("control.png", FakeRunner()), ("control_slow.png", FakeRunner("retreat", rt=0.6))]
    for name, runner in shots:
        ctrl = Control(FakeOverlay(), runner)
        ctrl.show()
        app.processEvents()
        ctrl.resize(760, 560)
        app.processEvents()
        ctrl.grab().save(str(out / name))
        if name == "control.png":
            tab_names = ["tab_live.png", "tab_behaviour.png", "tab_vision.png", "tab_pheromone.png", "tab_circuit.png",
                         "tab_view.png", "tab_learning.png"]
            for i, tname in enumerate(tab_names):
                ctrl._switch_tab(i)
                app.processEvents()
                ctrl.grab().save(str(out / tname))
            ctrl._switch_tab(0)
        tray = Tray(app, ctrl, runner)
        if name == "control.png":
            tray.menu.show()
            app.processEvents()
            tray.menu.grab().save(str(out / "tray_menu.png"))
            tray.menu.hide()
        ctrl.hide()
    sizes = [16, 24, 32, 64]
    pm = QPixmap(sum(sizes) * len(STATE_STYLE) + 200, 90)
    pm.fill(QColor("#2b2b30"))
    p = QPainter(pm)
    x = 10
    for state in STATE_STYLE:
        for s in sizes:
            p.drawPixmap(x, 10, fly_icon(state).pixmap(s, s))
            x += s + 10
        x += 20
    p.end()
    pm.save(str(out / "icons.png"))
    print("saved to", out.resolve(), "bg", BG)


if __name__ == "__main__":
    main()
