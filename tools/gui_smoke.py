"""Offscreen GUI check: overlay + control + worker process, then switch tier and circuit.

Run: uv run python tools/gui_smoke.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from neuropest.app import FLYWIRE_SIZES, Control, Overlay
from neuropest.runner import Runner


def main():
    app = QApplication(sys.argv)
    runner = Runner()
    overlay = Overlay(runner)
    ctrl = Control(overlay, runner)
    log = []

    def snap(tag):
        log.append(f"--- {tag} (circuit={runner.cfg.circuit}, n={runner.cfg.n})\n{ctrl.size_label.text()}\n"
                   f"{ctrl.tier_info.text()}\n{ctrl.telemetry.text()}\n{ctrl.warn.text()}\n"
                   f"state={runner.state} fly=({overlay.fly.x:.0f},{overlay.fly.y:.0f}) "
                   f"rect={tuple(int(v) for v in overlay.play_rect())}")

    timers = []

    def at(ms, fn):
        t = QTimer(singleShot=True, interval=ms)          # coarse timers can fire out of order
        t.setTimerType(Qt.PreciseTimer)
        t.timeout.connect(fn)
        t.start()
        timers.append(t)

    at(10000, lambda: snap("start"))
    at(10100, lambda: ctrl.size.setValue(FLYWIRE_SIZES.index(20_000)))
    at(18000, lambda: snap("after size change"))
    at(18100, lambda: ctrl.circ.setCurrentIndex(len(ctrl.circuits) - 1))   # toy circuit
    at(26000, lambda: snap("toy circuit"))
    at(26500, app.quit)
    app.exec()
    runner.stop()
    print("\n".join(log))


if __name__ == "__main__":
    main()
