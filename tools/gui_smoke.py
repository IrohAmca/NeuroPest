"""Offscreen GUI check: overlay + control + worker process; switch tier, circuit and CPU/GPU.

Run: uv run --extra gpu python tools/gui_smoke.py
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
        log.append(f"--- {tag} (circuit={runner.cfg.circuit}, n={runner.cfg.n}, backend={runner.cfg.backend}, "
                   f"adapter={runner.cfg.adapter}; hardware choices: {[ctrl.hw.itemText(i) for i in range(ctrl.hw.count())]})\n"
                   f"{ctrl.size_label.text()}\n{ctrl.tier_info.text()}\n{ctrl.telemetry.text()}\n{ctrl.warn.text()}\n"
                   f"state={runner.state} fly=({overlay.fly.x:.0f},{overlay.fly.y:.0f})")

    timers = []

    def at(ms, fn):
        t = QTimer(singleShot=True, interval=ms)          # coarse timers can fire out of order
        t.setTimerType(Qt.PreciseTimer)
        t.timeout.connect(fn)
        t.start()
        timers.append(t)

    at(10000, lambda: snap("start"))
    at(10100, lambda: ctrl.size.setValue(len(FLYWIRE_SIZES) - 1))        # full brain: automatic picks the GPU
    at(24000, lambda: snap("full brain, automatic hardware"))
    at(24100, lambda: ctrl.hw.setCurrentIndex(1))                         # force CPU
    at(36000, lambda: snap("full brain on the CPU"))
    at(36500, app.quit)
    app.exec()
    runner.stop()
    print("\n".join(log))


if __name__ == "__main__":
    main()
