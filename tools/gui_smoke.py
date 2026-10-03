"""Offscreen GUI check: overlay + control + worker process, then switch circuit size.

Run: uv run python tools/gui_smoke.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from neuropest.app import SIZES, Control, Overlay
from neuropest.runner import Runner


def main():
    app = QApplication(sys.argv)
    runner = Runner()
    overlay = Overlay(runner)
    ctrl = Control(overlay, runner)
    log = []

    def snap(tag):
        log.append(f"--- {tag}\n{ctrl.telemetry.text()}\n{ctrl.warn.text()}\nstate={runner.state} "
                   f"fly=({overlay.fly.x:.0f},{overlay.fly.y:.0f}) rect={tuple(int(v) for v in overlay.play_rect())}")

    QTimer.singleShot(8000, lambda: snap("146 neurons"))
    QTimer.singleShot(8100, lambda: ctrl.size.setValue(SIZES.index(50_000)))
    QTimer.singleShot(22000, lambda: snap("50k neurons, dt 0.5"))
    QTimer.singleShot(22100, lambda: ctrl.dt.setCurrentIndex(0))        # dt 0.1 ms: heavier
    QTimer.singleShot(36000, lambda: snap("50k neurons, dt 0.1"))
    QTimer.singleShot(36500, app.quit)
    app.exec()
    runner.stop()
    print("\n".join(log))


if __name__ == "__main__":
    main()
