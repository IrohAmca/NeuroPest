"""Tray icon and menu: the icon's eyes follow the fly's behaviour state."""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .theme import STATE_STYLE, fly_icon


class Tray(QSystemTrayIcon):
    def __init__(self, app: QApplication, ctrl, runner):
        super().__init__(fly_icon(), app)
        self.ctrl, self.runner = ctrl, runner
        self._icons = {s: fly_icon(s) for s in STATE_STYLE}
        self._shown = None

        menu = QMenu()
        self.status = QAction("NeuroPest", menu, enabled=False)
        menu.addAction(self.status)
        menu.addSeparator()
        self.show_ctrl = QAction("Open Control Panel", menu, triggered=self._open_control)
        menu.addAction(self.show_ctrl)
        self.show_3d = QAction("View 3D Neural Circuit…", menu, triggered=self._open_3d)
        menu.addAction(self.show_3d)
        menu.addSeparator()
        menu.addAction(QAction("Exit", menu, triggered=app.quit))
        self.menu = menu                    # QSystemTrayIcon does not own the menu
        self.setContextMenu(menu)
        self.activated.connect(self._clicked)

        self._t = QTimer(self, timeout=self._refresh, interval=250)
        self._t.start()
        self._refresh()

    def _open_control(self):
        self.ctrl.showNormal()
        self.ctrl.raise_()
        self.ctrl.activateWindow()

    def _open_3d(self):
        self.ctrl.showNormal()
        self.ctrl.raise_()
        self.ctrl.activateWindow()
        if hasattr(self.ctrl, "_open_3d_popout"):
            self.ctrl._open_3d_popout()

    def _clicked(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):   # left click opens the window
            self._open_control()

    def _refresh(self):
        r = self.runner
        state = r.state if r.ready and r.alive else None
        hunger_str = ""
        overlay = getattr(self.ctrl, "overlay", None)
        if overlay and hasattr(overlay, "metabolism") and overlay.metabolism.enabled:
            hunger_str = f" · Hunger: {overlay.metabolism.hunger_pct}%"
        key = (state, hunger_str)
        if key == self._shown:
            return
        self._shown = key
        name = STATE_STYLE[state][1] if state in STATE_STYLE else ("Starting" if r.alive else "Stopped")
        self.setIcon(self._icons.get(state, self._icons["stand"]))
        self.setToolTip(f"NeuroPest · {name}{hunger_str}")
        self.status.setText(f"Fly: {name}{hunger_str}")
