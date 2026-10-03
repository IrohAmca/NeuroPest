"""How much does grabbing and shrinking the screen cost? Only timings are printed, the image is not kept.

Run: uv run python tools/capture_cost.py
"""
import sys
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    screen = QApplication.primaryScreen()
    geo = screen.geometry()
    print(f"screen {geo.width()}x{geo.height()} (device pixel ratio {screen.devicePixelRatio()})")
    for region, target_w in (((0, 0, geo.width(), geo.height()), 640), ((200, 200, 800, 800), 200),
                             ((200, 200, 600, 600), 150), ((200, 200, 400, 400), 100)):
        t = []
        x, y, w, h = region
        for _ in range(30):
            t0 = time.perf_counter()
            pix = screen.grabWindow(0, x, y, w, h)
            t1 = time.perf_counter()
            img = pix.toImage().scaled(target_w, int(target_w * h / w), Qt.IgnoreAspectRatio,
                                       Qt.FastTransformation).convertToFormat(QImage.Format_Grayscale8)
            arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width()]
            t2 = time.perf_counter()
            t.append((t1 - t0, t2 - t1, float(arr.mean())))
        t = np.array(t)
        print(f"region {w}x{h}: grabWindow {1000 * t[:, 0].mean():5.1f} ms | shrink to {target_w}px gray "
              f"{1000 * t[:, 1].mean():5.1f} ms | total {1000 * (t[:, 0] + t[:, 1]).mean():5.1f} ms  (frame {arr.shape})")


if __name__ == "__main__":
    main()
