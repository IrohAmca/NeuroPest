"""Build data/circuits/eye.npz (photoreceptor viewing directions) and a diagnostic picture.

Run: uv run python tools/build_eye.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from neuropest import flywire
from neuropest.paths import CIRCUITS, RAW_DIR
from neuropest.eyebuild import build_retina, build_visual_field
from neuropest.vision import EYE_FILE, Retina


def picture(r: Retina, path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication(sys.argv)
    W, H = 760, 380
    img = QImage(W, H, QImage.Format_RGB32)
    img.fill(QColor(20, 20, 24))
    p = QPainter(img)
    for deg in range(-180, 181, 30):                      # grid
        x = (deg + 180) / 360 * (W - 40) + 20
        p.setPen(QColor(90, 90, 100) if deg % 90 == 0 else QColor(50, 50, 60))
        p.drawLine(int(x), 20, int(x), H - 20)
    for deg in range(-90, 91, 30):
        y = (90 - deg) / 180 * (H - 40) + 20
        p.drawLine(20, int(y), W - 20, int(y))
    colors = {0: QColor(255, 140, 80), 1: QColor(90, 190, 255)}           # left eye orange, right eye blue
    p.setPen(Qt.NoPen)
    for e in (0, 1):
        m = (r.eye == e) & (r.kind == 0)
        pts = np.unique(np.round(np.degrees(np.c_[r.az[m], r.el[m]]), 1), axis=0)     # one dot per column
        p.setBrush(colors[e])
        for a, b in pts:
            p.drawEllipse(int((a + 180) / 360 * (W - 40) + 20) - 2, int((90 - b) / 180 * (H - 40) + 20) - 2, 4, 4)
    p.end()
    img.save(str(path))


def main():
    net = flywire.load_cache()
    r = build_retina(net, RAW_DIR / "Supplemental_file1_neuron_annotations.tsv")
    r.save()
    print(f"{len(r)} photoreceptors -> {EYE_FILE}")
    for e, name in ((0, "left"), (1, "right")):
        for k, kn in enumerate(("R1-6", "R7", "R8", "L1", "L2", "L3")):
            m = (r.eye == e) & (r.kind == k)
            az, el = np.degrees(r.az[m]), np.degrees(r.el[m])
            print(f"  {name:5s} {kn:5s} n={m.sum():5d} az mean {az.mean():7.1f} [{np.percentile(az, 2):7.1f}, "
                  f"{np.percentile(az, 98):7.1f}]  el mean {el.mean():6.1f} [{np.percentile(el, 2):6.1f}, "
                  f"{np.percentile(el, 98):6.1f}]")
    # angular spacing between neighbouring columns (should be a few degrees and fairly uniform)
    for e, name in ((0, "left"), (1, "right")):
        m = (r.eye == e) & (r.kind == 0)
        pts = np.unique(np.c_[r.az[m], r.el[m]], axis=0)
        v = np.c_[np.cos(pts[:, 1]) * np.cos(pts[:, 0]), np.cos(pts[:, 1]) * np.sin(pts[:, 0]), np.sin(pts[:, 1])]
        from scipy.spatial import cKDTree
        d, _ = cKDTree(v).query(v, k=2)
        ang = np.degrees(2 * np.arcsin(np.clip(d[:, 1] / 2, 0, 1)))
        frontal = int((np.abs(np.degrees(pts[:, 0])) < 20).sum())
        print(f"  {name}: {len(pts)} columns, nearest-neighbour angle median {np.median(ang):.1f} deg "
              f"(10-90%: {np.percentile(ang, 10):.1f}-{np.percentile(ang, 90):.1f}); within 20 deg of straight ahead: {frontal}")
    picture(r, CIRCUITS / "eye_map.png")
    print("picture:", CIRCUITS / "eye_map.png")
    fld = build_visual_field(net, RAW_DIR / "Supplemental_file1_neuron_annotations.tsv")
    np.savez(CIRCUITS / "field.npz", **fld)
    print(f"\nvisual field: {len(fld['col_dir'])} columns, {len(fld['rf_row'])} projection neurons with a receptive field")
    types = list(fld["types"])
    print(f"{'type':10s} {'n':>4s} {'conc med':>8s} {'az range':>16s} {'el range':>14s}")
    for t in ("LPLC2", "LC4", "LC6", "LPLC1", "LPC1", "LC10a", "LC10c-2", "LC10d", "LC11", "LC17", "LC15", "LC9"):
        if t not in types:
            continue
        m = fld["rf_type"] == types.index(t)
        d = fld["rf_dir"][m]
        az, el = np.degrees(np.arctan2(d[:, 1], d[:, 0])), np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
        print(f"{t:10s} {m.sum():4d} {np.median(fld['rf_conc'][m]):8.2f} [{np.percentile(az, 5):6.0f},{np.percentile(az, 95):6.0f}] "
              f"[{np.percentile(el, 5):5.0f},{np.percentile(el, 95):5.0f}]")


if __name__ == "__main__":
    main()
