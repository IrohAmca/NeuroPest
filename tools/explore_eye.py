"""Where are the photoreceptors? Positions, spherical fit per eye, and body axes from the anatomy.

Run: uv run python tools/explore_eye.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from neuropest.paths import RAW_DIR

COLS = ["root_id", "super_class", "cell_class", "cell_type", "side", "pos_x", "pos_y", "pos_z"]


def fit_sphere(p):
    """Least-squares sphere through points p (n, 3): returns center, radius, rms residual."""
    a = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(1)
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    c = x[:3]
    r = np.sqrt(x[3] + (c ** 2).sum())
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(p - c, axis=1) - r) ** 2)))


def main():
    pd.set_option("display.width", 200, "display.max_rows", 100)
    ann = pd.read_csv(RAW_DIR / "Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=COLS, low_memory=False)
    print("cell_class (sensory):")
    print(ann[ann.super_class.isin(["sensory", "sensory_ascending"])].cell_class.value_counts().to_string())
    pr = ann[(ann.cell_class == "visual") & ann.cell_type.isin(["R1-6", "R7", "R8"])]
    print("\nphotoreceptors:", len(pr))
    for scale_name, scale in [("voxels x,y=4nm z=40nm", np.array([4.0, 4.0, 40.0])), ("all axes equal (nm)", np.ones(3))]:
        print(f"\n== units: {scale_name}")
        for side in ("left", "right"):
            for t in ("R1-6", "R7", "R8"):
                s = pr[(pr.side == side) & (pr.cell_type == t)]
                p = s[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * scale / 1000.0       # micrometres
                c, r, rms = fit_sphere(p)
                print(f"{side:5s} {t:5s} n={len(s):5d} centroid {np.round(p.mean(0), 0)} sphere center {np.round(c, 0)} "
                      f"radius {r:6.1f} um  rms {rms:5.1f} um  extent {np.round(np.ptp(p, axis=0), 0)}")
    # body-axis anchors: centroids (voxel coordinates scaled to um) of known anatomy
    sc = np.array([4.0, 4.0, 40.0]) / 1000.0

    def cen(mask):
        d = ann[mask].dropna(subset=["pos_x"])
        return d[["pos_x", "pos_y", "pos_z"]].to_numpy(float).mean(0) * sc, len(d)

    print("\ncentroids (um) of reference groups:")
    for name, mask in [("optic lobe left", (ann.super_class == "optic") & (ann.side == "left")),
                       ("optic lobe right", (ann.super_class == "optic") & (ann.side == "right")),
                       ("olfactory sensory", ann.cell_class == "olfactory"),
                       ("gustatory sensory", ann.cell_class == "gustatory"),
                       ("mechanosensory", ann.cell_class == "mechanosensory"),
                       ("descending", ann.super_class == "descending"),
                       ("ascending", ann.super_class == "ascending"),
                       ("central", ann.super_class == "central"),
                       ("DAN dopaminergic", ann.cell_class == "DAN"),
                       ("Kenyon cells", ann.cell_class == "Kenyon_Cell"),
                       ("CX", ann.cell_class == "CX")]:
        c, n = cen(mask)
        print(f"  {name:20s} n={n:6d} {np.round(c, 0)}")


if __name__ == "__main__":
    main()
