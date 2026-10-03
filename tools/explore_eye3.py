"""Eye geometry check 2: global sphere/ellipsoid fits per eye, and where the lamina/medulla column cells sit.

Run: uv run python tools/explore_eye3.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from neuropest.paths import RAW_DIR

COLS = ["root_id", "super_class", "cell_class", "cell_type", "side", "pos_x", "pos_y", "pos_z"]
SC = np.array([4.0, 4.0, 40.0]) / 1000.0


def fit_sphere(p):
    a = np.c_[2 * p, np.ones(len(p))]
    x = np.linalg.lstsq(a, (p ** 2).sum(1), rcond=None)[0]
    c = x[:3]
    r = np.sqrt(x[3] + (c ** 2).sum())
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(p - c, axis=1) - r) ** 2)))


def main():
    pd.set_option("display.width", 200)
    ann = pd.read_csv(RAW_DIR / "Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=COLS, low_memory=False)
    ann = ann.dropna(subset=["pos_x"])
    P = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * SC
    ann = ann.assign(x=P[:, 0], y=P[:, 1], z=P[:, 2])
    print("positions (um) by cell type and side: mean and std [x y z]")
    for ct in ["R1-6", "R7", "R8", "L1", "L2", "L3", "Mi1", "Tm1", "Tm3", "T4a", "T5a", "LPLC2", "LC4"]:
        for side in ("left", "right"):
            s = ann[(ann.cell_type == ct) & (ann.side == side)]
            if len(s) == 0:
                continue
            m, sd = s[["x", "y", "z"]].mean().to_numpy(), s[["x", "y", "z"]].std().to_numpy()
            print(f"{ct:6s} {side:5s} n={len(s):5d}  mean {np.round(m, 0)}  std {np.round(sd, 0)}")
    print("\nsphere fits (um) on photoreceptors and on column cells:")
    for ct in ["R1-6", "R7", "R8", "L1", "Mi1", "Tm1"]:
        for side in ("left", "right"):
            s = ann[(ann.cell_type == ct) & (ann.side == side)]
            p = s[["x", "y", "z"]].to_numpy()
            c, r, rms = fit_sphere(p)
            print(f"{ct:6s} {side:5s} center {np.round(c, 0)} radius {r:6.1f} rms {rms:5.1f}")


if __name__ == "__main__":
    main()
