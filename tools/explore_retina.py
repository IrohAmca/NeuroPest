"""Photoreceptors and optic lobe in the FlyWire annotation: what could a screen image drive?

Run: uv run python tools/explore_retina.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from neuropest.paths import RAW_DIR

COLS = ["root_id", "flow", "super_class", "cell_class", "cell_sub_class", "cell_type", "side",
        "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z"]


def main():
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
    ann = pd.read_csv(RAW_DIR / "Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=COLS, low_memory=False)
    vis = ann[(ann.super_class == "sensory") & (ann.cell_class == "visual")]
    print("sensory/visual neurons:", len(vis))
    print(vis.groupby(["cell_type", "side"]).size().unstack(fill_value=0).to_string())
    print("\nflow of these:", vis.flow.value_counts().to_dict())
    for t in sorted(vis.cell_type.dropna().unique())[:6]:
        s = vis[vis.cell_type == t]
        print(f"{t:8s} n={len(s):5d} pos range x[{s.pos_x.min():.0f},{s.pos_x.max():.0f}] "
              f"y[{s.pos_y.min():.0f},{s.pos_y.max():.0f}] z[{s.pos_z.min():.0f},{s.pos_z.max():.0f}] "
              f"pos NaN {int(s.pos_x.isna().sum())}")
    print("\noptic super_class, top cell_class:")
    print(ann[ann.super_class == "optic"].cell_class.value_counts().head(15).to_string())
    print("\nmedulla columnar types with the most cells (optic):")
    print(ann[ann.super_class == "optic"].cell_type.value_counts().head(25).to_string())


if __name__ == "__main__":
    main()
