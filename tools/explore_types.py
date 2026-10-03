"""Look up cell types in the FlyWire annotation table. Read-only helper.

Run: uv run python tools/explore_types.py [regex ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
COLS = ["root_id", "flow", "super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type",
        "top_nt", "side", "synonyms"]


def main(patterns):
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 60, "display.max_rows", 500)
    ann = pd.read_csv(RAW / "Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=COLS, low_memory=False)
    if not patterns:
        d = ann[ann.super_class == "descending"]
        print("descending cell types:", d.cell_type.nunique(), "| neurons:", len(d), "| unnamed:", int(d.cell_type.isna().sum()))
        print(d.cell_type.value_counts().to_string())
        return
    for p in patterns:
        hit = ann[ann.cell_type.fillna("").str.contains(p, regex=True) | ann.hemibrain_type.fillna("").str.contains(p, regex=True)
                  | ann.synonyms.fillna("").str.contains(p, regex=True)]
        print(f"\n=== {p}: {len(hit)} neurons")
        g = hit.groupby(["super_class", "cell_class", "cell_type", "hemibrain_type"], dropna=False).size()
        print(g.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
