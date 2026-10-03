"""Print what is in the downloaded FlyWire v783 files (data/raw). Read-only, memory-frugal
(this machine has little free commit memory, so only the needed columns are loaded, as int32).

Run: uv run python tools/inspect_flywire.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    pd.set_option("display.width", 220, "display.max_columns", 60, "display.max_colwidth", 40)
    pf = pq.ParquetFile(RAW / "Connectivity_783.parquet")
    print("connectivity rows:", pf.metadata.num_rows, "row groups:", pf.num_row_groups)
    print(pf.schema_arrow)
    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory", "Excitatory x Connectivity"]
    t = pq.read_table(RAW / "Connectivity_783.parquet", columns=cols)
    a = {c: t.column(c).to_numpy().astype(np.int32) for c in cols}
    del t
    pre, post, cnt, exc, sgn = (a[c] for c in cols)
    n = int(max(pre.max(), post.max())) + 1
    print("neurons (max index + 1):", n, "| pre sorted:", bool(np.all(np.diff(pre) >= 0)))
    print("Excitatory values:", dict(zip(*np.unique(exc, return_counts=True))))
    print("signed == exc*cnt:", bool(np.all(sgn == exc * cnt)), "| Excitatory==-1 means inhibitory:",
          int((exc < 0).sum()), "edges")
    print("synapse count percentiles (1,5,25,50,75,95,99,max):", np.percentile(cnt, [1, 5, 25, 50, 75, 95, 99, 100]))
    print("edges with count>=5:", int((cnt >= 5).sum()), "| >=3:", int((cnt >= 3).sum()))
    print("out-degree max/mean:", np.bincount(pre, minlength=n).max(), pre.size / n,
          "| in-degree max:", np.bincount(post, minlength=n).max())
    print("self connections:", int((pre == post).sum()))

    comp = pd.read_csv(RAW / "Completeness_783.csv", index_col=0)
    print("\ncompleteness:", comp.shape, "index name:", comp.index.name, comp.index.dtype)
    print(comp.head(3).to_string())
    print(comp.iloc[:, 0].value_counts().head().to_string())

    ann = pd.read_csv(RAW / "Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False)
    print("\nannotations:", ann.shape)
    print(ann.dtypes.to_string())
    for c in ("flow", "super_class", "cell_class", "side", "top_nt", "status"):
        if c in ann:
            print(f"\n{c}:\n{ann[c].value_counts(dropna=False).head(25).to_string()}")
    print("\nannotation root_ids found in completeness index:", float(ann["root_id"].isin(comp.index).mean()))
    print("completeness ids found in annotations:", float(comp.index.isin(ann["root_id"]).mean()))


if __name__ == "__main__":
    main()
