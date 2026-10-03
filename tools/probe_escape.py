"""Does the model carry the escape DIRECTION in descending neurons? DNp02 / DNp11 (one neuron per side each) and the Giant
Fiber (DNp01) in the full brain, for an 800 px/s approach of the cursor from different bearings.

Dombrovski et al. 2023: LC4 neurons with receptive fields on one side of the eye drive DNp02 / DNp11 on a gradient, which
is how the fly picks the take-off direction; the Giant Fiber (DNp01) takes the looming signal of both sides. The app
now picks the escape heading as `cursor bearing + pi` (fly.py). If DNp02 / DNp11 carry the bearing in the model, the
heading could come from them instead. This prints their spike counts per bearing (the real VisionDrive, sphere cursor,
full brain) and the correlation of (R - L) with the bearing, and writes data/probes/escape_direction.csv.

Run: uv run python tools/probe_escape.py [--tier N]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from neuropest import flywire, paths
from neuropest.engine import LIFEngine
from neuropest.vision import VisualField
from neuropest.visual import VisionDrive, VisionParams

FRAME_S = 0.02
EVENT_S = 0.5
BEARINGS = (-120, -90, -60, -30, 0, 30, 60, 90, 120)       # degrees, positive to the right of the heading


def groups(net):
    meta = flywire.load_annotations(net.ids)
    ct, side = meta["cell_type"].fillna(""), meta["side"].fillna("")
    pick = lambda t, s: np.flatnonzero(((ct == t) & (side == s)).to_numpy()).astype(np.int32)     # noqa: E731
    return {f"{t}_{s[0].upper()}": pick(t, s) for t in ("DNp02", "DNp11", "DNp01") for s in ("left", "right")}


def approach(deg, speed=800.0, d0=400.0, stop=50.0):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return lambda t: (c * max(stop, d0 - speed * max(0.0, t - EVENT_S)), s * max(stop, d0 - speed * max(0.0, t - EVENT_S)))


def run(net, g, field, deg, seconds=1.5, seed=21):
    drive = VisionDrive(net, VisionParams(), field)
    path = approach(deg)
    e = LIFEngine(net, dt=0.5, seed=seed)
    counts = {k: 0 for k in g}
    for f in range(int(seconds / FRAME_S)):
        t = f * FRAME_S
        idx, rates = drive.step_cursor(path(t), path(t - FRAME_S), 0.0, 0.0, 0.0, FRAME_S)
        e.set_drive(idx, rates)
        e.advance(FRAME_S * 1000.0)
        for k, v in g.items():
            counts[k] += int(e.pop_counts(v).sum())
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0, help="0: the full brain")
    a = ap.parse_args()
    full = flywire.load_cache()
    net = full.prefix(a.tier) if a.tier else full
    g = groups(net)
    missing = [k for k, v in g.items() if len(v) != 1]
    assert not missing, f"expected one neuron per group, got {missing}"
    field = VisualField.load()
    print(f"{'bearing':>8} | " + " ".join(f"{k:>9}" for k in g) + " | DNp02 R-L  DNp11 R-L")
    rows = []
    for deg in BEARINGS:
        c = run(net, g, field, deg)
        rows.append(dict(bearing_deg=deg, **c))
        print(f"{deg:>8} | " + " ".join(f"{c[k]:>9}" for k in g) + f" | {c['DNp02_R'] - c['DNp02_L']:>9} {c['DNp11_R'] - c['DNp11_L']:>10}", flush=True)
    b = np.array([r["bearing_deg"] for r in rows], float)
    for t in ("DNp02", "DNp11", "DNp01"):
        d = np.array([r[f"{t}_R"] - r[f"{t}_L"] for r in rows], float)
        print(f"{t}: corr(R - L, bearing) = {np.corrcoef(b, d)[0, 1] if d.std() > 0 else float('nan'):+.2f}")
    out = paths.ROOT / "data" / "probes" / "escape_direction.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
