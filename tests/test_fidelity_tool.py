import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fidelity  # noqa: E402


def test_families_split_single_group_and_mixed_protocols():
    protocols = [{"LOOM": 5.0}, {"RETREAT_IN": 7.0}, {"LC10_L": 9.0}, {"LOOM": 3.0, "LC10_R": 4.0}]
    fam, mixed = fidelity.families(protocols)
    assert fam["LOOM"] == [0] and fam["RETREAT_IN"] == [1] and fam["LC10_L"] == [2] and fam["TOUCH_L"] == []
    assert mixed == [3]


def test_errors_are_relative_to_the_full_brain_with_a_five_hz_floor():
    protocols = [{"LOOM": 5.0}, {"RETREAT_IN": 7.0}, {"LC10_L": 9.0}, {"LC10_R": 9.0}, {"TOUCH_L": 4.0}]
    fam, mixed = fidelity.families(protocols)
    full = np.full((5, len(fidelity.ANCHORS)), 100.0)
    tier = full.copy()
    tier[0, 0] = 50.0                       # GF half of the full brain's on the LOOM protocol
    tier[1, 1] = 2.0                        # MDN 98 % low
    err = fidelity.errors(tier, full, fam, mixed)
    assert err["gf_err"] == 50.0 and abs(err["mdn_err"] - 98.0) < 1e-9 and err["steer_err"] == 0.0
    assert abs(fidelity.rel_err(np.array([3.0]), np.array([0.0]))[0] - 0.6) < 1e-9   # |3 - 0| / max(0, 5)


def test_write_csv_keeps_every_column(tmp_path):
    out = tmp_path / "sub" / "t.csv"
    fidelity.write_csv(out, [dict(a=1, b=0.12345), dict(a=2, c="x")])
    rows = list(csv.DictReader(open(out, encoding="utf8")))
    assert rows[0]["b"] == "0.123" and rows[1]["c"] == "x" and set(rows[0]) == {"a", "b", "c"}
