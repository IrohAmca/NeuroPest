"""Locations of the data files. Light module: the GUI process imports it without numba."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CIRCUITS = ROOT / "data" / "circuits"
CACHE = CIRCUITS / "flywire_v783.npz"
TIERS = CIRCUITS / "tiers.json"
