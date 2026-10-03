"""Locations of the data files. Light module: the GUI process imports it without numba."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CIRCUITS = ROOT / "data" / "circuits"
CACHE = CIRCUITS / "flywire_v783.npz"
TIER_DIR = CIRCUITS / "tiers"       # one small npz per tier size (flywire.load_tier)
TIERS = CIRCUITS / "tiers.json"
EYE = CIRCUITS / "eye.npz"          # photoreceptor viewing directions (tools/build_eye.py)
FIELD = CIRCUITS / "field.npz"      # medulla columns and projection-neuron receptive fields
