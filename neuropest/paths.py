"""Locations of the data files. Light module: the GUI process imports it without numba."""
import os
import sys
from pathlib import Path

# Base application path resolution (supporting PyInstaller frozen binary)
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ROOT = Path(sys._MEIPASS)
else:
    ROOT = Path(__file__).resolve().parents[1]

# If running as frozen and data files are located beside the executable
if getattr(sys, "frozen", False) and not (ROOT / "data" / "circuits").exists():
    exe_dir = Path(sys.executable).resolve().parent
    if (exe_dir / "data" / "circuits").exists():
        ROOT = exe_dir

RAW_DIR = ROOT / "data" / "raw"
CIRCUITS = ROOT / "data" / "circuits"
CACHE = CIRCUITS / "flywire_v783.npz"
CACHE_SYM = CIRCUITS / "flywire_v783_sym.npz"
TIER_DIR = CIRCUITS / "tiers"       # one small npz per tier size (flywire.load_tier)
TIERS = CIRCUITS / "tiers.json"
EYE = CIRCUITS / "eye.npz"          # photoreceptor viewing directions (tools/build_eye.py)
MUSHROOM = CIRCUITS / "mushroom_flywire.npz"   # real KC/MBON/DAN/PN wiring (tools/build_mushroom.py)
MUSHROOM_3D = CIRCUITS / "mushroom_3d.npz"     # 3D coordinates & anatomy for FlyWire visualization
FIELD = CIRCUITS / "field.npz"      # medulla columns and projection-neuron receptive fields
SKINS_DIR = Path(__file__).resolve().parent / "assets" / "skins"
SKIN_THUMBNAILS = SKINS_DIR / "thumbnails"


def get_user_data_dir() -> Path:
    """Return OS-appropriate persistent user data directory."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "NeuroPest"
        return Path.home() / ".neuropest"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "NeuroPest"
    else:
        base = os.environ.get("XDG_DATA_HOME")
        if base:
            return Path(base) / "neuropest"
        return Path.home() / ".local" / "share" / "neuropest"


# In frozen release builds, persistent user files go to user app data to prevent permission errors
if getattr(sys, "frozen", False):
    USER_DATA_DIR = get_user_data_dir()
    PREFERENCES = USER_DATA_DIR / "preferences.json"
    MEMORY = USER_DATA_DIR / "memory" / "fly_memory.npz"
else:
    USER_DATA_DIR = ROOT / "data"
    PREFERENCES = ROOT / "data" / "preferences.json"
    MEMORY = ROOT / "data" / "memory" / "fly_memory.npz"
