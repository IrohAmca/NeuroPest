# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

block_cipher = None
ROOT = Path.cwd()

datas = [
    (str(ROOT / "neuropest" / "assets"), "neuropest/assets"),
    (str(ROOT / "data" / "circuits" / "eye.npz"), "data/circuits"),
    (str(ROOT / "data" / "circuits" / "field.npz"), "data/circuits"),
    (str(ROOT / "data" / "circuits" / "tiers.json"), "data/circuits"),
    (str(ROOT / "data" / "circuits" / "tiers"), "data/circuits/tiers"),
]

if (ROOT / "data" / "circuits" / "mushroom_flywire.npz").exists():
    datas.append((str(ROOT / "data" / "circuits" / "mushroom_flywire.npz"), "data/circuits"))

if (ROOT / "data" / "circuits" / "mushroom_3d.npz").exists():
    datas.append((str(ROOT / "data" / "circuits" / "mushroom_3d.npz"), "data/circuits"))

# Include the full flywire cache if present
if (ROOT / "data" / "circuits" / "flywire_v783.npz").exists():
    datas.append((str(ROOT / "data" / "circuits" / "flywire_v783.npz"), "data/circuits"))

hidden_imports = [
    "neuropest",
    "neuropest.app",
    "neuropest.control",
    "neuropest.fly",
    "neuropest.flywire",
    "neuropest.engine",
    "neuropest.engine.lif_numba",
    "neuropest.engine.lif_ref",
    "neuropest.engine.lif_wgpu",
    "neuropest.engine.network",
    "neuropest.engine.params",
    "neuropest.paths",
    "neuropest.render",
    "neuropest.runner",
    "neuropest.states",
    "neuropest.theme",
    "neuropest.tray",
    "neuropest.vision",
    "neuropest.visual",
    "neuropest.brain",
    "neuropest.capture",
    "neuropest.data_manager",
    "neuropest.eyebuild",
    "neuropest.mb_geometry",
    "neuropest.mb_view3d",
    "neuropest.metabolism",
    "neuropest.mushroom",
    "neuropest.pheromone",
    "neuropest.preferences",
    "neuropest.skins",
    "scipy.ndimage",
    "scipy.special",
    "scipy.spatial",
    "numba",
]

# Check optional wgpu
try:
    import wgpu
    hidden_imports.append("wgpu")
except ImportError:
    pass

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeuroPest",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "neuropest.ico") if (ROOT / "neuropest.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NeuroPest",
)
