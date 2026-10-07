# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

block_cipher = None
ROOT = Path.cwd()


def _project_version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version"):
            return line.split('"')[1]
    return "0.0.0"


def _version_info():
    """Windows file-properties block (Details tab). Does not replace a code signature."""
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
    )
    ver = _project_version()
    nums = tuple(int(x) for x in ver.split(".")[:3]) + (0,)
    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums, mask=0x3F, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0),
        kids=[
            StringFileInfo([StringTable("040904B0", [
                StringStruct("CompanyName", "IrohAmca"),
                StringStruct("FileDescription", "NeuroPest desktop fly pet"),
                StringStruct("FileVersion", ver),
                StringStruct("InternalName", "NeuroPest"),
                StringStruct("OriginalFilename", "NeuroPest.exe"),
                StringStruct("ProductName", "NeuroPest"),
                StringStruct("ProductVersion", ver),
                StringStruct("LegalCopyright", "PolyForm Noncommercial 1.0.0"),
            ])]),
            VarFileInfo([VarStruct("Translation", [1033, 1200])]),
        ],
    )

datas = [
    (str(ROOT / "neuropest" / "assets"), "neuropest/assets"),
]

# Conditionally include circuit and model files if present on disk
circuits_dir = ROOT / "data" / "circuits"
if circuits_dir.exists():
    for fname in [
        "eye.npz",
        "field.npz",
        "tiers.json",
        "mushroom_flywire.npz",
        "mushroom_3d.npz",
        "flywire_v783.npz",
    ]:
        p = circuits_dir / fname
        if p.exists():
            datas.append((str(p), "data/circuits"))
    tiers_dir = circuits_dir / "tiers"
    if tiers_dir.exists():
        datas.append((str(tiers_dir), "data/circuits/tiers"))

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
    upx=False,   # UPX-packed exes are flagged far more often by Defender/SmartScreen heuristics
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "neuropest.ico") if (ROOT / "neuropest.ico").exists() else None,
    version=_version_info(),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="NeuroPest",
)
