"""Build standalone NeuroPest Windows application (.exe) using PyInstaller."""
import subprocess
import sys
from pathlib import Path
from tools.build_icon import generate_ico

ROOT = Path(__file__).resolve().parents[1]


def build():
    ico = ROOT / "neuropest.ico"
    if not ico.exists():
        print("Generating icon...")
        generate_ico(ico)

    spec = ROOT / "neuropest.spec"
    if not spec.exists():
        print(f"Spec file not found: {spec}")
        sys.exit(1)

    cmd = [
        "uv", "run", "--with", "pyinstaller",
        "pyinstaller",
        str(spec),
        "--noconfirm",
    ]
    print(f"Running build command: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        print("Build failed!")
        sys.exit(res.returncode)

    exe_path = ROOT / "dist" / "NeuroPest" / "NeuroPest.exe"
    if exe_path.exists():
        print("\n" + "=" * 60)
        print("Build successful!")
        print(f"Standalone executable: {exe_path}")
        print("You can run it directly by double-clicking NeuroPest.exe")
        print("=" * 60 + "\n")
    else:
        print(f"Executable not found at expected location: {exe_path}")
        sys.exit(1)


if __name__ == "__main__":
    build()
