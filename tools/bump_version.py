"""Automated version bumper and release preparation tool for NeuroPest.

Updates all synchronized version locations across the codebase:
1. pyproject.toml
2. neuropest/__init__.py
3. neuropest/data_manager.py
4. CITATION.cff (version and date-released)
5. README.md (status badge and release download link)
6. .github/workflows/release.yml (fallback release tag)
7. uv.lock (via 'uv lock')

Usage:
    uv run python tools/bump_version.py patch          # 0.5.0 -> 0.5.1
    uv run python tools/bump_version.py minor          # 0.5.1 -> 0.6.0
    uv run python tools/bump_version.py major          # 0.5.1 -> 1.0.0
    uv run python tools/bump_version.py 0.5.2          # explicit version
    uv run python tools/bump_version.py patch --commit --tag
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def get_current_version() -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if not m:
        raise ValueError("Could not find version in pyproject.toml")
    return m.group(1)


def compute_next_version(current: str, bump_type: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", current)
    if not m:
        raise ValueError(f"Current version '{current}' does not match standard semver (X.Y.Z)")
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))

    if bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    if bump_type == "major":
        return f"{major + 1}.0.0"

    # Explicit version given
    if re.match(r"^\d+\.\d+\.\d+$", bump_type):
        return bump_type

    raise ValueError(f"Invalid bump type or version: '{bump_type}'. Choose 'patch', 'minor', 'major', or 'X.Y.Z'.")


def update_file(path: Path, pattern: str, replacement: str) -> None:
    content = path.read_text(encoding="utf-8")
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        print(f"Warning: pattern '{pattern}' not found in {path.name}", file=sys.stderr)
    path.write_text(new_content, encoding="utf-8")


def bump(new_version: str, today: str | None = None) -> None:
    curr_version = get_current_version()
    if today is None:
        today = date.today().isoformat()

    print(f"Bumping version: {curr_version} -> {new_version} (Release date: {today})")

    # 1. pyproject.toml
    update_file(
        ROOT / "pyproject.toml",
        r'(^version\s*=\s*")[^"]+(")',
        rf"\g<1>{new_version}\g<2>",
    )

    # 2. neuropest/__init__.py
    update_file(
        ROOT / "neuropest" / "__init__.py",
        r'(^__version__\s*=\s*")[^"]+(")',
        rf"\g<1>{new_version}\g<2>",
    )

    # 3. neuropest/data_manager.py (User-Agent header)
    update_file(
        ROOT / "neuropest" / "data_manager.py",
        r'NeuroPest-Downloader/[0-9.]+',
        f"NeuroPest-Downloader/{new_version}",
    )

    # 4. CITATION.cff
    update_file(
        ROOT / "CITATION.cff",
        r'(^version:\s*)[0-9.]+',
        rf"\g<1>{new_version}",
    )
    update_file(
        ROOT / "CITATION.cff",
        r'(^date-released:\s*")[^"]+(")',
        rf'\g<1>{today}\g<2>',
    )

    # 5. README.md
    update_file(
        ROOT / "README.md",
        r'\*\*Current State \(v[0-9.]+\):\*\*',
        f"**Current State (v{new_version}):**",
    )
    update_file(
        ROOT / "README.md",
        r'NeuroPest-v[0-9.]+-windows-x64\.zip',
        f"NeuroPest-v{new_version}-windows-x64.zip",
    )

    # 6. .github/workflows/release.yml
    update_file(
        ROOT / ".github" / "workflows" / "release.yml",
        r"else \{\s*'v[0-9.]+'\s*\}",
        f"else {{ 'v{new_version}' }}",
    )

    # 7. uv.lock
    print("Running 'uv lock' to synchronize uv.lock...")
    res = subprocess.run(["uv", "lock"], cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running uv lock: {res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
    print("uv.lock updated successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroPest automated version bumper")
    parser.add_argument("target", help="'patch', 'minor', 'major', or explicit version 'X.Y.Z'")
    parser.add_argument("--commit", action="store_true", help="Stage files and create a release commit")
    parser.add_argument("--tag", action="store_true", help="Create an annotated git tag")
    parser.add_argument("-m", "--message", help="Custom release message for commit and tag")
    args = parser.parse_args()

    curr_version = get_current_version()
    next_version = compute_next_version(curr_version, args.target)

    bump(next_version)

    files_to_add = [
        "pyproject.toml",
        "neuropest/__init__.py",
        "neuropest/data_manager.py",
        "CITATION.cff",
        "README.md",
        ".github/workflows/release.yml",
        "uv.lock",
    ]

    if args.commit or args.tag:
        subprocess.run(["git", "add"] + files_to_add, cwd=ROOT, check=True)
        msg = args.message or f"release: v{next_version}"
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT, check=True)
        print(f"Created commit: '{msg}'")

    if args.tag:
        tag_name = f"v{next_version}"
        tag_msg = args.message or f"Release v{next_version}"
        subprocess.run(["git", "tag", "-a", tag_name, "-m", tag_msg], cwd=ROOT, check=True)
        print(f"Created annotated tag: {tag_name}")


if __name__ == "__main__":
    main()
