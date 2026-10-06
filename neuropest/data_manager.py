"""Data management and automated download/build pipeline for NeuroPest.

Handles downloading FlyWire v783 raw data files from upstream public repositories
and building biophysical circuit caches at system startup or via CLI.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable
import urllib.request

from .paths import CACHE, CIRCUITS, EYE, FIELD, MUSHROOM, MUSHROOM_3D, RAW_DIR

_log = logging.getLogger(__name__)

# Upstream raw data URLs (FlyWire v783 & Annotations)
RAW_SOURCES: dict[str, dict[str, str | int]] = {
    "Completeness_783.csv": {
        "url": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv",
        "size_bytes": 3_327_347,
        "description": "FlyWire v783 Neuron Root IDs (3.3 MB)",
    },
    "Connectivity_783.parquet": {
        "url": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
        "size_bytes": 100_804_642,
        "description": "FlyWire v783 Synaptic Connectivity (100.8 MB)",
    },
    "Supplemental_file1_neuron_annotations.tsv": {
        "url": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        "size_bytes": 31_720_298,
        "description": "Cell Type Annotations & Neurotransmitters (31.7 MB)",
    },
}

TOTAL_DOWNLOAD_BYTES = sum(int(item["size_bytes"]) for item in RAW_SOURCES.values())


def is_raw_data_present(raw_dir: Path = RAW_DIR) -> bool:
    """Return True if all required raw FlyWire data files exist on disk."""
    return all((raw_dir / name).exists() and (raw_dir / name).stat().st_size > 0 for name in RAW_SOURCES)


def is_connectome_ready(cache: Path = CACHE) -> bool:
    """Return True if the main FlyWire circuit cache exists and is valid."""
    return cache.exists() and cache.stat().st_size > 0


def download_file(
    url: str,
    dest: Path,
    expected_size: int = 0,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> None:
    """Download a file with streaming chunks and progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".part")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NeuroPest-Downloader/0.5.0 (https://github.com/IrohAmca/NeuroPest)"},
    )

    downloaded = 0
    chunk_size = 64 * 1024  # 64 KB

    with urllib.request.urlopen(req, timeout=30) as resp, open(temp_dest, "wb") as out_f:
        total_size = int(resp.headers.get("Content-Length", expected_size))
        while True:
            if cancel_flag and cancel_flag():
                temp_dest.unlink(missing_ok=True)
                raise InterruptedError("Download canceled by user")
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if progress_cb:
                progress_cb(downloaded, total_size)

    temp_dest.replace(dest)


def download_all_raw(
    raw_dir: Path = RAW_DIR,
    progress_cb: Callable[[float, str], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> None:
    """Download all missing raw connectome files with aggregate progress reporting."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    overall_downloaded = 0

    # Count sizes of already existing files
    for name, info in RAW_SOURCES.items():
        p = raw_dir / name
        if p.exists():
            overall_downloaded += p.stat().st_size

    for name, info in RAW_SOURCES.items():
        p = raw_dir / name
        url = str(info["url"])
        expected_size = int(info["size_bytes"])

        if p.exists() and p.stat().st_size >= expected_size:
            continue

        file_offset = overall_downloaded

        def _sub_cb(current_file_bytes: int, file_total: int):
            nonlocal overall_downloaded
            current_total = file_offset + current_file_bytes
            pct = min(100.0, (current_total / max(1, TOTAL_DOWNLOAD_BYTES)) * 100.0)
            mb_cur = current_total / (1024 * 1024)
            mb_tot = TOTAL_DOWNLOAD_BYTES / (1024 * 1024)
            if progress_cb:
                progress_cb(pct, f"Downloading {name} ({mb_cur:.1f} / {mb_tot:.1f} MB)...")

        download_file(
            url=url,
            dest=p,
            expected_size=expected_size,
            progress_cb=_sub_cb,
            cancel_flag=cancel_flag,
        )
        overall_downloaded = file_offset + p.stat().st_size


def build_all_circuits(progress_cb: Callable[[float, str], None] | None = None) -> None:
    """Build FlyWire circuit tiers, eye optics, and mushroom body models."""
    from . import flywire
    from .eyebuild import build_retina, build_visual_field
    from .mb_geometry import build_mushroom_3d
    from .paths import RAW_DIR

    # 1. FlyWire Connectome Tiers
    if progress_cb:
        progress_cb(10.0, "Building FlyWire whole-brain circuit tiers (~30s)...")
    net = flywire.build(raw_dir=RAW_DIR)
    flywire.CACHE.parent.mkdir(parents=True, exist_ok=True)
    net.save(flywire.CACHE)

    if progress_cb:
        progress_cb(40.0, "Building reduced neuron tiers (2k to 50k)...")
    for n in flywire.TIER_SIZES:
        flywire.load_tier(n, cache=flywire.CACHE, tier_dir=flywire.TIER_DIR)

    # 2. Eye & Visual Receptive Fields
    if progress_cb:
        progress_cb(60.0, "Building compound eye optics & medulla columns...")
    ann_file = RAW_DIR / "Supplemental_file1_neuron_annotations.tsv"
    if ann_file.exists():
        try:
            import numpy as np

            retina = build_retina(net, ann_file)
            retina.save(EYE)
            fld = build_visual_field(net, ann_file)
            np.savez(FIELD, **fld)
        except Exception as exc:
            _log.warning("Could not build eye models: %s", exc)

    # 3. Mushroom Body Wiring
    if progress_cb:
        progress_cb(80.0, "Building real Mushroom Body associative wiring...")
    try:
        import sys
        tools_path = Path(__file__).resolve().parents[1] / "tools"
        if str(tools_path) not in sys.path:
            sys.path.insert(0, str(tools_path))
        import build_mushroom
        build_mushroom.build(raw_dir=RAW_DIR, out=MUSHROOM)
    except Exception as exc:
        _log.warning("Could not build biological mushroom body wiring: %s", exc)

    # 4. 3D Anatomical Geometry
    if progress_cb:
        progress_cb(95.0, "Building 3D neural coordinates & neuropil envelope...")
    try:
        if ann_file.exists() and MUSHROOM.exists():
            build_mushroom_3d(RAW_DIR, MUSHROOM, MUSHROOM_3D)
    except Exception as exc:
        _log.warning("Could not build 3D mushroom geometry: %s", exc)

    if progress_cb:
        progress_cb(100.0, "Initialization complete!")


class DownloadWorkerThread(threading.Thread):
    """Background worker for downloading and building connectome files without blocking Qt."""

    def __init__(self, on_progress: Callable[[float, str], None], on_finished: Callable[[bool, str], None]):
        super().__init__(daemon=True)
        self.on_progress = on_progress
        self.on_finished = on_finished
        self.canceled = False

    def cancel(self):
        self.canceled = True

    def run(self):
        try:
            download_all_raw(
                progress_cb=self.on_progress,
                cancel_flag=lambda: self.canceled,
            )
            if self.canceled:
                self.on_finished(False, "Download canceled.")
                return

            self.on_progress(5.0, "Extracting and building biological circuits...")
            build_all_circuits(progress_cb=self.on_progress)
            self.on_finished(True, "Success")
        except Exception as exc:
            _log.exception("Error during data download/build")
            self.on_finished(False, str(exc))


def prompt_startup_data(parent=None) -> bool:
    """Prompt user at GUI startup if connectome data is missing.

    Returns True if FlyWire data is ready, False if user chose Toy Circuit.
    """
    if is_connectome_ready():
        return True

    # If running headless or in tests, never show GUI prompts
    if "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("NEUROPEST_NO_PROMPT") == "1":
        return False

    try:
        from PySide6.QtCore import Qt, QTimer, Signal, QObject
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import (
            QApplication,
            QDialog,
            QHBoxLayout,
            QLabel,
            QProgressBar,
            QPushButton,
            QVBoxLayout,
        )
    except ImportError:
        return False

    app = QApplication.instance()
    if app is None:
        return False

    class Bridge(QObject):
        progress_signal = Signal(float, str)
        finished_signal = Signal(bool, str)

    bridge = Bridge()

    dialog = QDialog(parent)
    dialog.setWindowTitle("NeuroPest - Data Setup")
    dialog.setFixedSize(480, 260)
    dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
    dialog.setStyleSheet("""
        QDialog { background-color: #121217; color: #f4f4f5; }
        QLabel { color: #d4d4d8; font-size: 13px; }
        QPushButton {
            background-color: #27272a;
            color: #f4f4f5;
            border: 1px solid #3f3f46;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #3f3f46; }
        QPushButton#primaryBtn {
            background-color: #10b981;
            color: #042f2e;
            border: 1px solid #059669;
        }
        QPushButton#primaryBtn:hover { background-color: #34d399; }
        QProgressBar {
            background-color: #18181b;
            border: 1px solid #27272a;
            border-radius: 4px;
            text-align: center;
            color: #e4e4e7;
            height: 18px;
        }
        QProgressBar::chunk {
            background-color: #10b981;
            border-radius: 3px;
        }
    """)

    vbox = QVBoxLayout(dialog)
    vbox.setContentsMargins(24, 20, 24, 20)
    vbox.setSpacing(14)

    title_lbl = QLabel("Biological Connectome Setup", dialog)
    title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
    vbox.addWidget(title_lbl)

    info_lbl = QLabel(
        "FlyWire v783 connectome data (~135 MB) is not yet installed.\n"
        "Would you like to download and initialize the biological fruit fly "
        "brain circuit now, or run in lightweight Toy Circuit mode (146 neurons)?",
        dialog,
    )
    info_lbl.setWordWrap(True)
    vbox.addWidget(info_lbl)

    pbar = QProgressBar(dialog)
    pbar.setRange(0, 100)
    pbar.setValue(0)
    pbar.setVisible(False)
    vbox.addWidget(pbar)

    status_lbl = QLabel("", dialog)
    status_lbl.setStyleSheet("color: #a1a1aa; font-size: 12px;")
    status_lbl.setVisible(False)
    vbox.addWidget(status_lbl)

    btn_layout = QHBoxLayout()
    btn_layout.addStretch()

    toy_btn = QPushButton("Use Toy Circuit", dialog)
    download_btn = QPushButton("Download & Initialize", dialog)
    download_btn.setObjectName("primaryBtn")

    btn_layout.addWidget(toy_btn)
    btn_layout.addWidget(download_btn)
    vbox.addLayout(btn_layout)

    worker_ref: list[DownloadWorkerThread | None] = [None]
    success_ref = [False]

    def _start_download():
        toy_btn.setEnabled(False)
        download_btn.setEnabled(False)
        download_btn.setText("Downloading...")
        pbar.setVisible(True)
        status_lbl.setVisible(True)
        status_lbl.setText("Connecting to repository...")

        worker = DownloadWorkerThread(
            on_progress=lambda pct, txt: bridge.progress_signal.emit(pct, txt),
            on_finished=lambda ok, txt: bridge.finished_signal.emit(ok, txt),
        )
        worker_ref[0] = worker
        worker.start()

    def _on_progress(pct: float, txt: str):
        pbar.setValue(int(pct))
        status_lbl.setText(txt)

    def _on_finished(ok: bool, txt: str):
        if ok:
            success_ref[0] = True
            dialog.accept()
        else:
            status_lbl.setText(f"Error: {txt}")
            download_btn.setText("Retry")
            download_btn.setEnabled(True)
            toy_btn.setEnabled(True)

    def _use_toy():
        dialog.reject()

    bridge.progress_signal.connect(_on_progress)
    bridge.finished_signal.connect(_on_finished)
    download_btn.clicked.connect(_start_download)
    toy_btn.clicked.connect(_use_toy)

    dialog.exec()
    return success_ref[0] or is_connectome_ready()


def main():
    """CLI entrypoint for downloading and building data files."""
    print("=" * 60)
    print("NeuroPest - Connectome Data Downloader & Builder")
    print("=" * 60)

    if is_connectome_ready():
        print(f"[OK] FlyWire connectome is already installed at: {CACHE}")
        return

    print("Downloading upstream FlyWire v783 raw files (~135 MB)...")
    def _cli_progress(pct: float, txt: str):
        bars = int(pct / 2.5)
        sys.stdout.write(f"\r[{'=' * bars}{' ' * (40 - bars)}] {pct:5.1f}% | {txt}")
        sys.stdout.flush()

    download_all_raw(progress_cb=_cli_progress)
    print("\nDownload complete! Building circuits...")

    def _build_progress(pct: float, txt: str):
        print(f"[{pct:3.0f}%] {txt}")

    build_all_circuits(progress_cb=_build_progress)
    print("\n[SUCCESS] NeuroPest connectome successfully installed!")


if __name__ == "__main__":
    main()
