"""Persistent user preferences for NeuroPest.

Saved as JSON in data/preferences.json, mirroring the mushroom body memory
persistence pattern (fly_memory.npz).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any

from .paths import PREFERENCES

_log = logging.getLogger(__name__)


@dataclass
class Preferences:
    """User configuration preserved across NeuroPest application launches."""

    # Circuit & Simulation Engine
    circuit: str = "flywire"       # "flywire" or "toy"
    neurons: int = 15_000          # number of neurons
    dt: float = 0.5                # time step in ms
    symmetry: str = "individual"   # "individual" or "symmetric"
    learning: bool = True          # mushroom body learning enabled

    # Compute Hardware & GPU
    hardware: str = "auto"         # "auto", "cpu", "gpu"
    gpu_name: str = ""             # e.g. "NVIDIA GeForce GTX 1650"
    gpu_backend: str = ""          # e.g. "Vulkan", "D3D12"
    gpu_index: int | None = None   # adapter index

    # View & Appearance
    scale: float = 1.0             # 0.5 to 4.0
    skin: str = "classic"          # skin id: "classic", "dark", "cyber", "gold", "albino"
    monitor_index: int = 0         # 0 = All monitors, 1..N = specific monitor
    monitor_name: str = ""         # screen name for stable identification

    # Behavior & Metabolism
    hunger_enabled: bool = True
    metabolic_rate: float = 1.0    # 0.2 to 3.0
    touch_groom_enabled: bool = False
    bias: float = 0.65             # walking bias 0.0 to 1.0
    skittish: float = 1.0          # escape multiplier 0.25 to 4.0
    wall_pain: float = 1.0         # wall pain punishment multiplier 0.0 to 2.0

    # Vision & Screen Capture
    vision_enabled: bool = False
    eye_height: float = 140.0

    # Pheromone & Scent
    pheromone_enabled: bool = True
    cursor_phero_mode: str = "attract"  # "attract", "repel", "none"

    _path: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, path: Path | None = None) -> Preferences:
        """Load user preferences from JSON file, returning defaults on any error."""
        if path is None and "PYTEST_CURRENT_TEST" in os.environ and "NEUROPEST_PREFERENCES" not in os.environ:
            inst = cls()
            return inst
        target = path or (Path(os.environ["NEUROPEST_PREFERENCES"]) if "NEUROPEST_PREFERENCES" in os.environ else PREFERENCES)
        if not target.exists():
            inst = cls()
            inst._path = target
            return inst
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                inst = cls()
                inst._path = target
                return inst
            valid_keys = {k for k in cls.__dataclass_fields__.keys() if k != "_path"}
            filtered = {k: v for k, v in raw.items() if k in valid_keys}
            inst = cls(**filtered)
            inst._path = target
            inst._sanitize()
            return inst
        except Exception as exc:
            _log.warning("Could not load preferences from %s: %s", target, exc)
            inst = cls()
            inst._path = target
            return inst

    def _sanitize(self) -> None:
        """Sanitize numerical and categorical fields to valid ranges."""
        self.scale = max(0.5, min(4.0, float(self.scale)))
        self.metabolic_rate = max(0.2, min(3.0, float(self.metabolic_rate)))
        self.bias = max(0.0, min(1.0, float(self.bias)))
        self.skittish = max(0.1, min(10.0, float(self.skittish)))
        self.wall_pain = max(0.0, min(3.0, float(self.wall_pain)))
        self.eye_height = max(40.0, min(300.0, float(self.eye_height)))
        if self.circuit not in ("flywire", "toy"):
            self.circuit = "flywire"
        if self.symmetry not in ("individual", "symmetric"):
            self.symmetry = "individual"
        if self.hardware not in ("auto", "cpu", "gpu"):
            self.hardware = "auto"
        if self.cursor_phero_mode not in ("attract", "repel", "none"):
            self.cursor_phero_mode = "attract"

    def save(self, path: Path | None = None) -> None:
        """Atomically persist preferences to disk as formatted JSON."""
        if path is None and self._path is None and "PYTEST_CURRENT_TEST" in os.environ and "NEUROPEST_PREFERENCES" not in os.environ:
            return
        target = path or self._path or (Path(os.environ["NEUROPEST_PREFERENCES"]) if "NEUROPEST_PREFERENCES" in os.environ else PREFERENCES)
        self._path = target
        try:
            self._sanitize()
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target.with_suffix(".tmp")
            data = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "_path"}
            tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(target)
        except Exception as exc:
            _log.warning("Could not save preferences to %s: %s", target, exc)
