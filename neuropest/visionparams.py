"""Parameters of the image-driven input (visual.VisionDrive). Light module: the GUI imports it without numba."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionParams:
    cursor_model: str = "sphere"     # "sphere": a disc facing the fly at eye level (height independent);
                                     # "disk": a disc lying on the screen plane seen through the funnel (the first model)
    eye_height: float = 100.0        # px above the screen plane; larger = more top-down view (funnel / screen image only)
    halo_px: float = 30.0            # the cursor is drawn as a dark disk this big
    background: float = 0.5
    # gains found by grid search on approach / slide / recede scenes through THIS class (tools/vision_calibrate.py
    # --grid; the sphere defaults below, the legacy plane disk uses DISK_PLANE)
    gain_loom: float = 25.0          # Hz of LPLC2/LC4 drive per unit of expansion
    max_loom: float = 150.0
    gain_retreat: float = 12.0       # Hz of LPC1 drive per unit of eye-wide expansion
    max_retreat: float = 20.0
    flee_lo: float = 3.0             # retreat drive fades out between these expansion values: strong looming flees
    flee_hi: float = 4.5
    gain_object: float = 50.0        # LC10 (steering): at 100 Hz a near disk drove DNa02 past 100 Hz, twice what
    max_object: float = 50.0         # the cursor drive gives at its maximum (79 Hz)


# The first model: the cursor as a disc lying on the screen plane, seen through the funnel (its retreat gain was
# found for that geometry; the eye-level disc's expansion is larger for slow approaches, hence 12 above).
DISK_PLANE = VisionParams(cursor_model="disk", gain_retreat=18.0)
