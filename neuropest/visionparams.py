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
    gain_loom: float = 30.0          # Hz of LPLC2/LC4 drive per unit of expansion
    max_loom: float = 150.0
    # LPLC2 and LC4 are not the same cell: LC4 codes the angular VELOCITY of the expanding edge (the expansion detector's
    # output is a growth rate, so it drives LC4 as before) and LPLC2 the angular SIZE (von Reyn et al. 2017, Ache et al.
    # 2019): its drive is the same expansion scaled by how much of its field the object covers, from 0 at size_lo to the
    # full drive at size_hi (share of the 30 deg pool the object covers, as `Features` measures it: the default 30 px
    # disc reads ~0.13 at 400 px and ~0.9 at 50 px, a 15 px one ~0.04 and ~0.48; see tools/vision_calibrate.py --verbose).
    # loom_split False: both types get the same rate (the first version).
    loom_split: bool = True
    size_lo: float = 0.10
    size_hi: float = 0.80
    rf_sigma_deg: float = 0.0        # receptive-field weight of the expansion pool: exp(-a^2 / 2 sigma^2) at `a` degrees
                                     # from a neuron's field centre (0: flat top hat, the first version)
    gain_retreat: float = 12.0       # Hz of LPC1 drive per unit of eye-wide expansion
    max_retreat: float = 20.0
    flee_lo: float = 3.0             # retreat drive fades out between these expansion values: strong looming flees
    flee_hi: float = 4.5
    gain_object: float = 50.0        # LC10 (steering): at 100 Hz a near disk drove DNa02 past 100 Hz, twice what
    max_object: float = 50.0         # the cursor drive gives at its maximum (79 Hz)


# The first model: the cursor as a disc lying on the screen plane, seen through the funnel (its retreat gain was
# found for that geometry; the eye-level disc's expansion is larger for slow approaches, hence 12 above).
DISK_PLANE = VisionParams(cursor_model="disk", gain_loom=25.0, gain_retreat=18.0, loom_split=False)
