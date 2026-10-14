"""Behavior states; kept free of heavy imports so the GUI process doesn't load numba."""
STAND, WALK, FLY, RETREAT, GROOM, FREEZE = "stand", "walk", "fly", "retreat", "groom", "freeze"
STATES = (STAND, WALK, FLY, RETREAT, GROOM, FREEZE)       # the worker passes the INDEX: append, never reorder
GEARS = ("stand", "walk", "fly_short", "fly_long")       # V_motor gear (brain.gear); append, never reorder

# The neuroscience lab (3D view): neuron groups that can be silenced (clamped, they cannot spike) or stimulated (forced
# Poisson spikes, like a light-gated channel). Names are the circuit's group names (brain.Brain.g; LPC1 is its retreat
# input). The worker and the GUI pass bit masks over this tuple (bit k = LAB_GROUPS[k]): append, never reorder.
LAB_GROUPS = ("GF", "MDN", "WALK", "LC4", "LPLC2", "LPC1", "LC10_L", "LC10_R", "DNa02_L", "DNa02_R", "GROOM")
# Forced rate (Hz) of a full-strength stimulation per group, inside the range the circuit's responses were probed in
# (brain.py: LOOM input up to 150 Hz, LPC1 -> MDN 15-26 Hz at 20-30 Hz, LC10 on one side -> DNa02 24-79 Hz at 10-20 Hz)
LAB_FULL_HZ = {"GF": 150.0, "MDN": 120.0, "WALK": 120.0, "LC4": 80.0, "LPLC2": 80.0, "LPC1": 30.0,
               "LC10_L": 20.0, "LC10_R": 20.0, "DNa02_L": 100.0, "DNa02_R": 100.0, "GROOM": 80.0}


def lab_mask(names) -> int:
    """Bit mask of the lab groups in `names` (unknown names are ignored)."""
    names = set(names)
    return sum(1 << k for k, n in enumerate(LAB_GROUPS) if n in names)


def lab_names(mask: float | int) -> tuple[str, ...]:
    """The lab groups whose bits are set in `mask`, in LAB_GROUPS order."""
    m = int(mask)
    return tuple(n for k, n in enumerate(LAB_GROUPS) if m >> k & 1)
