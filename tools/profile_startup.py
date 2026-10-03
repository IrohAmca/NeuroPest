"""Where does worker startup time go? Run in a fresh interpreter.

Run: uv run python tools/profile_startup.py
"""
import time

t0 = time.perf_counter()


def lap(name):
    global t0
    now = time.perf_counter()
    print(f"{name:34s} {now - t0:5.2f} s", flush=True)
    t0 = now


import numpy  # noqa: E402,F401
lap("import numpy")
import scipy.sparse  # noqa: E402,F401
lap("import scipy.sparse")
import numba  # noqa: E402,F401
lap("import numba")
from neuropest.engine import LIFEngine  # noqa: E402,F401
lap("import neuropest.engine")
from neuropest import flywire  # noqa: E402
lap("import neuropest.flywire")
net = flywire.load_cache()
lap("load_cache (npz)")
sub = net.prefix(2000)
lap("prefix(2000) (induced)")
from neuropest.brain import Brain  # noqa: E402
lap("import neuropest.brain")
b = Brain(sub)
lap("Brain()")
b.advance(2.0)
lap("first advance (kernel load)")
b.advance(2.0)
lap("second advance")
