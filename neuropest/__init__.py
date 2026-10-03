import os

# The simulation is single-threaded numba and never needs BLAS threads. Each BLAS thread reserves
# memory up front, which fails outright on machines with little free commit memory.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

__version__ = "0.2.0"
