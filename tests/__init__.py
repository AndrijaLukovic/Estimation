"""Tests run single-threaded BLAS, like the estimation workflows (set before NumPy loads)."""
import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
