"""Conditional remaining-payoff CPT mixture estimation.

Importing the package caps BLAS/OpenMP at one thread per process (the
workflows parallelise across processes; set the variables beforehand to
override) and keeps ``model_spec``, the module name saved result pickles
reference, pointing at ``estimation.spec``.
"""
import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import sys

from . import spec as _spec

sys.modules.setdefault("model_spec", _spec)
