"""Project paths and estimation settings shared by every workflow."""
from pathlib import Path

import numpy as np

from .lotteries import LOTTERIES_V2

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "gamedata.csv"

# Lottery set to estimate on.
LOTTERIES = LOTTERIES_V2

VALUATION_VERSION = "conditional_remaining_v1"

# psi = quasi-hyperbolic present-bias on the within-lottery exponential discount
# rate r: rho(t=0)=1, rho(t>=1) = psi * exp(-r*t).  psi=1 recovers plain
# exponential discounting.
# Curvature and loss aversion are properties of the utility function and memory
# decay of the reference point, so TK and Prelec search the same ranges for them.
# Parameter order: [r, alpha_gain, alpha_loss, lamb, gamma, a1, a2, a3, delta, psi]
TK_BOUNDS = (
    (1e-6, 1), (1e-6, 5), (1e-6, 5), (1e-6, 6), (0.2, 1.0),
    (0, 1), (0.0, 1), (0.0, 1), (0.2, 1), (1e-6, 1),
)
# Parameter order: [r, alpha_gain, alpha_loss, lamb, beta, palpha, a1, a2, a3, delta, psi]
PRELEC_BOUNDS = (
    (1e-6, 1), (1e-6, 5), (1e-6, 5), (1e-6, 6), (1.0, 1.0), (1e-6, 1),
    (0, 1), (0.0, 1), (0.0, 1), (0.2, 1), (1e-6, 1),
)
# Default bound table; run-specific changes go through spec.replace_bound.
BOUNDS = {"tk": TK_BOUNDS, "prelec": PRELEC_BOUNDS}

# Individual noise: sigma_i = ksi_i * spread.
KSI_BOUNDS = (1e-4, 1.5)

MAX_ITER = 1000
TOL = 1e-6

N_SEEDS = 100
MASTER_SEED = 42
SEEDS = np.random.default_rng(MASTER_SEED).integers(0, 10000, size=N_SEEDS).tolist()

# R^SQ in the composite reference point:
#   "zt"   : R^SQ(t) = Z_t  (accumulated realized payoffs)
#   "zero" : R^SQ(t) = 0
RSQ_MODE = "zero"
