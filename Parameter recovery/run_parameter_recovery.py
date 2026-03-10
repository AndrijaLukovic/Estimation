"""
Full parameter recovery exercise.

Workflow
--------
1. Generate a pseudo dataset from known ground-truth parameters.
2. Feed that dataset into estimate_mle() from MLE.py.
3. Compare estimated parameters against the ground truth.

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import pandas as pd
from generate_pseudo_data import (
    generate_pseudo_data,
    TRUE_PARAMS,
    KSI_VALUES,
    METHOD,
    LOTTERY,
    SEED,
)
import functions as f
from lotteries import lotteries_full
from MLE import estimate_mle, format_results, bounds_tk, bounds_prelec
from MLE_parallel import estimate_mle_parallel

# ── SETTINGS (recovery-specific) ──────────────────────────────────────────────
N_STARTS = 150            # multistart restarts for MLE
# ─────────────────────────────────────────────────────────────────────────────


def run_recovery(
    true_params=None,
    ksi_values=None,
    method=METHOD,
    lottery=None,
    n_starts=N_STARTS,
    seed=SEED,
):
    """
    Generate pseudo data and estimate parameters. Returns (result, df).
    """
    if true_params is None:
        true_params = TRUE_PARAMS
    if ksi_values is None:
        ksi_values = KSI_VALUES
    if lottery is None:
        lottery = lotteries_full

    # Normalise ksi_values to dict
    if isinstance(ksi_values, (list, np.ndarray)):
        ksi_values = {f"sub_{i+1}": float(k) for i, k in enumerate(ksi_values)}

    df = generate_pseudo_data(
        true_params=true_params,
        ksi_values=ksi_values,
        method=method,
        lottery=lottery,
        seed=seed,
    )
    print(f" [UPDATE] Pseudo data generated in pseudo_data.csv. {df['participant_label'].nunique()} subjects with "
          f"{df['lottery_id'].nunique()} lotteries = {len(df)} observations")

    print("\n [UPDATE] Estimating the pseudo data now.")
    bounds = bounds_tk if method == "tk" else bounds_prelec
    lotteries_t = f.transform(lottery)
    result = estimate_mle_parallel(
        n_starts=n_starts,
        param_bounds=bounds,
        y=df,
        lotteries=lotteries_t,
        method=method,
    )
    return result, df, ksi_values


def print_recovery_report(result, true_params, ksi_values, method=METHOD):
    """Print a side-by-side comparison of true vs estimated parameters."""
    if method == "tk":
        struct_names  = ["r", "alpha", "lamb", "gamma", "R"]
        true_struct   = [true_params[k] for k in struct_names]
        ksi_offset    = 5
    else:
        struct_names  = ["r", "alpha", "lamb", "beta", "palpha", "R"]
        true_struct   = [true_params[k] for k in struct_names]
        ksi_offset    = 6

    est_struct = result.x[:ksi_offset]
    subjects   = sorted(ksi_values.keys())
    est_ksi    = result.x[ksi_offset:]
    true_ksi   = [ksi_values[s] for s in subjects]

    print("\n" + "=" * 55)
    print("parameter recovery results:")
    print("=" * 55)
    print(f"{'Parameter':<14} {'True':>10} {'Estimated':>12} {'Bias':>10}")
    print("-" * 55)
    for name, tv, ev in zip(struct_names, true_struct, est_struct):
        print(f"{name:<14} {tv:>10.4f} {ev:>12.4f} {ev - tv:>10.4f}")
    print("-" * 55)
    print("Individual ksi:")
    for subj, tv, ev in zip(subjects, true_ksi, est_ksi):
        print(f"  {subj:<12} {tv:>10.4f} {ev:>12.4f} {ev - tv:>10.4f}")
    print("=" * 55)
    print(f"Best Log-Likelihood: {-result.fun:.4f}")


if __name__ == "__main__":
    result, df, ksi_dict = run_recovery(
        true_params=TRUE_PARAMS,
        ksi_values=KSI_VALUES,
        method=METHOD,
        lottery=LOTTERY,
        n_starts=N_STARTS,
        seed=SEED,
    )
    print_recovery_report(result, TRUE_PARAMS, ksi_dict, method=METHOD)
