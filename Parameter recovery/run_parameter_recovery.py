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
    ALL_SEEDS
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


def _struct_names(method):
    if method == "tk":
        return ["r", "alpha", "lamb", "gamma", "R"], 5
    return ["r", "alpha", "lamb", "beta", "palpha", "R"], 6


def collect_recovery_row(result, true_params, method=METHOD, seed=SEED):
    """Return a dict with per-parameter biases for one seed."""
    names, ksi_offset = _struct_names(method)
    row = {"seed": seed}
    for name, tv, ev in zip(names, [true_params[k] for k in names], result.x[:ksi_offset]):
        row[f"bias_{name}"] = ev - tv
    row["log_likelihood"] = -result.fun
    return row


def print_summary_table(rows, method=METHOD):
    """One table: each row = one seed, columns = bias per parameter + avg row."""
    names, _ = _struct_names(method)
    cw = 10
    total_w = 6 + (len(names) + 1) * (cw + 1)
    sep = "-" * total_w

    print("\n" + "=" * total_w)
    print("PARAMETER RECOVERY  (bias = estimated − true)")
    print("=" * total_w)
    print(" ".join([f"{'Seed':>6}"] + [f"{n:>{cw}}" for n in names] + [f"{'LogLik':>{cw}}"]))
    print(sep)

    bias_accum = {n: [] for n in names}
    for row in rows:
        parts = [f"{row['seed']:>6}"]
        for n in names:
            b = row[f"bias_{n}"]
            bias_accum[n].append(b)
            parts.append(f"{b:>{cw}.4f}")
        parts.append(f"{row['log_likelihood']:>{cw}.4f}")
        print(" ".join(parts))

    print(sep)
    avg_parts = [f"{'Avg':>6}"] + [f"{np.mean(bias_accum[n]):>{cw}.4f}" for n in names] + [f"{'':>{cw}}"]
    print(" ".join(avg_parts))
    print("=" * total_w)


if __name__ == "__main__":
    rows = []
    for s in ALL_SEEDS:
        result, df, ksi_dict = run_recovery(
            true_params=TRUE_PARAMS,
            ksi_values=KSI_VALUES,
            method=METHOD,
            lottery=LOTTERY,
            n_starts=N_STARTS,
            seed=s,
        )
        rows.append(collect_recovery_row(result, TRUE_PARAMS, method=METHOD, seed=s))

    print_summary_table(rows, method=METHOD)
