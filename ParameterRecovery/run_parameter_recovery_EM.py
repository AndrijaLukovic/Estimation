"""
Parameter recovery for the EM mixture model.

Edit the TRUE_PARAMS, KSI_VALUES, METHOD, LOTTERY, and C blocks below,
then run:
    python run_parameter_recovery_EM.py

Generates multi-session pseudo CE data using the composite reference-point
specification (a1, a2, a3, delta), fits em_mixture(), and compares recovered
cluster parameters to the true values.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import functions as f
from lotteries import para_recov
from GlobalSettings import GlobalMethod, GlobalLottery, GlobalCluster, GlobalTol, GlobalInterMax, GlobalSeedsSet, GlobalTKBounds, GlobalPrelecBounds
from Mixture import em_mixture, em_mixture_best_of
from generate_pseudo_data import generate_pseudo_data_multisession


# ── TRUE STRUCTURAL PARAMETERS ────────────────────────────────────────────────
TRUE_PARAMS = {
    "r":      0.03,
    "alpha":  0.88,
    "lamb":   2.25,
    "beta":   1.0,    # fixed at 1 for single-parameter Prelec (ignored by TK)
    "palpha": 0.65,
    "a1":     0.30,
    "a2":     0.20,
    "a3":     0.10,
    "delta":  0.80,
    # a4 = 1 - a1 - a2 - a3 = 0.40  (forward-looking weight, residual)
}

# For C>1: one dict per cluster and true mixing proportions.
# Set to None / [1.0] to use the single-cluster TRUE_PARAMS above.
TRUE_PARAMS_LIST = [
    TRUE_PARAMS,
    {
        "r":      0.01,
        "alpha":  0.65,
        "lamb":   2.72,
        "beta":   1.0,
        "palpha": 0.43,
        "a1":     0.20,
        "a2":     0.30,
        "a3":     0.40,
        "delta":  0.90,
    },
]
TRUE_PI = [0.4, 0.6]


# ── KSI VALUES ────────────────────────────────────────────────────────────────
NUM_SUBJECTS = 750
KSI_SEED     = 10
np.random.seed(KSI_SEED)
KSI_VALUES = {
    f"sub_{i}": max(1e-4, np.random.normal(0.19, 0.106))
    for i in range(1, NUM_SUBJECTS + 1)
}


# ── SETTINGS ──────────────────────────────────────────────────────────────────
METHOD    = GlobalMethod    
LOTTERY   = GlobalLottery
C         = GlobalCluster

ALL_SEEDS = [10, 15, 20, 25]   
# ─────────────────────────────────────────────────────────────────────────────


def _param_names(method):
    if method == "tk":
        return ["r", "alpha", "lamb", "gamma", "a1", "a2", "a3", "delta"]
    return ["r", "alpha", "lamb", "beta", "palpha", "a1", "a2", "a3", "delta"]


def _param_vec(params_dict, method):
    return [params_dict[k] for k in _param_names(method)]


def _generate_data(true_params_list, pi_list, ksi_values, method, lottery, seed):
    """Generate multi-session pseudo CE data (mixture or single cluster)."""
    if len(true_params_list) == 1:
        return generate_pseudo_data_multisession(
            true_params=true_params_list[0],
            ksi_values=ksi_values,
            method=method,
            lottery=lottery,
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    n   = len(ksi_values)
    subjects = list(ksi_values.keys())

    sizes = np.round(np.array(pi_list) * n).astype(int)
    sizes[-1] = n - sizes[:-1].sum()

    dfs, idx = [], 0
    for tp, size in zip(true_params_list, sizes):
        sub_ksi = {s: ksi_values[s] for s in subjects[idx: idx + size]}
        dfs.append(generate_pseudo_data_multisession(
            true_params=tp,
            ksi_values=sub_ksi,
            method=method,
            lottery=lottery,
            seed=int(rng.integers(1_000_000)),
        ))
        idx += size
    return pd.concat(dfs, ignore_index=True)


def _match_clusters(true_pi, rec_pis, rec_thetas):
    """Align recovered clusters to true clusters by descending pi order."""
    true_order = np.argsort(true_pi)[::-1]
    rec_order  = np.argsort(rec_pis)[::-1]
    inv = np.empty_like(rec_order)
    inv[rec_order] = true_order
    return [rec_pis[k] for k in inv], [rec_thetas[k] for k in inv]


def print_recovery_table(all_results, true_params_list, pi_list, method):
    """
    Print a detailed recovery table across seeds.

    For each cluster shows a block with columns:
        Param | True | Seed-1 | Seed-2 | … | Mean Bias | RMSE
    """
    pnames = _param_names(method)
    c      = len(true_params_list)

    for j in range(c):
        true_vec = _param_vec(true_params_list[j], method)
        header   = (f"Cluster {j+1}  (true π = {pi_list[j]:.2f})"
                    if c > 1 else "Single cluster")

        seed_labels = [str(r["seed"]) for r in all_results]
        col_w       = max(9, max(len(s) for s in seed_labels) + 1)
        n_seeds     = len(all_results)

        # Collect recovered values and biases per seed
        rec_vals  = []   # list of param vectors, one per seed
        rec_pis   = []
        lls       = []
        for r in all_results:
            rv  = r["thetas"][j]
            rec_vals.append(rv)
            rec_pis.append(r["pis"][j])
            lls.append(r["log_likelihood"])

        biases = [[rec_vals[s][p] - true_vec[p] for p in range(len(pnames))]
                  for s in range(n_seeds)]

        sep = "=" * (14 + 8 + n_seeds * (col_w + 1) + 14 + 12)
        print(f"\n{sep}")
        print(f"  PARAMETER RECOVERY — {header}")
        print(sep)

        # Header row
        row  = f"  {'Param':<10}  {'True':>8}  "
        row += "  ".join(f"{'S'+lb:>{col_w}}" for lb in seed_labels)
        row += f"  {'MeanBias':>10}  {'RMSE':>8}"
        print(row)
        print("-" * len(sep))

        for p, name in enumerate(pnames):
            tv      = true_vec[p]
            bs      = [biases[s][p] for s in range(n_seeds)]
            mean_b  = np.mean(bs)
            rmse    = np.sqrt(np.mean([b**2 for b in bs]))

            row  = f"  {name:<10}  {tv:>8.4f}  "
            row += "  ".join(f"{rec_vals[s][p]:>{col_w}.4f}" for s in range(n_seeds))
            row += f"  {mean_b:>+10.4f}  {rmse:>8.4f}"
            print(row)

        # Pi row
        print("-" * len(sep))
        row  = f"  {'pi':<10}  {pi_list[j]:>8.4f}  "
        row += "  ".join(f"{rec_pis[s]:>{col_w}.4f}" for s in range(n_seeds))
        mean_pi_bias = np.mean([rec_pis[s] - pi_list[j] for s in range(n_seeds)])
        rmse_pi      = np.sqrt(np.mean([(rec_pis[s] - pi_list[j])**2 for s in range(n_seeds)]))
        row += f"  {mean_pi_bias:>+10.4f}  {rmse_pi:>8.4f}"
        print(row)

        # Log-likelihood row
        row  = f"  {'LogLik':<10}  {'':>8}  "
        row += "  ".join(f"{lls[s]:>{col_w}.2f}" for s in range(n_seeds))
        print(row)
        print(sep)


if __name__ == "__main__":
    lotteries_t = f.transform(LOTTERY)

    # Resolve cluster configuration
    if TRUE_PARAMS_LIST is not None:
        assert len(TRUE_PARAMS_LIST) == C
        assert TRUE_PI is not None and len(TRUE_PI) == C
        true_params_list = TRUE_PARAMS_LIST
        pi_list          = TRUE_PI
    else:
        true_params_list = [TRUE_PARAMS]
        pi_list          = [1.0]

    all_results = []
    for seed in ALL_SEEDS:
        df = _generate_data(true_params_list, pi_list, KSI_VALUES, METHOD, LOTTERY, seed)
        print(f"\n[Seed {seed}] {df['participant_label'].nunique()} subjects × "
              f"{df['lottery_id'].nunique()} lotteries × "
              f"{df['round_number'].nunique()} sessions = {len(df)} obs")

        result = em_mixture_best_of(
            n_restarts= None,
            seeds=GlobalSeedsSet,
            method=METHOD, c=C, y=df, lotteries=lotteries_t,
        )

        if C > 1:
            rec_pis, rec_thetas = _match_clusters(
                np.array(pi_list), np.array(result["pis"]), list(result["thetas"])
            )
            result = {**result, "pis": rec_pis, "thetas": rec_thetas}

        all_results.append({**result, "seed": seed})

    # ── Summary table ────────────────────────────────────────────────────────
    print_recovery_table(all_results, true_params_list, pi_list, METHOD)

    # ── Ksi summary (last seed) ───────────────────────────────────────────────
    last = all_results[-1]
    print(f"\nTrue ksi      (mean ± std): "
          f"{np.mean(list(KSI_VALUES.values())):.4f} ± "
          f"{np.std(list(KSI_VALUES.values())):.4f}")
    print(f"Recovered ksi (mean ± std): "
          f"{np.mean(last['ksi']):.4f} ± "
          f"{np.std(last['ksi']):.4f}")
