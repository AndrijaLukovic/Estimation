"""
Parameter recovery for the EM mixture model.

Edit the TRUE_PARAMS, KSI_VALUES, METHOD, LOTTERY, and C blocks below,
then run:
    python run_parameter_recovery_EM.py

Generates multi-session pseudo CE data (sessions 1, 2, 3) using the
composite reference-point specification (a1, a2, a3, delta), then fits
em_mixture() and compares recovered cluster parameters to the true values.

For a single-cluster test (C=1), set TRUE_PARAMS directly.
For a multi-cluster test (C>1), set TRUE_PARAMS_LIST (one dict per cluster)
and leave TRUE_PARAMS as the first entry (used only for ksi generation).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import functions as f
from lotteries import para_recov
from GlobalSettings import GlobalMethod, GlobalLottery, GlobalCluster, GlobalStarts, GlobalTol
from EM_parallel import run_em_multistart
from generate_pseudo_data import generate_pseudo_data_multisession


# ── TRUE STRUCTURAL PARAMETERS ────────────────────────────────────────────────
# Single-cluster recovery: one set of true parameters for all subjects.
# For TK method, replace beta/palpha with gamma.
TRUE_PARAMS = {
    "r":      0.03,
    "alpha":  0.88,
    "lamb":   2.25,
    "beta":   1.0,    # fixed at 1 for single-parameter Prelec (ignored by TK)
    "palpha": 0.65,   # Prelec elevation parameter (replace with gamma for TK)
    "a1":     0.30,   # status-quo weight            (R^SQ)
    "a2":     0.20,   # partial-adaptation weight     (R^A)
    "a3":     0.10,   # lagged-expectation weight     (R^LE)
    "delta":  0.80,   # memory decay factor δ ∈ [0,1]
    # a4 = 1 - a1 - a2 - a3 = 0.40  →  forward-looking weight (residual)
}

# For multi-cluster (C>1): define TRUE_PARAMS_LIST with one dict per cluster
# and TRUE_PI for the true mixing proportions.
# Leave these as None to use the single-cluster TRUE_PARAMS above.
TRUE_PARAMS_LIST = [TRUE_PARAMS,
         {           
            "r":      0.01,
            "alpha":  0.65,
            "lamb":   2.72,
            "beta":   1.0,    # fixed at 1 for single-parameter Prelec (ignored by TK)
            "palpha": 0.43,   # Prelec elevation parameter (replace with gamma for TK)
            "a1":     0.20,   # status-quo weight            (R^SQ)
            "a2":     0.30,   # partial-adaptation weight     (R^A)
            "a3":     0.40,   # lagged-expectation weight     (R^LE)
            "delta":  0.9,   # memory decay factor δ ∈ [0,1]
            # a4 = 1 - a1 - a2 - a3 = 0.40  →  forward-looking weight (residual)
        }

                    ]   # e.g. [TRUE_PARAMS, {...}] for C=2
TRUE_PI          = [0.4,0.6]   # e.g. [0.6, 0.4] for C=2


# ── KSI VALUES ────────────────────────────────────────────────────────────────
NUM_SUBJECTS = 50
KSI_VALUES = {
    f"sub_{i}": max(1e-4, np.random.normal(0.19, 0.106))
    for i in range(1, NUM_SUBJECTS + 1)
}


# ── SETTINGS ──────────────────────────────────────────────────────────────────
METHOD     = GlobalMethod      # "prelec" or "tk"
LOTTERY    = GlobalLottery    # lottery set used for both data generation and estimation
C          = GlobalCluster             # number of EM clusters
SEED       = 42            # seed for KSI_VALUES draw and single-seed run
ALL_SEEDS  = [10, 15, 20, 25]   # data-generation seeds; one full recovery per seed
N_RESTARTS = GlobalStarts             # EM restarts per seed (passed to run_em_multistart)
_HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_PATH   = os.path.join(_HERE, "pseudo_data_multisession.csv")
# ─────────────────────────────────────────────────────────────────────────────


def _param_vec(params_dict, method):
    """Extract the ordered parameter vector from a TRUE_PARAMS dict."""
    if method == "tk":
        return [params_dict[k] for k in ["r", "alpha", "lamb", "gamma",
                                          "a1", "a2", "a3", "delta"]]
    else:  # prelec
        return [params_dict[k] for k in ["r", "alpha", "lamb", "beta",
                                          "palpha", "a1", "a2", "a3", "delta"]]


def _param_names(method):
    if method == "tk":
        return ["r", "alpha", "lamb", "gamma", "a1", "a2", "a3", "delta"]
    else:
        return ["r", "alpha", "lamb", "beta", "palpha", "a1", "a2", "a3", "delta"]


def _generate_mixture_data(true_params_list, pi_list, ksi_values, method, lottery, seed):
    """
    Generate multi-session pseudo data for a mixture of C clusters.
    Subjects are assigned to clusters proportionally to pi_list, then combined.
    """
    rng = np.random.default_rng(seed)
    subjects = list(ksi_values.keys())
    n        = len(subjects)

    cluster_sizes = np.round(np.array(pi_list) * n).astype(int)
    cluster_sizes[-1] = n - cluster_sizes[:-1].sum()   # ensure sum = n

    idx, dfs = 0, []
    for tp, size in zip(true_params_list, cluster_sizes):
        subj_slice = {s: ksi_values[s] for s in subjects[idx: idx + size]}
        dfs.append(generate_pseudo_data_multisession(
            true_params=tp,
            ksi_values=subj_slice,
            method=method,
            lottery=lottery,
            seed=int(rng.integers(1_000_000)),
        ))
        idx += size

    return pd.concat(dfs, ignore_index=True)


def _match_clusters(true_pi, rec_pis, rec_thetas):
    """
    Match recovered clusters to true clusters by sorting both on pi (descending).
    This resolves the label-switching ambiguity when cluster weights are distinct.
    Returns reordered (rec_pis, rec_thetas) aligned with true_pi order.
    """
    true_order = np.argsort(true_pi)[::-1]
    rec_order  = np.argsort(rec_pis)[::-1]
    # Invert rec_order so that rec_order[i] → true_order[i]
    inv = np.empty_like(rec_order)
    inv[rec_order] = true_order
    return [rec_pis[k]    for k in inv], \
           [rec_thetas[k] for k in inv]


def run_recovery_em(seed, true_params_list, pi_list, ksi_values,
                    method, lottery, c, n_restarts):
    """
    Generate multi-session pseudo data for one seed and fit the EM.
    Returns (result, df).
    """
    lotteries_t = f.transform(lottery)

    if c == 1:
        df = generate_pseudo_data_multisession(
            true_params=true_params_list[0],
            ksi_values=ksi_values,
            method=method,
            lottery=lottery,
            seed=seed,
        )
    else:
        df = _generate_mixture_data(
            true_params_list=true_params_list,
            pi_list=pi_list,
            ksi_values=ksi_values,
            method=method,
            lottery=lottery,
            seed=seed,
        )

    print(f"\n[Seed {seed}] {df['participant_label'].nunique()} subjects × "
          f"{df['lottery_id'].nunique()} lotteries × "
          f"{df['round_number'].nunique()} sessions = {len(df)} obs")

    result = run_em_multistart(
        n_restarts=n_restarts,
        method=method,
        c=c,
        y=df,
        lotteries=lotteries_t,
    )
    return result, df


def collect_recovery_rows_em(result, true_params_list, pi_list, method, seed):
    """
    Return a list of C dicts — one per cluster — each containing the seed,
    per-parameter bias (recovered − true), recovered pi, and log-likelihood.
    Clusters are matched to true clusters via pi-sorted order.
    """
    pnames = _param_names(method)
    rec_pis    = np.array(result["pis"])
    rec_thetas = list(result["thetas"])

    if len(true_params_list) > 1:
        rec_pis, rec_thetas = _match_clusters(
            np.array(pi_list), rec_pis, rec_thetas
        )

    rows = []
    for j, (tp, rv, rpi) in enumerate(zip(true_params_list, rec_thetas, rec_pis)):
        true_vec = _param_vec(tp, method)
        row = {"seed": seed, "cluster": j + 1, "rec_pi": rpi,
               "true_pi": pi_list[j], "log_likelihood": result["log_likelihood"]}
        for name, tv, ev in zip(pnames, true_vec, rv):
            row[f"bias_{name}"] = ev - tv
        rows.append(row)
    return rows


def print_summary_table_em(all_rows_per_seed, method, c, pi_list):
    """
    Print a bias table across seeds.  For C>1, one sub-table per cluster.

    all_rows_per_seed : list of lists — all_rows_per_seed[s] is the output of
                        collect_recovery_rows_em() for seed s.
    """
    pnames = _param_names(method)
    cw     = 10
    sep    = "-" * (6 + (len(pnames) + 2) * (cw + 1))

    for j in range(c):
        header = f"Cluster {j+1}  (true π = {pi_list[j]:.2f})" if c > 1 else "Single cluster"
        print(f"\n{'='*len(sep)}")
        print(f"PARAMETER RECOVERY — {header}  (bias = recovered − true)")
        print(f"{'='*len(sep)}")
        print(" ".join(
            [f"{'Seed':>6}"] +
            [f"{n:>{cw}}" for n in pnames] +
            [f"{'rec_pi':>{cw}}", f"{'LogLik':>{cw}}"]
        ))
        print(sep)

        bias_accum = {n: [] for n in pnames}
        for seed_rows in all_rows_per_seed:
            row = seed_rows[j]
            parts = [f"{row['seed']:>6}"]
            for n in pnames:
                b = row[f"bias_{n}"]
                bias_accum[n].append(abs(b))
                parts.append(f"{b:>{cw}.4f}")
            parts.append(f"{row['rec_pi']:>{cw}.4f}")
            parts.append(f"{row['log_likelihood']:>{cw}.4f}")
            print(" ".join(parts))

        print(sep)
        avg_parts = (
            [f"{'Avg':>6}"] +
            [f"{np.mean(bias_accum[n]):>{cw}.4f}" for n in pnames] +
            [f"{'':>{cw}}", f"{'':>{cw}}"]
        )
        print(" ".join(avg_parts))
        print(f"{'='*len(sep)}")


if __name__ == "__main__":
    # ── Required on Windows for ProcessPoolExecutor ──────────────────────────
    np.random.seed(SEED)

    # Resolve true parameters
    if TRUE_PARAMS_LIST is not None:
        assert len(TRUE_PARAMS_LIST) == C, \
            "TRUE_PARAMS_LIST must have one entry per cluster."
        assert TRUE_PI is not None and len(TRUE_PI) == C, \
            "TRUE_PI must be provided and have one entry per cluster."
        true_params_list = TRUE_PARAMS_LIST
        pi_list          = TRUE_PI
    else:
        true_params_list = [TRUE_PARAMS]
        pi_list          = [1.0]

    # ── Loop over seeds ──────────────────────────────────────────────────────
    all_rows_per_seed = []
    for s in ALL_SEEDS:
        result, df = run_recovery_em(
            seed=s,
            true_params_list=true_params_list,
            pi_list=pi_list,
            ksi_values=KSI_VALUES,
            method=METHOD,
            lottery=LOTTERY,
            c=C,
            n_restarts=N_RESTARTS,
        )
        all_rows_per_seed.append(
            collect_recovery_rows_em(result, true_params_list, pi_list, METHOD, s)
        )

    # ── Summary table ────────────────────────────────────────────────────────
    print_summary_table_em(all_rows_per_seed, METHOD, C, pi_list)

    # ── Ksi summary (from last seed only) ────────────────────────────────────
    print(f"\nTrue ksi      (mean ± std): "
          f"{np.mean(list(KSI_VALUES.values())):.4f} ± "
          f"{np.std(list(KSI_VALUES.values())):.4f}")
    print(f"Recovered ksi (mean ± std): "
          f"{np.mean(result['ksi']):.4f} ± "
          f"{np.std(result['ksi']):.4f}")
