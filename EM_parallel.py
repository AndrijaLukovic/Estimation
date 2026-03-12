"""
Parallelized EM mixture model.

Two levels of parallelism:

  1. Parallel M-step  (em_mixture_parallel):
       Within each EM iteration the per-cluster structural-parameter
       optimisations run simultaneously via ProcessPoolExecutor.
       Speedup ≈ min(C, n_cores) on the M-step portion.
       Use when C (number of clusters) is 2 or more.

  2. Parallel restarts  (run_em_multistart):
       Runs the full EM from N different random initialisations simultaneously.
       Each worker runs its own serial EM so there is no nested parallelism.
       Keeps the result with the highest final log-likelihood.
       Recommended: always prefer this over a single run — EM has local optima.

Typical usage:

    from EM_parallel import run_em_multistart
    result = run_em_multistart(
        n_restarts=8, method="prelec", c=2, y=df, lotteries=lotteries_t
    )

IMPORTANT (Windows / spawn):
    All code that creates a ProcessPoolExecutor must be protected by
    `if __name__ == "__main__":` to avoid infinite worker-spawning on Windows.
"""

import contextlib
import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

import functions as f
from Mixture import compute_log_likelihoods, C, method, lottery
from main import get_observed_ce


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker helpers
# Must live at module scope (not closures) to be picklable by ProcessPoolExecutor.
# ─────────────────────────────────────────────────────────────────────────────

def _mstep_cluster_worker(args):
    """
    Optimise the structural parameters for one cluster (M-step).

    Called in a worker process; `objective` is a local closure defined here
    and never pickled — only `args` (plain Python / numpy / pandas objects)
    need to be picklable.

    args tuple:
        j             – cluster index
        theta_init    – current parameter vector for cluster j  (1-D ndarray)
        thetas_frozen – list of current parameter vectors for ALL clusters
        ksi           – per-subject noise vector  (1-D ndarray)
        resp_col      – responsibility weights for cluster j  (1-D ndarray)
        method        – "tk" or "prelec"
        c             – number of clusters
        subjects      – ordered list of subject labels
        y_df          – filtered CE DataFrame
        lotteries     – transformed lotteries dict
        bounds        – L-BFGS-B bounds for cluster j
    """
    (j, theta_init, thetas_frozen, ksi,
     resp_col, method_arg, c, subjects, y_df, lotteries, bounds) = args

    def objective(params):
        thetas_temp    = list(thetas_frozen)
        thetas_temp[j] = params
        log_L_temp     = compute_log_likelihoods(
            thetas_temp, ksi, method_arg, c, subjects, y_df, lotteries
        )
        return -float(np.sum(resp_col * log_L_temp[:, j]))

    res = minimize(objective, theta_init, method="L-BFGS-B", bounds=bounds)
    return j, res.x


def _restart_worker(args):
    """
    Run one full serial EM from a random initialisation.

    Stdout is suppressed so parallel workers do not interleave output.
    The final log-likelihood and parameters are returned for comparison.

    args tuple:
        seed      – integer seed for np.random (controls EM initialisation)
        method    – "tk" or "prelec"
        c         – number of clusters
        y_df      – CE DataFrame (passed in; no file I/O in workers)
        lotteries – transformed lotteries dict
        max_iter  – EM iteration cap
        tol       – convergence tolerance
    """
    from Mixture import em_mixture   # local import avoids circular issues at spawn

    seed, method_arg, c, y_df, lotteries, max_iter, tol = args
    np.random.seed(seed)

    with contextlib.redirect_stdout(io.StringIO()):
        result = em_mixture(
            method=method_arg, c=c, y=y_df, lotteries=lotteries,
            max_iter=max_iter, tol=tol,
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# em_mixture_parallel — drop-in replacement for Mixture.em_mixture
# with parallelised per-cluster M-step
# ─────────────────────────────────────────────────────────────────────────────

def em_mixture_parallel(
    thetas=None, pis=None, ksi=None, subjects=None,
    method=method, c=C, y=None, lotteries=None,
    max_iter=100, tol=1e-6, n_workers=None, verbose=True,
):
    """
    EM mixture model with a parallelised M-step.

    Identical interface and output to Mixture.em_mixture(), but the per-cluster
    structural-parameter optimisations in the M-step run simultaneously via
    ProcessPoolExecutor(max_workers=n_workers).

    Parameters
    ----------
    n_workers : int or None
        Number of worker processes for the M-step.
        None → use all available CPU cores (os.cpu_count()).
    verbose : bool
        Print iteration log-likelihood (True) or run silently (False).

    Returns
    -------
    dict  {"thetas", "pis", "ksi", "log_likelihood"}  — same as em_mixture().
    """
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lottery)
    if subjects is None:
        subjects = sorted(y["participant_label"].unique())

    n = len(subjects)

    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}
    y       = y[y["lottery_id"].isin(lotteries.keys())].copy()
    y["spread"] = y["lottery_id"].map(spreads)

    # ── Initialisation ────────────────────────────────────────────────────────
    if pis is None:
        pis = np.ones(c) / c

    if thetas is None:
        if method == "tk":
            base  = np.array([0.05, 0.88, 2.25, 0.65, 0.0, 0.0, 0.0, 1.0])
            noise = np.array([0.01, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05])
            lo    = np.array([1e-4, 0.5,  1.0,  0.2,  0.0,  0.0,  0.0,  0.0])
            hi    = np.array([0.2,  1.5,  3.0,  1.0,  1.0,  1.0,  1.0,  1.0])
        else:
            base  = np.array([0.05, 0.88, 2.25, 1.0, 0.65, 0.0, 0.0, 0.0, 1.0])
            noise = np.array([0.01, 0.05, 0.10, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05])
            lo    = np.array([1e-4, 0.5,  1.0,  1.0, 0.1,  0.0,  0.0,  0.0, 0.0])
            hi    = np.array([0.2,  1.5,  3.0,  1.0, 0.8,  1.0,  1.0,  1.0, 1.0])
        thetas = [
            np.clip(base + np.random.randn(len(base)) * noise, lo, hi)
            for _ in range(c)
        ]

    if ksi is None:
        ksi = np.ones(n) * 0.1

    # Bounds per cluster (same as in Mixture.em_mixture)
    bounds_tk = [
        (1e-4, 0.2), (0.5, 1.5), (1.0, 3.0), (0.2, 1.0),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    ]
    bounds_prelec = [
        (1e-4, 0.2), (0.5, 1.5), (1.0, 3.0), (1.0, 1.0), (0.1, 0.8),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    ]
    cluster_bounds = bounds_tk if method == "tk" else bounds_prelec

    # ── EM loop ───────────────────────────────────────────────────────────────
    prev_ll = -np.inf

    for iteration in range(max_iter):

        # E-step: compute responsibilities
        log_L     = compute_log_likelihoods(thetas, ksi, method, c, subjects, y, lotteries)
        log_pi    = np.log(pis)
        log_joint = log_L + log_pi[np.newaxis, :]
        log_sum   = logsumexp(log_joint, axis=1, keepdims=True)
        resp      = np.exp(log_joint - log_sum)

        ll = float(np.sum(log_sum))
        if verbose:
            print(f"Iter {iteration:3d} | LL = {ll:.4f}")

        if abs(ll - prev_ll) < tol:
            if verbose:
                print("Converged.")
            break
        prev_ll = ll

        # M-step: mixing weights
        pis = resp.mean(axis=0)

        # M-step: structural parameters — one optimisation per cluster in parallel
        worker_args = [
            (
                j,
                thetas[j],
                list(thetas),       # snapshot: workers must not share mutable state
                ksi.copy(),
                resp[:, j].copy(),
                method,
                c,
                subjects,
                y,
                lotteries,
                cluster_bounds,
            )
            for j in range(c)
        ]

        # For C=1 skip the process-pool overhead; just call the worker directly.
        if c == 1:
            _, thetas[0] = _mstep_cluster_worker(worker_args[0])
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_mstep_cluster_worker, a): a[0]
                           for a in worker_args}
                for future in as_completed(futures):
                    j_done, new_theta = future.result()
                    thetas[j_done] = new_theta

        # M-step: individual ksi — kept serial (single joint optimisation)
        def neg_weighted_ll_ksi(ksi_params):
            log_L_tmp = compute_log_likelihoods(thetas, ksi_params, method, c, subjects, y, lotteries)
            return -float(np.sum(resp * log_L_tmp))

        res = minimize(neg_weighted_ll_ksi, ksi, method="L-BFGS-B",
                       bounds=[(1e-4, 5)] * n)
        ksi = res.x

    # ── Summary ───────────────────────────────────────────────────────────────
    param_names = {
        "tk":     ["r", "α", "λ", "γ",  "a1", "a2", "a3", "δ"],
        "prelec": ["r", "α", "λ", "β",  "pα", "a1", "a2", "a3", "δ"],
    }[method]

    if verbose:
        print(f"\n{'Cluster':<10} {'π':<8} " + "  ".join(f"{p:<8}" for p in param_names))
        print("-" * (10 + 8 + 10 * len(param_names)))
        for j in range(c):
            print(f"{j+1:<10} {pis[j]:<8.3f} " +
                  "  ".join(f"{v:<8.4f}" for v in thetas[j]))
        print(f"\nFinal LL: {ll:.4f}")

    return {"thetas": thetas, "pis": pis, "ksi": ksi, "log_likelihood": ll}


# ─────────────────────────────────────────────────────────────────────────────
# run_em_multistart — parallel restarts, each running a full serial EM
# ─────────────────────────────────────────────────────────────────────────────

def run_em_multistart(
    n_restarts=8, method=method, c=C, y=None, lotteries=None,
    max_iter=100, tol=1e-6, n_workers=None,
):
    """
    Run the EM algorithm from n_restarts different random initialisations
    in parallel and return the result with the highest log-likelihood.

    Each worker runs a full serial EM (no nested parallelism).
    Stdout from workers is suppressed; progress is printed in the main process.

    Parameters
    ----------
    n_restarts : int
        Number of independent EM runs.  8–16 is a reasonable default.
    n_workers  : int or None
        Maximum parallel workers.  None → os.cpu_count().

    Returns
    -------
    dict  {"thetas", "pis", "ksi", "log_likelihood"}  — best result found.
    """
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lottery)

    # Filter y once here so every worker gets the same clean DataFrame
    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}
    y       = y[y["lottery_id"].isin(lotteries.keys())].copy()
    y["spread"] = y["lottery_id"].map(spreads)

    seeds = np.random.randint(0, 10_000, size=n_restarts).tolist()
    worker_args = [
        (seed, method, c, y, lotteries, max_iter, tol)
        for seed in seeds
    ]

    best_result = None
    best_ll     = -np.inf
    completed   = 0

    print(f"Running {n_restarts} EM restarts in parallel (C={c}, method={method!r})...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_restart_worker, a): i
                   for i, a in enumerate(worker_args)}
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            ll     = result["log_likelihood"]
            if ll > best_ll:
                best_ll     = ll
                best_result = result
                print(f"  [{completed}/{n_restarts}] New best LL: {best_ll:.4f}")
            else:
                print(f"  [{completed}/{n_restarts}] LL = {ll:.4f}",
                      end="\r", flush=True)

    print(f"\nBest LL across {n_restarts} restarts: {best_ll:.4f}")

    # Re-print the winning cluster summary
    param_names = {
        "tk":     ["r", "α", "λ", "γ",  "a1", "a2", "a3", "δ"],
        "prelec": ["r", "α", "λ", "β",  "pα", "a1", "a2", "a3", "δ"],
    }[method]
    print(f"\n{'Cluster':<10} {'π':<8} " + "  ".join(f"{p:<8}" for p in param_names))
    print("-" * (10 + 8 + 10 * len(param_names)))
    for j in range(c):
        print(f"{j+1:<10} {best_result['pis'][j]:<8.3f} " +
              "  ".join(f"{v:<8.4f}" for v in best_result['thetas'][j]))

    return best_result


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Required on Windows: all multiprocessing code must be inside this guard.

    y          = get_observed_ce(export_excel=False)
    lotteries  = f.transform(lottery)

    result = run_em_multistart(
        n_restarts=8,
        method=method,
        c=C,
        y=y,
        lotteries=lotteries,
    )

    print("\nFinal parameters:")
    param_names = {
        "tk":     ["r", "alpha", "lamb", "gamma", "a1", "a2", "a3", "delta"],
        "prelec": ["r", "alpha", "lamb", "beta",  "palpha", "a1", "a2", "a3", "delta"],
    }[method]
    for j in range(C):
        print(f"\n  Cluster {j+1} (π={result['pis'][j]:.3f}):")
        for name, val in zip(param_names, result["thetas"][j]):
            print(f"    {name:<10} = {val:.4f}")
    print(f"\nFinal LL: {result['log_likelihood']:.4f}")
