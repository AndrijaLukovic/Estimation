"""
Parallelized EM mixture model.

Two levels of parallelism:

  1. Parallel M-step  (em_mixture_parallel):
       Within each EM iteration the per-cluster structural-parameter
       optimisations run simultaneously via ProcessPoolExecutor.
       A single pool is kept alive for the full EM run to avoid
       repeated process-spawn overhead on Windows.
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

"""

import contextlib
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from GlobalSettings import GlobalMethod, GlobalLottery, GlobalCluster, GlobalStarts, GlobalTol, GlobalPrelecBounds, GlobalTKBounds, GlobalInterMax

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

import functions as f
from Mixture import compute_log_likelihoods, C, method, lottery, write_em_results
from main import get_observed_ce


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker helpers
# Must live at module scope (not closures) to be picklable by ProcessPoolExecutor.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_single_cluster_log_L(j, params, thetas_frozen, ksi,
                                   method_arg, c, subjects, y_df, lotteries):

    thetas_temp    = list(thetas_frozen)
    thetas_temp[j] = params
    # compute_log_likelihoods returns (n, C); we only need column j
    log_L_full = compute_log_likelihoods(
        thetas_temp, ksi, method_arg, c, subjects, y_df, lotteries
    )
    return log_L_full[:, j]


def _mstep_cluster_worker(args):
    """
    Optimise the structural parameters for one cluster (M-step).

    The objective re-evaluates only cluster j's log-likelihood at each step
    (via _compute_single_cluster_log_L) — other clusters are not recomputed.

    args tuple:
        j             – cluster index
        theta_init    – current parameter vector for cluster j  (1-D ndarray)
        thetas_frozen – list of parameter vectors for ALL clusters (snapshot)
        ksi           – per-subject noise vector  (1-D ndarray)
        resp_col      – responsibility weights for cluster j  (1-D ndarray)
        method_arg    – "tk" or "prelec"
        c             – number of clusters
        subjects      – ordered list of subject labels
        y_df          – filtered CE DataFrame (with "spread" column)
        lotteries     – transformed lotteries dict
        bounds        – L-BFGS-B bounds for cluster j
    """
    (j, theta_init, thetas_frozen, ksi,
     resp_col, method_arg, c, subjects, y_df, lotteries, bounds) = args

    def objective(params):
        log_L_j = _compute_single_cluster_log_L(
            j, params, thetas_frozen, ksi, method_arg, c, subjects, y_df, lotteries
        )
        return -float(np.sum(resp_col * log_L_j))

    res = minimize(objective, theta_init, method="L-BFGS-B", bounds=bounds)
    return j, res.x


def _restart_worker(args):
    """
    Run one full serial EM from a random initialisation.

    Captures stdout into a buffer and returns it alongside the result so the
    main process can print per-iteration diagnostics once the worker completes.

    args tuple:
        seed       – integer seed for np.random (controls EM initialisation)
        method     – "tk" or "prelec"
        c          – number of clusters
        y_df       – already-filtered CE DataFrame
        lotteries  – transformed lotteries dict
        max_iter   – EM iteration cap
        tol        – convergence tolerance
        restart_id – 1-based restart index shown in progress prefix
    """
    from Mixture import em_mixture   # local import avoids circular issues at spawn

    seed, method_arg, c, y_df, lotteries, max_iter, tol, restart_id = args
    np.random.seed(seed)

    t0  = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = em_mixture(
            method=method_arg, c=c, y=y_df, lotteries=lotteries,
            max_iter=max_iter, tol=tol,
        )
    elapsed = time.time() - t0
    return result, buf.getvalue(), restart_id, elapsed


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
    structural-parameter optimisations in the M-step run simultaneously.

    A single ProcessPoolExecutor is kept alive for the entire EM run to avoid
    repeated process-spawn overhead (significant on Windows).  For C=1 the
    pool is skipped entirely.

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
    cluster_bounds = GlobalTKBounds if method == "tk" else GlobalPrelecBounds

    # ── EM loop ───────────────────────────────────────────────────────────────
    # Keep a single pool alive across all iterations to avoid per-iteration
    # process-spawn overhead (especially costly on Windows with spawn start).
    # For C=1 the pool is skipped; the worker is called directly.
    prev_ll  = -np.inf
    iter_log = []
    executor = None if c == 1 else ProcessPoolExecutor(max_workers=n_workers)

    try:
        for iteration in range(max_iter):

            t_iter_start = time.time()

            # E-step: compute responsibilities
            log_L     = compute_log_likelihoods(thetas, ksi, method, c, subjects, y, lotteries)
            log_pi    = np.log(pis)
            log_joint = log_L + log_pi[np.newaxis, :]
            log_sum   = logsumexp(log_joint, axis=1, keepdims=True)
            resp      = np.exp(log_joint - log_sum)

            ll          = float(np.sum(log_sum))
            soft_n      = resp.sum(axis=0)
            improvement = ll - prev_ll

            if verbose:
                print(f"Iter {iteration:3d} | LL = {ll:.4f}  Improvement = {improvement:+.4f}")

            if abs(ll - prev_ll) < tol:
                iter_log.append({
                    "iteration": iteration + 1,
                    "log_likelihood": ll,
                    "improvement": improvement,
                    "soft_n": soft_n.copy(),
                    "pis": pis.copy(),
                    "elapsed": time.time() - t_iter_start,
                })
                if verbose:
                    print("Converged.")
                break
            prev_ll = ll

            # M-step: mixing weights
            pis = resp.mean(axis=0)

            # M-step: structural parameters — one optimisation per cluster
            # Each worker recomputes only its own cluster's log-likelihood.
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

            if executor is None:
                # C=1: no pool, call directly
                _, thetas[0] = _mstep_cluster_worker(worker_args[0])
            else:
                futures = {executor.submit(_mstep_cluster_worker, a): a[0]
                           for a in worker_args}
                for future in as_completed(futures):
                    j_done, new_theta = future.result()
                    thetas[j_done] = new_theta

            # M-step: individual ksi — single joint optimisation, kept serial
            def neg_weighted_ll_ksi(ksi_params):
                log_L_tmp = compute_log_likelihoods(
                    thetas, ksi_params, method, c, subjects, y, lotteries
                )
                return -float(np.sum(resp * log_L_tmp))

            res = minimize(neg_weighted_ll_ksi, ksi, method="L-BFGS-B",
                           bounds=[(1e-4, 5)] * n)
            ksi = res.x

            iter_log.append({
                "iteration": iteration + 1,
                "log_likelihood": ll,
                "improvement": improvement,
                "soft_n": soft_n.copy(),
                "pis": pis.copy(),
                "elapsed": time.time() - t_iter_start,
            })

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

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

    n_iter    = iteration + 1
    converged = abs(ll - prev_ll) < tol
    return {"thetas": thetas, "pis": pis, "ksi": ksi, "log_likelihood": ll,
            "n_iter": n_iter, "converged": converged, "seed": None,
            "resp": resp, "iter_log": iter_log, "subjects": subjects}


# ─────────────────────────────────────────────────────────────────────────────
# run_em_multistart — parallel restarts, each running a full serial EM
# ─────────────────────────────────────────────────────────────────────────────

def run_em_multistart(
    n_restarts=8, method=method, c=C, y=None, lotteries=None,
    max_iter=GlobalInterMax, tol=GlobalTol, n_workers=None,
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

    # Filter y once before sending to workers — em_mixture would filter again
    # anyway, but sending the pre-filtered DataFrame avoids redundant work and
    # keeps each worker's DataFrame smaller.
    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}
    y       = y[y["lottery_id"].isin(lotteries.keys())].copy()
    y["spread"] = y["lottery_id"].map(spreads)

    # Use full int32 range to avoid seed collisions for large n_restarts
    rng   = np.random.default_rng()
    seeds = rng.integers(0, 2**31, size=n_restarts).tolist()

    worker_args = [
        (seed, method, c, y, lotteries, max_iter, tol, i + 1)
        for i, seed in enumerate(seeds)
    ]

    best_result = None
    best_ll     = -np.inf
    completed   = 0

    print(f"Running {n_restarts} EM restarts in parallel (C={c}, method={method!r})...")
    print(f"  Output prints as each restart finishes.\n", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_restart_worker, a): i
                   for i, a in enumerate(worker_args)}
        for future in as_completed(futures):
            completed += 1
            result, log_output, restart_id, elapsed = future.result()
            ll = result["log_likelihood"]

            # Print the captured per-iteration log with restart prefix
            prefix = f"[R{restart_id:02d}]"
            for line in log_output.splitlines():
                print(f"  {prefix} {line}", flush=True)
            print(f"  {prefix} DONE in {elapsed:.1f}s | LL = {ll:.4f}", flush=True)

            if ll > best_ll:
                best_ll     = ll
                best_result = result
                print(f"  >>> [{completed}/{n_restarts}] New best LL: {best_ll:.4f}\n", flush=True)
            else:
                print(f"      [{completed}/{n_restarts}] completed\n", flush=True)

    if best_result is None:
        raise RuntimeError("All EM restarts failed to return a result.")

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

    best_result["method_used"] = method
    return best_result


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Required on Windows: all multiprocessing code must be inside this guard.

    y          = get_observed_ce(export_excel=False)
    lotteries  = f.transform(lottery)
    subjects   = sorted(y["participant_label"].unique())

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

    # Attach subjects if not already set by the best em_mixture run
    if not result.get("subjects"):
        result["subjects"] = subjects

    write_em_results(result, filepath="em_results.txt")
