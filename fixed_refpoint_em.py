"""
Fixed-reference-point mixture: four clusters, each locked to full weight on
exactly one of the four composite reference points:

    Cluster 1  R^SQ  (status quo)          a = (1, 0, 0, 0)
    Cluster 2  R^A   (anchor)               a = (0, 1, 0, 0)
    Cluster 3  R^LE  (last expectation)     a = (0, 0, 1, 0)
    Cluster 4  R^FE  (forward expectation)  a = (0, 0, 0, 1)

where (a1, a2, a3, a4=1-a1-a2-a3) are the composite reference-point weights
in Mixture._compute_ce_th.  Unlike the free-mixture model, a1/a2/a3 are not
estimated: they are pinned per cluster via degenerate L-BFGS-B bounds on the
stick-breaking slots (s1, s2, s3).  Because stick-breaking maps a one-hot
s-vector to the same one-hot a-vector, no space conversion is needed --
pinning (s1,s2,s3) to one of (1,0,0)/(0,1,0)/(0,0,1)/(0,0,0) pins
(a1,a2,a3) to the same value.

Only the remaining structural preference parameters (r, alpha_gain,
alpha_loss, lambda, beta/palpha or gamma, delta, psi) are estimated per
cluster.  The mixing weights pi (and hence cluster sizes) are estimated
freely via the normal EM E-step -- they answer "how many subjects behave as
if they use reference point X".

For clusters where a2 = a3 = 0 (R^SQ, R^FE), delta never enters the
likelihood (it only appears in the R^A / R^LE terms), so it is additionally
pinned to a constant to avoid reporting a meaningless/unidentified estimate.

Usage:
    python fixed_refpoint_em.py [--method prelec|tk|both] [--n-seeds 20]
"""

import argparse
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize
from scipy.special import logsumexp

import functions as f
from GlobalSettings import (
    GlobalMethod, GlobalLottery, GlobalSeedsSet, GlobalTol, GlobalInterMax,
    GlobalTKBounds, GlobalPrelecBounds,
)
from Mixture import compute_log_likelihoods, _ksi_mstep, write_em_results
from main import get_observed_ce

# ── Index constants (same parameter layout as two_step_em.py) ────────────────
# Prelec: [r, alpha_gain, alpha_loss, lambda, beta, palpha, s1, s2, s3, delta, psi]
_I_S1, _I_S2, _I_S3, _I_DELTA = 6, 7, 8, 9
# TK:     [r, alpha_gain, alpha_loss, lambda, gamma, a1, a2, a3, delta, psi]
_TK_A1, _TK_A2, _TK_A3, _TK_DELTA = 5, 6, 7, 8

C = 4
REFPOINT_LABELS = ["R^SQ", "R^A", "R^LE", "R^FE"]
REFPOINT_S      = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)]


def _s_indices(method):
    return (_I_S1, _I_S2, _I_S3, _I_DELTA) if method == "prelec" else \
           (_TK_A1, _TK_A2, _TK_A3, _TK_DELTA)


def _make_cluster_bounds(j, method):
    """Per-cluster bounds: pin (s1,s2,s3) to the one-hot target for cluster j.
    Also pin delta when a2=a3=0 (R^SQ, R^FE), where it is unidentified."""
    base = list(GlobalPrelecBounds if method == "prelec" else GlobalTKBounds)
    bounds = list(base)
    i1, i2, i3, i_d = _s_indices(method)
    v1, v2, v3 = REFPOINT_S[j]
    bounds[i1] = (v1, v1)
    bounds[i2] = (v2, v2)
    bounds[i3] = (v3, v3)
    if v2 == 0.0 and v3 == 0.0:
        d_lo, d_hi = base[i_d]
        d_fixed = min(max(1.0, d_lo), d_hi)
        bounds[i_d] = (d_fixed, d_fixed)
    return bounds


def _init_theta(j, method, bounds_j):
    if method == "tk":
        base  = np.array([0.05, 0.88, 0.88, 2.25, 0.65, 0.0, 0.0, 0.0, 1.0, 1.0])
        noise = np.array([0.01, 0.05, 0.05, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0, 0.05])
    else:
        base  = np.array([0.05, 0.88, 0.88, 2.25, 1.0, 0.65, 0.0, 0.0, 0.0, 1.0, 1.0])
        noise = np.array([0.01, 0.05, 0.05, 0.10, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.05])
    th = base + np.random.randn(len(base)) * noise
    lo = np.array([b[0] for b in bounds_j], dtype=float)
    hi = np.array([b[1] for b in bounds_j], dtype=float)
    return np.clip(th, lo, hi)


def _run_one_em(args):
    (seed, y, lotteries, method, max_iter, tol) = args
    np.random.seed(seed)

    subjects = sorted(y["participant_label"].unique())
    n = len(subjects)

    per_cluster_bounds = [_make_cluster_bounds(j, method) for j in range(C)]
    thetas = [_init_theta(j, method, per_cluster_bounds[j]) for j in range(C)]
    pis = np.ones(C) / C
    ksi = np.ones(n) * 0.1

    prev_ll = -np.inf
    ll = prev_ll
    for iteration in range(max_iter):
        log_L     = compute_log_likelihoods(thetas, ksi, method, C, subjects, y, lotteries)
        log_pi    = np.log(pis)
        log_joint = log_L + log_pi[np.newaxis, :]
        log_sum   = logsumexp(log_joint, axis=1, keepdims=True)
        resp      = np.exp(log_joint - log_sum)
        ll        = float(np.sum(log_sum))

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # M-step: mixing weights (= cluster sizes)
        pis = resp.mean(axis=0)

        # M-step: per-cluster structural params (s1,s2,s3 pinned via bounds)
        for j in range(C):
            bounds_j = per_cluster_bounds[j]
            resp_j   = resp[:, j]

            def neg_q(params, j=j, resp_j=resp_j):
                thetas_tmp    = list(thetas)
                thetas_tmp[j] = params
                ll_j = compute_log_likelihoods(
                    thetas_tmp, ksi, method, C, subjects, y, lotteries
                )[:, j]
                return -float(np.dot(resp_j, ll_j))

            res = minimize(neg_q, thetas[j], method="L-BFGS-B", bounds=bounds_j)
            thetas[j] = res.x

        # M-step: individual noise parameters
        ksi = _ksi_mstep(resp, thetas, ksi, method, C, subjects, y, lotteries)

    n_iter    = iteration + 1
    converged = abs(ll - prev_ll) < tol

    return {
        "thetas": thetas, "pis": pis, "ksi": ksi,
        "log_likelihood": ll, "n_iter": n_iter, "converged": converged,
        "resp": resp, "subjects": subjects, "seed": seed,
        "per_cluster_bounds": per_cluster_bounds,
    }


def count_free_params(per_cluster_bounds, n_subjects):
    """k = (free structural params summed over clusters) + (C-1) mixing weights
    + one ksi per subject.  A parameter is 'free' iff its bounds are non-degenerate."""
    k_struct = sum(1 for bounds_j in per_cluster_bounds for (lo, hi) in bounds_j if hi > lo)
    k_mix    = C - 1
    return k_struct + k_mix + n_subjects


def compute_bic(ll, k, n_obs):
    return k * np.log(n_obs) - 2.0 * ll


def print_cluster_sizes(result):
    resp   = result["resp"]
    pis    = result["pis"]
    n      = len(result["subjects"])
    hard   = np.argmax(resp, axis=1)
    soft_n = resp.sum(axis=0)

    print(f"\n{'='*72}")
    print("  FIXED REFERENCE-POINT CLUSTER SIZES")
    print(f"{'='*72}")
    print(f"  {'Cluster':<8} {'pi':>8}  {'soft N':>10}  {'hard N':>8}  {'hard %':>8}")
    print("  " + "-" * 60)
    for j in range(C):
        hard_n = int(np.sum(hard == j))
        print(f"  {REFPOINT_LABELS[j]:<8} {pis[j]:>8.4f}  {soft_n[j]:>10.2f}  "
              f"{hard_n:>8d}  {100*hard_n/n:>7.2f}%")
    print("  " + "-" * 60)
    print(f"  {'Total':<8} {'':>8}  {soft_n.sum():>10.2f}  {n:>8d}  {100.0:>7.2f}%")
    print(f"{'='*72}\n")


def run_fixed_refpoint(method, y, lotteries, seeds, n_workers=None):
    args_list = [
        (seed, y, lotteries, method, GlobalInterMax, GlobalTol)
        for seed in seeds
    ]

    results   = []
    completed = 0
    n_seeds   = len(seeds)

    print(f"\nRunning {n_seeds} EM restarts  (fixed 4 reference-point clusters, "
          f"method={method!r})")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_one_em, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            seed_done = futures[fut]
            completed += 1
            try:
                r = fut.result()
                results.append(r)
                print(f"  [{completed:>2}/{n_seeds}] seed={seed_done:>6}  "
                      f"LL={r['log_likelihood']:>12.4f}  "
                      f"{'converged' if r['converged'] else 'not converged'}",
                      flush=True)
            except Exception as exc:
                print(f"  [{completed:>2}/{n_seeds}] seed={seed_done} FAILED: {exc}",
                      flush=True)

    if not results:
        raise RuntimeError("All fixed-refpoint EM restarts failed.")

    best = max(results, key=lambda r: r["log_likelihood"])
    best["method_used"] = method
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixed 4-cluster EM: one cluster per composite reference point."
    )
    parser.add_argument("--method", "-m", default=GlobalMethod,
                         choices=["prelec", "tk", "both"])
    parser.add_argument("--n-seeds", type=int, default=20,
                         help="Number of random EM restarts (default 20).")
    args = parser.parse_args()

    OUTDIR = Path("em_outputs_fixed_refpoint")
    OUTDIR.mkdir(exist_ok=True)
    TS = datetime.now().strftime("%Y%m%d_%H%M%S")

    y         = get_observed_ce(export_excel=False)
    lotteries = f.transform(GlobalLottery)
    seeds     = GlobalSeedsSet[:args.n_seeds]

    methods = ["prelec", "tk"] if args.method == "both" else [args.method]

    for method in methods:
        t0     = time.time()
        result = run_fixed_refpoint(method, y, lotteries, seeds)

        valid  = y[y["lottery_id"].isin(lotteries.keys())]
        n_obs  = len(valid)
        k      = count_free_params(result["per_cluster_bounds"], len(result["subjects"]))
        bic    = compute_bic(result["log_likelihood"], k, n_obs)
        result["bic"]      = bic
        result["k_params"] = k
        result["n_obs"]    = n_obs

        print(f"\nBest seed: {result['seed']}   LL = {result['log_likelihood']:.4f}"
              f"   BIC = {bic:.4f}  (k={k}, n_obs={n_obs})"
              f"   {time.time()-t0:.1f}s")

        print_cluster_sizes(result)

        outfile = OUTDIR / f"fixed_refpoint_{method}_{TS}.txt"
        write_em_results(result, filepath=str(outfile))
        print(f"Saved results to: {outfile}")
