"""
Two-step EM estimation for reference-dependent preferences.

Step 1  (session 1 only, a2=a3=0 fixed)
    Data    : round_number < 15
    Fixed   : r=0, a2=0, a3=0
    Estimated: alpha, lambda, palpha, a1  (+delta, irrelevant in session 1)
    Identifies from: cross-lottery CE variation (different gain/loss profiles)

Step 2  (all sessions, preference params fixed)
    Data    : all sessions
    Fixed   : r, alpha, lambda, palpha, a1  from Step 1
    Estimated: a2, a3, delta  (reference-point dynamics)
    Identifies from: within-subject CE changes as Z1/Z2 vary across sessions

Usage:
    python two_step_em.py
"""

import time
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize
from scipy.special import logsumexp

import functions as f
import GlobalSettings
from GlobalSettings import (
    GlobalMethod, GlobalLottery, GlobalCluster,
    GlobalSeedsSet, GlobalTol, GlobalInterMax, GlobalFixTau
)
from EM_parallel import run_em_multistart
from Mixture import (
    compute_log_likelihoods, write_em_results,
    _ksi_mstep, _s_to_a, _a_to_s,
)
from main import get_observed_ce


# ── Index constants (Prelec parameter order) ─────────────────────────────────
# [r, alpha, lambda, beta, palpha, s1, s2, s3, delta]
_I_R, _I_ALPHA, _I_LAM, _I_BETA, _I_PALPHA = 0, 1, 2, 3, 4
_I_S1, _I_S2, _I_S3, _I_DELTA = 5, 6, 7, 8
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1
# ══════════════════════════════════════════════════════════════════════════════

def run_step1(y_full, lotteries, method=GlobalMethod, c=GlobalCluster,
              seeds=GlobalSeedsSet):
    """
    Step 1: session-1-only EM with a2=a3 pinned at 0.

    Patches GlobalPrelecBounds in-place so that child processes spawned by
    run_em_multistart inherit the modification (Linux fork semantics).
    The patch is always restored before returning.
    """
    y_s1 = y_full[y_full["round_number"] < 15].copy()

    print(f"\n{'='*60}")
    print(f"  STEP 1  |  session 1 only  |  {len(y_s1)} obs")
    print(f"  Fixed: a2=a3=delta=0  (only forward-looking + SQ components)")
    print(f"  Free : alpha, lambda, palpha, a1  (a4 = 1 - a1)")
    print(f"{'='*60}")

    # In session 1 (t=0): RA=RLE=0, so R_l = a4*EL = (1-a1)*EL.
    # a1 is identified from how much R_l deviates from EL across lotteries.
    # a2, a3, delta are pinned to 0 (unidentified without Z1/Z2 variation).
    orig_s2    = GlobalSettings.GlobalPrelecBounds[_I_S2]
    orig_s3    = GlobalSettings.GlobalPrelecBounds[_I_S3]
    orig_delta = GlobalSettings.GlobalPrelecBounds[_I_DELTA]
    GlobalSettings.GlobalPrelecBounds[_I_S2]    = (0.0, 0.0)
    GlobalSettings.GlobalPrelecBounds[_I_S3]    = (0.0, 0.0)
    GlobalSettings.GlobalPrelecBounds[_I_DELTA] = (0.0, 0.0)

    try:
        result = run_em_multistart(
            seeds=seeds, method=method, c=c,
            y=y_s1, lotteries=lotteries,
            max_iter=GlobalInterMax, tol=GlobalTol,
        )
    finally:
        GlobalSettings.GlobalPrelecBounds[_I_S2]    = orig_s2
        GlobalSettings.GlobalPrelecBounds[_I_S3]    = orig_s3
        GlobalSettings.GlobalPrelecBounds[_I_DELTA] = orig_delta

    result["step"]        = 1
    result["method_used"] = method
    result["data_note"]   = "session 1 only (round < 15)"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2
# ══════════════════════════════════════════════════════════════════════════════

def _make_step2_bounds(theta_a, method="prelec"):
    """
    Per-cluster L-BFGS-B bounds for step 2.

    theta_a   : step-1 theta in *a-space* (output of em_mixture / run_em_multistart)
    Returns bounds list in *s-space* (the optimisation space), with all step-1
    parameters pinned and only (s2, s3, delta) free.
    """
    if method != "prelec":
        raise NotImplementedError("two_step_em currently supports prelec only")

    r      = float(theta_a[_I_R])
    alpha  = float(theta_a[_I_ALPHA])
    lam    = float(theta_a[_I_LAM])
    beta   = float(theta_a[_I_BETA])
    palpha = float(theta_a[_I_PALPHA])

    # Step 1 fixed a1=a2=a3=delta=0, so Step 2 estimates ALL reference-point
    # weights freely. Only the preference params (alpha, lambda, palpha) are
    # carried over from Step 1.
    return [
        (r,      r     ),   # r      – fixed (= 0)
        (alpha,  alpha ),   # alpha  – fixed from Step 1
        (lam,    lam   ),   # lambda – fixed from Step 1
        (beta,   beta  ),   # beta   – fixed (always 1)
        (palpha, palpha),   # palpha – fixed from Step 1
        (0.0,    1.0   ),   # s1     – FREE → a1
        (0.0,    1.0   ),   # s2     – FREE → a2
        (0.0,    1.0   ),   # s3     – FREE → a3
        (0.0,    1.0   ),   # delta  – FREE
    ]


def _a_to_s_theta(theta_a, method="prelec"):
    """Convert an a-space theta vector to s-space (optimisation space)."""
    th = np.array(theta_a, dtype=float)
    if method == "prelec":
        th[_I_S1], th[_I_S2], th[_I_S3] = _a_to_s(
            th[_I_S1], th[_I_S2], th[_I_S3]
        )
    return th


def _s_to_a_theta(theta_s, method="prelec"):
    """Convert an s-space theta vector to a-space (interpretable)."""
    th = np.array(theta_s, dtype=float)
    if method == "prelec":
        th[_I_S1], th[_I_S2], th[_I_S3] = _s_to_a(
            th[_I_S1], th[_I_S2], th[_I_S3]
        )
    return th


def _run_step2_em_once(args):
    """Single step-2 EM run (one random initialisation of a2, a3, delta).

    Designed to be called directly or submitted to ProcessPoolExecutor.
    args = (seed, thetas_s1_s, pis, ksi, resp_fixed, y, lotteries, subjects,
            method, c, per_cluster_bounds, max_iter, tol)

    resp_fixed : (n, c) array or None
        If provided, cluster responsibilities are held fixed (no E-step).
        This implements the strict two-step: cluster assignments from Step 1
        are frozen; Step 2 only estimates a2, a3, delta conditional on them.
        If None, a full EM is run and cluster assignments can drift.
    """
    (seed, thetas_s1_s, pis, ksi, resp_fixed, y, lotteries, subjects,
     method, c, per_cluster_bounds, max_iter, tol) = args

    np.random.seed(seed)

    # Initialise free reference-point params (s1, s2, s3, delta) randomly
    thetas = [np.array(th, dtype=float) for th in thetas_s1_s]
    for j in range(c):
        thetas[j][_I_S1]    = np.random.uniform(0.0, 0.3)
        thetas[j][_I_S2]    = np.random.uniform(0.0, 0.3)
        thetas[j][_I_S3]    = np.random.uniform(0.0, 0.3)
        thetas[j][_I_DELTA] = np.random.uniform(0.0, 0.5)

    y = y[y["lottery_id"].isin(lotteries.keys())].copy()
    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}
    y["spread"] = y["lottery_id"].map(spreads)

    # Use fixed responsibilities if provided (strict two-step)
    resp = resp_fixed.copy() if resp_fixed is not None else None

    prev_ll = -np.inf

    for iteration in range(max_iter):

        # ── E-step (skipped when tau is fixed from Step 1) ────────────────────
        if resp_fixed is None:
            log_L     = compute_log_likelihoods(
                thetas, ksi, method, c, subjects, y, lotteries
            )
            log_pi    = np.log(pis)
            log_joint = log_L + log_pi[np.newaxis, :]
            log_sum   = logsumexp(log_joint, axis=1, keepdims=True)
            resp      = np.exp(log_joint - log_sum)
            ll        = float(np.sum(log_sum))
        else:
            # Fixed tau: compute LL just for convergence tracking
            log_L  = compute_log_likelihoods(
                thetas, ksi, method, c, subjects, y, lotteries
            )
            log_pi = np.log(pis)
            ll     = float(np.sum(
                logsumexp(log_L + log_pi[np.newaxis, :], axis=1)
            ))

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # ── M-step: mixing weights (skip when tau is fixed) ───────────────────
        if resp_fixed is None:
            pis = resp.mean(axis=0)

        # ── M-step: per-cluster structural params ─────────────────────────────
        for j in range(c):
            bounds_j = per_cluster_bounds[j]
            resp_j   = resp[:, j]

            def neg_q(params, j=j, resp_j=resp_j):
                thetas_tmp    = list(thetas)
                thetas_tmp[j] = params
                ll_j = compute_log_likelihoods(
                    thetas_tmp, ksi, method, c, subjects, y, lotteries
                )[:, j]
                return -float(np.dot(resp_j, ll_j))

            res = minimize(neg_q, thetas[j], method="L-BFGS-B", bounds=bounds_j)
            thetas[j] = res.x

        # ── M-step: ksi ───────────────────────────────────────────────────────
        ksi = _ksi_mstep(resp, thetas, ksi, method, c, subjects, y, lotteries)

    n_iter    = iteration + 1
    converged = abs(ll - prev_ll) < tol

    # Convert optimised s-space thetas to interpretable a-space
    thetas_a = [_s_to_a_theta(th, method) for th in thetas]

    return {
        "thetas": thetas_a, "pis": pis, "ksi": ksi,
        "log_likelihood": ll, "n_iter": n_iter, "converged": converged,
        "resp": resp, "subjects": subjects, "seed": seed,
    }


def run_step2(step1_result, y_full, lotteries, method=GlobalMethod,
              c=GlobalCluster, seeds=None, n_workers=None, fixed_tau=True):
    """
    Step 2: all-session EM with preference params fixed at step-1 values.

    Runs multiple random initialisations of (a2, a3, delta) in parallel and
    returns the best result by log-likelihood.
    """
    if seeds is None:
        seeds = GlobalSeedsSet[:16]   # fewer seeds: only 3 free params per cluster

    subjects = sorted(y_full["participant_label"].unique())

    # Step-1 thetas are in a-space; convert to s-space for the optimiser
    thetas_s1_s = [_a_to_s_theta(th, method) for th in step1_result["thetas"]]
    pis         = step1_result["pis"].copy()
    ksi         = step1_result["ksi"].copy()

    # Per-cluster bounds (pin step-1 params, free s2/s3/delta)
    per_cluster_bounds = [
        _make_step2_bounds(step1_result["thetas"][j], method)
        for j in range(c)
    ]

    tau_mode = "FIXED tau from Step 1" if fixed_tau else "tau free (full EM)"
    print(f"\n{'='*60}")
    print(f"  STEP 2  |  all sessions  |  {len(seeds)} restarts  |  {tau_mode}")
    print(f"  Fixed per cluster: r, alpha, lambda, palpha  (from Step 1)")
    print(f"  Free  per cluster: a1, a2, a3, delta")
    print(f"{'='*60}")
    for j in range(c):
        a1 = float(step1_result["thetas"][j][_I_S1])
        print(f"  C{j+1}: alpha={step1_result['thetas'][j][_I_ALPHA]:.3f}  "
              f"lambda={step1_result['thetas'][j][_I_LAM]:.3f}  "
              f"palpha={step1_result['thetas'][j][_I_PALPHA]:.3f}  "
              f"a1={a1:.3f}")

    # Compute fixed responsibilities from Step 1 (using session-1-only data)
    # so the weights reflect preference-based cluster assignment
    resp_fixed = step1_result.get("resp") if fixed_tau else None

    args_list = [
        (seed, thetas_s1_s, pis.copy(), ksi.copy(), resp_fixed,
         y_full, lotteries, subjects,
         method, c, per_cluster_bounds, GlobalInterMax, GlobalTol)
        for seed in seeds
    ]

    results   = []
    completed = 0
    n_seeds   = len(seeds)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_step2_em_once, a): a[0]
                   for a in args_list}
        for fut in as_completed(futures):
            seed_done = futures[fut]
            completed += 1
            try:
                r = fut.result()
                results.append(r)
                print(f"  [{completed:>2}/{n_seeds}] seed={seed_done:>5}  "
                      f"LL={r['log_likelihood']:>12.4f}  "
                      f"{'converged' if r['converged'] else 'not converged'}",
                      flush=True)
            except Exception as exc:
                print(f"  [{completed:>2}/{n_seeds}] seed={seed_done} FAILED: {exc}",
                      flush=True)

    if not results:
        raise RuntimeError("All step-2 restarts failed.")

    best = max(results, key=lambda r: r["log_likelihood"])
    best["step"]        = 2
    best["method_used"] = method
    best["data_note"]   = "all sessions"
    return best


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison(step1, step2, method="prelec"):
    """Side-by-side parameter table: step 1 vs step 2."""
    names = {
        "prelec": ["r", "alpha", "lambda", "beta", "palpha",
                   "a1", "a2", "a3", "delta"],
        "tk":     ["r", "alpha", "lambda", "gamma",
                   "a1", "a2", "a3", "delta"],
    }[method]

    c  = len(step1["thetas"])
    W  = 70
    print(f"\n{'='*W}")
    print("  TWO-STEP COMPARISON")
    print(f"{'='*W}")
    print(f"  {'':12}  {'Step 1 (S1 only)':>18}  {'Step 2 (all sessions)':>22}")
    print(f"  {'LL':12}  {step1['log_likelihood']:>18.4f}  {step2['log_likelihood']:>22.4f}")
    print(f"  {'N subjects':12}  {len(step1['subjects']):>18}  {len(step2['subjects']):>22}")
    print()

    for j in range(c):
        print(f"  {'─'*W}")
        print(f"  Cluster {j+1}   π(S1)={step1['pis'][j]:.3f}   π(S2)={step2['pis'][j]:.3f}")
        print(f"  {'─'*W}")
        print(f"  {'Param':12}  {'Step 1':>18}  {'Step 2':>22}  {'Δ':>8}")
        print(f"  {'-'*W}")
        for k, name in enumerate(names):
            v1   = step1["thetas"][j][k]
            v2   = step2["thetas"][j][k]
            diff = v2 - v1
            flag = "  ***" if abs(diff) > 0.05 else ""
            print(f"  {name:<12}  {v1:>18.4f}  {v2:>22.4f}  {diff:>+8.4f}{flag}")
        print()

    print(f"{'='*W}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    METHOD   = GlobalMethod
    C        = GlobalCluster
    OUTDIR   = Path("em_outputs")
    OUTDIR.mkdir(exist_ok=True)
    TS       = datetime.now().strftime("%Y%m%d_%H%M%S")

    y         = get_observed_ce(export_excel=False)
    lotteries = f.transform(GlobalLottery)

    t0 = time.time()

    # ── Step 1 ────────────────────────────────────────────────────────────────
    step1 = run_step1(y, lotteries, method=METHOD, c=C, seeds=GlobalSeedsSet)
    write_em_results(
        step1,
        filepath=str(OUTDIR / f"two_step1_{METHOD}_C{C}_{TS}.txt")
    )

    # ── Step 2 ────────────────────────────────────────────────────────────────
    # GlobalFixTau=True  → clusters can shift   (fixed_tau=False)
    # GlobalFixTau=False → clusters locked from Step 1 (fixed_tau=True)
    fixed_tau   = not GlobalFixTau
    tau_tag     = "freetau"  if GlobalFixTau else "fixedtau"
    tau_note    = "all sessions, tau free (full EM)" if GlobalFixTau \
                  else "all sessions, tau fixed from Step 1"

    step2 = run_step2(step1, y, lotteries, method=METHOD, c=C,
                      fixed_tau=fixed_tau)
    step2["data_note"] = tau_note
    write_em_results(
        step2,
        filepath=str(OUTDIR / f"two_step2_{tau_tag}_{METHOD}_C{C}_{TS}.txt")
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    print_comparison(step1, step2, method=METHOD)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"  GlobalFixTau = {GlobalFixTau}  →  {'clusters free' if GlobalFixTau else 'clusters locked'}")
    print(f"  Step 1 → em_outputs/two_step1_{METHOD}_C{C}_{TS}.txt")
    print(f"  Step 2 → em_outputs/two_step2_{tau_tag}_{METHOD}_C{C}_{TS}.txt")
