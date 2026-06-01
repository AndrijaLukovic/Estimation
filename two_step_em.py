"""
Two-step EM estimation for reference-dependent preferences.

Step 1  (session 1 only, a2=a3=delta=0 fixed)
    Data    : round_number <= 16
    Fixed   : r=0, a2=0, a3=0, delta=0
    Estimated: alpha, lambda, palpha  (preference params)
               a1                    (reference-point weight on status quo)
    Note    : In session 1, Z1=Z2=0 so a2·Z1=a3·Z2=0 identically;
              R_l = (1-a1)·EL, identifying a1 from cross-lottery CE variation.

Step 2  (all sessions, preference params fixed)
    Data    : all sessions
    Fixed   : r, alpha, lambda, palpha  (preference params from Step 1)
    Estimated: a1, a2, a3, delta  (all reference-point weights, jointly)
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
    GlobalSeedsSet, GlobalTol, GlobalInterMax, GlobalFixTau,
    GlobalTKBounds, GlobalPrelecBounds,
)
from EM_parallel import run_em_multistart
from Mixture import (
    compute_log_likelihoods, write_em_results,
    _ksi_mstep, _s_to_a, _a_to_s,
)
from main import get_observed_ce


# ── Index constants ───────────────────────────────────────────────────────────
# Prelec: [r, alpha, lambda, beta, palpha, s1, s2, s3, delta]
_I_R, _I_ALPHA, _I_LAM, _I_BETA, _I_PALPHA = 0, 1, 2, 3, 4
_I_S1, _I_S2, _I_S3, _I_DELTA = 5, 6, 7, 8
# TK:     [r, alpha, lambda, gamma, a1, a2, a3, delta]
_TK_R, _TK_ALPHA, _TK_LAM, _TK_GAMMA = 0, 1, 2, 3
_TK_A1, _TK_A2, _TK_A3, _TK_DELTA = 4, 5, 6, 7
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1
# ══════════════════════════════════════════════════════════════════════════════

def run_step1(y_full, lotteries, method=GlobalMethod, c=GlobalCluster,
              seeds=GlobalSeedsSet):
    """
    Step 1: session-1-only EM with a2=a3=delta pinned at 0.

    Patches GlobalPrelecBounds or GlobalTKBounds in-place so that child
    processes spawned by run_em_multistart inherit the modification
    (Linux fork semantics).  The patch is always restored before returning.
    """
    y_s1 = y_full[y_full["round_number"] <= 16].copy()

    if method == "prelec":
        bounds_obj         = GlobalSettings.GlobalPrelecBounds
        i_a2, i_a3, i_d   = _I_S2, _I_S3, _I_DELTA
        free_desc          = "alpha, lambda, palpha, a1"
    elif method == "tk":
        bounds_obj         = GlobalSettings.GlobalTKBounds
        i_a2, i_a3, i_d   = _TK_A2, _TK_A3, _TK_DELTA
        free_desc          = "alpha, lambda, gamma, a1"
    else:
        raise NotImplementedError(f"run_step1: unsupported method {method!r}")

    print(f"\n{'='*60}")
    print(f"  STEP 1  |  session 1 only  |  {len(y_s1)} obs")
    print(f"  Fixed: a2=a3=delta=0  (only forward-looking + status quo)")
    print(f"  Free : {free_desc}  (a4 = 1 - a1)")
    print(f"{'='*60}")

    # In session 1 (t=0): RA=RLE=0, so R_l = a4*EL = (1-a1)*EL.
    # a1 is identified from how much R_l deviates from EL across lotteries.
    # a2, a3, delta are pinned to 0 (unidentified without Z1/Z2 variation).
    orig_a2 = bounds_obj[i_a2]
    orig_a3 = bounds_obj[i_a3]
    orig_d  = bounds_obj[i_d]
    bounds_obj[i_a2] = (0.0, 0.0)
    bounds_obj[i_a3] = (0.0, 0.0)
    bounds_obj[i_d]  = (0.0, 0.0)

    try:
        result = run_em_multistart(
            seeds=seeds, method=method, c=c,
            y=y_s1, lotteries=lotteries,
            max_iter=GlobalInterMax, tol=GlobalTol,
        )
    finally:
        bounds_obj[i_a2] = orig_a2
        bounds_obj[i_a3] = orig_a3
        bounds_obj[i_d]  = orig_d

    result["step"]        = 1
    result["method_used"] = method
    result["data_note"]   = "session 1 only (round <= 16)"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2
# ══════════════════════════════════════════════════════════════════════════════

def _make_step2_bounds(theta_a, method="prelec"):
    """
    Per-cluster L-BFGS-B bounds for step 2.

    theta_a   : step-1 theta in *a-space* (output of em_mixture / run_em_multistart)
    Returns bounds list in *s-space* (the optimisation space), with all step-1
    preference parameters pinned and only (a1, a2, a3, delta) free.
    """
    if method == "prelec":
        r      = float(theta_a[_I_R])
        alpha  = float(theta_a[_I_ALPHA])
        lam    = float(theta_a[_I_LAM])
        beta   = float(theta_a[_I_BETA])
        palpha = float(theta_a[_I_PALPHA])
        d_lo, d_hi = GlobalPrelecBounds[_I_DELTA]
        return [
            (r,      r     ),   # r      – fixed
            (alpha,  alpha ),   # alpha  – fixed from Step 1
            (lam,    lam   ),   # lambda – fixed from Step 1
            (beta,   beta  ),   # beta   – fixed (always 1)
            (palpha, palpha),   # palpha – fixed from Step 1
            (0.0,    1.0   ),   # s1 slot → f2: fraction of C → a2
            (0.0,    1.0   ),   # s2 slot → f3: fraction of remaining C → a3
            (0.0,    0.0   ),   # s3 slot → unused (a1 is residual, a4 pinned)
            (d_lo,   d_hi  ),   # delta  – from GlobalPrelecBounds
        ]
    elif method == "tk":
        r     = float(theta_a[_TK_R])
        alpha = float(theta_a[_TK_ALPHA])
        lam   = float(theta_a[_TK_LAM])
        gamma = float(theta_a[_TK_GAMMA])
        d_lo, d_hi = GlobalTKBounds[_TK_DELTA]
        return [
            (r,     r    ),   # r     – fixed (= 0)
            (alpha, alpha),   # alpha – fixed from Step 1
            (lam,   lam  ),   # lambda – fixed from Step 1
            (gamma, gamma),   # gamma – fixed from Step 1
            (0.0,   1.0  ),   # s1 slot → f2: fraction of C → a2
            (0.0,   1.0  ),   # s2 slot → f3: fraction of remaining C → a3
            (0.0,   0.0  ),   # s3 slot → unused (a1 is residual, a4 pinned)
            (d_lo,  d_hi ),   # delta – from GlobalTKBounds
        ]
    else:
        raise NotImplementedError(f"_make_step2_bounds: unsupported method {method!r}")


def _a_to_s_theta(theta_a, method="prelec"):
    """Convert an a-space theta vector to s-space (optimisation space). [stick breaking]"""
    th = np.array(theta_a, dtype=float)
    if method == "prelec":
        th[_I_S1], th[_I_S2], th[_I_S3] = _a_to_s(
            th[_I_S1], th[_I_S2], th[_I_S3]
        )
    elif method == "tk":
        th[_TK_A1], th[_TK_A2], th[_TK_A3] = _a_to_s(
            th[_TK_A1], th[_TK_A2], th[_TK_A3]
        )
    return th


def _s_to_a_theta(theta_s, method="prelec"):
    """Convert an s-space theta vector to a-space (interpretable)."""
    th = np.array(theta_s, dtype=float)
    if method == "prelec":
        th[_I_S1], th[_I_S2], th[_I_S3] = _s_to_a(
            th[_I_S1], th[_I_S2], th[_I_S3]
        )
    elif method == "tk":
        th[_TK_A1], th[_TK_A2], th[_TK_A3] = _s_to_a(
            th[_TK_A1], th[_TK_A2], th[_TK_A3]
        )
    return th


def _f_to_a_theta(theta_f, C, method):
    """Convert step-2 f-space theta → interpretable a-space.

    s1 slot = f2,  s2 slot = f3,  s3 slot = 0 (unused).
      a2 = f2 * C
      a3 = f3 * (1-f2) * C
      a1 = (1-f2) * (1-f3) * C   [residual within non-forward budget C]
      a4 = 1 - C                  [pinned from Step 1]
    """
    th = np.array(theta_f, dtype=float)
    if method == "prelec":
        i1, i2, i3 = _I_S1, _I_S2, _I_S3
    else:  # tk
        i1, i2, i3 = _TK_A1, _TK_A2, _TK_A3
    f2 = float(th[i1])
    f3 = float(th[i2])
    th[i1] = (1.0 - f2) * (1.0 - f3) * C   # a1
    th[i2] = f2 * C                           # a2
    th[i3] = f3 * (1.0 - f2) * C             # a3
    return th


def _run_step2_em_once(args):
    """Single step-2 EM run (one random initialisation of f2, f3, delta).

    Designed to be called directly or submitted to ProcessPoolExecutor.
    args = (seed, thetas_s1_s, pis, ksi, resp_fixed, y, lotteries, subjects,
            method, c, per_cluster_bounds, max_iter, tol, C_per_cluster)

    C_per_cluster : list[float] of length c
        Total non-forward budget per cluster from Step 1 (= step-1 a1 = 1-a4).
        a4 = 1-C is held fixed; f2/f3 split C into a2, a3; a1 = residual.

    resp_fixed : (n, c) array or None
        If provided, cluster responsibilities are held fixed (no E-step).
    """
    (seed, thetas_s1_s, pis, ksi, resp_fixed, y, lotteries, subjects,
     method, c, per_cluster_bounds, max_iter, tol, C_per_cluster) = args

    np.random.seed(seed)

    # Indices depend on method
    if method == "prelec":
        i_a1, i_a2, i_a3, i_d = _I_S1, _I_S2, _I_S3, _I_DELTA
    else:  # tk
        i_a1, i_a2, i_a3, i_d = _TK_A1, _TK_A2, _TK_A3, _TK_DELTA

    # Initialise: randomise f2 (→a2 fraction of C) and f3 (→a3 fraction of rest)
    thetas = [np.array(th, dtype=float) for th in thetas_s1_s]
    for j in range(c):
        thetas[j][i_a1] = np.random.uniform(0.0, 1.0)  # f2
        thetas[j][i_a2] = np.random.uniform(0.0, 1.0)  # f3
        thetas[j][i_a3] = 0.0                            # unused (s3 slot fixed at 0)
        thetas[j][i_d]  = np.random.uniform(0.0, 1.0)

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
                thetas, ksi, method, c, subjects, y, lotteries,
                C_fixed_list=C_per_cluster
            )
            log_pi    = np.log(pis)
            log_joint = log_L + log_pi[np.newaxis, :]
            log_sum   = logsumexp(log_joint, axis=1, keepdims=True)
            resp      = np.exp(log_joint - log_sum)
            ll        = float(np.sum(log_sum))
        else:
            # Fixed tau: compute LL just for convergence tracking
            log_L  = compute_log_likelihoods(
                thetas, ksi, method, c, subjects, y, lotteries,
                C_fixed_list=C_per_cluster
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
                    thetas_tmp, ksi, method, c, subjects, y, lotteries,
                    C_fixed_list=C_per_cluster
                )[:, j]
                return -float(np.dot(resp_j, ll_j))

            res = minimize(neg_q, thetas[j], method="L-BFGS-B", bounds=bounds_j)
            thetas[j] = res.x

        # ── M-step: ksi ───────────────────────────────────────────────────────
        ksi = _ksi_mstep(resp, thetas, ksi, method, c, subjects, y, lotteries,
                         C_fixed_list=C_per_cluster)

    n_iter    = iteration + 1
    converged = abs(ll - prev_ll) < tol

    # Convert f-space thetas to interpretable a-space
    thetas_a = [_f_to_a_theta(th, C_per_cluster[j], method)
                for j, th in enumerate(thetas)]

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
        lenSeed = len(GlobalSeedsSet) // 2
        seeds = GlobalSeedsSet[:lenSeed]   # fewer seeds: only 4 free params per cluster (a1,a2,a3,delta)

    subjects = sorted(y_full["participant_label"].unique())

    # Step-1 thetas are in a-space; convert to s-space for the optimiser
    thetas_s1_s = [_a_to_s_theta(th, method) for th in step1_result["thetas"]]
    pis         = step1_result["pis"].copy()
    ksi         = step1_result["ksi"].copy()

    # Per-cluster bounds (pin step-1 params, free f2/f3/delta)
    per_cluster_bounds = [
        _make_step2_bounds(step1_result["thetas"][j], method)
        for j in range(c)
    ]

    # C per cluster = total non-forward budget from Step 1 (= step-1 "a1")
    # In Step 1 a2=a3=0 were forced, so step-1 a1 = a1+a2+a3 in the full model.
    if method == "prelec":
        fixed_desc = "r, alpha, lambda, palpha"
        i_C        = _I_S1
        def _pref_str(th):
            return (f"alpha={th[_I_ALPHA]:.3f}  lambda={th[_I_LAM]:.3f}  "
                    f"palpha={th[_I_PALPHA]:.3f}")
    else:  # tk
        fixed_desc = "r, alpha, lambda, gamma"
        i_C        = _TK_A1
        def _pref_str(th):
            return (f"alpha={th[_TK_ALPHA]:.3f}  lambda={th[_TK_LAM]:.3f}  "
                    f"gamma={th[_TK_GAMMA]:.3f}")

    C_per_cluster = [float(step1_result["thetas"][j][i_C]) for j in range(c)]

    tau_mode = "FIXED tau from Step 1" if fixed_tau else "tau free (full EM)"
    print(f"\n{'='*60}")
    print(f"  STEP 2  |  all sessions  |  {len(seeds)} restarts  |  {tau_mode}")
    print(f"  Fixed per cluster: {fixed_desc} + a4=1-C  (from Step 1)")
    print(f"  Free  per cluster: f2→a2, f3→a3, delta  (a1 = C-a2-a3 residual)")
    print(f"{'='*60}")
    for j in range(c):
        C_j = C_per_cluster[j]
        print(f"  C{j+1}: {_pref_str(step1_result['thetas'][j])}"
              f"  C(budget)={C_j:.3f}  a4(fixed)={1-C_j:.3f}")

    # Compute fixed responsibilities from Step 1 (using session-1-only data)
    # so the weights reflect preference-based cluster assignment
    resp_fixed = step1_result.get("resp") if fixed_tau else None

    args_list = [
        (seed, thetas_s1_s, pis.copy(), ksi.copy(), resp_fixed,
         y_full, lotteries, subjects,
         method, c, per_cluster_bounds, GlobalInterMax, GlobalTol, C_per_cluster)
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

def count_free_params(result, fixed_tau=True):
    """
    Count free parameters for BIC.

    Step 1: 3 preference params (alpha, lambda, palpha/gamma)
            + 1 reference-point param (a1) per cluster
            + (c-1) mixing weights + one ksi per subject.
    Step 2: 4 reference-point params (a1, a2, a3, delta) per cluster
            + (c-1) mixing weights if tau is free, 0 if fixed
            + one ksi per subject.
    """
    c          = len(result["thetas"])
    n_subjects = len(result["subjects"])
    step       = result.get("step", 2)

    structural = 4 * c
    if step == 1:
        mixing = c - 1
    else:
        mixing = 0 if fixed_tau else c - 1

    return structural + mixing + n_subjects


def compute_bic(result, n_obs, fixed_tau=True):
    """BIC = k * log(n) - 2 * LL."""
    k   = count_free_params(result, fixed_tau=fixed_tau)
    ll  = result["log_likelihood"]
    bic = k * np.log(n_obs) - 2.0 * ll
    return bic, k


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

    bic1 = step1.get("bic")
    bic2 = step2.get("bic")
    k1   = step1.get("k_params")
    k2   = step2.get("k_params")
    if bic1 is not None and bic2 is not None:
        print(f"  {'BIC':12}  {bic1:>18.4f}  {bic2:>22.4f}")
        print(f"  {'k (params)':12}  {k1:>18d}  {k2:>22d}")
        print(f"  {'N obs':12}  {step1.get('n_obs', '?'):>18}  {step2.get('n_obs', '?'):>22}")

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

def _run_one(method, c, y, lotteries, outdir, ts):
    """Run the full two-step pipeline for a single method and cluster count."""
    fixed_tau = not GlobalFixTau
    tau_tag   = "freetau" if GlobalFixTau else "fixedtau"
    tau_note  = ("all sessions, tau free (full EM)" if GlobalFixTau
                 else "all sessions, tau fixed from Step 1")

    valid      = y[y["lottery_id"].isin(lotteries.keys())]
    n_obs_s1   = len(valid[valid["round_number"] <= 16])
    n_obs_full = len(valid)

    t0 = time.time()

    # ── Step 1 ────────────────────────────────────────────────────────────────
    step1 = run_step1(y, lotteries, method=method, c=c, seeds=GlobalSeedsSet)
    bic1, k1 = compute_bic(step1, n_obs_s1, fixed_tau=True)
    step1["bic"] = bic1;  step1["k_params"] = k1;  step1["n_obs"] = n_obs_s1
    write_em_results(
        step1,
        filepath=str(outdir / f"two_step1_{method}_C{c}_{ts}.txt")
    )

    # ── Step 2 ────────────────────────────────────────────────────────────────
    step2 = run_step2(step1, y, lotteries, method=method, c=c,
                      fixed_tau=fixed_tau)
    step2["data_note"] = tau_note
    bic2, k2 = compute_bic(step2, n_obs_full, fixed_tau=fixed_tau)
    step2["bic"] = bic2;  step2["k_params"] = k2;  step2["n_obs"] = n_obs_full
    write_em_results(
        step2,
        filepath=str(outdir / f"two_step2_{tau_tag}_{method}_C{c}_{ts}.txt")
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    print_comparison(step1, step2, method=method)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"  GlobalFixTau = {GlobalFixTau}  →  "
          f"{'clusters free' if GlobalFixTau else 'clusters locked'}")
    print(f"  Step 1 → {outdir}/two_step1_{method}_C{c}_{ts}.txt")
    print(f"  Step 2 → {outdir}/two_step2_{tau_tag}_{method}_C{c}_{ts}.txt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-step EM estimation for reference-dependent preferences."
    )
    parser.add_argument(
        "--method", "-m",
        default=GlobalMethod,
        choices=["prelec", "tk", "both"],
        help=(
            "Probability-weighting specification. "
            "'both' runs prelec and tk sequentially. "
            f"(default from GlobalSettings: {GlobalMethod!r})"
        ),
    )
    parser.add_argument(
        "--clusters", "-c",
        type=int,
        default=GlobalCluster,
        help=f"Number of mixture clusters C. (default from GlobalSettings: {GlobalCluster})",
    )

    parser.add_argument(
        "--lotteries", "-l",
        type=int,
        default=GlobalLottery,
        help=f"Number of mixture clusters C. (default from GlobalSettings: {GlobalLottery})",
    )
    args = parser.parse_args()

    OUTDIR = Path("em_outputs_3")
    OUTDIR.mkdir(exist_ok=True)
    TS = datetime.now().strftime("%Y%m%d_%H%M%S")

    y         = get_observed_ce(export_excel=False)
    lotteries = f.transform(GlobalLottery)

    methods = ["prelec", "tk"] if args.method == "both" else [args.method]

    for method in methods:
        if len(methods) > 1:
            print(f"\n{'#'*60}")
            print(f"#  METHOD: {method.upper()}")
            print(f"{'#'*60}")
        _run_one(method, args.clusters, y, lotteries, OUTDIR, TS)
