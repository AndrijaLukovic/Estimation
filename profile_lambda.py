"""
Profile log-likelihood over lambda (loss-aversion parameter).

For each fixed lambda in LAMBDA_GRID, the EM is run from N_SEEDS different
initialisations and the best log-likelihood is recorded.  All other model
parameters are freely optimised.

Design note
-----------
We call em_mixture() directly (in-process, serial) rather than going through
run_em_multistart (which spawns subprocesses that re-import GlobalSettings and
would not see the in-place lambda patch).  Patching GlobalPrelecBounds[LAM_IDX]
in-place is safe here because the list object is shared between GlobalSettings
and every module that imported it via `from GlobalSettings import GlobalPrelecBounds`.

Usage:
    python profile_lambda.py
"""

import os
import time
import numpy as np

import functions as f
import GlobalSettings                        # module reference — lets us patch in-place
from GlobalSettings import GlobalLottery, GlobalMethod, GlobalCluster
from Mixture import em_mixture
from main import get_observed_ce

# ── Configuration ─────────────────────────────────────────────────────────────
LAMBDA_GRID = [1.0, 1.5, 2.25, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
SEEDS       = [10, 30, 80, 156]   # 4 restarts per lambda point (best seeds from prior run)
MAX_ITER    = 100
TOL         = 1e-3
METHOD      = GlobalMethod
C           = GlobalCluster
OUTPUT_FILE = "em_outputs/profile_lambda.txt"

# Index of lambda in GlobalPrelecBounds: [r, alpha, lambda, beta, palpha, s1, s2, s3, delta]
LAM_IDX = 2
# ──────────────────────────────────────────────────────────────────────────────


def run_profile():
    y         = get_observed_ce(export_excel=False)
    lotteries = f.transform(GlobalLottery)

    orig_bound = GlobalSettings.GlobalPrelecBounds[LAM_IDX]   # save for restore

    profile = []   # list of (lambda, best_ll, n_converged, best_seed)

    print(f"\nProfile LL   method={METHOD!r}   C={C}   seeds={SEEDS}")
    print(f"  {len(LAMBDA_GRID)} lambda points × {len(SEEDS)} restarts × max {MAX_ITER} iter\n")
    print(f"  {'Lambda':>8}  {'Best LL':>14}  {'Conv':>6}  {'Best seed':>10}  {'Time':>8}")
    print("  " + "-" * 56)

    t_total = time.time()

    for lam in LAMBDA_GRID:
        # Pin lambda: L-BFGS-B treats lb == ub as a fixed (equality-constrained) value
        GlobalSettings.GlobalPrelecBounds[LAM_IDX] = (lam, lam)

        best_ll     = -np.inf
        best_seed   = None
        n_converged = 0
        t0          = time.time()

        for seed in SEEDS:
            np.random.seed(seed)
            try:
                res = em_mixture(
                    method=METHOD, c=C,
                    y=y, lotteries=lotteries,
                    max_iter=MAX_ITER, tol=TOL,
                    seed=seed, verbose=False,
                )
                if res["log_likelihood"] > best_ll:
                    best_ll   = res["log_likelihood"]
                    best_seed = seed
                if res["converged"]:
                    n_converged += 1
            except Exception as exc:
                print(f"  [lambda={lam}  seed={seed}] ERROR: {exc}")

        elapsed = time.time() - t0
        profile.append((lam, best_ll, n_converged, best_seed))
        print(f"  {lam:8.2f}  {best_ll:14.4f}  {n_converged:>2}/{len(SEEDS)}  "
              f"{str(best_seed):>10}  {elapsed:7.1f}s", flush=True)

    # Restore original bound
    GlobalSettings.GlobalPrelecBounds[LAM_IDX] = orig_bound

    elapsed_total = time.time() - t_total
    print(f"\n  Total time: {elapsed_total/60:.1f} min")

    # ── Find optimum ──────────────────────────────────────────────────────────
    best     = max(profile, key=lambda x: x[1])
    ll_max   = best[1]
    print(f"  Optimal lambda: {best[0]:.2f}   LL = {ll_max:.4f}")

    # ── ASCII chart ───────────────────────────────────────────────────────────
    lls    = [p[1] for p in profile]
    ll_min = min(lls)
    span   = max(ll_max - ll_min, 1e-6)
    width  = 40

    print("\n  Profile LL  (bar length = relative fit)")
    print(f"  {'Lambda':>8}  {'LL':>12}  {'ΔLL vs max':>12}  Chart")
    print("  " + "-" * 72)
    for lam, ll, nc, bs in profile:
        bar = int((ll - ll_min) / span * width)
        delta = ll - ll_max            # ≤ 0; maximum is 0.0
        mark  = "  ← max" if ll == ll_max else ""
        print(f"  {lam:8.2f}  {ll:12.4f}  {delta:+12.4f}  {'█' * bar}{mark}")

    # ── Save to file ──────────────────────────────────────────────────────────
    os.makedirs("em_outputs", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write(f"Profile LL over lambda\n")
        fh.write(f"method={METHOD}  C={C}  seeds={SEEDS}  max_iter={MAX_ITER}  tol={TOL}\n\n")
        fh.write(f"{'Lambda':>10}  {'Best_LL':>14}  {'Delta_LL':>10}  "
                 f"{'Converged':>10}  {'Best_seed':>10}\n")
        fh.write("-" * 60 + "\n")
        for lam, ll, nc, bs in profile:
            fh.write(f"{lam:10.2f}  {ll:14.4f}  {ll - ll_max:+10.4f}  "
                     f"{nc:>4}/{len(SEEDS)}  {str(bs):>10}\n")
    print(f"\n  Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    run_profile()
