import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import logsumexp


import functions as f
from lotteries import lotteries_full, one, all_high_stake, all_low_stake
from main import get_observed_ce
from GlobalSettings import GlobalTKBounds, GlobalPrelecBounds, GlobalMethod, GlobalLottery, GlobalCluster, GlobalTol
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# ── Lottery set to estimate on ──────────────────────────────────────────────
# Switch between all_low_stake / all_high_stake / lotteries_full
lottery = GlobalLottery
# ────────────────────────────────────────────────────────────────────────────

### Choice of the method "prelec" or "tk"
prob_weighter = "prelec"


### Number of clusters
C = GlobalCluster

### Method
method = GlobalMethod




'''
def mixture(thetas, pis, ksi, method, c=1, y=None, lotteries=None, subjects=None):

    if y is None:

        y = get_observed_ce(export_excel=False)

    if lotteries is None:

        lotteries = f.transform(lottery)

    if subjects is None:

        subjects = sorted(y["participant_label"].unique())

    # n is the number of participants

    n = len(subjects)

    # Setup individual error terms
    # Create a mapping from participant_label to their corresponding ksi_i value from the params vector.

    ksi_map = {subj: ksi[i] for i, subj in enumerate(subjects)}


    # Map the specification of ksi
    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}

    y = y[y["lottery_id"].isin(lotteries.keys())].copy()

    y["spread"] = y["lottery_id"].map(spreads)

    y["sigma"] = y["participant_label"].map(ksi_map) * y["spread"]

    L = 0

    # Group the data once outside the loop for speed
    grouped_y = y.groupby("participant_label")

    for subj_label in subjects:

        # Get only the data for THIS specific person

        y_subj = grouped_y.get_group(subj_label)

        log_terms = []

        for j in range(c):  # Changed to 'j' to avoid shadowing
            params = thetas[j]
            pi_j = pis[j]


            if method == "tk":
                r, alpha, lamb, gamma = params[:4]
                beta, palpha = 1, 1   # defaults passed to ce_dict but unused by TK

            elif method == "prelec":
                r, alpha, lamb, beta, palpha = params[:5]
                gamma = 0.61          # default passed to ce_dict but unused by Prelec

            R = 0

            # Get theoretical CEs and map to the subset
            ce_theoretical = f.ce_dict(r, gamma, alpha, lamb, R, lotteries=lotteries, method=method, beta=beta, palpha=palpha)

            log_likelihood_j = np.sum(norm.logpdf(y_subj["ce_observed"],
                                          loc=y_subj["lottery_id"].map(ce_theoretical),
                                          scale=y_subj["sigma"]))

            log_terms.append(np.log(pi_j) + log_likelihood_j)

        # Add the log of the weighted sum to the total Log-Likelihood
        # We add a tiny epsilon (1e-100) to prevent log(0)
        L += logsumexp(log_terms)

    return L
'''

def _s_to_a(s1, s2, s3):
    """Stick-breaking: (s1,s2,s3) ∈ [0,1]^3  →  (a1,a2,a3) with a1+a2+a3 ≤ 1."""
    a1 = s1
    a2 = s2 * (1.0 - s1)
    a3 = s3 * (1.0 - s1) * (1.0 - s2)
    return a1, a2, a3


def _a_to_s(a1, a2, a3):
    """Inverse stick-breaking: (a1,a2,a3) with sum ≤ 1  →  (s1,s2,s3) ∈ [0,1]^3."""
    s1 = a1
    rem1 = 1.0 - a1
    s2 = a2 / rem1 if rem1 > 1e-12 else 0.0
    rem2 = rem1 * (1.0 - s2)
    s3 = a3 / rem2 if rem2 > 1e-12 else 0.0
    return s1, s2, s3


def _cluster_ll(j, params, method, n, EL_arr, t_arr, Z1_arr, Z2_arr,
                Zt_arr, si_arr, sig_arr, obs_arr, lotteries, y):

        if method == "tk":
            r, alpha, lamb, gamma, s1, s2, s3, delta = params[:8]
            beta, palpha = 1, 1
        elif method == "prelec":
            r, alpha, lamb, beta, palpha, s1, s2, s3, delta = params[:9]
            gamma = 0.61
        else:
            raise ValueError(f"Unknown method: {method!r}")

        # Stick-breaking: enforce a1+a2+a3 ≤ 1 using (s1,s2,s3) ∈ [0,1]^3
        a1, a2, a3 = _s_to_a(s1, s2, s3)
        a4 = 1.0 - a1 - a2 - a3    # always ≥ 0 by construction

        # ── Vectorised R_l for every row ──────────────────────────────────────
        # Closed-form expansions of partial_adaptation and lagged_expectation
        # for t ∈ {0, 1, 2} (three sessions only).
        #
        # t=0: RA=0,  RLE=0            → R_l = a4*EL
        # t=1: RA = Z1/(δ+1),  RLE=EL  → R_l = a2*Z1/(δ+1) + (a3+a4)*EL
        # t=2: RA = (δ·Z1+Z2)/(δ²+δ+1), RLE=EL
        #           → R_l = a2*(δ·Z1+Z2)/(δ²+δ+1) + (a3+a4)*EL
        RA  = np.zeros(len(y))
        m1  = t_arr == 1
        m2  = t_arr == 2
        RA[m1] = Z1_arr[m1] / (delta + 1.0)
        RA[m2] = (delta * Z1_arr[m2] + Z2_arr[m2]) / (delta**2 + delta + 1.0)
        RLE = np.where(t_arr >= 1, EL_arr, 0.0)   # EL for s2/s3, 0 for s1
        R_l = a2 * RA + a3 * RLE + a4 * EL_arr    # RSQ=0 always

        # ── Cached evaluation → ce_th_base ───────────────────────────────────
        # Calls f.evaluation() once per unique (lottery_id, t, Z1, Z2) group
        # instead of once per observation.  For session 1 this reduces calls
        # from n_subjects × n_lotteries down to n_lotteries alone.
        ce_th_base = np.empty(len(y))
        cache      = {}
        for key, idx in y.groupby("_ckey", sort=False).groups.items():
            if key not in cache:
                lid = key[0]
                rl  = float(R_l[idx[0]])
                ev  = f.evaluation(r=r, R=rl, alpha=alpha, lamb=lamb,
                                   gamma=gamma, lotteries={lid: lotteries[lid]},
                                   method=method, beta=beta, palpha=palpha)
                cache[key] = f.u_inv(ev[lid]["V"], rl, alpha, lamb)
            ce_th_base[idx] = cache[key]

        ce_th = ce_th_base - Zt_arr

        # ── Accumulate log-likelihood per subject ─────────────────────────────
        log_pdf = norm.logpdf(obs_arr, loc=ce_th, scale=sig_arr)
        col = np.zeros(n)
        np.add.at(col, si_arr, log_pdf)
        return j, col



def compute_log_likelihoods(thetas, ksi, method=method, c=C, subjects=None, y=None, lotteries=None):
    """
    Returns an (n, c) matrix where entry [i, j] is the total log-likelihood
    of subject i's observed CEs under cluster j's structural parameters.

      Session 1 (round_number < 15):  t=0, Z_t = 0
        Only the forward-looking component is active:
        R_l = (1 - a1 - a2 - a3) * E[L_l]

      Session 2 (round_number == 15): t=1, Z_t = Z_1
        R_l = composite(a1, a2, a3,
                        R^SQ = 0,
                        R^A  = partial_adaptation([0, Z_1], delta),
                        R^LE = lagged_expectation([E[L_l]], delta),
                        R^FE = E[L_l])

      Session 3 (round_number == 16): t=2, Z_t = Z_1 + Z_2
        R_l = composite(a1, a2, a3,
                        R^SQ = 0,
                        R^A  = partial_adaptation([0, Z_1, Z_2], delta),
                        R^LE = lagged_expectation([E[L_l], E[L_l]], delta),
                        R^FE = E[L_l])
                                          
    Parameter vector per cluster:
      TK:     [r, alpha, lamb, gamma, a1, a2, a3, delta]         (8 params)
      Prelec: [r, alpha, lamb, beta, palpha, a1, a2, a3, delta]  (9 params)
    """

    assert len(thetas) == c, \
        "The length of the thetas list must correspond to the number of clusters c."

    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lottery)
    if subjects is None:
        subjects = sorted(y["participant_label"].unique())

    n = len(subjects)
    subj_index = {s: i for i, s in enumerate(subjects)}

    y = y[y["lottery_id"].isin(lotteries.keys())].copy().reset_index(drop=True)
    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}
    y["spread"] = y["lottery_id"].map(spreads)

    # ── Precompute per-row quantities that don't depend on cluster params ────
    # EL: expected payoff of each lottery — same for every row with that lottery_id
    EL_map = {lid: f.expected_payoff(lotteries[lid]["outcomes"]) for lid in lotteries}
    y["_EL"] = y["lottery_id"].map(EL_map)

    # Session indicator: t=0 (s1), t=1 (s2), t=2 (s3)
    s2 = y["round_number"] == 15
    s3 = y["round_number"] >= 16
    y["_t"] = s2.astype(int) + s3.astype(int)

    # Realised payoffs — vectorized label parsing; 0 where not yet observed
    def _parse_col(col):
        return pd.to_numeric(
            col.str.replace("£", "", regex=False), errors="coerce"
        ).fillna(0).astype(int)

    y["_Z1"] = np.where(s2 | s3, _parse_col(y["realized_period1_label"]), 0)
    y["_Z2"] = np.where(s3,      _parse_col(y["realized_period2_label"]), 0)
    y["_Zt"] = y["_Z1"] + y["_Z2"]

    # Subject integer index per row (for np.add.at accumulation)
    y["_si"] = y["participant_label"].map(subj_index)

    # sigma = ksi_i * spread_l  (same across clusters for a given ksi)
    ksi_arr  = np.array([ksi[subj_index[s]] for s in y["participant_label"]])
    y["_sigma"] = ksi_arr * y["spread"].values

    # Cache key: uniquely determines R_l for any fixed cluster params
    # (lottery_id, t, Z1, Z2) → same EL, same RA, same RLE → same R_l
    y["_ckey"] = list(zip(y["lottery_id"], y["_t"], y["_Z1"], y["_Z2"]))

    # Numpy arrays for hot-path computation
    EL_arr  = y["_EL"].values.astype(float)
    t_arr   = y["_t"].values.astype(float)
    Z1_arr  = y["_Z1"].values.astype(float)
    Z2_arr  = y["_Z2"].values.astype(float)
    Zt_arr  = y["_Zt"].values.astype(float)
    si_arr  = y["_si"].values
    sig_arr = y["_sigma"].values
    obs_arr = y["ce_observed"].values.astype(float)

    log_L = np.zeros((n, c))

    import multiprocessing
    fixed = (method, n, EL_arr, t_arr, Z1_arr, Z2_arr,
             Zt_arr, si_arr, sig_arr, obs_arr, lotteries, y)
    in_worker = multiprocessing.current_process().name != "MainProcess"
    if in_worker or c == 1:
        for j in range(c):
            _, col = _cluster_ll(j, thetas[j], *fixed)
            log_L[:, j] = col
    else:
        with ThreadPoolExecutor(max_workers=c) as executor:
            futures = [executor.submit(_cluster_ll, j, thetas[j], *fixed) for j in range(c)]
            for future in futures:
                j, col = future.result()
                log_L[:, j] = col
            
    return log_L


## Add this to start randomly. This helps to eliminate the local maxima.

def _run_one_em(args):
    seed, kwargs = args
    np.random.seed(seed)
    return em_mixture(**kwargs, seed=seed)

def em_mixture_best_of(n_restarts, seeds=None, n_workers=None, **em_kwargs):
    if seeds is None:
        seeds = list(range(n_restarts))

    em_kwargs["verbose"] = False
    args_list = [(s, em_kwargs) for s in seeds]

    W = 62
    print(f"\n{'─'*W}")
    print(f"  Parallel EM  |  {n_restarts} restarts  C={em_kwargs.get('c', C)}  method={em_kwargs.get('method', method)!r}")
    print(f"{'─'*W}")
    print(f"  {'Seed':>6}  │  {'Log-Likelihood':>16}  │  {'Iters':>5}  │  Status")
    print(f"  {'─'*6}──┼──{'─'*16}──┼──{'─'*5}──┼──{'─'*14}")

    results  = []
    best_ll  = -np.inf

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_seed = {executor.submit(_run_one_em, a): a[0] for a in args_list}
        for future in as_completed(future_to_seed):
            r = future.result()
            results.append(r)
            is_best = r["log_likelihood"] > best_ll
            if is_best:
                best_ll = r["log_likelihood"]
            status = ("converged" if r["converged"] else "not converged")
            mark   = "  <- best so far" if is_best else ""
            print(f"  {str(r['seed']):>6}  │  {r['log_likelihood']:>16.4f}  │  {r['n_iter']:>5}  │  {status}{mark}", flush=True)

    best = max(results, key=lambda r: r["log_likelihood"])
    print(f"{'─'*W}")
    print(f"  Winner: Seed {best['seed']}  |  LL = {best['log_likelihood']:.4f}  |  {best['n_iter']} iters")
    print(f"{'─'*W}")

    # ── Print winner's parameter estimates ───────────────────────────────────
    method_used = em_kwargs.get("method", method)
    param_names = {
        "tk":     ["r", "α", "λ", "γ",  "a1", "a2", "a3", "δ"],
        "prelec": ["r", "α", "λ", "β",  "pα", "a1", "a2", "a3", "δ"],
    }[method_used]
    c_used = em_kwargs.get("c", C)
    print(f"\n{'Cluster':<10} {'π':<8} " + "  ".join(f"{p:<8}" for p in param_names))
    print("-" * (10 + 8 + 10 * len(param_names)))
    for j in range(c_used):
        print(f"{j+1:<10} {best['pis'][j]:<8.3f} " + "  ".join(f"{v:<8.4f}" for v in best['thetas'][j]))

    return best


def em_mixture(thetas=None, pis=None, ksi=None, subjects=None, method=method, c=C,
               y=None, lotteries=None, max_iter=100, tol=1e-6, seed=None, verbose=True):

    if y is None:
        y = get_observed_ce(export_excel=False)

    if lotteries is None:
        lotteries = f.transform(lottery)

    if subjects is None:
        subjects = sorted(y["participant_label"].unique())

    n = len(subjects)

    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}

    y = y[y["lottery_id"].isin(lotteries.keys())].copy()

    y["spread"] = y["lottery_id"].map(spreads)

    grouped_y = y.groupby("participant_label")


    # ---- Initialisation ---- #
    if pis is None:
        # Equal mixing weights across all clusters
        pis = np.ones(c) / c

    if thetas is None:
        if method == "tk":
            # Initial values: typical CPT estimates for structural params,
            # zero reference-point weights (a1=a2=a3=0 → pure forward-looking),
            # delta=1 (equal weighting of past observations).
            # Small random perturbation breaks symmetry between clusters.
            # Parameter order: [r, alpha, lamb, gamma, a1, a2, a3, delta]
            base_tk   = np.array([0.05, 0.88, 2.25, 0.65, 0.0, 0.0, 0.0, 1.0])
            noise_tk  = np.array([0.01, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05])
            lo_tk     = np.array([1e-4, 0.5,  1.0,  0.2,  0.0,  0.0,  0.0,  0.0])
            hi_tk     = np.array([0.2,  1.5,  3.0,  1.0,  1.0,  1.0,  1.0,  1.0])
            def _init_tk():
                th = np.clip(base_tk + np.random.randn(8) * noise_tk, lo_tk, hi_tk)
                # Convert sampled a1,a2,a3 (indices 4,5,6) to s-space
                th[4], th[5], th[6] = _a_to_s(th[4], th[5], th[6])
                return th
            thetas = [_init_tk() for _ in range(c)]

        elif method == "prelec":
            # Parameter order: [r, alpha, lamb, beta, palpha, a1, a2, a3, delta]
            # beta is fixed to 1 (single-parameter Prelec); bounds enforce this.
            base_p  = np.array([0.05, 0.88, 2.25, 1.0, 0.65, 0.0, 0.0, 0.0, 1.0])
            noise_p = np.array([0.01, 0.05, 0.10, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05])
            lo_p    = np.array([1e-4, 0.5,  1.0,  1.0, 0.1,  0.0,  0.0,  0.0, 0.0])
            hi_p    = np.array([0.2,  1.5,  3.0,  1.0, 0.8,  1.0,  1.0,  1.0, 1.0])
            def _init_p():
                th = np.clip(base_p + np.random.randn(9) * noise_p, lo_p, hi_p)
                # Convert sampled a1,a2,a3 (indices 5,6,7) to s-space
                th[5], th[6], th[7] = _a_to_s(th[5], th[6], th[7])
                return th
            thetas = [_init_p() for _ in range(c)]

        else:
            raise ValueError(f"Unknown method: {method!r}")

    if ksi is None:
        # Initialise all individual noise parameters at a small positive value
        ksi = np.ones(n) * 0.1

    # ----------------------------------------


    # ---- EM loop ---- #
    # NOTE: the E-step and M-step structure below is unchanged from the original.
    # Only the parameter vectors (thetas) and bounds have been expanded to include
    # reference-point weights (a1, a2, a3) and memory factor (delta).

    # 'resp' holds the (n, c) responsibility matrix from the E-step.
    # Using 'resp' here to avoid shadowing the discount-rate variable 'r' that
    # appears inside compute_log_likelihoods().
    prev_ll  = -np.inf
    iter_log = []

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  EM start  |  C={c}  method={method!r}  n={n}  tol={tol}")
        print(f"{'─'*55}")

    for iteration in range(max_iter):

        t_iter_start = time.time()

        if verbose:
            print(f"\n[Seed {str(seed):>3}, Iter {iteration+1:>3}] E-step  — computing log-likelihoods ...", flush=True)

        # ── E-step ───────────────────────────────────────────────────────────────
        log_L     = compute_log_likelihoods(thetas, ksi, method, c, subjects, y, lotteries)
        log_pi    = np.log(pis)
        log_joint = log_L + log_pi[np.newaxis, :]
        log_sum   = logsumexp(log_joint, axis=1, keepdims=True)
        resp      = np.exp(log_joint - log_sum)   # (n, c) responsibility matrix

        ll = float(np.sum(log_sum))

        # Print per-cluster soft assignment sizes
        soft_n = resp.sum(axis=0)
        improvement = ll - prev_ll
        assign_str = "  ".join(f"C{j+1}: {soft_n[j]:.1f} (π={pis[j]:.3f})" for j in range(c))
        if verbose:
            print(f"         LL = {ll:.4f}   Improvement = {improvement:+.4f}   [{assign_str}]", flush=True)

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
                print(f"\n  Converged after {iteration+1} iteration(s)  (|ΔLL| < {tol})")
            break
        prev_ll = ll

        # ── M-step: mixing weights ────────────────────────────────────────────────
        pis = resp.mean(axis=0)

        # ── M-step: cluster structural + reference-point parameters ──────────────

        bounds = GlobalTKBounds if method == "tk" else GlobalPrelecBounds

        for j in range(c):
            if verbose:
                print(f"[Seed {str(seed):>3} Iter {iteration+1:>3}] M-step  — cluster {j+1}/{c}: Estimating preference parameters...", flush=True)
            resp_j = resp[:, j]

            def neg_weighted_ll_theta(params, j=j, resp_j=resp_j):
                thetas_temp    = list(thetas)
                thetas_temp[j] = params
                log_L_temp     = compute_log_likelihoods(thetas_temp, ksi, method, c, subjects, y, lotteries)
                return -np.sum(resp_j * log_L_temp[:, j])

            result    = minimize(neg_weighted_ll_theta, thetas[j], method="L-BFGS-B", bounds=bounds)
            thetas[j] = result.x
            if verbose:
                print(f"         cluster {j+1} done  (converged={result.success})", flush=True)

        # ── M-step: individual noise parameters ksi ──────────────────────────────
        if verbose:
            print(f"[Seed {str(seed):>3} Iter {iteration+1:>3}] M-step  — Estimating inidividual errors ({n} subjects) ...", flush=True)

        def neg_weighted_ll_ksi(ksi_params):
            log_L_temp = compute_log_likelihoods(thetas, ksi_params, method, c, subjects, y, lotteries)
            return -np.sum(resp * log_L_temp)

        result = minimize(neg_weighted_ll_ksi, ksi, method="L-BFGS-B", bounds=[(1e-4, 5)] * n)
        ksi    = result.x
        if verbose:
            print(f"         Done:  (ksi-mean={ksi.mean():.4f}  std={ksi.std():.4f})", flush=True)

        iter_log.append({
            "iteration": iteration + 1,
            "log_likelihood": ll,
            "improvement": improvement,
            "soft_n": soft_n.copy(),
            "pis": pis.copy(),
            "elapsed": time.time() - t_iter_start,
        })

    n_iter    = iteration + 1
    converged = abs(ll - prev_ll) < tol

    # Convert optimised s-params back to interpretable a-params
    if method == "tk":
        i1, i2, i3 = 4, 5, 6
    else:
        i1, i2, i3 = 5, 6, 7
    for j in range(c):
        th = np.array(thetas[j], dtype=float)
        th[i1], th[i2], th[i3] = _s_to_a(th[i1], th[i2], th[i3])
        thetas[j] = th

    # ---- Summary ---- #
    param_names = {
        "tk":     ["r", "α", "λ", "γ",  "a1", "a2", "a3", "δ"],
        "prelec": ["r", "α", "λ", "β",  "pα", "a1", "a2", "a3", "δ"],
    }[method]

    if verbose:
        print(f"\n{'Cluster':<10} {'π':<8} " + "  ".join(f"{p:<8}" for p in param_names))
        print("-" * (10 + 8 + 10 * len(param_names)))
        for j in range(c):
            print(f"{j+1:<10} {pis[j]:<8.3f} " + "  ".join(f"{v:<8.4f}" for v in thetas[j]))
        print(f"\nFinal LL: {ll:.4f}")

    return {"thetas": thetas, "pis": pis, "ksi": ksi, "log_likelihood": ll,
            "n_iter": n_iter, "converged": converged, "seed": seed,
            "resp": resp, "iter_log": iter_log, "subjects": subjects}



def write_em_results(result, filepath="em_results.txt"):
    """
    Write EM estimation results to a plain-text file.

    Includes:
      - Run metadata (method, clusters, convergence, seed, final LL)
      - Iteration overview (LL, improvement, soft cluster sizes, wall time per iter)
      - Cluster parameter table (thetas + pis)
      - Individual estimates: ksi and tau (responsibility) per subject
    """
    method_used = result.get("method_used", method)
    c_used      = len(result["thetas"])
    subjects    = result.get("subjects", [])
    n           = len(subjects)

    param_names = {
        "tk":     ["r", "alpha", "lambda", "gamma", "a1", "a2", "a3", "delta"],
        "prelec": ["r", "alpha", "lambda", "beta",  "palpha", "a1", "a2", "a3", "delta"],
    }[method_used]

    lines = []
    sep   = "=" * 72
    thin  = "─" * 72

    lines += [sep, "EM MIXTURE MODEL — ESTIMATION RESULTS", sep, ""]
    lines.append(f"  Method        : {method_used}")
    lines.append(f"  Clusters (C)  : {c_used}")
    lines.append(f"  Subjects (N)  : {n}")
    lines.append(f"  Iterations    : {result.get('n_iter', '?')}")
    lines.append(f"  Converged     : {result.get('converged', '?')}")
    lines.append(f"  Seed          : {result.get('seed', 'N/A')}")
    lines.append(f"  Log-Likelihood: {result['log_likelihood']:.6f}")
    lines.append("")

    # ── Iteration overview ───────────────────────────────────────────────────
    iter_log = result.get("iter_log", [])
    if iter_log:
        lines += [thin, "ITERATION OVERVIEW", thin]
        col_w   = 12
        hdr  = f"{'Iter':>5}  {'Log-Lik':>14}  {'Improvement':>13}  {'Time(s)':>8}"
        for j in range(c_used):
            hdr += f"  {'C'+str(j+1)+' soft_n':>{col_w}}  {'pi_'+str(j+1):>8}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for entry in iter_log:
            row = (f"{entry['iteration']:>5}  {entry['log_likelihood']:>14.4f}"
                   f"  {entry['improvement']:>+13.4f}  {entry['elapsed']:>8.2f}")
            for j in range(c_used):
                row += f"  {entry['soft_n'][j]:>{col_w}.2f}  {entry['pis'][j]:>8.4f}"
            lines.append(row)
        lines.append("")

    # ── Cluster parameters ───────────────────────────────────────────────────
    lines += [thin, "CLUSTER PARAMETERS", thin]
    hdr = f"{'Cluster':<10} {'pi':<10}" + "  ".join(f"{p:<12}" for p in param_names)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for j in range(c_used):
        row = f"{j+1:<10} {result['pis'][j]:<10.6f}" + "  ".join(f"{v:<12.6f}" for v in result["thetas"][j])
        lines.append(row)
    lines.append("")

    # ── Individual estimates: ksi and tau ────────────────────────────────────
    lines += [thin, "INDIVIDUAL ESTIMATES (ksi and tau)", thin]
    hdr = f"{'Subject':<30}  {'ksi':>12}"
    for j in range(c_used):
        hdr += f"  {'tau_C'+str(j+1):>12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    resp = result.get("resp")
    for i, subj in enumerate(subjects):
        row = f"{str(subj):<30}  {result['ksi'][i]:>12.6f}"
        if resp is not None:
            for j in range(c_used):
                row += f"  {resp[i, j]:>12.6f}"
        lines.append(row)
    lines.append("")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Results written to {filepath}")


if __name__ == "__main__":

    SEEDS = [10, 20, 30, 40, 50, 60, 70, 80]

    y        = get_observed_ce(export_excel=False)
    subjects = sorted(y["participant_label"].unique())

    best = em_mixture_best_of(
        n_restarts=len(SEEDS),
        seeds=SEEDS,
        y=y,
    )

    # Attach metadata needed by write_em_results
    best["method_used"] = method
    if "subjects" not in best or best["subjects"] is None:
        best["subjects"] = subjects

    write_em_results(best, filepath="em_results.txt")
