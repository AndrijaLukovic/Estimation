import sys
import time
import multiprocessing
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
from scipy.special import logsumexp


import functions as f
from lotteries import lotteries_full, one, all_high_stake, all_low_stake
from main import get_observed_ce
from GlobalSettings import GlobalTKBounds, GlobalPrelecBounds, GlobalMethod, GlobalLottery, GlobalCluster, GlobalTol
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED


# ── ANSI colour helpers (terminal output) ────────────────────────────────────
_RST = "\033[0m"   # reset
_B   = "\033[1m"   # bold
_DG  = "\033[90m"  # dark grey   — timing, secondary info
_CY  = "\033[36m"  # cyan        — step labels
_YL  = "\033[33m"  # yellow      — LL values
_GR  = "\033[32m"  # green       — improvement / converged
_RD  = "\033[31m"  # red         — not converged / diverging
_MG  = "\033[35m"  # magenta     — parameter values
_BL  = "\033[34m"  # blue        — headers / separators
# ─────────────────────────────────────────────────────────────────────────────

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

# Stick breaking method to control a1, a2, a3.
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


def _display_params(th, method):
    """Convert s-space params → interpretable a-space for verbose display."""
    th = np.array(th, dtype=float)
    if method == "tk":
        a1, a2, a3 = _s_to_a(th[4], th[5], th[6])
        a4 = 1.0 - a1 - a2 - a3
        names = ["r",  "α",  "λ",  "γ",  "a1", "a2", "a3", "a4", "δ"]
        vals  = [th[0], th[1], th[2], th[3], a1, a2, a3, a4, th[7]]
    else:
        a1, a2, a3 = _s_to_a(th[5], th[6], th[7])
        a4 = 1.0 - a1 - a2 - a3
        names = ["r",  "α",  "λ",  "β",  "pα", "a1", "a2", "a3", "a4", "δ"]
        vals  = [th[0], th[1], th[2], th[3], th[4], a1, a2, a3, a4, th[8]]
    return "  ".join(f"{_MG}{n}{_RST}={v:.4f}" for n, v in zip(names, vals))


def _compute_ce_th(params, method, EL_arr, t_arr, Z1_arr, Z2_arr,
                   Zt_arr, lotteries, y):
    """
    Compute theoretical CEs for all rows under one cluster's params.
    Extracted from _cluster_ll so it can be reused by the ksi M-step.
    Does NOT depend on ksi.  Returns array of shape (len(y),).
    """
    if method == "tk":
        r, alpha, lamb, gamma, s1, s2, s3, delta = params[:8]
        beta, palpha = 1, 1
    elif method == "prelec":
        r, alpha, lamb, beta, palpha, s1, s2, s3, delta = params[:9]
        gamma = 0.61
    else:
        raise ValueError(f"Unknown method: {method!r}")

    a1, a2, a3 = _s_to_a(s1, s2, s3)
    a4 = 1.0 - a1 - a2 - a3

    RA  = np.zeros(len(y))
    m1  = t_arr == 1
    m2  = t_arr == 2
    RA[m1] = Z1_arr[m1] / (delta + 1.0)
    RA[m2] = (delta * Z1_arr[m2] + Z2_arr[m2]) / (delta**2 + delta + 1.0)
    RLE = np.where(t_arr >= 1, EL_arr, 0.0)
    R_l = a2 * RA + a3 * RLE + a4 * EL_arr

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

    return ce_th_base - Zt_arr


def _cluster_ll(j, params, method, n, EL_arr, t_arr, Z1_arr, Z2_arr,
                Zt_arr, si_arr, sig_arr, obs_arr, lotteries, y):

        ce_th = _compute_ce_th(params, method, EL_arr, t_arr, Z1_arr, Z2_arr,
                               Zt_arr, lotteries, y)

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


def _ksi_mstep(resp, thetas, ksi, method, c, subjects, y, lotteries):
    """
    M-step for individual noise parameters ksi.

    Replaces the previous n-dimensional joint L-BFGS-B optimisation with
    n independent 1-D scalar minimisations (one per subject), executed in
    parallel via ThreadPoolExecutor.

    Why this is valid:
      ksi_i enters the likelihood only through sigma_il = ksi_i * spread_l,
      so the weighted objective Q(ksi) = Σ_i Q_i(ksi_i) is fully separable.

    Why this is fast:
      1. Algorithmic: n scalar (Brent) minimisations × ~15 evals each,
         versus the old n-D L-BFGS-B × (n+1) evals per gradient step.
      2. Each scalar evaluation uses a tiny pre-sliced numpy array for
         subject i only — no full (n×c) matrix recomputation.
      3. Subjects run in parallel (numpy releases the GIL for logpdf).
    """
    n = len(subjects)
    subj_index = {s: i for i, s in enumerate(subjects)}

    # ── Preprocessing (identical to compute_log_likelihoods, ksi-independent) ──
    y_proc = y[y["lottery_id"].isin(lotteries.keys())].copy().reset_index(drop=True)
    spreads_map = {lid: lotteries[lid]["spread"] for lid in lotteries}
    y_proc["spread"] = y_proc["lottery_id"].map(spreads_map)

    EL_map = {lid: f.expected_payoff(lotteries[lid]["outcomes"]) for lid in lotteries}
    y_proc["_EL"] = y_proc["lottery_id"].map(EL_map)

    s2 = y_proc["round_number"] == 15
    s3 = y_proc["round_number"] >= 16
    y_proc["_t"] = s2.astype(int) + s3.astype(int)

    def _parse_col(col):
        return pd.to_numeric(
            col.str.replace("£", "", regex=False), errors="coerce"
        ).fillna(0).astype(int)

    y_proc["_Z1"] = np.where(s2 | s3, _parse_col(y_proc["realized_period1_label"]), 0)
    y_proc["_Z2"] = np.where(s3,      _parse_col(y_proc["realized_period2_label"]), 0)
    y_proc["_Zt"] = y_proc["_Z1"] + y_proc["_Z2"]
    y_proc["_si"] = y_proc["participant_label"].map(subj_index)
    y_proc["_ckey"] = list(zip(y_proc["lottery_id"], y_proc["_t"],
                               y_proc["_Z1"], y_proc["_Z2"]))

    EL_arr     = y_proc["_EL"].values.astype(float)
    t_arr      = y_proc["_t"].values.astype(float)
    Z1_arr     = y_proc["_Z1"].values.astype(float)
    Z2_arr     = y_proc["_Z2"].values.astype(float)
    Zt_arr     = y_proc["_Zt"].values.astype(float)
    si_arr     = y_proc["_si"].values
    spread_arr = y_proc["spread"].values.astype(float)
    obs_arr    = y_proc["ce_observed"].values.astype(float)

    # ── Precompute ce_th for every cluster — ONE pass, no ksi needed ──────────
    # ce_th_all shape: (n_rows, c)
    ce_th_all = np.column_stack([
        _compute_ce_th(thetas[j], method, EL_arr, t_arr, Z1_arr, Z2_arr,
                       Zt_arr, lotteries, y_proc)
        for j in range(c)
    ])

    # ── Build per-subject slices (obs, ce_th, spread, resp) ───────────────────
    subj_args = []
    for i in range(n):
        mask = si_arr == i
        subj_args.append((
            i,
            obs_arr[mask],          # observed CEs for subject i
            ce_th_all[mask, :],     # theoretical CEs: (n_obs_i, c)
            spread_arr[mask],       # lottery spreads for subject i
            resp[i, :],             # soft responsibilities: (c,)
        ))

    # ── Scalar minimisation for a single subject ───────────────────────────────
    def _opt_one(args):
        i, obs_i, ce_th_i, spread_i, resp_i = args

        def obj(ksi_i):
            sigma_i = ksi_i * spread_i                         # (n_obs_i,)
            ll_ij   = norm.logpdf(obs_i[:, None],
                                  ce_th_i,
                                  sigma_i[:, None]).sum(axis=0) # (c,)
            return -np.dot(resp_i, ll_ij)

        res = minimize_scalar(obj, bounds=(1e-4, 5.0), method="bounded")
        return i, res.x

    # ── Parallel execution (numpy logpdf releases the GIL) ────────────────────
    new_ksi = ksi.copy()
    with ThreadPoolExecutor(max_workers=min(8, n)) as executor:
        futures = [executor.submit(_opt_one, a) for a in subj_args]
        for fut in as_completed(futures):
            i, ksi_i = fut.result()
            new_ksi[i] = ksi_i

    return new_ksi


## Add this to start randomly. This helps to eliminate the local maxima.

def _run_one_em(args):
    seed, kwargs = args
    np.random.seed(seed)
    return em_mixture(**kwargs, seed=seed)

def em_mixture_best_of(n_restarts, seeds=None, n_workers=None, **em_kwargs):
    if seeds is None:
        seeds = list(range(n_restarts))

    n_seeds     = len(seeds)
    method_used = em_kwargs.get("method", method)
    c_used      = em_kwargs.get("c", C)

    # ── Live-table state: one row per seed ────────────────────────────────────
    # status: "waiting" | "running" | "converged" | "not_converged" | "failed"
    state = {s: {"ll": None, "iters": 0, "status": "waiting"} for s in seeds}

    W = 68
    _prev_n_lines = [0]   # mutable: how many lines to erase on next redraw

    def _render():
        best_ll = max(
            (st["ll"] for st in state.values() if st["ll"] is not None),
            default=None,
        )
        elapsed = time.time() - t_wall_start
        sep = f"  {'─'*6}─┼─{'─'*18}─┼─{'─'*6}─┼─{'─'*20}"
        rows = [
            f"  {_B}{'Seed':>6}  │  {'Log-Likelihood':>18}  │  {'Iters':>6}  │  Status{_RST}",
            sep,
        ]
        for s in seeds:
            st     = state[s]
            ll_str = f"{st['ll']:>18.4f}" if st["ll"] is not None else f"{'—':>18}"
            it_str = f"{st['iters']:>6}"  if st["iters"] > 0       else f"{'—':>6}"
            status = st["status"]
            if status == "waiting":
                st_str = f"{_DG}○  waiting{_RST}"
            elif status == "running":
                st_str = f"{_YL}●  running{_RST}"
            elif status == "converged":
                st_str = f"{_GR}✓  converged{_RST}"
            elif status == "not_converged":
                st_str = f"{_RD}✗  not converged{_RST}"
            else:
                st_str = f"{_RD}✗  failed{_RST}"
            is_best   = st["ll"] is not None and st["ll"] == best_ll
            ll_col    = _GR if is_best else (_YL if st["ll"] is not None else "")
            best_mark = f"  {_GR}{_B}← best{_RST}" if is_best else ""
            rows.append(
                f"  {str(s):>6}  │  {ll_col}{ll_str}{_RST}  │  {it_str}  │  {st_str}{best_mark}"
            )
        rows.append(sep)
        rows.append(f"  {_DG}elapsed {elapsed:.0f}s{_RST}")
        return rows

    def _draw():
        rows = _render()
        n_up = _prev_n_lines[0]
        if n_up > 0:
            sys.stdout.write(f"\033[{n_up}A")
        for row in rows:
            sys.stdout.write(f"\033[2K\r{row}\n")
        sys.stdout.flush()
        _prev_n_lines[0] = len(rows)

    # ── Header (printed once, above the live table) ───────────────────────────
    print(f"\n{_B}{_BL}{'─'*W}{_RST}")
    print(f"{_B}{_BL}  Parallel EM  │  {n_seeds} restarts  C={c_used}  method={method_used!r}{_RST}")
    print(f"{_B}{_BL}{'─'*W}{_RST}", flush=True)

    t_wall_start = time.time()

    # ── Shared progress queue (Manager proxy — picklable across processes) ────
    with multiprocessing.Manager() as mgr:
        q = mgr.Queue()
        em_kwargs["verbose"]        = False
        em_kwargs["progress_queue"] = q
        args_list = [(s, em_kwargs) for s in seeds]

        _draw()   # initial table (all "waiting")

        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_seed = {executor.submit(_run_one_em, a): a[0] for a in args_list}

            # All seeds move to "running" once submitted
            for s in seeds:
                state[s]["status"] = "running"
            _draw()

            pending = set(future_to_seed.keys())
            while pending:
                # Drain all queued progress messages from workers
                updated = False
                while True:
                    try:
                        msg = q.get_nowait()
                        sd  = msg["seed"]
                        state[sd]["ll"]     = msg["ll"]
                        state[sd]["iters"]  = msg["iter"]
                        state[sd]["status"] = "running"
                        updated = True
                    except Exception:
                        break
                if updated:
                    _draw()

                # Wait up to 0.5 s for any future to complete
                done_futs, pending = wait(pending, timeout=0.5,
                                          return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    sd = future_to_seed[fut]
                    try:
                        r = fut.result()
                        results.append(r)
                        state[sd]["ll"]     = r["log_likelihood"]
                        state[sd]["iters"]  = r["n_iter"]
                        state[sd]["status"] = "converged" if r["converged"] else "not_converged"
                    except Exception as exc:
                        state[sd]["status"] = "failed"
                        _prev_n_lines[0] = 0
                        print(f"\n  {_RD}Seed {sd} failed: {exc}{_RST}", flush=True)
                    _draw()

    # ── Final summary (printed below the live table) ──────────────────────────
    if not results:
        raise RuntimeError("All EM restarts failed.")

    best    = max(results, key=lambda r: r["log_likelihood"])
    total_t = time.time() - t_wall_start
    print(f"\n{_B}{_BL}{'─'*W}{_RST}")
    print(f"  {_B}{_GR}Winner:{_RST}  Seed {best['seed']}"
          f"  │  LL = {_YL}{best['log_likelihood']:.4f}{_RST}"
          f"  │  {best['n_iter']} iters"
          f"  │  {_DG}total {total_t:.1f}s{_RST}")
    print(f"{_B}{_BL}{'─'*W}{_RST}")

    # ── Print winner's parameter estimates ───────────────────────────────────
    print(f"\n  {_B}{'Cluster':<10} {'π':<8}{_RST}")
    print("  " + "─" * 60)
    for j in range(c_used):
        print(f"  {_B}C{j+1}{_RST}{'':8} {best['pis'][j]:<8.3f} {_display_params(best['thetas'][j], method_used)}")

    return best


def em_mixture(thetas=None, pis=None, ksi=None, subjects=None, method=method, c=C,
               y=None, lotteries=None, max_iter=100, tol=1e-6, seed=None, verbose=True,
               progress_queue=None):

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

    _sid = f"S{str(seed):>3}"   # e.g. "S 10"

    if verbose:
        W2 = 65
        print(f"\n{_B}{_BL}{'─'*W2}{_RST}")
        print(f"{_B}{_BL}  EM  │  C={c}  method={method!r}  n={n}  seed={seed}  tol={tol}{_RST}")
        print(f"{_B}{_BL}{'─'*W2}{_RST}")

    for iteration in range(max_iter):

        t_iter_start = time.time()
        _it = f"I{iteration+1:>3}"   # e.g. "I  1"

        # ── E-step ───────────────────────────────────────────────────────────────
        if verbose:
            print(f"\n  {_CY}[{_sid} {_it}] E-step{_RST} computing ...", end="  ", flush=True)
        t_e = time.time()
        log_L     = compute_log_likelihoods(thetas, ksi, method, c, subjects, y, lotteries)
        log_pi    = np.log(pis)
        log_joint = log_L + log_pi[np.newaxis, :]
        log_sum   = logsumexp(log_joint, axis=1, keepdims=True)
        resp      = np.exp(log_joint - log_sum)   # (n, c) responsibility matrix
        t_e = time.time() - t_e

        ll = float(np.sum(log_sum))
        soft_n = resp.sum(axis=0)
        improvement = ll - prev_ll

        if verbose:
            impr_col   = _GR if improvement > 1e-3 else (_RD if improvement < -1e-3 else _DG)
            impr_str   = "  +∞  " if prev_ll == -np.inf else f"{improvement:>+10.4f}"
            assign_str = "  ".join(
                f"{_B}C{j+1}{_RST}: {soft_n[j]:.1f} {_DG}(π={pis[j]:.3f}){_RST}"
                for j in range(c)
            )
            print(f"LL = {_YL}{ll:.4f}{_RST}   Δ = {impr_col}{impr_str}{_RST}"
                  f"   [{assign_str}]  {_DG}{t_e:.2f}s{_RST}", flush=True)

        if progress_queue is not None:
            try:
                progress_queue.put_nowait({"seed": seed, "iter": iteration + 1, "ll": ll})
            except Exception:
                pass

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
                print(f"\n  {_GR}{_B}✓ Converged{_RST} after {iteration+1} iteration(s)"
                      f"  (|ΔLL| = {abs(ll-prev_ll):.2e} < {tol})")
            break
        prev_ll = ll

        # ── M-step: mixing weights ────────────────────────────────────────────────
        pis = resp.mean(axis=0)

        # ── M-step: cluster structural + reference-point parameters ──────────────
        bounds = GlobalTKBounds if method == "tk" else GlobalPrelecBounds

        for j in range(c):
            if verbose:
                print(f"  {_CY}[{_sid} {_it}] θ C{j+1}/{c}{_RST} optimising ...", end="  ", flush=True)
            resp_j  = resp[:, j]
            t_j     = time.time()

            def neg_weighted_ll_theta(params, j=j, resp_j=resp_j):
                thetas_temp    = list(thetas)
                thetas_temp[j] = params
                log_L_temp     = compute_log_likelihoods(thetas_temp, ksi, method, c, subjects, y, lotteries)
                return -np.sum(resp_j * log_L_temp[:, j])

            result    = minimize(neg_weighted_ll_theta, thetas[j], method="L-BFGS-B", bounds=bounds)
            thetas[j] = result.x
            t_j       = time.time() - t_j

            if verbose:
                ok_str = f"{_GR}ok{_RST}" if result.success else f"{_RD}WARN (not converged){_RST}"
                print(f"{ok_str}  {_DG}{t_j:.2f}s{_RST}")
                print(f"           {_display_params(thetas[j], method)}", flush=True)

        # ── M-step: individual noise parameters ksi ──────────────────────────────
        if verbose:
            print(f"  {_CY}[{_sid} {_it}] κ {n} subj{_RST} parallel scalar ...", end="  ", flush=True)
        t_ksi = time.time()
        ksi = _ksi_mstep(resp, thetas, ksi, method, c, subjects, y, lotteries)
        t_ksi = time.time() - t_ksi
        if verbose:
            ksi_med = float(np.median(ksi))
            print(f"mean={_MG}{ksi.mean():.4f}{_RST}  std={ksi.std():.4f}"
                  f"  [{ksi.min():.4f} … {ksi_med:.4f} … {ksi.max():.4f}]"
                  f"  {_DG}{t_ksi:.2f}s{_RST}", flush=True)

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
    if verbose:
        conv_str = f"{_GR}{_B}converged{_RST}" if converged else f"{_RD}{_B}NOT converged{_RST}"
        print(f"\n  {conv_str}  after {n_iter} iterations  │  "
              f"Final LL = {_YL}{ll:.4f}{_RST}")
        print(f"\n  {_B}{'Cluster':<10} {'π':<8}{_RST}")
        print("  " + "─" * 60)
        for j in range(c):
            print(f"  {_B}C{j+1}{_RST}{'':8} {pis[j]:<8.3f} {_display_params(thetas[j], method)}")
        print()

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

    write_em_results(best, filepath="em_results_2.txt")
