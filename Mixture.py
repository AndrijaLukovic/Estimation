import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import logsumexp


import functions as f
from lotteries import lotteries_full, one, all_high_stake, all_low_stake
from main import get_observed_ce



# ── Lottery set to estimate on ──────────────────────────────────────────────
# Switch between all_low_stake / all_high_stake / lotteries_full
lottery = lotteries_full
# ────────────────────────────────────────────────────────────────────────────

### Choice of the method "prelec" or "tk"
prob_weighter = "prelec"


### Number of clusters
C = 2

### Method
method = "tk"




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



def compute_log_likelihoods(thetas, ksi, method=method, c=C, subjects=None, y=None, lotteries=None):
    """
    Returns an (n, c) matrix where entry [i, j] is the total log-likelihood
    of subject i's observed CEs under cluster j's structural parameters.

    All three sessions are handled jointly via a composite reference point:

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

    delta controls memory decay in R^A and R^LE:
      delta = 1  → equal weighting of all past observations (arithmetic mean)
      delta → 0  → only the most recent observation matters

    CE formula:  CE = u^{-1}(V(L; R_l)) - Z_t

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

    # Map each subject to their noise parameter ksi
    ksi_map = {subj: ksi[i] for i, subj in enumerate(subjects)}

    y = y[y["lottery_id"].isin(lotteries.keys())].copy()

    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}

    y["spread"] = y["lottery_id"].map(spreads)

    # sigma_i = ksi_i * spread_l  (subject-specific noise scaled by lottery range)
    y["sigma"] = y["participant_label"].map(ksi_map) * y["spread"]

    grouped_y = y.groupby("participant_label", observed=True)

    log_L = np.zeros((n, c))

    for j, params in enumerate(thetas):

        if method == "tk":
            # TK weighting: unpack structural params + reference-point weights + memory factor
            r, alpha, lamb, gamma, a1, a2, a3, delta = params[:8]
            beta, palpha = 1, 1          # unused by TK but required by evaluation()

        elif method == "prelec":
            # Single-parameter Prelec (beta fixed to 1 via bounds): unpack accordingly
            r, alpha, lamb, beta, palpha, a1, a2, a3, delta = params[:9]
            gamma = 0.61                 # unused by Prelec but required by evaluation()

        else:
            raise ValueError(f"Unknown method: {method!r}")


        for i, subj_label in enumerate(subjects):

            if subj_label not in grouped_y.groups:
                continue

            y_subj = grouped_y.get_group(subj_label)

            ce_th_vals  = []   # theoretical CEs for this subject under cluster j
            ce_obs_vals = []   # corresponding observed CEs
            sigma_vals  = []   # corresponding noise scales

            # Iterate row-by-row because the reference point is observation-specific:
            # Z_t depends on which lottery was played AND how its path was realised,
            # so it differs across lotteries within the same session.
            for _, row in y_subj.iterrows():

                lid = row["lottery_id"]
                if lid not in lotteries:
                    continue

                lot = lotteries[lid]

                # Expected total payoff of this lottery (sum over paths of p * sum(payoffs))
                # Used as the forward-looking and lagged-expectation reference component
                EL = f.expected_payoff(lot["outcomes"])

                rn = row["round_number"]

                if rn < 15:
                    # ── Session 1 ────────────────────────────────────────────────
                    # No past realisations yet.
                    # R^SQ = 0, R^A = 0, R^LE = 0 (all collapse to starting wealth).
                    # Only the forward-looking component R^FE = E[L_l] is active.
                    t      = 0
                    Z_seq  = [0.0]   # only starting wealth
                    EL_seq = []      # no past sessions → lagged_expectation returns Z_0 = 0
                    Z_t    = 0.0

                elif rn == 15:
                    # ── Session 2 ────────────────────────────────────────────────
                    # Participant has observed the period-1 payoff Z_1 for this lottery.
                    Z1     = f._parse_label(row["realized_period1_label"])
                    t      = 1
                    Z_seq  = [0.0, float(Z1)]
                    EL_seq = [EL]    # one past session; lagged expectation = E[L_l]
                    Z_t    = float(Z1)

                else:
                    # ── Session 3 ────────────────────────────────────────────────
                    # Participant has observed period-1 payoff Z_1 and period-2 payoff Z_2.
                    Z1 = f._parse_label(row["realized_period1_label"])
                    # Guard against missing period-2 label (should not occur after data.dropna,
                    # but handled gracefully just in case)
                    Z2 = (f._parse_label(row["realized_period2_label"])
                          if pd.notna(row.get("realized_period2_label")) else 0)
                    t      = 2
                    Z_seq  = [0.0, float(Z1), float(Z2)]
                    EL_seq = [EL, EL]  # two past sessions, both for the same lottery
                    Z_t    = float(Z1) + float(Z2)

                # ── Composite reference point ─────────────────────────────────────
                # R^SQ  = status quo = starting wealth (always 0 in session 1)
                # R^A   = partial adaptation to realised payoffs, decayed by delta
                # R^LE  = lagged expectation of past lottery E[L], decayed by delta
                # R^FE  = forward-looking expectation of the current lottery
                # a4    = 1 - a1 - a2 - a3  (residual weight, clipped ≥ 0 in composite())
                RSQ = f.status_quo(0.0)
                RA  = f.partial_adaptation(t, Z_seq, delta)
                RLE = f.lagged_expectation(t, EL_seq, 0.0, delta)
                RFE = f.forward_looking(EL)
                R_l = f.composite(a1, a2, a3, RSQ, RA, RLE, RFE)

                # ── Theoretical CE ────────────────────────────────────────────────
                # Evaluate the full lottery at reference point R_l, then invert utility
                # and subtract the already-realised cumulative payoff Z_t.
                ev    = f.evaluation(r=r, R=R_l, alpha=alpha, lamb=lamb,
                                     gamma=gamma, lotteries={lid: lot},
                                     method=method, beta=beta, palpha=palpha)
                ce_th = f.u_inv(ev[lid]["V"], R_l, alpha, lamb) - Z_t

                ce_th_vals .append(ce_th)
                ce_obs_vals.append(row["ce_observed"])
                sigma_vals .append(row["sigma"])

            if ce_th_vals:
                log_L[i, j] = np.sum(norm.logpdf(
                    np.array(ce_obs_vals),
                    loc=np.array(ce_th_vals),
                    scale=np.array(sigma_vals)
                ))

    return log_L




def em_mixture(thetas=None, pis=None, ksi=None, subjects=None, method=method, c=C,
               y=None, lotteries=None, max_iter=100, tol=1e-6):

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
            thetas = [
                np.clip(base_tk + np.random.randn(8) * noise_tk, lo_tk, hi_tk)
                for _ in range(c)
            ]

        elif method == "prelec":
            # Parameter order: [r, alpha, lamb, beta, palpha, a1, a2, a3, delta]
            # beta is fixed to 1 (single-parameter Prelec); bounds enforce this.
            base_p  = np.array([0.05, 0.88, 2.25, 1.0, 0.65, 0.0, 0.0, 0.0, 1.0])
            noise_p = np.array([0.01, 0.05, 0.10, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05])
            lo_p    = np.array([1e-4, 0.5,  1.0,  1.0, 0.1,  0.0,  0.0,  0.0, 0.0])
            hi_p    = np.array([0.2,  1.5,  3.0,  1.0, 0.8,  1.0,  1.0,  1.0, 1.0])
            thetas = [
                np.clip(base_p + np.random.randn(9) * noise_p, lo_p, hi_p)
                for _ in range(c)
            ]

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
    prev_ll = -np.inf

    for iteration in range(max_iter):

        # ── E-step ───────────────────────────────────────────────────────────────
        # Compute posterior responsibility resp[i, j] = P(subject i in cluster j | data).
        # Uses log-sum-exp trick for numerical stability.
        log_L     = compute_log_likelihoods(thetas, ksi, method, c, subjects, y, lotteries)
        log_pi    = np.log(pis)
        log_joint = log_L + log_pi[np.newaxis, :]
        log_sum   = logsumexp(log_joint, axis=1, keepdims=True)
        resp      = np.exp(log_joint - log_sum)   # (n, c) responsibility matrix

        ll = np.sum(log_sum)
        print(f"Iter {iteration:3d} | LL = {ll:.4f}")

        if abs(ll - prev_ll) < tol:
            print("Converged.")
            break
        prev_ll = ll

        # ── M-step: mixing weights ────────────────────────────────────────────────
        # pi_j = average responsibility for cluster j across all subjects
        pis = resp.mean(axis=0)

        # ── M-step: cluster structural + reference-point parameters ──────────────
        # Bounds for TK: [r, alpha, lamb, gamma, a1, a2, a3, delta]
        bounds_tk = [
            (1e-4, 0.2),  # r:     discount rate
            (0.5,  1.5),  # alpha: utility curvature
            (1.0,  3.0),  # lamb:  loss aversion
            (0.2,  1.0),  # gamma: TK probability weighting
            (0.0,  1.0),  # a1:    status quo weight
            (0.0,  1.0),  # a2:    partial adaptation weight
            (0.0,  1.0),  # a3:    lagged expectation weight
            (0.0,  1.0),  # delta: memory decay (0 = only recent, 1 = equal weights)
        ]
        # Bounds for Prelec: [r, alpha, lamb, beta, palpha, a1, a2, a3, delta]
        bounds_prelec = [
            (1e-4, 0.2),  # r
            (0.5,  1.5),  # alpha
            (1.0,  3.0),  # lamb
            (1.0,  1.0),  # beta:   fixed at 1 (single-parameter Prelec)
            (0.1,  0.8),  # palpha: Prelec elevation parameter
            (0.0,  1.0),  # a1
            (0.0,  1.0),  # a2
            (0.0,  1.0),  # a3
            (0.0,  1.0),  # delta
        ]
        bounds = bounds_tk if method == "tk" else bounds_prelec

        for j in range(c):
            resp_j = resp[:, j]   # responsibility weights for cluster j

            def neg_weighted_ll_theta(params, j=j, resp_j=resp_j):
                thetas_temp    = list(thetas)
                thetas_temp[j] = params
                log_L_temp     = compute_log_likelihoods(thetas_temp, ksi, method, c, subjects, y, lotteries)
                return -np.sum(resp_j * log_L_temp[:, j])

            result    = minimize(neg_weighted_ll_theta, thetas[j], method="L-BFGS-B", bounds=bounds)
            thetas[j] = result.x

        # ── M-step: individual noise parameters ksi ──────────────────────────────
        def neg_weighted_ll_ksi(ksi_params):
            log_L_temp = compute_log_likelihoods(thetas, ksi_params, method, c, subjects, y, lotteries)
            return -np.sum(resp * log_L_temp)

        result = minimize(neg_weighted_ll_ksi, ksi, method="L-BFGS-B", bounds=[(1e-4, 5)] * n)
        ksi    = result.x

    # ---- Summary ---- #
    param_names = {
        "tk":     ["r", "α", "λ", "γ",  "a1", "a2", "a3", "δ"],
        "prelec": ["r", "α", "λ", "β",  "pα", "a1", "a2", "a3", "δ"],
    }[method]

    print(f"\n{'Cluster':<10} {'π':<8} " + "  ".join(f"{p:<8}" for p in param_names))
    print("-" * (10 + 8 + 10 * len(param_names)))
    for j in range(c):
        print(f"{j+1:<10} {pis[j]:<8.3f} " + "  ".join(f"{v:<8.4f}" for v in thetas[j]))

    print(f"\nFinal LL: {ll:.4f}")

    return {"thetas": thetas, "pis": pis, "ksi": ksi, "log_likelihood": ll}



if __name__ == "__main__":

    em_mixture()
