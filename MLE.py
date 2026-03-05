import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import functions as f
from lotteries import lotteries_full, one, all_high_stake, all_low_stake
from main import get_observed_ce
import openpyxl

prob_weighter = "tk"  # "tk" or "prelec"

# Structural parameter bounds — only the probability weighting bounds differ
_shared_front = [(1e-4, 0.2), (0.5, 1.5), (1e-3, 7.0)]  # r, alpha, lamb
_shared_R     = [(-100, 100)]                              # R (always last)

bounds_tk     = _shared_front + [(0.2, 1)]          + _shared_R  # + gamma
bounds_prelec = _shared_front + [(0.1, 1), (0.1, 1)] + _shared_R  # + beta, palpha


bounds = bounds_tk if prob_weighter == "tk" else bounds_prelec

# Iteration time
n_starts = 100

# ── Lottery set to estimate on ──────────────────────────────────────────────
# Switch between all_low_stake / all_high_stake / lotteries_full
lottery = all_low_stake
# ────────────────────────────────────────────────────────────────────────────

# random setup
np.random.seed(10)

def loglikelihood(params, y=None, lotteries=None, subjects=None, method=prob_weighter):
    """
    Negative log-likelihood with individual error terms.

    TK:     params = [r, alpha, lamb, gamma, R,           ksi_1, ..., ksi_N]
    Prelec: params = [r, alpha, lamb, beta, palpha, R,    ksi_1, ..., ksi_N]

    subjects is the ordered list of participant_label values that maps
    the ksi block to individual subjects.

    method should be either "tk" or "prelec".
    """
    # Get data ready
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lottery)
    if subjects is None:
        subjects = sorted(y["participant_label"].unique())
    # Setup systematic parameters
    if method == "tk":
        r, alpha, lamb, gamma, R = params[:5]
        beta, palpha = 1, 1   # defaults passed to ce_dict but unused by TK
        ksi_offset = 5
    elif method == "prelec":
        r, alpha, lamb, beta, palpha, R = params[:6]
        gamma = 0.61          # default passed to ce_dict but unused by Prelec
        ksi_offset = 6
    # Setup individual error terms
    ksi_map = {subj: params[ksi_offset + i] for i, subj in enumerate(subjects)}  # Create a mapping from participant_label to their corresponding ksi_i value from the params vector.

    # Compute the theoretical CE
    ce_theoretical = f.ce_dict(r, gamma, alpha, lamb, R, lotteries=lotteries, method=method, beta=beta, palpha=palpha)

    # Map the specification of ksi
    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}
    y = y[y["lottery_id"].isin(lotteries.keys())].copy()
    y["ce_th"] = y["lottery_id"].map(ce_theoretical)
    y["spread"] = y["lottery_id"].map(spreads)
    y["sigma"] = y["participant_label"].map(ksi_map) * y["spread"]

    # Compute the loglik for each observation and return the negative sum for minimization.
    log_lik = norm.logpdf(y["ce_observed"], loc=y["ce_th"], scale=y["sigma"])
    return -float(log_lik.sum())


def run_multistart_mle(obj_func, n_starts=n_starts, param_bounds=None):
    """
    Run bounded multistart optimization and return the best successful result.

    """
    best_result = None
    best_f = np.inf  # We minimize -LogLikelihood, so smaller is better.

    lower_bounds = np.array([b[0] if b[0] is not None else -2 for b in param_bounds])
    upper_bounds = np.array([b[1] if b[1] is not None else 10 for b in param_bounds])

    for i in range(n_starts):
        print(f"Run {i + 1}/{n_starts}...", end="\r", flush=True)
        random_guess = np.random.uniform(lower_bounds, upper_bounds)

        res = minimize(
            obj_func,
            x0=random_guess,
            bounds=param_bounds,
            method="L-BFGS-B",
        ) #scipy minimize with L-BFGS-B method for bounded optimization

        if res.fun < best_f:
            best_f = res.fun
            best_result = res
            print(f"\nRun {i + 1}: New best Log-Likelihood found: {-res.fun:.4f}")

    print()
    return best_result


def estimate_mle(n_starts=n_starts, param_bounds=bounds, y=None, lotteries=None):
    """
    Estimate parameters with multistart MLE and fail loudly if nothing converged.

    TK:     full vector = [r, alpha, lamb, gamma, R,        ksi_1, ..., ksi_N]
    Prelec: full vector = [r, alpha, lamb, beta, palpha, R, ksi_1, ..., ksi_N]
    Per-subject ksi bounds (1e-3, None) are appended to param_bounds dynamically
    based on the number of unique subjects in y.
    """
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lottery)

    subjects = sorted(y["participant_label"].unique())
    full_bounds = list(param_bounds) + [(1e-3, None)] * len(subjects) # bounds for structural parameters + bounds for individual error terms

    def objective(theta):
        return loglikelihood(theta, y=y, lotteries=lotteries, subjects=subjects)

    result = run_multistart_mle(objective, n_starts=n_starts, param_bounds=full_bounds)
    if result is None:
        raise RuntimeError("No successful optimization run was found in multistart MLE.")
    return result


def format_results(result, method=prob_weighter):
    if method == "tk":
        param_names = ["r", "Alpha", "Lambda", "Gamma", "R"]
        n_structural = 5
    else:  # prelec
        param_names = ["r", "Alpha", "Lambda", "Beta", "PAlpha", "R"]
        n_structural = 6
    return pd.DataFrame(
        {"Parameter": param_names, "Estimate": result.x[:n_structural]}
    )


if __name__ == "__main__":
    y = get_observed_ce(export_excel=False)
    subjects = sorted(y["participant_label"].unique())

    result = estimate_mle(n_starts=n_starts, param_bounds=bounds, y=y)
    results_df = format_results(result)

    print("\nMAXIMUM LIKELIHOOD ESTIMATES")
    print(results_df.to_string(index=False))
    print(f"Best Log-Likelihood: {-result.fun:.4f}")
    print(f"Estimated lottery choice data: {lottery}")

    # Write individual ksi values to txt file
    ksi_offset = 5 if prob_weighter == "tk" else 6
    ksi_values = result.x[ksi_offset:]
    ksi_lines = [f"{subj}\t{ksi:.6f}" for subj, ksi in zip(subjects, ksi_values)]
    with open("ksi_estimates.txt", "w") as fh:
        fh.write("participant_label\tksi\n")
        fh.write("\n".join(ksi_lines) + "\n")
    print("Individual ksi estimates written to ksi_estimates.txt")
