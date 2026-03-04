import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import functions as f
from lotteries import lotteries_full, one, all_high_stake, all_low_stake
from main import get_observed_ce
import openpyxl

# r, alpha, lamb, gamma, R
params = [0.97, 0.88, 2.25, 0.61, 0]

bounds = [
    (1e-4, 1),    # r
    (0.5, 1.5),       # alpha  <-- too little alpha causes underflow
    (1e-3, 7.0),    # lamb
    (0.2, 1),     # gamma
    (-100, 100),    # R
    # per-subject ksi bounds added dynamically in estimate_mle
]


# Iteration time
n_starts = 100

# ── Lottery set to estimate on ──────────────────────────────────────────────
# Switch between all_low_stake / all_high_stake / lotteries_full
lottery = all_low_stake
# ────────────────────────────────────────────────────────────────────────────

# random setup
np.random.seed(10)

def loglikelihood(params, y=None, lotteries=None, subjects=None):
    """
    Negative log-likelihood with individual error terms.

    params = [r, alpha, lamb, gamma, R, ksi_1, ksi_2, ..., ksi_N]

    subjects is the ordered list of participant_label values that maps
    params[5], params[6], ... to individual subjects.

    """
    # Get data ready
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lottery)
    if subjects is None:
        subjects = sorted(y["participant_label"].unique())
    # Setup systematic parameters
    r, alpha, lamb, gamma, R = params[:5]
    # Setup individual error terms
    ksi_map = {subj: params[5 + i] for i, subj in enumerate(subjects)} # Create a mapping from participant_label to their corresponding ksi_i value from the params vector.

    # Compute the theoretical CE
    ce_theoretical = f.ce_dict(r, gamma, alpha, lamb, R, lotteries=lotteries)

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

    The full parameter vector is [r, alpha, lamb, gamma, R, ksi_1, ..., ksi_N].
    Per-subject ksi bounds (1e-3, 3.0) are appended to param_bounds dynamically
    based on the number of unique subjects in y. Only the 5 CPT parameters are
    reported; ksi_i values are estimated but not surfaced in format_results.
    """
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lottery)

    subjects = sorted(y["participant_label"].unique())
    full_bounds = list(param_bounds) + [(1e-3, 3.0)] * len(subjects) # bounds for structural parameters + bounds for individual error terms

    def objective(theta):
        return loglikelihood(theta, y=y, lotteries=lotteries, subjects=subjects)

    result = run_multistart_mle(objective, n_starts=n_starts, param_bounds=full_bounds)
    if result is None:
        raise RuntimeError("No successful optimization run was found in multistart MLE.")
    return result


def format_results(result):
    param_names = ["r", "Alpha", "Lambda", "Gamma", "R"]
    return pd.DataFrame(
        {"Parameter": pd.Series(param_names), "Estimate": pd.Series(result.x[:5])}
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
    ksi_values = result.x[5:]
    ksi_lines = [f"{subj}\t{ksi:.6f}" for subj, ksi in zip(subjects, ksi_values)]
    with open("ksi_estimates.txt", "w") as fh:
        fh.write("participant_label\tksi\n")
        fh.write("\n".join(ksi_lines) + "\n")
    print("Individual ksi estimates written to ksi_estimates.txt")
