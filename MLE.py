import numpy as np
import pandas as pd
from scipy.optimize import minimize

import functions as f
from lotteries import lotteries_full, one
from main import get_observed_ce
import openpyxl

# r, alpha, lamb, gamma, R
params = [0.97, 0.88, 2.25, 0.61, 0]

bounds = [
    (1e-4, 3.0),    # r
    (0.2, 2),       # alpha  <-- too little alpha causes underflow
    (1e-3, 6.0),    # lamb
    (0.2, 1.5),     # gamma
    (-100, 100),    # R
    # ksi removed: profiled out analytically per subject
]


# Iteration time
n_starts = 1000
lottery = lotteries_full

# random setup
np.random.seed(5)

def loglikelihood(params, y=None, lotteries=None):
    """
    Concentrated negative log-likelihood with per-subject ksi profiled out analytically.

    For each subject i with J_i lottery responses, the MLE of ksi_i at fixed CPT
    parameters theta is:
        ksi_hat_i = sqrt( S_i / J_i )
        where S_i = sum_j [ (ce_obs[i,j] - ce_th[j])^2 / spread[j]^2 ]

    Substituting ksi_hat_i back gives the concentrated objective:
        min_theta  sum_i  J_i * log(S_i(theta))

    ksi_i is never in the optimizer; it can be recovered post-estimation.
    """
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lottery)

    r, alpha, lamb, gamma, R = params
    ce_theoretical = f.ce_dict(r, gamma, alpha, lamb, R, lotteries=lotteries)

    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}

    y = y.copy()
    y["ce_th"] = y["lottery_id"].map(ce_theoretical)
    y["spread"] = y["lottery_id"].map(spreads)
    y["std_resid_sq"] = ((y["ce_observed"] - y["ce_th"]) / y["spread"]) ** 2

    s = 0.0
    for _, group in y.groupby("participant_label"):
        J_i = len(group)
        S_i = group["std_resid_sq"].sum()
        if S_i <= 0:
            return np.inf
        s += J_i * np.log(S_i)

    return float(s)


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
        random_guess = np.random.uniform(lower_bounds, upper_bounds) # A random starting guess within bounds for each parameter

        res = minimize(
            obj_func,
            x0=random_guess,
            bounds=param_bounds,
            method="L-BFGS-B",
        ) #scipy minimize with L-BFGS-B method for bounded optimization

        if res.success and res.fun < best_f:
            best_f = res.fun
            best_result = res
            print(f"\nRun {i + 1}: New best Log-Likelihood found: {-res.fun:.4f}")

    print()
    return best_result


def estimate_mle(n_starts=n_starts, param_bounds=bounds, y=None, lotteries=None):
    """
    Estimate parameters with multistart MLE and fail loudly if nothing converged.
    """
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lottery)

    # Bind data/lotteries once so each optimizer call only varies parameters.
    def objective(theta):
        return loglikelihood(theta, y=y, lotteries=lotteries)

    result = run_multistart_mle(objective, n_starts=n_starts, param_bounds=param_bounds)
    if result is None:
        raise RuntimeError("No successful optimization run was found in multistart MLE.")
    return result


def format_results(result):
    param_names = ["r", "Alpha", "Lambda", "Gamma", "R"]
    return pd.DataFrame(
        {"Parameter": pd.Series(param_names), "Estimate": pd.Series(result.x)}
    )


if __name__ == "__main__":
    result = estimate_mle(n_starts=n_starts, param_bounds=bounds)
    results_df = format_results(result)

    print("\nMAXIMUM LIKELIHOOD ESTIMATES")
    print(results_df.to_string(index=False))
    print(f"Best Log-Likelihood: {-result.fun:.4f}")
