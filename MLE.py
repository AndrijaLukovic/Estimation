import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import functions as f
from lotteries import lotteries_full
from main import get_observed_ce


# r, alpha, lamb, gamma, R, ksi
params = [0.97, 0.88, 2.25, 0.61, 0, 1]

# Keep your original bounds for now (not changed in this patch request).
bounds = [(1e-10, 3)] * 6


def loglikelihood(params, y=None, lotteries=None):
    """Negative log-likelihood under Normal errors around CPT-implied CE."""
    if y is None:
        # Lazy-load observed CE so importing this module has no heavy side effects.
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lotteries_full)

    r, alpha, lamb, gamma, R, ksi = params
    ce_theoretical = f.ce_dict(r, gamma, alpha, lamb, R, lotteries=lotteries)

    s = 0.0
    for lottery_id, lottery in lotteries.items():
        sigma = ksi * lottery["spread"]
        ce_observed = y.loc[y["lottery_id"] == lottery_id, "ce_observed"]
        ce = ce_theoretical[lottery_id]
        x = norm.logpdf(ce_observed, loc=ce, scale=sigma)
        s += np.sum(x)

    return -float(s)


def run_multistart_mle(obj_func, n_starts=100, param_bounds=None):
    """Run bounded multistart optimization and return the best successful result."""
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
            options={"maxiter": 100000},
        )

        if res.success and res.fun < best_f:
            best_f = res.fun
            best_result = res
            print(f"\nRun {i + 1}: New best Log-Likelihood found: {-res.fun:.4f}")

    print()
    return best_result


def estimate_mle(n_starts=1000, param_bounds=bounds, y=None, lotteries=None):
    """Estimate parameters with multistart MLE and fail loudly if nothing converged."""
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lotteries_full)

    # Bind data/lotteries once so each optimizer call only varies parameters.
    def objective(theta):
        return loglikelihood(theta, y=y, lotteries=lotteries)

    result = run_multistart_mle(objective, n_starts=n_starts, param_bounds=param_bounds)
    if result is None:
        raise RuntimeError("No successful optimization run was found in multistart MLE.")
    return result


def format_results(result):
    """Return a clean estimate table."""
    param_names = ["r", "Alpha", "Lambda", "Gamma", "R", "ksi"]
    return pd.DataFrame(
        {"Parameter": pd.Series(param_names), "Estimate": pd.Series(result.x)}
    )


if __name__ == "__main__":
    result = estimate_mle(n_starts=1000, param_bounds=bounds)
    results_df = format_results(result)

    print("\nMAXIMUM LIKELIHOOD ESTIMATES")
    print(results_df.to_string(index=False))
