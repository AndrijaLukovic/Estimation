from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np
from scipy.optimize import minimize

import functions as f
from main import get_observed_ce
from MLE import loglikelihood, bounds, params, n_starts, lottery, format_results

from lotteries import low_stake, high_stake


def _single_run(random_guess, obj_func, param_bounds):
    """
    Module-level helper for parallel execution.
    Must be at module level (not a closure) to be picklable by ProcessPoolExecutor.
    """
    return minimize(obj_func, x0=random_guess, bounds=param_bounds, method="L-BFGS-B")


def run_multistart_mle_parallel(obj_func, n_starts=n_starts, param_bounds=None):
    """
    Run all multistart optimizations simultaneously across all CPU cores.
    All n_starts random starting points are submitted at once; results are
    collected as they complete.
    """
    best_result = None
    best_f = np.inf

    lower_bounds = np.array([b[0] if b[0] is not None else -2 for b in param_bounds])
    upper_bounds = np.array([b[1] if b[1] is not None else 10 for b in param_bounds])
    random_guesses = [np.random.uniform(lower_bounds, upper_bounds) for _ in range(n_starts)]

    completed = 0
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_single_run, guess, obj_func, param_bounds)
            for guess in random_guesses
        ]
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res.fun < best_f:
                best_f = res.fun
                best_result = res
                print(f"[{completed}/{n_starts}] New best Log-Likelihood: {-res.fun:.4f}")
            else:
                print(f"[{completed}/{n_starts}] done", end="\r", flush=True)

    print()
    return best_result


def estimate_mle_parallel(n_starts=n_starts, param_bounds=bounds, y=None, lotteries=None):
    """
    Parallel multistart MLE. Identical specification to estimate_mle() in MLE.py
    but uses all CPU cores simultaneously.

    Uses functools.partial instead of a closure for the objective — partial objects
    are picklable, which is required to send work to ProcessPoolExecutor workers.
    """
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        lotteries = f.transform(lottery)

    subjects = sorted(y["participant_label"].unique())
    full_bounds = list(param_bounds) + [(1e-3, 3.0)] * len(subjects)

    objective = partial(loglikelihood, y=y, lotteries=lotteries, subjects=subjects)

    result = run_multistart_mle_parallel(objective, n_starts=n_starts, param_bounds=full_bounds)
    if result is None:
        raise RuntimeError("No successful optimization run was found.")
    return result


if __name__ == "__main__":
    result = estimate_mle_parallel(n_starts=n_starts, param_bounds=bounds)
    results_df = format_results(result)

    print("\nMAXIMUM LIKELIHOOD ESTIMATES (PARALLEL)")
    print(results_df.to_string(index=False))
    print(f"Best Log-Likelihood: {-result.fun:.4f}")
