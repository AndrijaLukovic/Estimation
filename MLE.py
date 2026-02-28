import pandas as pd

from scipy.optimize import minimize

from scipy.stats import norm

from lotteries import lotteries, lotteries_full, one

import functions as f

from main import y

import numpy as np





# r, alpha, lamb, gamma, R, ksi, desired = 0.97, 0.88, 2.25, 0.61, 0, 1, "lottery_3"

params = [0.97, 0.88, 2.25, 0.61, 0, 1]



def loglikelihood(params, y=y, lotteries = f.transform(one)):

    r, alpha, lamb, gamma, R, ksi = params

    l = lotteries.keys()

    ce_theoretical = f.ce_dict(r, gamma, alpha, lamb, R, lotteries=lotteries)

    s = 0

    for i in l:

        lottery = lotteries[i]

        sigma = ksi*lottery["spread"]

        ce_observed = y[y["lottery_id"] == i]["selected_amount"].copy()

        ce = ce_theoretical[i]

        temp = np.array(norm.pdf((ce_observed - ce) / sigma)*(1/sigma))

        x = [np.log(temp[i]) for i in range(len(temp))]

        s += np.sum(x)

    
    return -s



bounds = [(1e-10, 3)] * 6

result = minimize(loglikelihood, x0=[1, 2, 3, 4, 0, 6], bounds=bounds)



def run_multistart_mle(obj_func, n_starts=100, param_bounds=None):
    
    best_result = None
    best_f = np.inf  # We minimize -LogLikelihood, so look for the lowest value

    lower_bounds = np.array([b[0] if b[0] is not None else -2 for b in param_bounds])
    upper_bounds = np.array([b[1] if b[1] is not None else 10 for b in param_bounds])

    for i in range(n_starts):

        random_guess = np.random.uniform(lower_bounds, upper_bounds)
        
        res = minimize(obj_func, x0=random_guess, bounds=param_bounds, method='L-BFGS-B')
        
        if res.success and res.fun < best_f:
            best_f = res.fun
            best_result = res
            print(f"Run {i+1}: New best Log-Likelihood found: {-res.fun:.4f}")

    return best_result


result = run_multistart_mle(loglikelihood, n_starts=1000, param_bounds=bounds)



# Define your parameter names
param_names = ["r", 'Alpha', 'Lambda', "Gamma", 'R', 'ksi']


# Create a clean table
results_df = pd.DataFrame({
    'Parameter': pd.Series(param_names),
    'Estimate': pd.Series(result.x)
})


print("\nMAXIMUM LIKELIHOOD ESTIMATES")
print(results_df.to_string(index=False))
