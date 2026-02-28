import pandas as pd

from scipy.optimize import minimize

from scipy.stats import norm

from tabulate import tabulate

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






# print(loglikelihood(params, lotteries=f.transform(one)))


bounds = [(1e-6, None)] * 6

result = minimize(loglikelihood, x0=[1, 2, 3, 4, 0, 6], bounds=bounds)

print(result.x)

print(len(result.x))


# Define your parameter names
param_names = ["r", 'Alpha', 'Lambda', "Gamma", 'R', 'ksi']


# Create a clean table
results_df = pd.DataFrame({
    'Parameter': pd.Series(param_names),
    'Estimate': pd.Series(result.x)
})


print("\n### MAXIMUM LIKELIHOOD ESTIMATES ###")
print(results_df.to_string(index=False))

