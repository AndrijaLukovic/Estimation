from scipy.optimize import minimize

from lotteries import lotteries_full

import functions as f



def loglikelihood(y, r, gamma, alpha, lamb, R, ksi):

    l = f.lotteries_transformed.keys()

    s = 0

    for i in l:

        lottery = lotteries_full[i]


        sigma = ksi*lotter

        ce_observed = y[y[i] == "lottery_1"]["selected_amount"].copy()

