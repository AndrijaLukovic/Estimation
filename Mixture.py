import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import functions as f
from lotteries import lotteries_full, one, all_high_stake, all_low_stake
from main import get_observed_ce
import openpyxl


# ── Lottery set to estimate on ──────────────────────────────────────────────
# Switch between all_low_stake / all_high_stake / lotteries_full
lottery = lotteries_full
# ────────────────────────────────────────────────────────────────────────────


prob_weighter = "prelec"


# n number of individuals

# params = [[]]

# mixture(thetas = [[]], pis = [], clusters=c, individual =[], y=y, method=prob_weighter, n=n, lotteries=lotteries, subjects=None)

# Thetas is a list of lists, each sublist a collection of cluster-specific parameters

# pis 



def mixture(thetas, pis, individual, clusters, method, c=1, y=None, lotteries=None, subjects=None):
    
    ksi = individual
    
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

    for i in range(n):

        s = 0

        for i in range(c):

            params = thetas[i]

            if method == "tk":
                r, alpha, lamb, gamma = params[:4]
                beta, palpha = 1, 1   # defaults passed to ce_dict but unused by TK                

            elif method == "prelec":
                r, alpha, lamb, beta, palpha = params[:5]
                gamma = 0.61          # default passed to ce_dict but unused by Prelec
        
            R = 0

            ce_theoretical = f.ce_dict(r, gamma, alpha, lamb, R, lotteries=lotteries, method=method, beta=beta, palpha=palpha)

            y["ce_th"] = y["lottery_id"].map(ce_theoretical)

            likelihood_individual = np.prod(norm.pdf(y["ce_observed"], loc=y["ce_th"], scale=y["sigma"]))

            s += pis[i] * likelihood_individual

        L += np.log(s)



        def EM(objective):

            pass