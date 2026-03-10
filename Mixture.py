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


# Choice of the method

prob_weighter = "prelec"



def mixture(thetas, pis, individual, method, c=1, y=None, lotteries=None, subjects=None):
    
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





def compute_log_likelihoods(thetas, ksi, method, c, subjects, grouped_y, y, lotteries):
    
    """Returns (n, c) matrix of log-likelihoods: log L_ij"""

    n = len(subjects)

    ksi_map = {subj: ksi[i] for i, subj in enumerate(subjects)}

    y = y.copy()

    y["sigma"] = y["participant_label"].map(ksi_map) * y["spread"]

    log_L = np.zeros((n, c))

    for j, params in enumerate(thetas):

        if method == "tk":

            r, alpha, lamb, gamma = params[:4]

            beta, palpha = 1, 1

        elif method == "prelec":

            r, alpha, lamb, beta, palpha = params[:5]

            gamma = 0.61

        else:

            raise ValueError(f"Unknown method: {method!r}")


        R = 0

        ce_theoretical = f.ce_dict(r, gamma, alpha, lamb, R = R, lotteries=lotteries, method=method, beta=beta, palpha=palpha)


        for i, subj_label in enumerate(subjects):

            y_subj   = grouped_y.get_group(subj_label)

            loc_vals = y_subj["lottery_id"].map(ce_theoretical)

            log_L[i, j] = np.sum(norm.logpdf(y_subj["ce_observed"], loc=loc_vals, scale=y_subj["sigma"]))
    
    return log_L






def em_mixture(thetas=None, pis=None, individual=None, method="prelec", c=1, y=None, lotteries=None, subjects=None, max_iter=100, tol=1e-6):

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

        pis = np.ones(c) / c

    if thetas is None:

        if method == "tk":

            thetas = [np.array([0.5, 0.88, 2.25, 0.65]) + np.random.randn(4) * 0.05 for _ in range(c)]
        
        elif method == "prelec":  
            
            thetas = [np.array([0.5, 0.88, 2.25, 1.0, 0.65]) + np.random.randn(5) * 0.05 for _ in range(c)]
        
        else:
            
            raise ValueError(f"Unknown method: {method!r}")

    
    if individual is None:

        ksi = np.ones(n) * 0.1

    else:
        
        ksi = individual


    # ---- EM loop ---- #
    prev_ll = -np.inf

    for iteration in range(max_iter):

        # E-step
        log_L      = compute_log_likelihoods(thetas, ksi, method, c, subjects, grouped_y, y, lotteries)
        log_pi     = np.log(pis)
        log_joint  = log_L + log_pi[np.newaxis, :]
        log_sum    = logsumexp(log_joint, axis=1, keepdims=True)
        r          = np.exp(log_joint - log_sum)

        ll = np.sum(log_sum)
        print(f"Iter {iteration:3d} | LL = {ll:.4f}")

        if abs(ll - prev_ll) < tol:
            print("Converged.")
            break
        prev_ll = ll

        # M-step: mixing weights
        pis = r.mean(axis=0)

        # M-step: cluster params
        bounds_tk     = [(0.01, 2), (0.01, 2), (0.01, 10), (0.01, 2)]
        bounds_prelec = [(0.01, 2), (0.01, 2), (0.01, 10), (0.01, 5), (0.01, 2)]
        bounds = bounds_tk if method == "tk" else bounds_prelec

        for j in range(c):
            r_j = r[:, j]

            def neg_weighted_ll_theta(params, j=j, r_j=r_j):
                thetas_temp    = list(thetas)
                thetas_temp[j] = params
                log_L_temp     = compute_log_likelihoods(thetas_temp, ksi, method, c, subjects, grouped_y, y, lotteries)
                return -np.sum(r_j * log_L_temp[:, j])

            result     = minimize(neg_weighted_ll_theta, thetas[j], method="L-BFGS-B", bounds=bounds)
            thetas[j]  = result.x

        # M-step: ksi
        def neg_weighted_ll_ksi(ksi_params):
            log_L_temp = compute_log_likelihoods(thetas, ksi_params, method, c, subjects, grouped_y, y, lotteries)
            return -np.sum(r * log_L_temp)

        result = minimize(neg_weighted_ll_ksi, ksi, method="L-BFGS-B", bounds=[(1e-4, 5)] * n)
        ksi    = result.x

    return {"thetas": thetas, "pis": pis, "ksi": ksi, "log_likelihood": ll}