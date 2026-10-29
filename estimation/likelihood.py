"""Gaussian mixture likelihood of observed CEs, noise update and BIC."""
import numpy as np
from scipy.special import logsumexp

from .config import KSI_BOUNDS
from .spec import k_free


def predict_states(thetas, delta_les, method, data):
    """(state, cluster) predicted CEs."""
    return np.column_stack([data.predict(theta, method, delta)
                            for theta, delta in zip(thetas, delta_les)])


def component_loglik(state_mu, ksi, data):
    """(subject, cluster) log-likelihood with sigma_il = ksi_i * spread_l."""
    row_mu = state_mu[data.state_index]
    sigma = ksi[data.subject_index] * data.spread
    lpdf = -.5 * ((data.observed[:, None] - row_mu) / sigma[:, None]) ** 2 - np.log(sigma[:, None]) - .5 * np.log(2 * np.pi)
    return np.column_stack([np.bincount(data.subject_index, weights=lpdf[:, j], minlength=len(data.subjects))
                            for j in range(state_mu.shape[1])])


def posterior(log_l, pis):
    """Observed-data log-likelihood and (subject, cluster) responsibilities."""
    joint = log_l + np.log(np.maximum(pis, 1e-300))[None, :]
    sums = logsumexp(joint, axis=1)
    return float(sums.sum()), np.exp(joint - sums[:, None])


def result_loglik(result, data):
    """Recompute a saved result's log-likelihood and responsibilities."""
    mu = predict_states(result["thetas"], result["delta_les"], result["spec"].pw, data)
    if not np.isfinite(mu[data.state_index]).all():
        raise FloatingPointError("Nonfinite predictions")
    return posterior(component_loglik(mu, np.asarray(result["ksi"]), data), result["pis"])


def noise_mstep(resp, row_mu, data):
    """Exact bounded Gaussian M-step for ksi; same objective as scalar minimization."""
    squared = ((data.observed[:, None] - row_mu) / data.spread[:, None]) ** 2
    weighted = np.sum(squared * resp[data.subject_index], axis=1)
    variance = np.bincount(data.subject_index, weights=weighted,
                          minlength=len(data.subjects)) / data.counts
    return np.clip(np.sqrt(variance), *KSI_BOUNDS)


def count_free_params(result):
    """k = sum of per-cluster free structural params + (C-1) mixing weights
    + one ksi per subject. Only parameters actually estimated count."""
    layouts = result["layouts"]
    c = len(layouts)
    n_subjects = len(result["subjects"])
    k_struct = sum(k_free(layout) for layout in layouts)
    k_mix = c - 1
    return k_struct + k_mix + n_subjects


def compute_bic(result, n_obs):
    k = count_free_params(result)
    ll = result["log_likelihood"]
    bic = k * np.log(n_obs) - 2.0 * ll
    return bic, k
