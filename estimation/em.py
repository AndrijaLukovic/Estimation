"""One EM run (one initialisation) of the conditional CPT mixture for a ModelSpec.

E-step: Gaussian mixture responsibilities. M-step: mixing weights, then per
cluster an L-BFGS-B fit of the free structural parameters to the
state-aggregated weighted squared errors, then the exact noise update.
Final LL and responsibilities are always recomputed at the returned parameters.
"""
import json
import time

import numpy as np
from scipy.optimize import minimize

from .config import KSI_BOUNDS, LOTTERIES, MAX_ITER, TOL, VALUATION_VERSION
from .data import observed_ce
from .likelihood import component_loglik, noise_mstep, posterior, predict_states
from .lottery_paths import transform
from .prepare import prepare_grid_data
from .spec import ModelSpec, bounds_free, expand_all, expand_free, init_free, k_free, spec_cluster_layouts
from .store import atomic_write_text


def grid_em_mixture(spec: ModelSpec, y=None, lotteries=None,
                    max_iter=MAX_ITER, tol=TOL, seed=None,
                    verbose=False, initial_values=None, prepared=None,
                    progress_path=None, bounds=None, layouts=None, free_bounds=None):
    """One EM run for ``spec``.

    bounds      : bound table for initial draws and default free bounds (None: config.BOUNDS).
    layouts     : cluster layouts replacing spec_cluster_layouts(spec), e.g. with
                  a profiled parameter pinned.
    free_bounds : per-cluster L-BFGS-B bounds replacing bounds_free(layout, bounds).
    """
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if prepared is None:
        prepared = prepare_grid_data(observed_ce() if y is None else y,
                                     transform(LOTTERIES) if lotteries is None else lotteries)
    data = prepared
    subjects = data.subjects
    n = len(subjects)

    default_layouts, rp_identities = spec_cluster_layouts(spec)
    layouts = default_layouts if layouts is None else layouts
    if free_bounds is None:
        free_bounds = [bounds_free(layout, bounds) for layout in layouts]
    c = spec.C
    method = spec.pw

    rng = np.random.default_rng(seed)
    pis = np.ones(c) / c
    free_vecs = [init_free(layouts[j], rng, bounds) for j in range(c)]
    ksi = np.ones(n) * 0.1
    if initial_values is not None:
        free_vecs = [np.asarray(vector, dtype=float).copy() for vector in initial_values["free_vecs"]]
        pis = np.asarray(initial_values["pis"], dtype=float).copy()
        ksi = np.asarray(initial_values["ksi"], dtype=float).copy()
        assert len(free_vecs) == c and len(ksi) == n
        for vector, layout in zip(free_vecs, layouts):
            assert len(vector) == k_free(layout)
        pis = np.maximum(pis, 1e-300)
        pis /= pis.sum()
        ksi = np.clip(ksi, *KSI_BOUNDS)

    prev_ll = -np.inf
    ll = prev_ll
    iter_log = []
    optimizer_log = []
    converged = False
    exit_reason = "max_iter"
    started = time.time()

    def write_progress(iteration, likelihood, status="running"):
        if progress_path is None:
            return
        atomic_write_text(progress_path, json.dumps(dict(seed=seed, iteration=iteration,
            log_likelihood=likelihood, pis=pis.tolist(), status=status,
            elapsed_seconds=time.time()-started)))

    def expectation():
        thetas, deltas = expand_all(free_vecs, layouts)
        state_mu = predict_states(thetas, deltas, method, data)
        ll, resp = posterior(component_loglik(state_mu, ksi, data), pis)
        return ll, resp, state_mu

    for iteration in range(max_iter):
        t0 = time.time()

        ll, resp, state_mu = expectation()
        if not np.isfinite(ll):
            raise FloatingPointError("Nonfinite observed-data likelihood")
        soft_n = resp.sum(axis=0)
        improvement = ll - prev_ll
        if iteration % 10 == 0:
            write_progress(iteration + 1, ll)

        if verbose:
            print(f"  [seed {seed}] iter {iteration+1:3d}  LL={ll:.4f}  "
                  f"d={improvement:+.4f}", flush=True)

        if abs(ll - prev_ll) < tol:
            converged = True
            exit_reason = "ll_tolerance"
            iter_log.append(dict(iteration=iteration + 1, log_likelihood=ll,
                                 improvement=improvement, soft_n=soft_n.copy(),
                                 pis=pis.copy(), elapsed=time.time() - t0))
            break
        prev_ll = ll

        # ── M-step: mixing weights ──────────────────────────────────────
        pis = np.maximum(resp.mean(axis=0), 1e-300)
        pis /= pis.sum()

        # ── M-step: per-cluster structural parameters ───────────────────
        for j in range(c):
            layout_j = layouts[j]
            resp_j = resp[:, j]
            weights = resp_j[data.subject_index] / (ksi[data.subject_index] * data.spread) ** 2
            weight_sum = np.bincount(data.state_index, weights=weights, minlength=len(state_mu))
            weighted_obs = np.bincount(data.state_index, weights=weights * data.observed, minlength=len(state_mu))
            target = np.divide(weighted_obs, weight_sum, out=np.zeros_like(weight_sum), where=weight_sum > 0)

            def neg_q(fv, layout_j=layout_j, weight_sum=weight_sum, target=target):
                th_j, dle_j = expand_free(fv, layout_j)
                mu = data.predict(th_j, method, dle_j)
                if not np.all(np.isfinite(mu)):
                    return 1e100
                return .5 * float(np.dot(weight_sum, (mu - target) ** 2))

            q_before = neg_q(free_vecs[j])
            res = minimize(neg_q, free_vecs[j], method="L-BFGS-B", bounds=free_bounds[j])
            accepted = np.isfinite(res.fun) and res.fun <= q_before + 1e-8
            if accepted:
                free_vecs[j] = res.x
            optimizer_log.append(dict(iteration=iteration + 1, cluster=j,
                success=bool(res.success), status=int(res.status), message=str(res.message),
                q_before=q_before, q_after=float(res.fun), accepted=bool(accepted),
                nfev=int(res.nfev)))

        # ── M-step: ksi ──────────────────────────────────────────────────
        thetas, delta_les = expand_all(free_vecs, layouts)
        state_mu = predict_states(thetas, delta_les, method, data)
        ksi = noise_mstep(resp, state_mu[data.state_index], data)

        iter_log.append(dict(iteration=iteration + 1, log_likelihood=ll,
                             improvement=improvement, soft_n=soft_n.copy(),
                             pis=pis.copy(), elapsed=time.time() - t0))

    n_iter = iteration + 1
    thetas, delta_les = expand_all(free_vecs, layouts)
    # Always save likelihood and responsibilities at the returned parameters,
    # including an iteration-limit exit after the final M-step.
    final_ll, resp, _ = expectation()
    final_ll_change = final_ll - ll
    ll = final_ll
    write_progress(n_iter, ll, exit_reason)

    return {
        "spec": spec, "thetas": thetas, "delta_les": delta_les,
        "free_vecs": [fv.copy() for fv in free_vecs], "layouts": layouts,
        "rp_identities": rp_identities, "pis": pis, "ksi": ksi,
        "log_likelihood": ll, "n_iter": n_iter, "converged": converged,
        "seed": seed, "resp": resp, "subjects": subjects, "iter_log": iter_log,
        "exit_reason": exit_reason, "final_ll_change": final_ll_change,
        "optimizer_log": optimizer_log, "valuation_version": VALUATION_VERSION,
        "n_obs_effective": len(data.observed),
    }
