"""Spec-driven EM for conditional remaining-payoff lotteries.

Uses the existing layouts, parameter bounds, Gaussian likelihood and L-BFGS-B
structural M-step. Unique-state predictions match Mixture._compute_ce_th;
Q aggregates by state and the noise M-step uses its exact Gaussian solution.
Final LL/responsibilities are always recomputed. Restarts are checkpointed.
"""

import time
import json
import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import functions as f
from GlobalSettings import GlobalInterMax, GlobalTol, GlobalKsiBounds, GlobalRSQMode
from Mixture import _preprocess_rows
from main import get_observed_ce
from model_spec import (
    ModelSpec, build_cluster_layout, expand_free, bounds_free, init_free,
    param_names_free, k_free, RP_LABELS_4,
)


def spec_cluster_layouts(spec: ModelSpec):
    """Return (layouts, rp_identities) — one entry per cluster."""
    if spec.rp_mode == "mixed":
        return ([build_cluster_layout(spec) for _ in range(spec.C)],
                [None] * spec.C)
    elif spec.rp_mode == "fixed_pure":
        assert spec.C == 1, "fixed_pure rp_mode always has C=1"
        return ([build_cluster_layout(spec, rp_identity=spec.rp_identity)],
                [spec.rp_identity])
    elif spec.rp_mode == "fixed_4":
        assert spec.C == 4, "fixed_4 rp_mode always has C=4"
        return ([build_cluster_layout(spec, rp_identity=lbl) for lbl in RP_LABELS_4],
                list(RP_LABELS_4))
    raise ValueError(f"Unknown rp_mode: {spec.rp_mode!r}")


def _expand_all(free_vecs, layouts):
    thetas, delta_les = [], []
    for fv, layout in zip(free_vecs, layouts):
        th, dle = expand_free(fv, layout)
        thetas.append(th)
        delta_les.append(dle)
    return thetas, delta_les


@dataclass
class PreparedGridData:
    """Fixed conditional states and row indices, prepared once per EM restart."""
    subjects: list
    state_y: object
    reference_inputs: tuple
    streams: np.ndarray
    probabilities: np.ndarray
    path_mask: np.ndarray
    state_index: np.ndarray
    subject_index: np.ndarray
    spread: np.ndarray
    observed: np.ndarray
    counts: np.ndarray

    def predict(self, theta, method, delta_le=None):
        """Vectorized equivalent of _compute_ce_th over unique current states."""
        el, fe, c1, stage, z1, z2, zt = self.reference_inputs
        if method == "tk":
            r, ag, al, lam, gamma, s1, s2, s3, delta, psi = theta[:10]
            beta, palpha = 1.0, 1.0
        else:
            r, ag, al, lam, beta, palpha, s1, s2, s3, delta, psi = theta[:11]
            gamma = 1.0
        a1 = s1
        a2 = s2 * (1 - s1)
        a3 = s3 * (1 - s1) * (1 - s2)
        a4 = 1 - a1 - a2 - a3
        ra = np.where(stage == 1, z1 / (delta + 1), np.where(
            stage == 2, (delta * z1 + z2) / (delta ** 2 + delta + 1), 0))
        dle = delta if delta_le is None else delta_le
        rle = np.where(stage == 1, el, np.where(stage == 2, (dle * el + c1) / (dle + 1), 0))
        reference = a1 * (zt if GlobalRSQMode == "zt" else 0) + a2 * ra + a3 * rle + a4 * fe
        delays = np.arange(self.streams.shape[2])
        discount = psi * np.exp(-r * delays)
        discount[0] = 1.0
        pv = np.sum(self.streams * discount, axis=2)
        distance = pv - reference[:, None]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            utility = np.where(distance >= 0, np.maximum(distance, 0) ** ag,
                               -lam * np.maximum(-distance, 0) ** al)
            order = np.argsort(np.where(self.path_mask, -utility, np.inf), axis=1, kind="stable")
            u_sorted = np.take_along_axis(utility, order, axis=1)
            p_sorted = np.take_along_axis(self.probabilities, order, axis=1)
            mask_sorted = np.take_along_axis(self.path_mask, order, axis=1)
            u_sorted = np.where(mask_sorted, u_sorted, 0)
            left = np.cumsum(p_sorted, axis=1)
            right = np.cumsum(p_sorted[:, ::-1], axis=1)[:, ::-1]
            full_prefix = np.arange(p_sorted.shape[1])[None, :] >= mask_sorted.sum(axis=1)[:, None] - 1
            left[full_prefix] = 1.0
            right[:, 0] = 1.0
            left_before = np.column_stack([np.zeros(len(left)), left[:, :-1]])
            right_after = np.column_stack([right[:, 1:], np.zeros(len(right))])

            def weight(p):
                p = np.clip(p, 0, 1)
                out = np.zeros_like(p)
                inside = (p > 0) & (p < 1)
                value = p[inside]
                if method == "tk":
                    out[inside] = value ** gamma / (value ** gamma + (1 - value) ** gamma) ** (1 / gamma)
                else:
                    out[inside] = np.exp(-beta * (-np.log(value)) ** palpha)
                out[p == 1] = 1
                return out

            gain_weights = weight(left) - weight(left_before)
            loss_weights = weight(right) - weight(right_after)
            value = np.sum(u_sorted * np.where(u_sorted > 0, gain_weights, loss_weights), axis=1)
            ce = reference + np.where(value >= 0, np.maximum(value, 0) ** (1 / ag),
                                     -(np.maximum(-value, 0) / lam) ** (1 / al))
        return ce


def prepare_grid_data(y, lotteries):
    active = y[y.lottery_id.isin(lotteries)].dropna(subset=["ce_observed"])
    subjects = sorted(active.participant_label.unique())
    prepared = _preprocess_rows(active, lotteries, {s: i for i, s in enumerate(subjects)})
    yp, el, fe, c1, stage, z1, z2, zt, si, spread, obs = prepared
    groups = yp.attrs["_ce_state_groups"]
    first = np.array([indices[0] for _, indices, _ in groups])
    state_index = np.empty(len(yp), dtype=int)
    for index, (_, indices, _) in enumerate(groups):
        state_index[indices] = index
    state_y = yp.iloc[first].reset_index(drop=True)
    state_y.attrs = {"_ce_state_groups": [
        (key, np.array([index]), current) for index, (key, _, current) in enumerate(groups)]}
    max_paths = max(len(current["outcomes"]) for _, _, current in groups)
    max_delays = max(len(stream) for _, _, current in groups
                     for path in current["outcomes"].values() for stream in path.values())
    streams = np.zeros((len(groups), max_paths, max_delays))
    probabilities = np.zeros((len(groups), max_paths))
    mask = np.zeros((len(groups), max_paths), dtype=bool)
    for state, (_, _, current) in enumerate(groups):
        for path_index, path in enumerate(current["outcomes"].values()):
            probability, stream = next(iter(path.items()))
            streams[state, path_index, :len(stream)] = stream
            probabilities[state, path_index] = probability
            mask[state, path_index] = True
    return PreparedGridData(subjects, state_y,
        tuple(x[first] for x in (el, fe, c1, stage, z1, z2, zt)), streams, probabilities,
        mask, state_index, si, spread, obs, np.bincount(si, minlength=len(subjects)))


def noise_mstep(resp, row_mu, data):
    """Exact bounded Gaussian M-step; same objective as scalar minimization."""
    squared = ((data.observed[:, None] - row_mu) / data.spread[:, None]) ** 2
    weighted = np.sum(squared * resp[data.subject_index], axis=1)
    variance = np.bincount(data.subject_index, weights=weighted,
                          minlength=len(data.subjects)) / data.counts
    return np.clip(np.sqrt(variance), *GlobalKsiBounds)


def grid_em_mixture(spec: ModelSpec, y=None, lotteries=None,
                     max_iter=GlobalInterMax, tol=GlobalTol, seed=None,
                     verbose=False, initial_values=None, prepared=None,
                     progress_path=None):
    """One EM run (one random initialisation) for the given ModelSpec."""
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        from GlobalSettings import GlobalLottery
        lotteries = f.transform(GlobalLottery)

    data = prepare_grid_data(y, lotteries) if prepared is None else prepared
    subjects = data.subjects
    n = len(subjects)

    layouts, rp_identities = spec_cluster_layouts(spec)
    c = spec.C
    method = spec.pw

    rng = np.random.default_rng(seed)
    pis = np.ones(c) / c
    free_vecs = [init_free(layouts[j], rng) for j in range(c)]
    ksi = np.ones(n) * 0.1
    if initial_values is not None:
        free_vecs = [np.asarray(vector, dtype=float).copy() for vector in initial_values["free_vecs"]]
        pis = np.asarray(initial_values["pis"], dtype=float).copy()
        ksi = np.asarray(initial_values["ksi"], dtype=float).copy()
        assert len(free_vecs) == c and len(ksi) == n
        for vector, layout in zip(free_vecs, layouts):
            assert len(vector) == k_free(layout)
        pis = np.maximum(pis, 1e-300); pis /= pis.sum()
        ksi = np.clip(ksi, *GlobalKsiBounds)

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
        path = Path(progress_path)
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(dict(seed=seed, iteration=iteration,
            log_likelihood=likelihood, pis=pis.tolist(), status=status,
            elapsed_seconds=time.time()-started)))
        temporary.replace(path)

    def expectation():
        expanded, deltas = _expand_all(free_vecs, layouts)
        state_mu = np.column_stack([data.predict(theta, method, delta)
                                    for theta, delta in zip(expanded, deltas)])
        row_mu = state_mu[data.state_index]
        sigma = ksi[data.subject_index] * data.spread
        lpdf = -.5 * ((data.observed[:, None] - row_mu) / sigma[:, None]) ** 2 - np.log(sigma[:, None]) - .5 * np.log(2 * np.pi)
        log_l = np.column_stack([np.bincount(data.subject_index, weights=lpdf[:, j], minlength=n) for j in range(c)])
        joint = log_l + np.log(np.maximum(pis, 1e-300))[None, :]
        sums = logsumexp(joint, axis=1)
        return float(sums.sum()), np.exp(joint - sums[:, None]), state_mu, row_mu

    for iteration in range(max_iter):
        t0 = time.time()

        ll, resp, state_mu, row_mu = expectation()
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
            res = minimize(neg_q, free_vecs[j], method="L-BFGS-B",
                            bounds=bounds_free(layout_j))
            accepted = np.isfinite(res.fun) and res.fun <= q_before + 1e-8
            if accepted:
                free_vecs[j] = res.x
            optimizer_log.append(dict(iteration=iteration + 1, cluster=j,
                success=bool(res.success), status=int(res.status), message=str(res.message),
                q_before=q_before, q_after=float(res.fun), accepted=bool(accepted),
                nfev=int(res.nfev)))

        thetas, delta_les = _expand_all(free_vecs, layouts)

        # ── M-step: ksi ──────────────────────────────────────────────────
        state_mu = np.column_stack([data.predict(theta, method, delta)
                                    for theta, delta in zip(thetas, delta_les)])
        ksi = noise_mstep(resp, state_mu[data.state_index], data)

        iter_log.append(dict(iteration=iteration + 1, log_likelihood=ll,
                              improvement=improvement, soft_n=soft_n.copy(),
                              pis=pis.copy(), elapsed=time.time() - t0))

    n_iter = iteration + 1
    thetas, delta_les = _expand_all(free_vecs, layouts)
    # Always save likelihood and responsibilities at the returned parameters,
    # including an iteration-limit exit after the final M-step.
    final_ll, resp, _, _ = expectation()
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
        "optimizer_log": optimizer_log, "valuation_version": "conditional_remaining_v1",
        "n_obs_effective": len(data.observed),
    }


def _restart_worker(args):
    spec, seed, y, lotteries, max_iter, tol, initial_values, prepared, checkpoint_dir, restart_id = args
    t0 = time.time()
    result = grid_em_mixture(spec, y=y, lotteries=lotteries, max_iter=max_iter,
                              tol=tol, seed=seed, verbose=False,
                              initial_values=initial_values, prepared=prepared,
                              progress_path=Path(checkpoint_dir)/f"{restart_id}.progress.json" if checkpoint_dir else None)
    result["elapsed"] = time.time() - t0
    result["restart_id"] = restart_id
    if checkpoint_dir:
        path = Path(checkpoint_dir)/f"{restart_id}.pkl"
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        with temporary.open("wb") as fh:
            pickle.dump(result, fh)
        temporary.replace(path)
    return result


def grid_em_multistart(spec: ModelSpec, seeds, y=None, lotteries=None,
                        max_iter=GlobalInterMax, tol=GlobalTol, n_workers=None,
                        verbose=True, initializations=None, checkpoint_dir=None):
    """Run grid_em_mixture from multiple seeds in parallel; keep the best LL.
    Mirrors EM_parallel.run_em_multistart's restart strategy exactly."""
    if y is None:
        y = get_observed_ce(export_excel=False)
    if lotteries is None:
        from GlobalSettings import GlobalLottery
        lotteries = f.transform(GlobalLottery)

    spreads = {lid: lotteries[lid]["spread"] for lid in lotteries}
    y = y[y["lottery_id"].isin(lotteries.keys())].copy()
    y["spread"] = y["lottery_id"].map(spreads)

    prepared = prepare_grid_data(y, lotteries)
    starts = [(label, None, values) for label, values in (initializations or {}).items()]
    starts += [(f"seed_{seed}", seed, None) for seed in seeds]
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    args_list = []

    results = []
    best = None
    best_ll = -np.inf
    n_seeds = len(starts)

    for restart_id, seed, initial_values in starts:
        checkpoint = Path(checkpoint_dir)/f"{restart_id}.pkl" if checkpoint_dir else None
        if checkpoint and checkpoint.exists():
            with checkpoint.open("rb") as fh:
                r = pickle.load(fh)
            if r.get("valuation_version") != "conditional_remaining_v1" or r["spec"] != spec:
                raise ValueError(f"Incompatible restart checkpoint: {checkpoint}")
            results.append(r)
            if r["log_likelihood"] > best_ll:
                best_ll, best = r["log_likelihood"], r
        else:
            args_list.append((spec, seed, y, lotteries, max_iter, tol, initial_values,
                              prepared, str(checkpoint_dir) if checkpoint_dir else None, restart_id))

    if verbose:
        print(f"\n[{spec.config_id()}]  {n_seeds} restarts  C={spec.C}  "
              f"pw={spec.pw}", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_restart_worker, a): a[-1] for a in args_list}
        completed = len(results)
        for fut in as_completed(futures):
            completed += 1
            seed_done = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                if verbose:
                    conv = "converged" if r["converged"] else "NOT converged"
                    print(f"  [{completed:>2}/{n_seeds}] start={seed_done}  "
                          f"LL={r['log_likelihood']:>12.4f}  {conv}  "
                          f"{r['elapsed']:.1f}s", flush=True)
                if r["log_likelihood"] > best_ll:
                    best_ll = r["log_likelihood"]
                    best = r
            except Exception as exc:
                if verbose:
                    print(f"  [{completed:>2}/{n_seeds}] seed={seed_done} "
                          f"FAILED: {exc}", flush=True)

    if best is None:
        raise RuntimeError(f"All restarts failed for spec {spec.config_id()}")

    best["seeds_used"] = seeds
    best["n_converged"] = sum(1 for r in results if r["converged"])
    best["n_restarts_ok"] = len(results)
    best["n_random_restarts"] = len(seeds)
    best["n_warm_starts"] = len(initializations or {})
    best["restart_summaries"] = [dict(restart_id=r.get("restart_id"), seed=r["seed"],
        log_likelihood=r["log_likelihood"], converged=r["converged"], n_iter=r["n_iter"],
        exit_reason=r["exit_reason"], elapsed=r["elapsed"]) for r in results]
    return best


def count_free_params(result):
    """k = sum of per-cluster free structural params + (C-1) mixing weights
    + one ksi per subject. Matches the BIC prescription: only parameters
    actually estimated (per layout.free_positions / has_delta_le) are counted."""
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
