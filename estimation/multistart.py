"""Parallel EM restarts (random seeds and warm starts) with per-restart checkpoints."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np

from .config import LOTTERIES, MAX_ITER, TOL, VALUATION_VERSION
from .data import observed_ce
from .em import grid_em_mixture
from .lottery_paths import transform
from .prepare import prepare_grid_data
from .spec import ModelSpec
from .store import atomic_pickle, load_pickle


def _restart_worker(args):
    spec, seed, max_iter, tol, initial_values, prepared, checkpoint_dir, restart_id, bounds = args
    t0 = time.time()
    result = grid_em_mixture(spec, max_iter=max_iter, tol=tol, seed=seed, verbose=False,
                             initial_values=initial_values, prepared=prepared, bounds=bounds,
                             progress_path=Path(checkpoint_dir)/f"{restart_id}.progress.json" if checkpoint_dir else None)
    result["elapsed"] = time.time() - t0
    result["restart_id"] = restart_id
    if checkpoint_dir:
        atomic_pickle(Path(checkpoint_dir)/f"{restart_id}.pkl", result)
    return result


def grid_em_multistart(spec: ModelSpec, seeds, y=None, lotteries=None,
                       max_iter=MAX_ITER, tol=TOL, n_workers=None,
                       verbose=True, initializations=None, checkpoint_dir=None, bounds=None):
    """Run grid_em_mixture from every warm start (``initializations``: label ->
    initial values) and seed in parallel; return the best LL with restart
    summaries. Finished restarts in ``checkpoint_dir`` are reused."""
    prepared = prepare_grid_data(observed_ce() if y is None else y,
                                 transform(LOTTERIES) if lotteries is None else lotteries)
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
            r = load_pickle(checkpoint)
            if r.get("valuation_version") != VALUATION_VERSION or r["spec"] != spec:
                raise ValueError(f"Incompatible restart checkpoint: {checkpoint}")
            results.append(r)
            if r["log_likelihood"] > best_ll:
                best_ll, best = r["log_likelihood"], r
        else:
            args_list.append((spec, seed, max_iter, tol, initial_values, prepared,
                              str(checkpoint_dir) if checkpoint_dir else None, restart_id, bounds))

    if verbose:
        print(f"\n[{spec.config_id()}]  {n_seeds} restarts  C={spec.C}  "
              f"pw={spec.pw}", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_restart_worker, a): a[7] for a in args_list}
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
