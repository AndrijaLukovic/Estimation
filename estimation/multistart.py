"""Parallel EM restarts (random seeds and warm starts) with per-restart checkpoints.

Optional short-run screening of the random starts (Biernacki, Celeux & Govaert
2003, "emEM"): every random start runs a few EM iterations, the best share
continues to convergence and the rest stop. Warm starts always run to the end.
Continuing a screening run is the same computation as one uninterrupted run, so
the choice only decides which starts are finished. How often the best
log-likelihood is reached by independent random starts (Hipp & Bauer 2006) is
reported from the finished random starts (run_summary.restart_hits)."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
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


def start_label(seed, alpha_init="normal"):
    """Checkpoint label of a random start; default-draw labels predate the option."""
    return f"seed_{seed}" if alpha_init == "normal" else f"seed_{seed}_{alpha_init}"


def ll_after(result, n_iter):
    """Log-likelihood after ``n_iter`` EM iterations (the final one if the run
    stopped earlier): iter_log[i] holds the value after i M-steps."""
    log = result["iter_log"]
    return log[n_iter]["log_likelihood"] if n_iter < len(log) else result["log_likelihood"]


def merge_runs(first, second):
    """One run from a screening run and its continuation."""
    offset = first["n_iter"]
    merged = dict(second)
    merged["n_iter"] = offset + second["n_iter"]
    for key in ("iter_log", "optimizer_log"):
        merged[key] = first[key] + [dict(row, iteration=row["iteration"] + offset) for row in second[key]]
    merged["elapsed"] = first["elapsed"] + second["elapsed"]
    merged["screened_at"] = offset
    return merged


def _restart_worker(task):
    """One EM run; ``previous`` (a screening run) is continued and merged."""
    task = dict(task)
    restart_id, checkpoint, previous = task.pop("restart_id"), task.pop("checkpoint"), task.pop("previous")
    if previous is not None:
        task["initial_values"] = dict(free_vecs=previous["free_vecs"], pis=previous["pis"], ksi=previous["ksi"],
                                      prev_ll=previous["iter_log"][-1]["log_likelihood"])
        task["max_iter"] -= previous["n_iter"]
    t0 = time.time()
    result = grid_em_mixture(verbose=False, **task)
    result.update(elapsed=time.time() - t0, restart_id=restart_id, alpha_init=task["alpha_init"])
    if previous is not None:
        result = merge_runs(previous, result)
    if checkpoint:
        atomic_pickle(checkpoint, result)
    return result


def _check(result, spec, subjects, path):
    if (result.get("valuation_version") != VALUATION_VERSION or result["spec"] != spec
            or list(result["subjects"]) != list(subjects)):
        raise ValueError(f"Incompatible restart checkpoint: {path}")
    return result


def grid_em_multistart(spec: ModelSpec, seeds, y=None, lotteries=None,
                       max_iter=MAX_ITER, tol=TOL, n_workers=None,
                       verbose=True, initializations=None, checkpoint_dir=None, bounds=None,
                       logunif_seeds=(), screen_iter=None, screen_keep=None, reuse_dir=None):
    """Run grid_em_mixture from every warm start (``initializations``: label ->
    initial values) and random seed in parallel; return the best LL with restart
    summaries.

    logunif_seeds : seeds (a subset of ``seeds``) that draw CARA alpha
                    log-uniformly (spec.init_free); the others use the default draw.
    screen_iter, screen_keep : run every random start ``screen_iter`` iterations,
                    then continue the share ``screen_keep`` with the highest
                    log-likelihood (ties to the earlier start). A start that already
                    finished counts with its log-likelihood at ``screen_iter``.
    checkpoint_dir : finished runs are saved as <label>.pkl, screening runs as
                    screen/<label>.pkl; both are reused.
    reuse_dir     : restart directory of another run of the same model and protocol;
                    its finished random starts are copied instead of re-estimated."""
    prepared = prepare_grid_data(observed_ce() if y is None else y,
                                 transform(LOTTERIES) if lotteries is None else lotteries)
    seeds, logunif = [int(seed) for seed in seeds], {int(seed) for seed in logunif_seeds}
    if len(set(seeds)) != len(seeds):
        raise ValueError("Random-start seeds must be distinct")
    if not logunif <= set(seeds):
        raise ValueError("logunif_seeds must be a subset of seeds")
    screening = screen_iter is not None
    if screening and not (1 <= screen_iter < max_iter and 0 < screen_keep <= 1):
        raise ValueError("Need 1 <= screen_iter < max_iter and 0 < screen_keep <= 1")
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_dir:
        (checkpoint_dir / "screen" if screening else checkpoint_dir).mkdir(parents=True, exist_ok=True)

    starts = [dict(label=label, seed=None, alpha_init="normal", values=values)
              for label, values in (initializations or {}).items()]
    random_labels = []
    for seed in seeds:
        alpha_init = "logunif" if seed in logunif else "normal"
        random_labels.append(start_label(seed, alpha_init))
        starts.append(dict(label=random_labels[-1], seed=seed, alpha_init=alpha_init, values=None))
    by_label = {start["label"]: start for start in starts}
    if len(by_label) != len(starts):
        raise ValueError("Start labels must be distinct")

    def task(start, n_iter, checkpoint, previous=None):
        return dict(spec=spec, seed=start["seed"], max_iter=n_iter, tol=tol, initial_values=start["values"],
                    prepared=prepared, bounds=bounds, alpha_init=start["alpha_init"],
                    progress_path=checkpoint.with_suffix(".progress.json") if checkpoint else None,
                    restart_id=start["label"], checkpoint=checkpoint, previous=previous)

    finished, screened = {}, {}
    to_run = []
    for start in starts:
        label = start["label"]
        final_path = checkpoint_dir / f"{label}.pkl" if checkpoint_dir else None
        screen_path = checkpoint_dir / "screen" / f"{label}.pkl" if checkpoint_dir else None
        copied = Path(reuse_dir) / f"{label}.pkl" if reuse_dir and start["seed"] is not None else None
        if final_path and final_path.exists():
            finished[label] = _check(load_pickle(final_path), spec, prepared.subjects, final_path)
        elif copied and copied.exists():
            result = _check(load_pickle(copied), spec, prepared.subjects, copied)
            result["copied_from"] = str(copied)
            if final_path:
                atomic_pickle(final_path, result)
            finished[label] = result
        elif screening and start["seed"] is not None:
            if screen_path and screen_path.exists():
                screened[label] = _check(load_pickle(screen_path), spec, prepared.subjects, screen_path)
            else:
                to_run.append(task(start, screen_iter, screen_path))
        else:
            to_run.append(task(start, max_iter, final_path))

    if verbose:
        print(f"\n[{spec.config_id()}]  {len(starts)} starts ({len(seeds)} random, {len(logunif)} log-uniform "
              f"alpha)  C={spec.C}  pw={spec.pw}  {len(finished)} saved, {len(screened)} screened", flush=True)

    def run(tasks, stage):
        if not tasks:
            return []
        done = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_restart_worker, item): item["restart_id"] for item in tasks}
            for count, future in enumerate(as_completed(futures), 1):
                label = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    if verbose:
                        print(f"  [{stage} {count:>3}/{len(tasks)}] {label} FAILED: {exc}", flush=True)
                    continue
                done.append(result)
                if verbose:
                    print(f"  [{stage} {count:>3}/{len(tasks)}] {label}  LL={result['log_likelihood']:>12.4f}  "
                          f"{result['exit_reason']}  {result['elapsed']:.1f}s", flush=True)
        return done

    for result in run(to_run, "screen" if screening else "run"):
        label = result["restart_id"]
        if screening and by_label[label]["seed"] is not None and result["n_iter"] == screen_iter \
                and result["exit_reason"] == "max_iter":
            screened[label] = result
        else:
            if screening and checkpoint_dir and by_label[label]["seed"] is not None:
                atomic_pickle(checkpoint_dir / f"{label}.pkl", result)   # converged while screening
            finished[label] = result
    for label, result in list(screened.items()):
        if result["exit_reason"] != "max_iter":    # a reused screening run that converged
            finished[label] = screened.pop(label)

    kept = set()
    if screening:
        pool = [label for label in random_labels if label in finished or label in screened]
        score = {label: ll_after(finished.get(label) or screened[label], screen_iter) for label in pool}
        ranked = sorted(pool, key=lambda label: (-score[label], pool.index(label)))
        kept = set(ranked[:math.ceil(screen_keep * len(pool))])
        continued = [task(by_label[label], max_iter, checkpoint_dir / f"{label}.pkl" if checkpoint_dir else None,
                          previous=screened[label]) for label in random_labels if label in kept and label in screened]
        for result in run(continued, "continue"):
            finished[result["restart_id"]] = result
            screened.pop(result["restart_id"])

    results = [finished[start["label"]] for start in starts if start["label"] in finished]
    if not results:
        raise RuntimeError(f"All restarts failed for spec {spec.config_id()}")
    best = max(results, key=lambda result: result["log_likelihood"])

    def summary(result, stage):
        return dict(restart_id=result.get("restart_id"), seed=result["seed"],
                    log_likelihood=result["log_likelihood"], converged=result["converged"] and stage != "screened_out",
                    n_iter=result["n_iter"], exit_reason="screened_out" if stage == "screened_out"
                    else result["exit_reason"], elapsed=result["elapsed"], stage=stage,
                    alpha_init=result.get("alpha_init", "normal"), random=result["seed"] is not None)

    best["seeds_used"] = seeds
    best["n_converged"] = sum(1 for r in results if r["converged"])
    best["n_restarts_ok"] = len(results)
    best["n_random_restarts"] = len(seeds)
    best["n_logunif_alpha"] = len(logunif)
    best["n_warm_starts"] = len(initializations or {})
    best["n_screened_out"] = len(screened)
    best["screen"] = dict(iter=screen_iter, keep=screen_keep, n_kept=len(kept)) if screening else None
    best["restart_summaries"] = (
        [summary(r, "continued" if r.get("screened_at") else "full") for r in results]
        + [summary(screened[label], "screened_out") for label in random_labels if label in screened])
    return best
