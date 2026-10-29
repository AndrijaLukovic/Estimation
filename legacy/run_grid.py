"""
Master orchestration script for the 144-job model-comparison grid
(Test 1: cluster-number benchmark, Test 2: pure-RP horserace, Test 3: fixed
four-RP-cluster model), each crossed with the 2^4 factorial of
(discounting x curvature x memory-decay x probability-weighting).

Usage:
    python run_grid.py                          # run everything, 20 seeds/job
    python run_grid.py --seeds 8                # fewer restarts per job (faster)
    python run_grid.py --seeds 50 --which test1 # only Test-1 jobs, 50 restarts
    python run_grid.py --list                   # print the 144 config IDs and exit
    python run_grid.py --consolidate-only        # skip estimation, just rebuild
                                                  # summary.csv / BIC tables / plots
                                                  # from whatever is already saved

Results land in RESULTS_DIR (gitignored), one subfolder per config ID:
    results_grid/<config_id>/result.pkl   (full EM result: thetas, resp, ksi, ...)
    results_grid/<config_id>/meta.json    (small summary + status, used for resume)

Resuming: jobs already marked status="done" in meta.json are skipped. Jobs
that raised an exception are marked status="failed" and are retried on the
next invocation (a "done" result was never produced for them).

GlobalSeedsSet (the existing reproducible seed list, seeded from
GlobalMasterSeed=42) is reused, truncated to --seeds -- so results stay
reproducible and consistent with the existing multi-start convention; --seeds
only trades off how many of those seeds are spent per job, not their values.
"""

import argparse
import json
import pickle
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

import functions as f
from GlobalSettings import GlobalLottery, GlobalSeedsSet, GlobalInterMax, GlobalTol
from main import get_observed_ce
from model_spec import ModelSpec, all_specs_16, RP_LABELS_4
from grid_em import grid_em_multistart, count_free_params, compute_bic

RESULTS_DIR = Path("results_grid")


# ── Job generation (deterministic order, 144 jobs total) ─────────────────────

def generate_jobs():
    """Yield (config_id, ModelSpec) for all 144 jobs, in a fixed order:
    Test 1 (64) -> Test 2 (64) -> Test 3 (16)."""
    jobs = []

    # Test 1: mixed simplex RP, C in {1,2,3,4}  -> 16 * 4 = 64
    for (disc, curv, mem, pw) in all_specs_16():
        for C in (1, 2, 3, 4):
            spec = ModelSpec(test="test1", discounting=disc, curvature=curv,
                              memory=mem, pw=pw, rp_mode="mixed", C=C)
            jobs.append((spec.config_id(), spec))

    # Test 2: pure RP, C=1, one of SQ/PA/LE/FE  -> 16 * 4 = 64
    for (disc, curv, mem, pw) in all_specs_16():
        for rp in RP_LABELS_4:
            spec = ModelSpec(test="test2", discounting=disc, curvature=curv,
                              memory=mem, pw=pw, rp_mode="fixed_pure", C=1,
                              rp_identity=rp)
            jobs.append((spec.config_id(), spec))

    # Test 3: four fixed RP clusters, C=4  -> 16
    for (disc, curv, mem, pw) in all_specs_16():
        spec = ModelSpec(test="test3", discounting=disc, curvature=curv,
                          memory=mem, pw=pw, rp_mode="fixed_4", C=4)
        jobs.append((spec.config_id(), spec))

    assert len(jobs) == 144, f"expected 144 jobs, got {len(jobs)}"
    assert len(set(cid for cid, _ in jobs)) == 144, "config_id collision"
    return jobs


# ── Per-job persistence / resume ──────────────────────────────────────────

def job_dir(config_id):
    return RESULTS_DIR / config_id


def load_meta(config_id):
    p = job_dir(config_id) / "meta.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def is_done(config_id):
    meta = load_meta(config_id)
    return meta is not None and meta.get("status") == "done"


def save_result(config_id, spec, result, n_obs, elapsed, seeds_used):
    d = job_dir(config_id)
    d.mkdir(parents=True, exist_ok=True)

    bic, k = compute_bic(result, n_obs)
    result["bic"] = bic
    result["k_params"] = k
    result["n_obs"] = n_obs

    with open(d / "result.pkl", "wb") as fh:
        pickle.dump(result, fh)

    meta = {
        "config_id": config_id,
        "status": "done",
        **asdict(spec),
        "log_likelihood": result["log_likelihood"],
        "k_params": k,
        "bic": bic,
        "n_obs": n_obs,
        "n_subjects": len(result["subjects"]),
        "converged": result["converged"],
        "n_iter": result["n_iter"],
        "n_restarts_ok": result.get("n_restarts_ok"),
        "n_converged": result.get("n_converged"),
        "seeds_used": seeds_used,
        "elapsed_seconds": elapsed,
        "pis": list(map(float, result["pis"])),
        "rp_identities": result["rp_identities"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(d / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


def save_failure(config_id, spec, exc):
    d = job_dir(config_id)
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "config_id": config_id,
        "status": "failed",
        **asdict(spec),
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(d / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


# ── Queue execution ───────────────────────────────────────────────────────

def run_queue(jobs, seeds, max_iter, tol, n_workers, force=False,
              debug_max_subjects=None):
    y = get_observed_ce(export_excel=False)
    if debug_max_subjects is not None:
        # DEV/SMOKE-TEST ONLY: silently subsampling the sample is exactly
        # what the brief says never to do for real results -- this path
        # exists solely to let the plumbing be verified quickly and must
        # never be used for a production run.
        print(f"  *** DEBUG MODE: subsampling to {debug_max_subjects} subjects "
              f"-- DO NOT use these results for real estimates ***")
        subs = sorted(y["participant_label"].unique())[:debug_max_subjects]
        y = y[y["participant_label"].isin(subs)].copy()
    lotteries = f.transform(GlobalLottery)
    n_obs = len(y[y["lottery_id"].isin(lotteries.keys())])

    n_total = len(jobs)
    n_skipped = n_run = n_failed = 0

    for i, (config_id, spec) in enumerate(jobs, start=1):
        if not force and is_done(config_id):
            n_skipped += 1
            print(f"[{i:>3}/{n_total}] {config_id}  -- already done, skipping")
            continue

        print(f"\n{'='*72}")
        print(f"[{i:>3}/{n_total}] {config_id}")
        print(f"{'='*72}")

        t0 = time.time()
        try:
            result = grid_em_multistart(
                spec, seeds=seeds, y=y, lotteries=lotteries,
                max_iter=max_iter, tol=tol, n_workers=n_workers, verbose=True,
            )
            elapsed = time.time() - t0
            save_result(config_id, spec, result, n_obs, elapsed, seeds)
            n_run += 1
            print(f"  -> saved  LL={result['log_likelihood']:.4f}  "
                  f"BIC={result['bic']:.2f}  k={result['k_params']}  "
                  f"{elapsed:.1f}s")
        except Exception as exc:
            n_failed += 1
            save_failure(config_id, spec, exc)
            print(f"  -> FAILED: {exc}")

    print(f"\n{'='*72}")
    print(f"QUEUE DONE  total={n_total}  ran={n_run}  skipped={n_skipped}  failed={n_failed}")
    print(f"{'='*72}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=20,
                         help="Number of multistart restarts per job, drawn as a "
                              "prefix of GlobalSeedsSet (default: 20).")
    parser.add_argument("--max-iter", type=int, default=GlobalInterMax)
    parser.add_argument("--tol", type=float, default=GlobalTol)
    parser.add_argument("--n-workers", type=int, default=None,
                         help="Parallel workers per job's multistart (default: all cores).")
    parser.add_argument("--which", choices=["all", "test1", "test2", "test3"],
                         default="all")
    parser.add_argument("--list", action="store_true",
                         help="Print the config IDs that would run and exit.")
    parser.add_argument("--force", action="store_true",
                         help="Re-run jobs even if already marked done.")
    parser.add_argument("--consolidate-only", action="store_true",
                         help="Skip estimation; just rebuild summary tables/plots "
                              "from whatever is already saved in results_grid/.")
    parser.add_argument("--debug-max-subjects", type=int, default=None,
                         help="DEV/SMOKE-TEST ONLY: subsample to this many subjects "
                              "to verify the pipeline quickly. Never use for real "
                              "results -- silently changing the sample is exactly "
                              "what production runs must not do.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    all_jobs = generate_jobs()
    jobs = all_jobs if args.which == "all" else [
        (cid, spec) for cid, spec in all_jobs if spec.test == args.which
    ]

    if args.list:
        for cid, _ in jobs:
            print(cid)
        raise SystemExit(0)

    seeds = GlobalSeedsSet[:args.seeds]

    if not args.consolidate_only:
        run_queue(jobs, seeds=seeds, max_iter=args.max_iter, tol=args.tol,
                  n_workers=args.n_workers, force=args.force,
                  debug_max_subjects=args.debug_max_subjects)

    from consolidate_grid import consolidate
    consolidate(RESULTS_DIR)
