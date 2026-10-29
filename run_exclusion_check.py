"""Robustness check: refit Test 3 and Test 1 without top-endpoint responders.

A top-endpoint answer keeps the lottery until the sure amount equals the best
remaining outcome (effective selected amount == maximum remaining payoff; list
lengths differ across lotteries, so the row number is not used). Participants
with at least --threshold of such answers are excluded. Everything else matches
run_conditional_grid.py. Warm starts: the full-sample fit of the same model,
the same model in --warm-dirs, the nested Test 3 fit for Test 1 C=4, and every
class of the C-1 fit split in two for Test 1 C>=2 (so C cannot end below C-1).

--lambda-upper / --gamma-lower (TK gamma) change bounds for this run only;
GlobalSettings is not edited. TK weighting is monotone only for gamma >= 0.2792.

    python run_exclusion_check.py --list      # show exclusions and jobs only
    python run_exclusion_check.py
    python run_exclusion_check.py --lambda-upper 20 \
        --output-dir results_grid_conditional_stricter_no_endpoint_lam20 \
        --warm-dirs results_grid_conditional_stricter_no_endpoint
    python run_exclusion_check.py --pw tk --lambda-upper 20 --gamma-lower 0.28 \
        --no-test3 --clusters 4 5 6 \
        --output-dir results_grid_conditional_stricter_no_endpoint_tk_lam20_g028 \
        --warm-dirs results_grid_conditional_stricter_no_endpoint_tk_lam20
    python run_exclusion_check.py --pw tk --lambda-upper 20 --gamma-lower 0.28 --test2 \
        --output-dir results_grid_conditional_stricter_no_endpoint_tk_main \
        --warm-dirs results_grid_conditional_stricter_no_endpoint_tk_lam20_g028 \
                    results_grid_conditional_stricter_no_endpoint_tk_lam20
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import functions as f
from GlobalSettings import GlobalLottery, GlobalSeedsSet, GlobalTol, GlobalInterMax, GlobalPrelecBounds, GlobalTKBounds
from main import get_observed_ce
from model_spec import CANON, RP_LABELS_4, ModelSpec, bounds_free
from grid_em import grid_em_multistart, prepare_grid_data, spec_cluster_layouts
from run_conditional_grid import (ROOT, VERSION, SOURCE_FILES, initial_values, save, write_protocol,
                                  load_completed, write_summary)


def model_jobs(pw, clusters, test3=True, test2=False):
    spec = dict(discounting="exp", curvature="sepalpha", memory="commem", pw=pw)
    return ([ModelSpec(test="test3", rp_mode="fixed_4", C=4, **spec)] if test3 else []) + \
           ([ModelSpec(test="test2", rp_mode="fixed_pure", C=1, rp_identity=rp, **spec) for rp in RP_LABELS_4]
            if test2 else []) + \
           [ModelSpec(test="test1", rp_mode="mixed", C=c, **spec) for c in sorted(clusters)]


def endpoint_shares(observed, lotteries):
    """Share of each participant's CEs at the top end of the choice list."""
    active = observed[observed.lottery_id.isin(lotteries)].dropna(subset=["ce_observed"])
    data = prepare_grid_data(observed, lotteries)
    raw = pd.read_csv(ROOT / "gamedata.csv", sep=",", skipinitialspace=True, encoding="utf-8-sig", low_memory=False)
    raw.columns = raw.columns.str.strip()
    raw = raw.loc[active.index]
    chosen = raw.fine_selected_amount.combine_first(raw.selected_amount).to_numpy(float)
    payoff = data.streams.sum(axis=2)
    best = np.where(data.path_mask, payoff, -np.inf).max(axis=1)[data.state_index]
    if not np.allclose(active.ce_observed.to_numpy(), data.observed):
        raise ValueError("Raw rows and prepared CE rows are misaligned")
    return pd.Series(np.isclose(chosen, best)).groupby(active.participant_label.to_numpy()).mean()


def set_bound(name, side, value):
    """Change one canonical bound in place; forked workers inherit it."""
    if multiprocessing.get_start_method() != "fork":
        raise RuntimeError("Run-specific bounds rely on forked workers")
    for bounds, method in ((GlobalPrelecBounds, "prelec"), (GlobalTKBounds, "tk")):
        if name in CANON[method]:
            position = CANON[method][name]
            low, high = bounds[position]
            bounds[position] = (value, high) if side == "low" else (low, value)


def split_starts(result, spec, subjects, scale=.02):
    """Turn a C-1 class fit into C warm starts by splitting each class in two."""
    smaller = initial_values(result, dataclasses.replace(spec, C=spec.C - 1), subjects=subjects)
    layout = spec_cluster_layouts(spec)[0][0]
    low, high = np.array(bounds_free(layout)).T
    starts = {}
    for j, vector in enumerate(smaller["free_vecs"]):
        shift = scale * (high - low) * np.random.default_rng(j).choice([-1., 1.], size=len(vector))
        halves = [np.clip(vector + shift, low, high), np.clip(vector - shift, low, high)]
        pis = np.r_[np.delete(smaller["pis"], j), [smaller["pis"][j] / 2] * 2]
        starts[f"split_C{spec.C - 1}_class{j + 1}"] = dict(
            free_vecs=[v for i, v in enumerate(smaller["free_vecs"]) if i != j] + halves,
            pis=pis, ksi=smaller["ksi"].copy())
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "results_grid_conditional_stricter")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_grid_conditional_stricter_no_endpoint")
    parser.add_argument("--pw", choices=["prelec", "tk"], default="prelec")
    parser.add_argument("--clusters", type=int, nargs="+", default=[1, 2, 3, 4], help="Test 1 class counts")
    parser.add_argument("--no-test3", action="store_true")
    parser.add_argument("--test2", action="store_true", help="Also fit the four pure-reference Test 2 models")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=24)
    parser.add_argument("--max-iter", type=int, default=GlobalInterMax)
    parser.add_argument("--tol", type=float, default=GlobalTol)
    parser.add_argument("--lambda-upper", type=float, help="Upper lambda bound for this run (default: GlobalSettings)")
    parser.add_argument("--gamma-lower", type=float, help="Lower TK gamma bound for this run (default: GlobalSettings)")
    parser.add_argument("--warm-dirs", type=Path, nargs="*", default=[], help="Extra warm starts: same config IDs")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.lambda_upper is not None:
        set_bound("LAM", "high", args.lambda_upper)
    if args.gamma_lower is not None:
        set_bound("GAM", "low", args.gamma_lower)
    bounds = GlobalPrelecBounds if args.pw == "prelec" else GlobalTKBounds
    lambda_bound = list(bounds[CANON[args.pw]["LAM"]])
    weight_bound = list(bounds[CANON[args.pw]["PALPHA" if args.pw == "prelec" else "GAM"]])
    jobs = model_jobs(args.pw, args.clusters, test3=not args.no_test3, test2=args.test2)

    lotteries = f.transform(GlobalLottery)
    observed = get_observed_ce(export_excel=False)
    shares = endpoint_shares(observed, lotteries)
    excluded = sorted(shares.index[shares >= args.threshold])
    observed = observed[~observed.participant_label.isin(excluded)]
    valid = observed[observed.lottery_id.isin(lotteries)].dropna(subset=["ce_observed"])
    subjects = sorted(valid.participant_label.unique())
    print(f"Excluded {len(excluded)} of {len(shares)} participants (top-endpoint share >= {args.threshold}); "
          f"remaining {len(subjects)} participants, {len(valid)} CEs; lambda bound {lambda_bound}, "
          f"weighting bound {weight_bound}", flush=True)
    for spec in jobs:
        print("  " + spec.config_id(), flush=True)
    if args.list:
        return

    output = args.output_dir.resolve()
    if output == args.source_dir.resolve():
        raise ValueError("Write the exclusion refit to a separate directory")
    output.mkdir(exist_ok=True)
    pd.DataFrame(dict(participant_label=excluded, top_endpoint_share=shares[excluded].to_numpy())) \
        .to_csv(output / "excluded_participants.csv", index=False)
    files = SOURCE_FILES + ["run_exclusion_check.py"]
    protocol = dict(valuation_version=VERSION, exclusion=f"top-endpoint share >= {args.threshold}",
        excluded_sha256=hashlib.sha256("\n".join(excluded).encode()).hexdigest(), n_excluded=len(excluded),
        source_dir=str(args.source_dir.resolve()), pw=args.pw, lambda_bound=lambda_bound,
        weight_bound=weight_bound, warm_dirs=[str(path.resolve()) for path in args.warm_dirs],
        source_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        seeds=list(map(int, GlobalSeedsSet[:args.seeds])), max_iter=args.max_iter, tol=args.tol)
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["fingerprint"] = fingerprint
    write_protocol(output, protocol)

    def find(cid):
        """Fit of cid from this run, else from the first --warm-dirs entry that has it."""
        result = load_completed(output, cid, fingerprint)
        for warm_dir in args.warm_dirs:
            if result is None and (warm_dir / cid / "result.pkl").exists():
                with (warm_dir / cid / "result.pkl").open("rb") as fh:
                    result = pickle.load(fh)
        return result

    seeds = GlobalSeedsSet[:args.seeds]
    for index, spec in enumerate(jobs, 1):
        cid = spec.config_id()
        if load_completed(output, cid, fingerprint) is not None:
            print(f"[{index}/{len(jobs)}] {cid}: checkpoint already complete", flush=True)
            continue
        print(f"[{index}/{len(jobs)}] START {cid}", flush=True)
        initializations = {}
        if (args.source_dir / cid / "result.pkl").exists():
            with (args.source_dir / cid / "result.pkl").open("rb") as fh:
                initializations["full_sample_" + cid] = initial_values(pickle.load(fh), spec, subjects=subjects)
        for warm_dir in args.warm_dirs:
            if (warm_dir / cid / "result.pkl").exists():
                with (warm_dir / cid / "result.pkl").open("rb") as fh:
                    initializations[f"warm_{warm_dir.name}"] = initial_values(pickle.load(fh), spec, subjects=subjects)
        if spec.test == "test1" and spec.C == 4:
            test3_id = dataclasses.replace(spec, test="test3", rp_mode="fixed_4").config_id()
            nested = find(test3_id)
            if nested is not None:
                initializations["nested_" + test3_id] = initial_values(nested, spec, subjects=subjects)
        if spec.test == "test1" and spec.C >= 2:
            smaller = find(dataclasses.replace(spec, C=spec.C - 1).config_id())
            if smaller is not None:
                initializations.update(split_starts(smaller, spec, subjects))
        print(f"  warm starts: {sorted(initializations)}", flush=True)
        started = time.time()
        result = grid_em_multistart(spec, seeds, y=observed, lotteries=lotteries, max_iter=args.max_iter,
            tol=args.tol, n_workers=args.n_workers, initializations=initializations,
            checkpoint_dir=output / cid / "restarts", verbose=False)
        result["excluded_participants"] = excluded
        meta = save(output, spec, result, len(valid), time.time() - started, fingerprint)
        write_summary(output)
        print(f"  SAVED {cid}: LL={meta['log_likelihood']:.4f}, BIC={meta['bic']:.2f}, "
              f"{(time.time() - started) / 60:.1f} min", flush=True)
    print("ALL EXCLUSION REFITS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
