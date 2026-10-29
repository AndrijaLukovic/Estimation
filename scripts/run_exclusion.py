"""Robustness check: refit Test 3 and Test 1 without top-endpoint responders.

A top-endpoint answer keeps the lottery until the sure amount equals the best
remaining outcome (effective selected amount == maximum remaining payoff; list
lengths differ across lotteries, so the row number is not used). Participants
with at least --threshold of such answers are excluded. Everything else matches
scripts/run_grid.py. Warm starts: the full-sample fit of the same model, the
same model in --warm-dirs, the nested Test 3 fit for Test 1 C=4, and every
class of the C-1 fit split in two for Test 1 C>=2 (so C cannot end below C-1).

--lambda-upper / --gamma-lower (TK gamma) change bounds for this run only.
TK weighting is monotone only for gamma >= 0.2792.

    python scripts/run_exclusion.py --list      # show exclusions and jobs only
    python scripts/run_exclusion.py
    python scripts/run_exclusion.py --lambda-upper 20 \
        --output-dir results_grid_conditional_stricter_no_endpoint_lam20 \
        --warm-dirs results_grid_conditional_stricter_no_endpoint
    python scripts/run_exclusion.py --pw tk --lambda-upper 20 --gamma-lower 0.28 \
        --no-test3 --clusters 4 5 6 \
        --output-dir results_grid_conditional_stricter_no_endpoint_tk_lam20_g028 \
        --warm-dirs results_grid_conditional_stricter_no_endpoint_tk_lam20
    python scripts/run_exclusion.py --pw tk --lambda-upper 20 --gamma-lower 0.28 --test2 \
        --output-dir results_grid_conditional_stricter_no_endpoint_tk_main \
        --warm-dirs results_grid_conditional_stricter_no_endpoint_tk_lam20_g028 \
                    results_grid_conditional_stricter_no_endpoint_tk_lam20
"""
import argparse
import hashlib
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from estimation.config import BOUNDS, LOTTERIES, MAX_ITER, ROOT, SEEDS, TOL  # noqa: E402
from estimation.data import endpoint_shares, exclude_participants, observed_ce  # noqa: E402
from estimation.jobs import exclusion_jobs  # noqa: E402
from estimation.lottery_paths import transform  # noqa: E402
from estimation.multistart import grid_em_multistart  # noqa: E402
from estimation.spec import CANON, replace_bound  # noqa: E402
from estimation.store import (load_completed, save_model, scientific_inputs, seal_protocol,  # noqa: E402
                              write_protocol, write_summary)
from estimation.warm_starts import exclusion_initializations  # noqa: E402


def parse_args(argv=None):
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
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--lambda-upper", type=float, help="Upper lambda bound for this run (default: config)")
    parser.add_argument("--gamma-lower", type=float, help="Lower TK gamma bound for this run (default: config)")
    parser.add_argument("--warm-dirs", type=Path, nargs="*", default=[], help="Extra warm starts: same config IDs")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bounds = BOUNDS
    if args.lambda_upper is not None:
        bounds = replace_bound("LAM", "high", args.lambda_upper, bounds)
    if args.gamma_lower is not None:
        bounds = replace_bound("GAM", "low", args.gamma_lower, bounds)
    lambda_bound = list(bounds[args.pw][CANON[args.pw]["LAM"]])
    weight_bound = list(bounds[args.pw][CANON[args.pw]["PALPHA" if args.pw == "prelec" else "GAM"]])
    jobs = exclusion_jobs(args.pw, args.clusters, test3=not args.no_test3, test2=args.test2)

    lotteries = transform(LOTTERIES)
    observed = observed_ce()
    shares = endpoint_shares(observed, lotteries)
    excluded = sorted(shares.index[shares >= args.threshold])
    observed = exclude_participants(observed, excluded)
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
    seeds = SEEDS[:args.seeds]
    protocol = seal_protocol(dict(
        inputs=scientific_inputs(observed, bounds), exclusion=f"top-endpoint share >= {args.threshold}",
        excluded_sha256=hashlib.sha256("\n".join(excluded).encode()).hexdigest(), n_excluded=len(excluded),
        source_dir=str(args.source_dir.resolve()), pw=args.pw,
        warm_dirs=[str(path.resolve()) for path in args.warm_dirs],
        seeds=list(map(int, seeds)), max_iter=args.max_iter, tol=args.tol))
    fingerprint = protocol["fingerprint"]
    write_protocol(output, protocol)

    for index, spec in enumerate(jobs, 1):
        cid = spec.config_id()
        if load_completed(output, cid, fingerprint) is not None:
            print(f"[{index}/{len(jobs)}] {cid}: checkpoint already complete", flush=True)
            continue
        print(f"[{index}/{len(jobs)}] START {cid}", flush=True)
        initializations = exclusion_initializations(spec, subjects, output, fingerprint,
                                                    args.source_dir, args.warm_dirs, bounds)
        print(f"  warm starts: {sorted(initializations)}", flush=True)
        started = time.time()
        result = grid_em_multistart(spec, seeds, y=observed, lotteries=lotteries, max_iter=args.max_iter,
                                    tol=args.tol, n_workers=args.n_workers, initializations=initializations,
                                    checkpoint_dir=output / cid / "restarts", verbose=False, bounds=bounds)
        result["excluded_participants"] = excluded
        meta = save_model(output, spec, result, len(valid), time.time() - started, fingerprint)
        write_summary(output)
        print(f"  SAVED {cid}: LL={meta['log_likelihood']:.4f}, BIC={meta['bic']:.2f}, "
              f"{(time.time() - started) / 60:.1f} min", flush=True)
    print("ALL EXCLUSION REFITS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
