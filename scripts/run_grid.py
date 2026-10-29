"""Estimate the complete 144-model grid with conditional remaining lotteries.

    python scripts/run_grid.py --list
    python scripts/run_grid.py --output-dir results_grid_conditional_new
    python scripts/run_grid.py --which test3 --seeds 20 --n-workers 20

Default order: Test 3, Test 1, Test 2. The original results_grid is read-only
and supplies legacy warm starts. Pure-reference memory labels are equivalent:
fit once, then save an explicitly labelled alias with the corresponding
canonical/free parameter layout. Finished models and restarts resume from
checkpoints when the protocol fingerprint matches.
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimation.config import LOTTERIES, MAX_ITER, ROOT, SEEDS, TOL  # noqa: E402
from estimation.data import observed_ce  # noqa: E402
from estimation.jobs import ordered_jobs  # noqa: E402
from estimation.lottery_paths import transform  # noqa: E402
from estimation.multistart import grid_em_multistart  # noqa: E402
from estimation.store import (load_completed, save_model, scientific_inputs, seal_protocol,  # noqa: E402
                              snapshot_sources, write_protocol, write_summary)
from estimation.warm_starts import equivalent_result, grid_initializations  # noqa: E402

LEGACY_RESULTS = ROOT / "results_grid"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--which", choices=["all", "test3", "test1", "test2"], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results_grid_conditional")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    jobs = ordered_jobs(args.which)
    if args.list:
        print("\n".join(cid for cid, _ in jobs))
        return
    output = args.output_dir.resolve()
    if output == LEGACY_RESULTS.resolve():
        raise ValueError("Original results_grid is protected; use a separate output directory")
    output.mkdir(exist_ok=True)
    seeds = SEEDS[:args.seeds]
    observed = observed_ce()
    protocol = seal_protocol(dict(
        inputs=scientific_inputs(observed), seeds=list(map(int, seeds)), max_iter=args.max_iter, tol=args.tol,
        reference_basis="current remaining payoff; original historical benchmark formulas",
        time_origin="current decision, next payment delay 1", noise_scale="original full-lottery spread",
        memory_equivalence="fit 104 distinct models; save 144 configurations with labelled pure-reference aliases"))
    fingerprint = protocol["fingerprint"]
    write_protocol(output, protocol)
    snapshot_sources(output, [__file__])

    lotteries = transform(LOTTERIES)
    valid = observed[observed.lottery_id.isin(lotteries)].dropna(subset=["ce_observed"])
    subjects = sorted(valid.participant_label.unique())
    print(f"CONDITIONAL REFIT: {len(jobs)} configurations, {valid.participant_label.nunique()} subjects, "
          f"{len(valid)} CEs; priority Test 3", flush=True)
    current_test = None
    for index, (cid, spec) in enumerate(jobs, 1):
        if spec.test != current_test:
            if current_test:
                print(f"TEST COMPLETE: {current_test}", flush=True)
            current_test = spec.test
        if load_completed(output, cid, fingerprint) is not None:
            print(f"[{index}/{len(jobs)}] {cid}: checkpoint already complete", flush=True)
            continue
        print(f"[{index}/{len(jobs)}] START {cid}", flush=True)
        if spec.memory == "sepmem" and spec.rp_mode != "mixed":
            source_cid = cid.replace("_sepmem_", "_commem_")
            source = load_completed(output, source_cid, fingerprint)
            if source is None:
                raise RuntimeError(f"Missing equivalent source {source_cid}")
            meta = save_model(output, spec, equivalent_result(source, spec), len(valid), 0, fingerprint)
            print(f"  equivalent alias of {source_cid}, BIC={meta['bic']:.2f}", flush=True)
            write_summary(output)
            continue
        initializations = grid_initializations(spec, subjects, output, fingerprint, LEGACY_RESULTS)
        started = time.time()
        result = grid_em_multistart(spec, seeds, y=observed, lotteries=lotteries, max_iter=args.max_iter,
                                    tol=args.tol, n_workers=args.n_workers, initializations=initializations,
                                    checkpoint_dir=output/cid/"restarts")
        meta = save_model(output, spec, result, len(valid), time.time()-started, fingerprint)
        write_summary(output)
        print(f"  SAVED {cid}: LL={meta['log_likelihood']:.6f}, BIC={meta['bic']:.2f}, exit={meta['exit_reason']}", flush=True)
    if current_test:
        print(f"TEST COMPLETE: {current_test}", flush=True)
    print("ALL REQUESTED CONDITIONAL REFITS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
