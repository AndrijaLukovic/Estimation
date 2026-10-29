"""Component-specific profile likelihood for saved conditional Grid EM models.

Examples (use --list to inspect jobs without estimating or writing output):
    python scripts/run_profile.py --parameters boundary --list
    python scripts/run_profile.py --parameters palpha --cluster PA
    python scripts/run_profile.py --parameters alpha_gain --cluster FE
    python scripts/run_profile.py --parameters lambda alpha_loss r --cluster all
    python scripts/run_profile.py --result results_grid_conditional_stricter/test1_exp_sepalpha_commem_prelec_C4/result.pkl --parameters delta_pa --cluster 4
    python scripts/run_profile.py --parameters palpha --cluster PA --grid 1e-8 1e-6 1e-4 .001 .01 .1

Default source: stricter-sample Test 3, exp/sepalpha/commem/Prelec. The default
parameter remains lambda. Cluster numbers are ONE-based; fixed-pure reference
names (SQ/PA/LE/FE) are also accepted. s1/s2/s3 are stick coordinates, NOT actual
SQ/PA/LE weights. Common-alpha profiles vary both exponents; common-memory
delta_pa/delta_le are aliases for the same decay parameter.

At each point only one component's selected parameter is fixed. All remaining
free structural parameters, mixing shares and participant noise are reestimated
using the current conditional Gaussian EM. Source/near-optimal saved restarts,
random restarts, and ascending/descending continuation explore different modes.
Mixed-model warm starts are aligned to the source component order. Subsequent
component reassignment and collapsing shares are reported: a flat mixed-model
profile can include component switching, not just weak parameter identification.

The target parameter's unrestricted bounds cover the grid and original bounds;
other nuisance bounds are unchanged. The unrestricted reference is refitted in
that SAME domain, also starting from the best profile point. CSV/JSON diagnostics
and per-point pickle checkpoints go outside the input directory. No HTML or
automatic chi-square confidence intervals are made. Identical commands resume
checkpoints; changed grids/seeds/inputs/source results require a different
--output-dir.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from estimation.config import LOTTERIES, ROOT, SEEDS  # noqa: E402
from estimation.data import observed_ce  # noqa: E402
from estimation.likelihood import result_loglik  # noqa: E402
from estimation.lottery_paths import transform  # noqa: E402
from estimation.prepare import prepare_grid_data  # noqa: E402
from estimation.profile import (choose_jobs, default_output, init_worker, load_starts,  # noqa: E402
                                run_one)
from estimation.store import atomic_json, load_result, scientific_inputs, seal_protocol, sha256_file  # noqa: E402

DEFAULT_RESULT = ROOT / "results_grid_conditional_stricter/test3_exp_sepalpha_commem_prelec/result.pkl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--parameters", "--parameter", "--param", nargs="+", default=["lambda"],
        help="lambda, alpha_gain, alpha_loss, palpha, gamma, r, psi, delta_pa, delta_le, s1/s2/s3; or all/boundary")
    parser.add_argument("--cluster", nargs="+", default=["all"], help="ONE-based numbers, SQ/PA/LE/FE, or all")
    parser.add_argument("--grid", type=float, nargs="+", help="Custom grid for ONE explicit parameter; source estimate is added")
    parser.add_argument("--seeds", type=int, default=4, help="Number of reproducible random restarts per point")
    parser.add_argument("--warm-starts", type=int, default=3, help="Additional near-optimal saved restarts")
    parser.add_argument("--warm-ll-window", type=float, default=10.)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list", action="store_true", help="Validate and print jobs without estimating or writing output")
    args = parser.parse_args(argv)
    if args.grid is not None and (len(args.parameters) != 1 or args.parameters[0] in {"all", "boundary"}):
        parser.error("--grid requires one explicit parameter")
    if not 0 <= args.seeds <= len(SEEDS) or args.warm_starts < 0 or not np.isfinite(args.warm_ll_window) or args.warm_ll_window < 0:
        parser.error("Invalid restart count/window")
    if args.max_iter < 1 or args.n_workers < 1 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("max-iter, n-workers and tol must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    path = args.result.resolve()
    source = load_result(path)
    observed = observed_ce()
    data = prepare_grid_data(observed, transform(LOTTERIES))
    if source["subjects"] != data.subjects or source.get("n_obs_effective", source.get("n_obs")) != len(data.observed):
        raise ValueError("Saved result and current filtered sample differ; choose a result fitted on the current sample")
    source_ll, _ = result_loglik(source, data)
    if abs(source_ll - source["log_likelihood"]) > 1e-5:
        raise ValueError(f"Saved likelihood does not reproduce: gap={source_ll - source['log_likelihood']}")
    jobs, skipped = choose_jobs(source, args)
    for reason in skipped:
        print("SKIP " + reason, flush=True)
    if not jobs:
        raise ValueError("No eligible free parameters selected")
    starts, warm_hashes = load_starts(path, source, args.warm_starts, args.warm_ll_window)
    seeds = list(map(int, SEEDS[:args.seeds]))
    print(f"Source: {source['spec'].config_id()}; {len(data.subjects)} participants, {len(data.observed)} CEs; LL={source_ll:.8f}")
    print(f"Profiles: {len(jobs)}; saved starts={len(starts)}, random starts={len(seeds)}, workers={args.n_workers}")
    for target, grid in jobs:
        label = source["rp_identities"][target.cluster] or f"mixed component {target.cluster + 1}"
        print(f"  {label}: {target.name}={target.source_value:.8g}; grid={grid}; unrestricted bounds={target.reference_bound}")
    if args.list:
        return
    output = args.output_dir.resolve() if args.output_dir else default_output(path, source, jobs)
    source_root = path.parent.parent
    if output in path.parents or source_root in output.parents:
        raise ValueError("Profile output must be separate from the estimation input directory")
    output.mkdir(parents=True, exist_ok=True)
    protocol = seal_protocol(dict(version="conditional_component_profile_v1", inputs=scientific_inputs(observed),
        result=str(path), result_sha256=sha256_file(path), warm_sha256=warm_hashes,
        spec=asdict(source["spec"]), jobs=[dict(target=asdict(target), grid=grid) for target, grid in jobs],
        seeds=seeds, max_iter=args.max_iter, tol=args.tol,
        n_subjects=len(data.subjects), n_obs=len(data.observed), warm_ll_window=args.warm_ll_window))
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != protocol["fingerprint"]:
        raise ValueError("Profile output has a different protocol; choose another --output-dir")
    atomic_json(manifest, protocol, ensure_ascii=True, allow_nan=False)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers, initializer=init_worker, initargs=(data,)) as pool:
        for target, grid in jobs:
            run_one(pool, source, target, grid, starts, seeds, args, output)
    print(f"Completed in {(time.time() - started) / 60:.1f} minutes. Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
