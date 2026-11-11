"""Estimate any selection of models on any sample.

    python scripts/run_grid.py --list                        # full 144-model grid; nothing is estimated
    python scripts/run_grid.py --tests test3 --pw tk --output-dir results/sample_stricter/tk_test3
    python scripts/run_grid.py --tests test1 --clusters 4 5 6 \\
        --discounting exp --curvature sepalpha --memory commem --pw tk \\
        --lambda-upper 20 --gamma-lower 0.28 --output-dir results/sample_stricter/tk_test1
    python scripts/run_grid.py --only test3_exp_sepalpha_commem_tk test1_exp_sepalpha_commem_tk_C4 \\
        --exclude-endpoint 0.5 --output-dir results/sample_stricter_no_endpoint/my_run
    python scripts/run_grid.py --tests test3 --exclude-file my_exclusions.csv --output-dir results/sample_my_rule/test3

Sample: the cleaned data (questionnaire completed, task not rated very or too
complicated), minus participants with at least --exclude-endpoint of their CEs
at the top of the choice list, minus participants listed in --exclude-file
(CSV with a participant_label column: any rule decided outside the model, e.g.
page times). Excluded participants and reasons go to excluded_participants.csv.

Models: --tests, --discounting, --curvature, --memory and --pw take one or more
values (default: all); --clusters sets the Test 1 class counts (any C >= 1) and
--rp the Test 2 reference points. --only names config IDs instead of filters.
--utility exponential estimates CARA utility instead of power utility (config
IDs end in _cara, curvature bounds config.CARA_BOUNDS); one utility per
output directory.
Pure-reference sepmem models are saved as aliases of their commem fit, which
is added automatically.

Warm starts: the original unconditional fit (results/legacy_unconditional), the same model in
--warm-dirs, the closest models already fitted in this run, the nested Test 3
fit for Test 1 C=4 and the split C-1 fits for Test 1 C>=2 (this run's and every
warm directory's).

Random starts: --seeds per model (the first 100 are config.SEEDS). With CARA,
--logunif-alpha N of them (the last N) draw alpha log-uniformly instead of near
zero. --screen-iter K --screen-keep F runs every random start K EM iterations
and continues only the share F with the highest log-likelihood (warm starts
always run to the end). --reuse-restarts DIR copies the random starts already
finished in an earlier run of the same protocol instead of re-estimating them.
summary.txt reports how many finished random starts reach the best fit.

Output: a folder per model (result.pkl, meta.json), summary.csv, summary.txt
(readable fit table and class parameters, see scripts/summarize_run.py) and
run_protocol.json. The protocol fingerprint covers the sample, bounds, seeds,
tolerances and warm-start directories but not the model selection, so one
directory collects every model run under the same protocol and an identical
command resumes from checkpoints. Changing any of those inputs needs a new
--output-dir. TK weighting is monotone only for gamma >= 0.2792.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimation.config import ALL_SEEDS, BOUNDS, LEGACY_RESULTS, LOTTERIES, MAX_ITER, SEEDS, TOL, random_seeds  # noqa: E402
from estimation.data import exclude_participants, exclusions, observed_ce  # noqa: E402
from estimation.jobs import CHOICES, TESTS, select_jobs  # noqa: E402
from estimation.lottery_paths import transform  # noqa: E402
from estimation.multistart import grid_em_multistart  # noqa: E402
from estimation.run_summary import write_run_summary  # noqa: E402
from estimation.spec import CANON, RP_LABELS_4, UTILITIES, default_bounds, replace_bound  # noqa: E402
from estimation.store import (load_completed, protocol_differences, save_model, scientific_inputs,  # noqa: E402
                              seal_protocol, sha256_bytes, sha256_file, snapshot_sources, write_protocol,
                              write_summary)
from estimation.warm_starts import equivalent_result, job_initializations  # noqa: E402

FILTERS = ["tests", "discounting", "curvature", "memory", "pw", "clusters", "rp"]

# terminal call commands
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    models = parser.add_argument_group("models (default: all)")
    models.add_argument("--tests", nargs="+", choices=TESTS)
    for name, values in CHOICES.items():
        models.add_argument(f"--{name}", nargs="+", choices=values)
    models.add_argument("--clusters", type=int, nargs="+", help="Test 1 class counts (default: 1 2 3 4)")
    models.add_argument("--rp", nargs="+", choices=RP_LABELS_4, help="Test 2 reference points")
    models.add_argument("--utility", choices=UTILITIES,
                        help="power (default) or exponential (CARA, config IDs end in _cara)")
    models.add_argument("--only", nargs="+", metavar="CONFIG_ID", help="Config IDs instead of the filters above")
    sample = parser.add_argument_group("sample (default: everyone in the cleaned data)")
    sample.add_argument("--exclude-endpoint", type=float, metavar="SHARE",
                        help="Drop participants with at least this share of top-of-list answers")
    sample.add_argument("--exclude-file", type=Path, nargs="+", default=[], metavar="CSV",
                        help="Drop participants listed in a CSV with a participant_label column")
    search = parser.add_argument_group("estimation")
    search.add_argument("--lambda-upper", type=float, help="Upper lambda bound (default: config)")
    search.add_argument("--gamma-lower", type=float, help="Lower TK gamma bound (default: config)")
    search.add_argument("--warm-dirs", type=Path, nargs="+", default=[], help="Saved fits of the same models")
    search.add_argument("--seeds", type=int, default=len(SEEDS), help="Random starts per model")
    search.add_argument("--logunif-alpha", type=int, default=0, metavar="N",
                        help="CARA only: the last N random starts draw alpha log-uniformly")
    search.add_argument("--screen-iter", type=int, metavar="K", help="Screen random starts after K EM iterations")
    search.add_argument("--screen-keep", type=float, metavar="SHARE", help="Share of screened starts continued")
    search.add_argument("--reuse-restarts", type=Path, metavar="DIR",
                        help="Earlier run of the same protocol whose finished random starts are copied")
    search.add_argument("--n-workers", type=int, default=24)
    search.add_argument("--max-iter", type=int, default=MAX_ITER)
    search.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--output-dir", type=Path, help="New directory, or one written under the same protocol")
    parser.add_argument("--verbose", action="store_true", help="Print every restart")
    parser.add_argument("--list", action="store_true", help="Show the sample and models; estimate nothing")
    args = parser.parse_args(argv)
    if args.only and any(getattr(args, name) is not None for name in FILTERS + ["utility"]):
        parser.error("--only replaces --tests/--discounting/--curvature/--memory/--pw/--clusters/--rp/--utility")
    if args.clusters and min(args.clusters) < 1:
        parser.error("--clusters must be positive")
    if args.exclude_endpoint is not None and not 0 < args.exclude_endpoint <= 1:
        parser.error("--exclude-endpoint is a share in (0, 1]")
    if not 1 <= args.seeds <= len(ALL_SEEDS):
        parser.error(f"--seeds must be between 1 and {len(ALL_SEEDS)}")
    if not 0 <= args.logunif_alpha <= args.seeds:
        parser.error("--logunif-alpha must be between 0 and --seeds")
    if (args.screen_iter is None) != (args.screen_keep is None):
        parser.error("--screen-iter and --screen-keep go together")
    if args.screen_iter is not None and not (1 <= args.screen_iter < args.max_iter and 0 < args.screen_keep <= 1):
        parser.error("--screen-iter must be in [1, --max-iter) and --screen-keep in (0, 1]")
    if args.output_dir is None and not args.list:
        parser.error("--output-dir is required (use a new directory for a new sample, bounds or warm starts)")
    return args


# Protocol entries that must agree before restarts of another run are copied
# (the starts themselves are matched by seed and alpha draw).
REUSE_KEYS = ("inputs", "sample_filters", "max_iter", "tol", "utility", "reference_basis", "time_origin",
              "noise_scale")


def reused_protocol(directory):
    path = directory / "run_protocol.json"
    if not path.exists():
        raise ValueError(f"--reuse-restarts: no run_protocol.json in {directory}")
    return dict(directory=str(directory), fingerprint=json.loads(path.read_text())["fingerprint"])


def check_reuse(directory, protocol):
    """Refuse to copy restarts estimated under different inputs or tolerances."""
    old = json.loads((directory / "run_protocol.json").read_text())
    differences = protocol_differences({key: old.get(key) for key in REUSE_KEYS},
                                       {key: protocol.get(key) for key in REUSE_KEYS})
    if differences:
        raise ValueError(f"--reuse-restarts {directory} has a different protocol: {', '.join(differences)}")


def write_summaries(output):
    """summary.csv, then summary.txt; a failing text summary must not stop the
    remaining estimates (rewrite it later with scripts/summarize_run.py)."""
    write_summary(output)
    try:
        write_run_summary(output)
    except Exception as error:  # noqa: BLE001
        print(f"  WARNING summary.txt not written: {error!r}", flush=True)

# define jobs to run
def chosen_jobs(args):
    if args.only:
        return select_jobs(only=args.only)
    defaults = dict(tests=TESTS, clusters=(1, 2, 3, 4), rp=RP_LABELS_4, **CHOICES)
    chosen = {name: getattr(args, name) or defaults[name] for name in FILTERS}
    return select_jobs(tests=chosen["tests"], discounting=chosen["discounting"], curvature=chosen["curvature"],
                       memory=chosen["memory"], pw=chosen["pw"], clusters=chosen["clusters"], rps=chosen["rp"],
                       utility=args.utility or "power")


def main(argv=None):
    args = parse_args(argv)
    jobs = chosen_jobs(args)
    utilities = sorted({spec.utility for _, spec in jobs})
    if len(utilities) != 1:
        raise SystemExit(f"Estimate one utility per output directory, not {utilities}")
    utility = utilities[0]
    if args.logunif_alpha and utility != "exponential":
        raise SystemExit("--logunif-alpha draws CARA alpha: it needs --utility exponential")
    bounds = default_bounds(utility)
    if args.lambda_upper is not None:
        bounds = replace_bound("LAM", "high", args.lambda_upper, bounds)
    if args.gamma_lower is not None:
        bounds = replace_bound("GAM", "low", args.gamma_lower, bounds)

    lotteries = transform(LOTTERIES)
    cleaned = observed_ce()
    dropped = exclusions(cleaned, lotteries, args.exclude_endpoint, args.exclude_file)
    excluded = sorted(dropped.participant_label[dropped.in_sample])
    observed = exclude_participants(cleaned, excluded)
    valid = observed[observed.lottery_id.isin(lotteries)].dropna(subset=["ce_observed"])
    subjects = sorted(valid.participant_label.unique())
    print(f"Sample: {len(subjects)} participants, {len(valid)} CEs; excluded {len(excluded)} of "
          f"{cleaned.participant_label.nunique()}", flush=True)
    if (~dropped.in_sample).any():
        print(f"  NOTE {(~dropped.in_sample).sum()} listed participants are not in the cleaned data", flush=True)
    for method in sorted({spec.pw for _, spec in jobs}):
        weight = "PALPHA" if method == "prelec" else "GAM"
        print(f"  {method}: {utility} utility, alpha bound {list(bounds[method][CANON[method]['AG']])}, "
              f"lambda bound {list(bounds[method][CANON[method]['LAM']])}, "
              f"weighting bound {list(bounds[method][CANON[method][weight]])}", flush=True)
    print(f"Models: {len(jobs)}", flush=True)
    for cid, _ in jobs:
        print("  " + cid, flush=True)
    if args.list:
        return

    output = args.output_dir.resolve()
    if output == LEGACY_RESULTS.resolve() or output in [path.resolve() for path in args.warm_dirs]:
        raise ValueError("Write to a new directory: the output may not be the legacy grid or a warm directory")
    output.mkdir(exist_ok=True)
    seeds = random_seeds(args.seeds)
    logunif_seeds = seeds[len(seeds) - args.logunif_alpha:] if args.logunif_alpha else []
    reuse = None
    if args.reuse_restarts is not None:
        reuse = args.reuse_restarts.resolve()
        if reuse == output:
            raise ValueError("--reuse-restarts must name another run directory")
    protocol = seal_protocol(dict(
        inputs=scientific_inputs(observed, bounds),
        sample_filters=dict(exclude_endpoint=args.exclude_endpoint,
                            exclude_files={path.name: sha256_file(path) for path in args.exclude_file},
                            excluded_sha256=sha256_bytes("\n".join(excluded).encode()), n_excluded=len(excluded)),
        warm_dirs=[str(path.resolve()) for path in args.warm_dirs],
        seeds=list(map(int, seeds)), max_iter=args.max_iter, tol=args.tol,
        reference_basis="current remaining payoff; original historical benchmark formulas",
        time_origin="current decision, next payment delay 1", noise_scale="original full-lottery spread",
        memory_equivalence="pure-reference sepmem models are labelled aliases of their commem fit",
        # Power-utility protocols predate this key; leaving it out keeps their fingerprints.
        **({} if utility == "power" else dict(utility=utility)),
        # Likewise the random-start options, recorded only when used.
        **(dict(alpha_logunif_seeds=list(map(int, logunif_seeds))) if logunif_seeds else {}),
        **(dict(screening=dict(iter=args.screen_iter, keep=args.screen_keep)) if args.screen_iter else {}),
        **(dict(reuse_restarts=reused_protocol(reuse)) if reuse else {})))
    fingerprint = protocol["fingerprint"]
    if reuse:
        check_reuse(reuse, protocol)
    write_protocol(output, protocol)
    snapshot_sources(output, [__file__])
    if len(dropped):
        dropped.to_csv(output / "excluded_participants.csv", index=False)

    for index, (cid, spec) in enumerate(jobs, 1):
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
            write_summaries(output)
            continue
        initializations = job_initializations(spec, subjects, output, fingerprint, args.warm_dirs,
                                              LEGACY_RESULTS, bounds)
        print(f"  warm starts: {sorted(initializations)}", flush=True)
        started = time.time()
        result = grid_em_multistart(spec, seeds, y=observed, lotteries=lotteries, max_iter=args.max_iter,
                                    tol=args.tol, n_workers=args.n_workers, initializations=initializations,
                                    checkpoint_dir=output / cid / "restarts", verbose=args.verbose, bounds=bounds,
                                    logunif_seeds=logunif_seeds, screen_iter=args.screen_iter,
                                    screen_keep=args.screen_keep,
                                    reuse_dir=reuse / cid / "restarts" if reuse else None) #main estimation function
        if excluded:
            result["excluded_participants"] = excluded
        meta = save_model(output, spec, result, len(valid), time.time() - started, fingerprint)
        write_summaries(output)
        print(f"  SAVED {cid}: LL={meta['log_likelihood']:.4f}, BIC={meta['bic']:.2f}, "
              f"exit={meta['exit_reason']}, {(time.time() - started) / 60:.1f} min", flush=True)
    print("ALL REQUESTED MODELS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
