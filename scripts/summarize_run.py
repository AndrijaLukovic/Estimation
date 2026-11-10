"""Write summary.txt for estimation runs: sample and bounds, the model fit
table, model comparison and every model's class parameters, read from the
saved files (nothing is re-estimated).

    python scripts/summarize_run.py results/sample_stricter_no_endpoint/tk_lam20_g028_main
    python scripts/summarize_run.py tk_lam20_g028_main tk_lam20      # run names under results/<sample>/
    python scripts/summarize_run.py sample_stricter/grid             # or <sample>/<run>
    python scripts/summarize_run.py --all                            # every run under results/
    python scripts/summarize_run.py tk_lam20_g028_main --stdout      # print instead of writing

Each summary goes to <run>/summary.txt (next to summary.csv) unless --output
is given. run_grid.py rewrites the same file after every model it saves.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimation.config import RESULTS  # noqa: E402
from estimation.run_summary import display_path, find_runs, run_summary, write_run_summary  # noqa: E402


def resolve(name):
    """A run directory from a path, a run name or <sample>/<run> under results/."""
    path = Path(name)
    if path.is_dir():
        return path
    runs = find_runs()
    matches = [run for run in runs if run.name == name or str(run.relative_to(RESULTS)) == name.strip("/")]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise SystemExit(f"{name} names several runs; give one of: "
                         + ", ".join(str(run.relative_to(RESULTS)) for run in matches))
    raise SystemExit(f"No run {name!r}; runs under {display_path(RESULTS)}: "
                     + ", ".join(str(run.relative_to(RESULTS)) for run in runs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="Run directories, run names or <sample>/<run>")
    parser.add_argument("--all", action="store_true", help="Every run under results/")
    parser.add_argument("--stdout", action="store_true", help="Print the summaries instead of writing them")
    parser.add_argument("--output", type=Path, help="Write to this file instead of <run>/summary.txt (one run)")
    args = parser.parse_args(argv)
    if bool(args.runs) == args.all:
        parser.error("give run directories or names, or --all")
    runs = find_runs() if args.all else [resolve(name) for name in args.runs]
    if args.output and len(runs) != 1:
        parser.error("--output takes one run")
    for run in runs:
        if args.stdout:
            print(run_summary(run))
        else:
            print(f"Wrote {display_path(write_run_summary(run, args.output))}")


if __name__ == "__main__":
    main()
