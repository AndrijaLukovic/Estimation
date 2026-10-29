"""Export conditional-grid pickles to a bounded, offline HTML explorer.

Usage:
    python scripts/export_report.py
    python scripts/export_report.py --language en  # separate English report
    python scripts/export_report.py --audit none     # saved diagnostics only
    python scripts/export_report.py --models-only    # omit restart pickles

Inputs are this project's locally generated result/checkpoint pickle files.
The exporter never estimates models or changes any input result.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimation.config import ROOT  # noqa: E402
from estimation.report.export import default_output_dir, export  # noqa: E402
from estimation.report.i18n import localize  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=ROOT/"results_grid_conditional")
    parser.add_argument("--language", "--lang", choices=["zh", "en"], default="zh",
        help="report language (default: zh); English uses a separate default output directory")
    parser.add_argument("--output-dir", type=Path,
        help="override output directory (default: reports/grid_em_conditional_explorer[_en])")
    parser.add_argument("--audit", choices=["all", "models", "none"], default="all")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--boundary-tol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.boundary_tol < 0:
        parser.error(localize("--boundary-tol 必须非负", args.language))
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.language)
    return args


def main(argv=None):
    result = export(parse_args(argv))
    if result["counts"]["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
