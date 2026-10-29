"""Compact existing reports losslessly, preserving saved audits and languages.

Usage:
    python scripts/organize_reports.py
    python scripts/organize_reports.py --check
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimation.config import ROOT  # noqa: E402
from estimation.report.organize import organize  # noqa: E402
from estimation.report.storage import validate_report_tree  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=ROOT/"reports")
    parser.add_argument("--check", action="store_true", help="verify the combined count and byte limits without changing files")
    args = parser.parse_args(argv)
    result = validate_report_tree(args.reports_dir) if args.check else organize(args.reports_dir)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
