"""
Compares EM estimates between full lottery set and low-stake-only subset.

Runs two multistart EM estimations and prints a side-by-side parameter table.
Results are also saved to em_outputs/compare_stakes.txt.
"""

import os
import time
import numpy as np
import functions as f
from datetime import datetime
from pathlib import Path

import GlobalSettings
from GlobalSettings import (GlobalMethod, GlobalLottery, GlobalLotteryLowStake,
                             GlobalCluster, GlobalSeedsSet, GlobalTol, GlobalInterMax)
from EM_parallel import run_em_multistart
from main import get_observed_ce
from Mixture import write_em_results, _s_to_a

PARAM_NAMES = {
    "tk":     ["r", "alpha_gain", "alpha_loss", "lambda", "gamma",
               "a1", "a2", "a3", "delta"],
    "prelec": ["r", "alpha_gain", "alpha_loss", "lambda", "beta",
               "palpha", "a1", "a2", "a3", "delta"],
}


def run_one(label, lottery_set):
    print(f"\n{'='*60}")
    print(f"  Running: {label}  ({len(lottery_set)} lotteries)")
    print(f"{'='*60}")
    y         = get_observed_ce(export_excel=False)
    lotteries = f.transform(lottery_set)
    t0        = time.time()
    result    = run_em_multistart(
        seeds=GlobalSeedsSet,
        method=GlobalMethod,
        c=GlobalCluster,
        y=y,
        lotteries=lotteries,
        max_iter=GlobalInterMax,
        tol=GlobalTol,
    )
    result["method_used"] = GlobalMethod
    result["elapsed"]     = time.time() - t0
    result["label"]       = label
    result["n_lotteries"] = len(lottery_set)
    result["lottery_ids"] = sorted(lottery_set.keys())
    return result


def format_table(results, method):
    pnames  = PARAM_NAMES[method]
    labels  = [r["label"] for r in results]
    C       = len(results[0]["thetas"])

    lines = []
    sep   = "=" * 80

    lines += [sep, "STAKE COMPARISON — EM Mixture Model", sep, ""]
    lines.append(f"  Method: {method}   C={C}   "
                 f"Seeds={len(GlobalSeedsSet)}   Tol={GlobalTol}")
    lines.append("")

    for j in range(C):
        lines += [f"  {'─'*76}", f"  Cluster {j+1}", f"  {'─'*76}"]
        col_w = 14
        hdr   = f"  {'Parameter':<12}" + "".join(f"  {lb:>{col_w}}" for lb in labels)
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))

        for p_idx, pname in enumerate(pnames):
            row = f"  {pname:<12}"
            for res in results:
                val = res["thetas"][j][p_idx]
                row += f"  {val:>{col_w}.4f}"
            lines.append(row)

        # mixing weight
        row = f"  {'pi':<12}"
        for res in results:
            row += f"  {res['pis'][j]:>{col_w}.4f}"
        lines.append(row)
        lines.append("")

    lines += [f"  {'─'*76}", "  Summary", f"  {'─'*76}"]
    hdr = f"  {'Metric':<20}" + "".join(f"  {lb:>{col_w}}" for lb in labels)
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for key, fmt in [("log_likelihood", ".4f"), ("n_iter", "d"),
                     ("converged", ""), ("n_lotteries", "d"), ("elapsed", ".0f")]:
        row = f"  {key:<20}"
        for res in results:
            val = res.get(key, "?")
            row += f"  {str(val) if fmt == '' else format(val, fmt):>{col_w}}"
        lines.append(row)
    lines.append("")

    lines.append("  Lotteries used:")
    for res in results:
        lines.append(f"  [{res['label']}]  {', '.join(res['lottery_ids'])}")

    return "\n".join(lines)


if __name__ == "__main__":
    results = [
        run_one("Full (16 lotteries)", GlobalLottery),
        run_one("Low-stake (11 lotteries)", GlobalLotteryLowStake),
    ]

    table = format_table(results, GlobalMethod)
    print("\n\n" + table)

    # Save
    outdir = Path("em_outputs")
    outdir.mkdir(exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = outdir / f"compare_stakes_{GlobalMethod}_C{GlobalCluster}_{ts}.txt"
    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write(table + "\n")

    # Also save individual result files for reference
    for res in results:
        slug = res["label"].split()[0].lower()
        write_em_results(res, filepath=str(outdir / f"em_{slug}_{GlobalMethod}_C{GlobalCluster}_{ts}.txt"))

    print(f"\n  Saved → {outfile}")
