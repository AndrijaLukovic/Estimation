"""Figures for the results slides.

Main results: exp/sepalpha/commem/TK after excluding top-endpoint responders
(scripts/run_exclusion.py, results_grid_conditional_stricter_no_endpoint_tk_main).
Backup profile figures: full sample, Prelec.

    python slides/make_figures.py                 # PDF figures into slides/figures
    python slides/make_figures.py --format png --out /some/dir
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from estimation.spec import CANON  # noqa: E402

MAIN = ROOT / "results_grid_conditional_stricter_no_endpoint_tk_main"
PROFILES = ROOT / "profiles/results_grid_conditional_stricter/test3_exp_sepalpha_commem_prelec"
BLUE, ORANGE, INK, MUTED, GRID, SHADE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3de", "#f0efec"
THRESHOLD = 1.92   # chi2(1) 95% quantile / 2; a heuristic only (boundary + mixture)
EXCLUSION = .5     # scripts/run_exclusion.py default

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": .6,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "savefig.bbox": "tight", "savefig.pad_inches": .02,
})


def profile_points(name):
    path = PROFILES / f"c2_{name}" / f"cluster_2_{name}" / "profile.json"
    return json.loads(path.read_text())["points"]


def threshold_line(axis):
    axis.axhline(THRESHOLD, color=MUTED, lw=.8, ls="--", zorder=1)
    axis.annotate("1.92", (1, THRESHOLD), xycoords=("axes fraction", "data"), xytext=(2, 0),
                  textcoords="offset points", va="center", fontsize=6.5, color=MUTED)


def lambda_profile(out, fmt):
    points = profile_points("lambda")
    value = np.array([p["value"] for p in points])
    delta = np.array([p["delta_ll"] for p in points])
    gain = np.array([p["focal_parameters"]["alpha_gain"] for p in points])
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(3.3, 2.75),
                                      gridspec_kw=dict(height_ratios=[2.3, 1], hspace=.12))
    for axis in (top, bottom):
        axis.axvspan(6, 130, color=SHADE, lw=0, zorder=0)
    top.plot(value, delta, color=BLUE, marker="o", zorder=3)
    threshold_line(top)
    top.text(8, 11.5, "beyond original\nbound $\\lambda\\leq 6$", fontsize=6.5, color=MUTED, va="top")
    top.set_ylabel("$\\Delta$LL to profile maximum")
    top.set_ylim(-.4, 13.5)
    bottom.plot(value, gain, color=BLUE, marker="o", zorder=3)
    bottom.set_ylabel("$\\alpha^{+}$")
    bottom.set_ylim(.4, .9)
    bottom.set_xscale("log")
    bottom.set_xlim(.2, 130)
    bottom.set_xticks([.25, 1, 6, 20, 100])
    bottom.set_xticklabels(["0.25", "1", "6", "20", "100"])
    bottom.minorticks_off()
    bottom.set_xlabel("$\\lambda$ of the PA type (held fixed)")
    fig.savefig(out / f"profile_lambda.{fmt}")
    plt.close(fig)


def palpha_profile(out, fmt):
    points = profile_points("palpha")
    value = np.array([p["value"] for p in points])
    delta = np.array([p["delta_ll"] for p in points])
    corner = value <= .1
    suspect = np.isin(value, [.6, .8])   # non-monotone; only one start reached them
    fig, axis = plt.subplots(figsize=(3.3, 2.45))
    threshold_line(axis)
    for mask, color in ((corner, BLUE), (~corner, ORANGE)):
        axis.plot(value[mask], delta[mask], color=color, zorder=2)
        solid = mask & ~suspect
        axis.plot(value[solid], delta[solid], ls="none", marker="o", color=color, zorder=3)
        hollow = mask & suspect
        axis.plot(value[hollow], delta[hollow], ls="none", marker="o", mfc="white", mec=color, mew=1.2, zorder=3)
    axis.text(1e-5, 6, "corner", color=INK, fontsize=7, ha="center")
    axis.text(.35, 48, "regular PT", color=INK, fontsize=7, ha="right")
    handles = [Line2D([], [], color=BLUE, marker="o", label="Corner: $\\alpha^{-}\\to 0$, $\\lambda\\approx 6$"),
               Line2D([], [], color=ORANGE, marker="o", label="Regular: $\\alpha^{\\pm}\\approx 1$, $\\lambda\\approx 1.5$-$2.8$"),
               Line2D([], [], ls="none", marker="o", mfc="white", mec=MUTED, mew=1.2,
                      label="Likely not at profile max")]
    axis.legend(handles=handles, loc="upper left", frameon=False, handlelength=1.6, borderaxespad=.2)
    axis.set_xscale("log")
    axis.set_xlim(5e-9, 1.6)
    axis.set_xticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    axis.minorticks_off()
    axis.set_ylim(-2, 85)
    axis.set_xlabel("Prelec $\\gamma$ of the PA type (held fixed)")
    axis.set_ylabel("$\\Delta$LL to profile maximum")
    fig.savefig(out / f"profile_gamma.{fmt}")
    plt.close(fig)


def screening(out, fmt):
    from estimation.config import LOTTERIES
    from estimation.data import endpoint_shares, observed_ce
    from estimation.lottery_paths import transform
    shares = endpoint_shares(observed_ce(), transform(LOTTERIES)).to_numpy()
    edges = np.linspace(0, 1, 21)
    counts, _ = np.histogram(shares, bins=edges)
    fig, axis = plt.subplots(figsize=(3.4, 2.3))
    colors = [ORANGE if left >= EXCLUSION else BLUE for left in edges[:-1]]
    axis.bar(edges[:-1], counts, width=.05 * .92, align="edge", color=colors, zorder=2)
    axis.axvline(EXCLUSION, color=MUTED, lw=.8, ls="--", zorder=1)
    kept, dropped = np.sum(shares < EXCLUSION), np.sum(shares >= EXCLUSION)
    axis.text(.25, counts.max() * .92, f"kept: {kept}", ha="center", fontsize=7.5)
    axis.text(.75, counts.max() * .45, f"excluded: {dropped}", ha="center", fontsize=7.5)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Share of a participant's CEs at the top of the list")
    axis.set_ylabel("Participants")
    fig.savefig(out / f"screening.{fmt}")
    plt.close(fig)


def class_label(theta, share):
    ix = CANON["tk"]
    s1, s2, s3 = theta[ix["S1"]], theta[ix["S2"]], theta[ix["S3"]]
    weights = dict(SQ=s1, PA=s2 * (1 - s1), LE=s3 * (1 - s1) * (1 - s2))
    weights["FE"] = 1 - sum(weights.values())
    reference = max(weights, key=weights.get) if max(weights.values()) >= .6 else "mixed RP"
    return f"{reference} ({share:.0%})"


def class_crosstab(out, fmt):
    def load(name):
        with (MAIN / name / "result.pkl").open("rb") as handle:
            return pickle.load(handle)
    test1 = load("test1_exp_sepalpha_commem_tk_C4")
    test3 = load("test3_exp_sepalpha_commem_tk")
    assert test1["subjects"] == test3["subjects"]
    order = np.argsort(-np.asarray(test1["pis"]))
    counts = np.zeros((4, 4), dtype=int)
    np.add.at(counts, (np.argsort(order)[np.asarray(test1["resp"]).argmax(1)], np.asarray(test3["resp"]).argmax(1)), 1)
    rows = [f"{k + 1}: " + class_label(test1["thetas"][j], test1["pis"][j]) for k, j in enumerate(order)]
    columns = [f"{label}\n{share:.0%}" for label, share in zip(test3["rp_identities"], test3["pis"])]
    ramp = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    fig, axis = plt.subplots(figsize=(2.5, 2.0))
    axis.imshow(counts, cmap=ramp, vmin=0, vmax=counts.max(), aspect="auto")
    for (i, j), count in np.ndenumerate(counts):
        axis.text(j, i, str(count), ha="center", va="center", fontsize=9,
                  color="white" if count > .45 * counts.max() else INK)
    axis.set_xticks(range(4))
    axis.set_xticklabels(columns, fontsize=8)
    axis.set_yticks(range(4))
    axis.set_yticklabels(rows, fontsize=8)
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.set_xlabel("Test 3 type (modal posterior)")
    axis.set_ylabel("Test 1, $C=4$ class (modal)")
    axis.tick_params(length=0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.savefig(out / f"crosstab_test1_test3.{fmt}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.format == "png":
        plt.rcParams["savefig.dpi"] = 300
    screening(args.out, args.format)
    class_crosstab(args.out, args.format)
    lambda_profile(args.out, args.format)
    palpha_profile(args.out, args.format)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
