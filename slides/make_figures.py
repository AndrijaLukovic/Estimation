"""Figures for the results slides.

Main results: exp/sepalpha/commem/TK after excluding top-endpoint responders
(results/sample_stricter_no_endpoint/tk_lam20_g028_main; scripts/run_grid.py --exclude-endpoint 0.5).
CARA slides (results_cara.tex): the same sample and protocol with exponential utility
(Test 3 and Test 1 C=4 from the 500-start rerun results/sample_stricter_no_endpoint/cara_sepalpha_tk_wide_s500).
Backup profile figures: full sample, Prelec.

    python slides/make_figures.py                 # PDF figures into slides/figures
    python slides/make_figures.py --only cara     # only the CARA figures
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
from estimation.config import RESULTS  # noqa: E402
from estimation.spec import CANON  # noqa: E402

MAIN = RESULTS / "sample_stricter_no_endpoint" / "tk_lam20_g028_main"
CARA = RESULTS / "sample_stricter_no_endpoint" / "cara_sepalpha_tk_wide_s500"
PROFILES = RESULTS / "sample_stricter" / "profiles" / "grid" / "test3_exp_sepalpha_commem_prelec"
BLUE, ORANGE, INK, MUTED, GRID, SHADE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3de", "#f0efec"
TYPE_COLORS = dict(SQ=BLUE, PA=ORANGE, LE="#1baf7a", FE="#eda100")   # fixed per type, categorical slots 1-4
THRESHOLD = 1.92   # chi2(1) 95% quantile / 2; a heuristic only (boundary + mixture)
EXCLUSION = .5     # --exclude-endpoint used for the main results

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


def load_result(run, name):
    with (run / name / "result.pkl").open("rb") as handle:
        return pickle.load(handle)


def modal(result):
    return np.asarray(result["resp"]).argmax(1)


def heatmap(counts, rows, columns, xlabel, ylabel, path):
    ramp = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    fig, axis = plt.subplots(figsize=(2.5, 2.0))
    axis.imshow(counts, cmap=ramp, vmin=0, vmax=counts.max(), aspect="auto")
    for (i, j), count in np.ndenumerate(counts):
        axis.text(j, i, str(count), ha="center", va="center", fontsize=9,
                  color="white" if count > .45 * counts.max() else INK)
    axis.set_xticks(range(counts.shape[1]))
    axis.set_xticklabels(columns, fontsize=8)
    axis.set_yticks(range(counts.shape[0]))
    axis.set_yticklabels(rows, fontsize=8)
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.tick_params(length=0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def type_columns(result):
    return [f"{label}\n{share:.0%}" for label, share in zip(result["rp_identities"], result["pis"])]


def class_crosstab(out, fmt, run=MAIN, suffix="", name="crosstab_test1_test3"):
    """Test 1 C=4 class (rows, numbered by share) against Test 3 type (columns)."""
    test1 = load_result(run, f"test1_exp_sepalpha_commem_tk{suffix}_C4")
    test3 = load_result(run, f"test3_exp_sepalpha_commem_tk{suffix}")
    assert test1["subjects"] == test3["subjects"]
    order = np.argsort(-np.asarray(test1["pis"]))
    counts = np.zeros((4, 4), dtype=int)
    np.add.at(counts, (np.argsort(order)[modal(test1)], modal(test3)), 1)
    rows = [f"{k + 1}: " + class_label(test1["thetas"][j], test1["pis"][j]) for k, j in enumerate(order)]
    heatmap(counts, rows, type_columns(test3), "Test 3 type (modal posterior)", "Test 1, $C=4$ class (modal)",
            out / f"{name}.{fmt}")


def cara_crosstab(out, fmt):
    class_crosstab(out, fmt, CARA, "_cara", "cara_crosstab_test1_test3")


def utility_crosstab(out, fmt):
    """Test 3 type under power utility (rows) against CARA (columns)."""
    power = load_result(MAIN, "test3_exp_sepalpha_commem_tk")
    cara = load_result(CARA, "test3_exp_sepalpha_commem_tk_cara")
    assert power["subjects"] == cara["subjects"]
    counts = np.zeros((4, 4), dtype=int)
    np.add.at(counts, (modal(power), modal(cara)), 1)
    rows = [f"{label} ({share:.0%})" for label, share in zip(power["rp_identities"], power["pis"])]
    heatmap(counts, rows, type_columns(cara), "CARA: Test 3 type", "Power: Test 3 type",
            out / f"cara_crosstab_utility.{fmt}")


def spread_labels(values, gap):
    """Label heights near ``values``, at least ``gap`` apart, in the same order."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)[order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    placed -= max(0, placed[-1] - max(values))   # shift back so the top label stays at its line
    result = np.empty_like(placed)
    result[order] = placed
    return result


def risk_premium_curves(out, fmt):
    """Risk premium of 50/50 gambles by stake for each CARA Test 3 type (utility only)."""
    from estimation import cpt
    cara = load_result(CARA, "test3_exp_sepalpha_commem_tk_cara")
    index = CANON["tk"]
    stakes = np.geomspace(1, 1000, 91)
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.25), gridspec_kw=dict(wspace=.55))
    titles = ["Gains: 50/50 of \u00a30 or +\u00a3$x$", "Losses: 50/50 of \u00a30 or \u2212\u00a3$x$"]
    for side, (axis, title) in enumerate(zip(axes, titles)):
        axis.axhline(0, color=MUTED, lw=.8, zorder=1)
        ends, labels = [], []
        for theta, label in zip(cara["thetas"], cara["rp_identities"]):
            options = dict(alpha=float(theta[index["AG"]]), alpha_loss=float(theta[index["AL"]]), utility="exponential")
            values = 100 * np.array([cpt.risk_premium(x, **options)[side] for x in stakes])
            axis.plot(stakes, values, color=TYPE_COLORS[label], lw=1.6, zorder=3)
            ends.append(values[-1])
            labels.append(label)
        low, high = axis.get_ylim()
        for label, height in zip(labels, spread_labels(ends, .075 * (high - low))):
            axis.text(1150, height, label, fontsize=7, color=INK, va="center", ha="left", clip_on=False)
        axis.set_xscale("log")
        axis.set_xlim(1, 1000)
        axis.set_xticks([1, 10, 100, 1000])
        axis.set_xticklabels(["\u00a31", "\u00a310", "\u00a3100", "\u00a31,000"])
        axis.minorticks_off()
        axis.set_xlabel("Stake $x$")
        axis.set_title(title, fontsize=7.5, loc="left", color=INK)
    axes[0].set_ylabel("Risk premium (% of |EV|)")
    fig.savefig(out / f"cara_risk_premium.{fmt}")
    plt.close(fig)


FIGURES = dict(main=(screening, class_crosstab, lambda_profile, palpha_profile),
               cara=(cara_crosstab, utility_crosstab, risk_premium_curves))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    parser.add_argument("--only", choices=sorted(FIGURES), help="one group of figures (default: all)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.format == "png":
        plt.rcParams["savefig.dpi"] = 300
    for group, figures in FIGURES.items():
        if args.only in (None, group):
            for figure in figures:
                figure(args.out, args.format)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
