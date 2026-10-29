"""Summary tables and plots for a results directory of the model grid.

Reads every <results_dir>/<config_id>/{meta.json,result.pkl} and builds

    summary.csv                 one row per completed job
    failed.csv                  one row per failed job (if any)
    test1_bic_by_C.csv          BIC(C=1..4) per 16-spec combo, best C flagged
    test1_pairwise_<spec>.png   cluster preference-param pairplot, all C
    test2_bic_ranking.csv       SQ/PA/LE/FE ranked by BIC per 16-spec combo
    test2_pairwise_<spec>.png   preference-param pairplot, 4 RP points
    test3_map_summary.csv       model-implied shares vs MAP counts/percentages

    python scripts/consolidate.py results_grid
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimation.spec import CANON  # noqa: E402
from estimation.store import load_pickle  # noqa: E402

SPEC_COLS = ["discounting", "curvature", "memory", "pw"]


def _spec_key(row):
    return "_".join(row[c] for c in SPEC_COLS)


def load_all_meta(results_dir: Path):
    rows = []
    for d in sorted(results_dir.iterdir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as fh:
            rows.append(json.load(fh))
    return pd.DataFrame(rows)


def load_result(results_dir: Path, config_id: str):
    return load_pickle(results_dir / config_id / "result.pkl")


def pref_params(theta, method):
    idx = CANON[method]
    d = {"r": theta[idx["R"]], "alpha_gain": theta[idx["AG"]],
         "alpha_loss": theta[idx["AL"]], "lambda": theta[idx["LAM"]]}
    if method == "tk":
        d["gamma"] = theta[idx["GAM"]]
    else:
        d["palpha"] = theta[idx["PALPHA"]]
    return d


def _pairplot(points_by_group, param_names, out_path, size_key=None, title=""):
    """points_by_group: {group_label: [ {param: val, ..., '_size': s}, ... ]}
    Draws a lower-triangle scatter matrix over param_names, one subplot per
    (i>j) pair, colored by group."""
    p = len(param_names)
    if p < 2:
        return
    fig, axes = plt.subplots(p - 1, p - 1, figsize=(3 * (p - 1), 3 * (p - 1)))
    if p - 1 == 1:
        axes = np.array([[axes]])

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(points_by_group), 1)))

    for row_i in range(1, p):
        for col_j in range(0, row_i):
            ax = axes[row_i - 1, col_j]
            xname, yname = param_names[col_j], param_names[row_i]
            for gi, (label, pts) in enumerate(points_by_group.items()):
                xs = [pt[xname] for pt in pts]
                ys = [pt[yname] for pt in pts]
                sizes = [pt.get("_size", 60) for pt in pts]
                ax.scatter(xs, ys, s=sizes, color=colors[gi], label=str(label),
                           alpha=0.75, edgecolors="black", linewidths=0.5)
            if row_i == p - 1:
                ax.set_xlabel(xname)
            if col_j == 0:
                ax.set_ylabel(yname)
        for col_j in range(row_i, p - 1):
            axes[row_i - 1, col_j].axis("off")

    handles, labels = axes[-1, 0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.legend(handles, labels, loc="upper left",
               bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ── Test 1: BIC-by-C table + pairwise plots ───────────────────────────────

def test1_bic_by_C(df: pd.DataFrame, results_dir: Path):
    t1 = df[df["test"] == "test1"].copy()
    if t1.empty:
        return
    t1["spec_key"] = t1.apply(_spec_key, axis=1)

    rows = []
    for spec_key, g in t1.groupby("spec_key"):
        row = {"spec": spec_key}
        bic_by_C = {}
        for _, r in g.iterrows():
            bic_by_C[int(r["C"])] = r["bic"]
            row[f"BIC_C{int(r['C'])}"] = r["bic"]
            row[f"LL_C{int(r['C'])}"] = r["log_likelihood"]
            row[f"k_C{int(r['C'])}"] = r["k_params"]
            row[f"converged_C{int(r['C'])}"] = r["converged"]
        if bic_by_C:
            row["best_C"] = min(bic_by_C, key=bic_by_C.get)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("spec")
    out.to_csv(results_dir / "test1_bic_by_C.csv", index=False)
    print(f"  wrote {results_dir/'test1_bic_by_C.csv'}  ({len(out)} specs)")

    for spec_key, g in t1.groupby("spec_key"):
        pw = g["pw"].iloc[0]
        param_names = ["r", "alpha_gain", "alpha_loss", "lambda"] + (
            ["gamma"] if pw == "tk" else ["palpha"])
        points_by_C = {}
        for _, r in g.sort_values("C").iterrows():
            res = load_result(results_dir, r["config_id"])
            pts = []
            for j, theta in enumerate(res["thetas"]):
                d = pref_params(theta, pw)
                d["_size"] = 400 * float(res["pis"][j])
                pts.append(d)
            points_by_C[f"C={int(r['C'])}"] = pts
        out_path = results_dir / f"test1_pairwise_{spec_key}.png"
        _pairplot(points_by_C, param_names, out_path,
                  title=f"Test 1 — {spec_key} (marker size ∝ cluster share)")
    print(f"  wrote test1_pairwise_*.png  ({t1['spec_key'].nunique()} specs)")


# ── Test 2: BIC ranking + pairwise plot ───────────────────────────────────

def test2_bic_ranking(df: pd.DataFrame, results_dir: Path):
    t2 = df[df["test"] == "test2"].copy()
    if t2.empty:
        return
    t2["spec_key"] = t2.apply(_spec_key, axis=1)

    rows = []
    for spec_key, g in t2.groupby("spec_key"):
        row = {"spec": spec_key}
        bic_by_rp = {}
        for _, r in g.iterrows():
            rp = r["rp_identity"]
            bic_by_rp[rp] = r["bic"]
            row[f"BIC_{rp}"] = r["bic"]
            row[f"LL_{rp}"] = r["log_likelihood"]
            row[f"k_{rp}"] = r["k_params"]
        if bic_by_rp:
            ranked = sorted(bic_by_rp, key=bic_by_rp.get)
            row["ranking_best_to_worst"] = " < ".join(ranked)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("spec")
    out.to_csv(results_dir / "test2_bic_ranking.csv", index=False)
    print(f"  wrote {results_dir/'test2_bic_ranking.csv'}  ({len(out)} specs)")

    for spec_key, g in t2.groupby("spec_key"):
        pw = g["pw"].iloc[0]
        param_names = ["r", "alpha_gain", "alpha_loss", "lambda"] + (
            ["gamma"] if pw == "tk" else ["palpha"])
        points_by_rp = {}
        for _, r in g.iterrows():
            res = load_result(results_dir, r["config_id"])
            d = pref_params(res["thetas"][0], pw)
            points_by_rp[r["rp_identity"]] = [d]
        out_path = results_dir / f"test2_pairwise_{spec_key}.png"
        _pairplot(points_by_rp, param_names, out_path,
                  title=f"Test 2 — {spec_key} (one point per pure RP model)")
    print(f"  wrote test2_pairwise_*.png  ({t2['spec_key'].nunique()} specs)")


# ── Test 3: model-implied shares vs MAP assignment ────────────────────────

def test3_map_summary(df: pd.DataFrame, results_dir: Path):
    t3 = df[df["test"] == "test3"].copy()
    if t3.empty:
        return
    t3["spec_key"] = t3.apply(_spec_key, axis=1)

    rows = []
    for _, r in t3.iterrows():
        res = load_result(results_dir, r["config_id"])
        resp = res["resp"]
        n = resp.shape[0]
        hard = np.argmax(resp, axis=1)
        rp_ids = res["rp_identities"]
        for j, label in enumerate(rp_ids):
            hard_n = int(np.sum(hard == j))
            rows.append({
                "spec": r["spec_key"], "config_id": r["config_id"],
                "cluster_rp": label,
                "pi_model_implied": float(res["pis"][j]),
                "map_count": hard_n,
                "map_pct": 100.0 * hard_n / n,
            })
    out = pd.DataFrame(rows).sort_values(["spec", "cluster_rp"])
    out.to_csv(results_dir / "test3_map_summary.csv", index=False)
    print(f"  wrote {results_dir/'test3_map_summary.csv'}  ({t3.shape[0]} specs)")


# ── Entry point ─────────────────────────────────────────────────────────

def consolidate(results_dir: Path):
    results_dir = Path(results_dir)
    df = load_all_meta(results_dir)
    if df.empty:
        print("No completed jobs found in", results_dir)
        return

    done = df[df["status"] == "done"].copy()
    failed = df[df["status"] == "failed"].copy()

    done.to_csv(results_dir / "summary.csv", index=False)
    print(f"wrote {results_dir/'summary.csv'}  ({len(done)} done jobs)")
    if not failed.empty:
        failed.to_csv(results_dir / "failed.csv", index=False)
        print(f"wrote {results_dir/'failed.csv'}  ({len(failed)} failed jobs)")

    if done.empty:
        return

    print("\nTest 1:")
    test1_bic_by_C(done, results_dir)
    print("\nTest 2:")
    test2_bic_ranking(done, results_dir)
    print("\nTest 3:")
    test3_map_summary(done, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    consolidate(parser.parse_args().results_dir)
