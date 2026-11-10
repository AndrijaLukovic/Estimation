"""Plain-text summary of one estimation run: <run>/summary.txt.

Built from the saved files only (about.txt, run_protocol.json, every
test*/meta.json and result.pkl): nothing is re-estimated or recomputed, so it
works for every run under results/, including runs written before the refactor
(older protocol formats) and the legacy unconditional grid (no protocol).
run_grid.py rewrites it after each saved model; scripts/summarize_run.py writes
it for existing runs.

Sections: sample and protocol, model fit table, model comparison (Test 1 BIC by
C, Test 2 ranking of reference points), then every model's class parameters.
"""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from . import cpt
from .config import KSI_BOUNDS, RESULTS, ROOT
from .diagnostics import probability_shape
from .spec import CANON, RP_LABELS_4, default_bounds
from .store import atomic_write_text, load_pickle
from .valuation import s_to_a

BOUNDARY_TOL = 1e-4      # same default as export_report.py --boundary-tol
STAKES = (5, 20, 50)     # loss aversion A(x) = -u(-x)/u(x) at these amounts (£)
RISK_STAKES = (20, 100, 500)   # risk premium of 50/50 gambles of these sizes (£), see cpt.risk_premium
HIT_TOL = 0.01           # a restart "hits" the optimum within this log-likelihood
TESTS = ("test1", "test2", "test3")
LABELS = dict(R="r", AG="alpha+", AL="alpha-", LAM="lambda", GAM="gamma", PALPHA="palpha",
              DELTA="delta", PSI="psi")
RULE = "-" * 100


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def find_runs(root=RESULTS):
    """Run directories under results/: results/<sample>/<run>/ or a sample
    folder that holds models itself (profiles are not runs)."""
    metas = [*root.glob("*/test*/meta.json"), *root.glob("*/*/test*/meta.json")]
    return sorted({path.parent.parent for path in metas if "profiles" not in path.parts})


# ── Formatting ────────────────────────────────────────────────────────────

def num(value, digits=3):
    """Parameter value: fixed decimals, or 3 significant digits below 0.01."""
    if value is None:
        return "-"
    value = float(value)
    if not np.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    if value != 0 and abs(value) < 0.01:
        return f"{value:.3g}"
    return f"{value:.{digits}f}"


def big(value, digits=2):
    return "-" if value is None else f"{float(value):,.{digits}f}"


def table(rows, left=(0,), indent="  "):
    """Rows of cells as aligned lines; columns in ``left`` are left-aligned."""
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [indent + "  ".join(cell.ljust(width) if i in left else cell.rjust(width)
                               for i, (cell, width) in enumerate(zip(row, widths))).rstrip()
            for row in rows]


def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


# ── Protocol ──────────────────────────────────────────────────────────────

def run_bounds(protocol, pw, utility="power"):
    """Canonical bounds of ``pw`` used by the run. Current protocols store the
    table; pre-refactor ones only the changed lambda and TK gamma bounds; the
    legacy grid (no protocol) used the defaults."""
    tables = protocol.get("inputs", {}).get("bounds")
    if tables:
        return [tuple(bound) for bound in tables[pw]]
    table_ = list(default_bounds(utility)[pw])
    if "lambda_bound" in protocol:
        table_[CANON[pw]["LAM"]] = tuple(protocol["lambda_bound"])
    if "weight_bound" in protocol and pw == "tk":
        table_[CANON[pw]["GAM"]] = tuple(protocol["weight_bound"])
    return table_


def exclusion_text(protocol):
    filters = protocol.get("sample_filters")
    if filters:
        rules = ([f"top-endpoint share >= {filters['exclude_endpoint']}"] if filters.get("exclude_endpoint") else []) + \
                [f"listed in {name}" for name in filters.get("exclude_files", {})]
        return f"{' + '.join(rules)} ({filters['n_excluded']} excluded)" if rules else "none"
    if "exclusion" in protocol:
        return f"{protocol['exclusion']} ({protocol.get('n_excluded', '?')} excluded)"
    return "none"


def utility_of(meta):
    return meta.get("utility") or "power"


def spec_key(meta):
    key = "_".join(meta[name] for name in ("discounting", "curvature", "memory", "pw"))
    return key + ("_cara" if utility_of(meta) == "exponential" else "")


def model_order(meta):
    rp = meta.get("rp_identity")
    return (TESTS.index(meta["test"]) if meta["test"] in TESTS else len(TESTS), spec_key(meta),
            int(meta.get("C") or 0), RP_LABELS_4.index(rp) if rp in RP_LABELS_4 else -1)


# ── Sections ──────────────────────────────────────────────────────────────

def protocol_section(run, protocol, models):
    first = models[0]["meta"]
    utilities = sorted({utility_of(model["meta"]) for model in models})
    lines = ["SAMPLE AND PROTOCOL"]
    rows = [["Participants / CEs", f"{first['n_subjects']:,} / {first['n_obs']:,}"],
            ["Excluded", exclusion_text(protocol)],
            ["Utility", ", ".join(utilities) + ("  (alpha = absolute risk aversion, 1/£)"
                                                if "exponential" in utilities else "")]]
    for pw in sorted({model["meta"]["pw"] for model in models}):
        bounds = run_bounds(protocol, pw, utilities[0])
        rows.append([f"Bounds ({pw})", ", ".join(f"{label} [{bounds[CANON[pw][key]][0]:g}, {bounds[CANON[pw][key]][1]:g}]"
                                                 for key, label in LABELS.items() if key in CANON[pw])])
    ksi = protocol.get("inputs", {}).get("ksi_bounds", KSI_BOUNDS)
    rows.append(["Noise bounds", f"ksi [{ksi[0]:g}, {ksi[1]:g}]"])
    seeds = len(protocol.get("seeds", [])) or first.get("n_random_restarts") or len(first.get("seeds_used") or [])
    if seeds:
        tolerance = f"; max_iter {protocol['max_iter']}, tol {protocol['tol']:g}" if "max_iter" in protocol else ""
        warm = " + warm starts" if any(model["meta"].get("n_warm_starts") for model in models) else ""
        rows.append(["Starts", f"{seeds} random per model{warm}{tolerance}"])
    for index, directory in enumerate(protocol.get("warm_dirs", [])):
        rows.append(["Warm-start dirs" if index == 0 else "", display_path(directory)])
    version = protocol.get("valuation_version") or protocol.get("inputs", {}).get("valuation_version") \
        or first.get("valuation_version") or "unconditional (legacy)"
    rows.append(["Valuation", version])
    if protocol.get("fingerprint"):
        rows.append(["Fingerprint", protocol["fingerprint"][:16]])
    code = protocol.get("provenance")
    if code:
        rows.append(["Code", f"git {(code.get('git_commit') or '?')[:10]}"
                     + (" + uncommitted changes" if code.get("code_modified") else "")])
    elif protocol:
        rows.append(["Code", "not recorded (protocol written before the refactor)"])
    else:
        rows.append(["Protocol", "none saved (bounds are the defaults)"])
    tests = {}
    for model in models:
        tests[model["meta"]["test"]] = tests.get(model["meta"]["test"], 0) + 1
    stamps = sorted(str(model["meta"]["timestamp"]) for model in models if model["meta"].get("timestamp"))
    rows.append(["Models", f"{len(models)} done ({', '.join(f'{test} x{n}' for test, n in sorted(tests.items()))})"
                 + (f"; last saved {stamps[-1][:16].replace('T', ' ')}" if stamps else "")])
    return lines + table(rows, left=(0, 1))


def restart_hits(result):
    lls = [row["log_likelihood"] for row in result.get("restart_summaries", [])
           if row.get("log_likelihood") is not None and np.isfinite(row["log_likelihood"])]
    return (sum(ll >= max(lls) - HIT_TOL for ll in lls), len(lls)) if lls else None


def fit_section(models):
    n_subjects = models[0]["meta"]["n_subjects"]
    lowest = min(model["meta"]["bic"] for model in models)
    best = {}
    for model in models:
        test = model["meta"]["test"]
        if test not in best or model["meta"]["bic"] < best[test]:
            best[test] = model["meta"]["bic"]
    rows = [["Model", "k", "Log-lik.", "BIC", "", "dBIC", "med ksi", "Exit", "Iter", "Hits", "Min", "Note"]]
    for model in models:
        meta, result = model["meta"], model["result"]
        hits = restart_hits(result)
        rows.append([meta["config_id"], meta["k_params"] - meta["n_subjects"], big(meta["log_likelihood"]),
                     big(meta["bic"]), "*" if meta["bic"] == best[meta["test"]] else "",
                     big(meta["bic"] - lowest), num(np.median(result["ksi"])), meta.get("exit_reason") or
                     ("converged" if meta.get("converged") else "not converged"), meta.get("n_iter", "-"),
                     f"{hits[0]}/{hits[1]}" if hits else "-",
                     f"{meta['elapsed_seconds'] / 60:.1f}" if meta.get("elapsed_seconds") is not None else "-",
                     f"alias of {meta['equivalent_to']}" if meta.get("equivalent_to") else ""])
    return (["MODEL FIT"]
            + table(rows, left=(0, 4, 7, 11))
            + [f"  k: structural parameters + mixing weights; the {n_subjects:,} noise parameters ksi_i also enter "
               "the BIC.", "  *: lowest BIC of its test. dBIC: from the lowest BIC in this run. "
               f"Hits: restarts within {HIT_TOL} log-lik. of the best restart / restarts finished. Min: minutes."])


def comparison_section(models):
    lines = ["MODEL COMPARISON"]
    fitted = [model["meta"] for model in models]
    lowest = min(fitted, key=lambda meta: meta["bic"])
    lines.append(f"  Lowest BIC: {lowest['config_id']} ({big(lowest['bic'])})")
    for test in TESTS:
        metas = [meta for meta in fitted if meta["test"] == test]
        if metas:
            top = min(metas, key=lambda meta: meta["bic"])
            lines.append(f"  Best {test}: {top['config_id']} (dBIC {big(top['bic'] - lowest['bic'])})")

    test1 = [meta for meta in fitted if meta["test"] == "test1"]
    if test1:
        classes = sorted({int(meta["C"]) for meta in test1})
        rows = [["Specification", *[f"C={c}" for c in classes], "best C"]]
        for key in sorted({spec_key(meta) for meta in test1}):
            bics = {int(meta["C"]): meta["bic"] for meta in test1 if spec_key(meta) == key}
            rows.append([key, *[big(bics.get(c), 1) if c in bics else "-" for c in classes], min(bics, key=bics.get)])
        lines += ["", "  Test 1: BIC by number of classes"] + table(rows, indent="    ")

    test2 = [meta for meta in fitted if meta["test"] == "test2"]
    if test2:
        rows = [["Specification", *RP_LABELS_4, "ranking (best first)"]]
        for key in sorted({spec_key(meta) for meta in test2}):
            bics = {meta["rp_identity"]: meta["bic"] for meta in test2 if spec_key(meta) == key}
            rows.append([key, *[big(bics[rp], 1) if rp in bics else "-" for rp in RP_LABELS_4],
                         " < ".join(sorted(bics, key=bics.get))])
        lines += ["", "  Test 2: BIC by pure reference point"] + table(rows, left=(0, 5), indent="    ")
    return lines


def describe(meta):
    if meta["rp_mode"] == "mixed":
        what = f"Test 1: {meta['C']} latent class{'es' if meta['C'] != 1 else ''}, reference-point weights free in each"
    elif meta["rp_mode"] == "fixed_pure":
        what = f"Test 2: one class with the pure {meta['rp_identity']} reference point"
    else:
        what = "Test 3: four classes, each pinned to one pure reference point"
    return (f"{what}; {meta['discounting']} discounting, {meta['curvature']}, {meta['memory']}, "
            f"{meta['pw']} weighting, {utility_of(meta)} utility")


def loss_aversion(alpha_gain, alpha_loss, lamb, stake, utility="power"):
    """A(x) = -u(-x)/u(x): value of losing £x relative to winning £x
    (lambda * x**(alpha_loss - alpha_gain) for power utility)."""
    options = dict(alpha=float(alpha_gain), alpha_loss=float(alpha_loss), lamb=float(lamb), utility=utility)
    gain = cpt.u(stake, **options)
    return -cpt.u(-stake, **options) / gain if gain else float("inf")


def parameter_rows(meta, result, bounds):
    """Class-by-class parameter table; † at an estimation bound, -- not used."""
    method, utility = result["spec"].pw, utility_of(meta)
    index = CANON[method]
    resp = np.asarray(result["resp"], dtype=float)
    modal = np.bincount(resp.argmax(axis=1), minlength=resp.shape[1])
    mixed = meta["rp_mode"] == "mixed"
    weighting = "GAM" if method == "tk" else "PALPHA"
    names = (["DELTA"] if meta["memory"] == "commem" else ["DELTA", "DELTA_LE"]) + ["R"] \
        + (["PSI"] if meta["discounting"] == "qh" else []) \
        + (["AG"] if meta["curvature"] == "commonalpha" else ["AG", "AL"]) + ["LAM", weighting]
    label = dict(LABELS, AG="alpha" if meta["curvature"] == "commonalpha" else "alpha+",
                 DELTA="delta" if meta["memory"] == "commem" else "delta_PA", DELTA_LE="delta_LE")

    columns = [[] for _ in names]
    stakes = [[] for _ in STAKES]
    premiums = [[] for _ in RISK_STAKES]
    weights, monotone = [], []
    for j, (theta, delta_le, layout) in enumerate(zip(result["thetas"], result["delta_les"], result["layouts"])):
        a1, a2, a3 = s_to_a(*(float(theta[index[key]]) for key in ("S1", "S2", "S3")))
        weights.append([a1, a2, a3, 1 - a1 - a2 - a3])
        for row, name in enumerate(names):
            if name == "DELTA_LE":
                value, free, used = (theta[index["DELTA"]] if delta_le is None else delta_le,
                                     layout.has_delta_le, a3 > 1e-12)
                bound = bounds[index["DELTA"]]
            else:
                value, free, bound = theta[index[name]], index[name] in layout.free_positions, bounds[index[name]]
                used = {"DELTA": (a2 if meta["memory"] == "sepmem" else a2 + a3) > 1e-12}.get(name, True)
            at_bound = free and min(abs(value - bound[0]), abs(value - bound[1])) <= BOUNDARY_TOL
            columns[row].append(num(value) + ("†" if at_bound else "") if used else "--")
        for row, stake in enumerate(STAKES):
            stakes[row].append(num(loss_aversion(theta[index["AG"]], theta[index["AL"]], theta[index["LAM"]],
                                                 stake, utility), 2))
        for row, stake in enumerate(RISK_STAKES):
            premiums[row].append(cpt.risk_premium(stake, float(theta[index["AG"]]), float(theta[index["AL"]]), utility))
        monotone.append(probability_shape(method, float(theta[index[weighting]]))[0])

    header = [f"Class {j + 1}" if mixed else identity for j, identity in enumerate(result["rp_identities"])]
    rows = [["", *header], ["Share pi", *[f"{100 * pi:.1f}%" for pi in result["pis"]]],
            ["Modal n", *map(str, modal)]]
    if mixed:
        rows += [[f"a_{rp}", *[num(weight[i]) for weight in weights]] for i, rp in enumerate(RP_LABELS_4)]
    rows += [[label[name], *column] for name, column in zip(names, columns)]
    rows += [[f"A({stake})", *column] for stake, column in zip(STAKES, stakes)]
    for side, sign in enumerate("+-"):
        rows += [[f"Premium{sign}({stake})", *[f"{100 * premium[side]:+.1f}%" for premium in column]]
                 for stake, column in zip(RISK_STAKES, premiums)]
    if method == "tk":
        rows.append(["TK monotone", *["yes" if flag else "NO" for flag in monotone]])
    population = np.asarray(result["pis"]) @ np.asarray(weights) if mixed and len(weights) > 1 else None
    return rows, population


def model_section(model, models_by_id, bounds_of):
    meta, result = model["meta"], model["result"]
    lines = [RULE, meta["config_id"], "  " + describe(meta)]
    if meta.get("equivalent_to"):
        source = meta["equivalent_to"]
        state = "listed above" if source in models_by_id else "not in this run"
        return lines + [f"  Same estimates as {source} ({state}): a pure-reference sepmem model is saved "
                        "as an alias of its commem fit.", ""]
    n_subjects = meta["n_subjects"]
    lines.append(f"  Log-lik. {big(meta['log_likelihood'], 3)}, BIC {big(meta['bic'], 3)}, "
                 f"k {meta['k_params'] - n_subjects} + {n_subjects:,} ksi, n {meta['n_obs']:,}; "
                 f"{meta.get('exit_reason') or ('converged' if meta.get('converged') else 'not converged')} "
                 f"after {meta.get('n_iter', '?')} EM iterations")
    starts = [f"{meta['n_random_restarts']} random + {meta['n_warm_starts']} warm"
              if meta.get("n_random_restarts") is not None else None,
              f"{meta['n_restarts_ok']} finished" if meta.get("n_restarts_ok") is not None else None,
              f"{meta['n_converged']} converged" if meta.get("n_converged") is not None else None,
              f"best from {result['restart_id']}" if result.get("restart_id") else None]
    hits = restart_hits(result)
    if hits:
        starts.append(f"{hits[0]} within {HIT_TOL} log-lik. of the best")
    if any(starts):
        lines.append("  Starts: " + ", ".join(item for item in starts if item))
    ksi = np.asarray(result["ksi"], dtype=float)
    low, high = bounds_of["ksi"]
    q05, q50, q95 = np.quantile(ksi, [0.05, 0.5, 0.95])
    lines.append(f"  Noise ksi_i: median {num(q50)}, 5-95% {num(q05)}-{num(q95)}, "
                 f"at lower/upper bound {int(np.sum(np.abs(ksi - low) <= BOUNDARY_TOL))}"
                 f"/{int(np.sum(np.abs(ksi - high) <= BOUNDARY_TOL))}")
    resp = np.asarray(result["resp"], dtype=float)
    if resp.shape[1] > 1:
        top = resp.max(axis=1)
        lines.append(f"  Posterior: mean max {top.mean():.3f}; max > 0.5 for {100 * np.mean(top > 0.5):.0f}%, "
                     f"> 0.8 for {100 * np.mean(top > 0.8):.0f}% of participants")
    if "optimizer_unsuccessful_calls" in meta:
        lines.append(f"  Optimizer: unsuccessful calls {meta['optimizer_unsuccessful_calls']}, "
                     f"rejected steps {meta['optimizer_rejected_steps']}; final LL change {meta['final_ll_change']:g}")
    rows, population = parameter_rows(meta, result, bounds_of[meta["pw"]])
    lines += [""] + table(rows, indent="    ")
    if population is not None:
        lines.append("    Population RP weights (sum of pi * a): "
                     + ", ".join(f"{rp} {weight:.3f}" for rp, weight in zip(RP_LABELS_4, population)))
    notes = ["† at an estimation bound", "--: not used by this class's reference point",
             "A(x) = -u(-x)/u(x)",
             "Premium+/-(x): risk premium of a 50/50 gamble between 0 and +/-£x under the utility alone "
             "(no probability weighting), % of its expected value; > 0 risk averse, < 0 risk seeking"]
    if meta["discounting"] == "exp":
        notes.append("psi = 1")
    return lines + ["    " + "; ".join(notes) + ".", ""]


# ── Entry points ──────────────────────────────────────────────────────────

def load_models(run):
    """Finished models of a run (meta + result), in test/spec/C/RP order, and
    the config IDs of folders without a finished model."""
    models, unfinished = [], []
    for path in sorted(Path(run).glob("test*/meta.json")):
        meta = json.loads(path.read_text())
        if meta.get("status") == "done" and (path.parent / "result.pkl").exists():
            models.append(dict(meta=meta, result=load_pickle(path.parent / "result.pkl")))
        else:
            unfinished.append(f"{meta.get('config_id', path.parent.name)} ({meta.get('status', 'no result')})")
    return sorted(models, key=lambda model: model_order(model["meta"])), unfinished


def run_summary(run):
    """Text of <run>/summary.txt."""
    run = Path(run)
    models, unfinished = load_models(run)
    if not models:
        raise ValueError(f"No finished models in {run}")
    protocol = read_json(run / "run_protocol.json")
    about = (run / "about.txt").read_text().strip() if (run / "about.txt").exists() else ""
    stamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d %H:%M")
    lines = [f"ESTIMATION RUN  {display_path(run)}"]
    if about:
        lines.append(f"About: {about}")
    lines += [f"Written {stamp} from the saved files (estimation/run_summary.py); nothing is re-estimated.", ""]
    lines += protocol_section(run, protocol, models) + [""]
    if unfinished:
        lines += ["NOT FINISHED", *[f"  {item}" for item in unfinished], ""]
    lines += fit_section(models) + [""] + comparison_section(models) + ["", "CLASS PARAMETERS BY MODEL"]
    utility = utility_of(models[0]["meta"])
    bounds_of = {pw: run_bounds(protocol, pw, utility) for pw in CANON}
    bounds_of["ksi"] = protocol.get("inputs", {}).get("ksi_bounds", KSI_BOUNDS)
    models_by_id = {model["meta"]["config_id"] for model in models}
    for model in models:
        lines += model_section(model, models_by_id, bounds_of)
    return "\n".join(lines).rstrip() + "\n"


def write_run_summary(run, path=None):
    """Write the summary of ``run`` (default <run>/summary.txt); return the path."""
    path = Path(run) / "summary.txt" if path is None else Path(path)
    atomic_write_text(path, run_summary(run))
    return path
