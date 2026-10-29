"""Component-specific profile likelihood for saved conditional Grid EM models.

Examples (use --list to inspect jobs without estimating or writing output):
    python profile_lambda.py --parameters boundary --list
    python profile_lambda.py --parameters palpha --cluster PA
    python profile_lambda.py --parameters alpha_gain --cluster FE
    python profile_lambda.py --parameters lambda alpha_loss r --cluster all
    python profile_lambda.py --result results_grid_conditional_stricter/test1_exp_sepalpha_commem_prelec_C4/result.pkl --parameters delta_pa --cluster 4
    python profile_lambda.py --parameters palpha --cluster PA --grid 1e-8 1e-6 1e-4 .001 .01 .1

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
and per-point pickle checkpoints go outside the input directory. No HTML,
automatic chi-square confidence intervals or edits to GlobalSettings/grid_em/
model_spec are made. Constraints are installed only inside isolated profile
workers and restored after each EM call. Identical commands resume checkpoints;
changed grids/seeds/code/source results require a different --output-dir.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import csv
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.special import logsumexp

import functions as f
import grid_em as engine
from GlobalSettings import GlobalLottery, GlobalSeedsSet, GlobalKsiBounds
from main import get_observed_ce
from model_spec import CANON, bounds_free, param_names_free

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT = ROOT / "results_grid_conditional_stricter/test3_exp_sepalpha_commem_prelec/result.pkl"
LAMBDA_GRID = [.25, .5, .75, 1., 1.5, 2.25, 3., 5., 7., 10., 15., 20., 30., 50., 100.]
GRIDS = {
    "lambda": LAMBDA_GRID,
    "alpha_gain": [1e-6, .1, .25, .5, .75, 1., 1.5, 2., 3., 4., 5., 6., 8., 10.],
    "alpha_loss": [1e-6, 1e-5, 1e-4, .001, .01, .05, .1, .25, .5, .75, 1., 1.5, 2., 3., 4., 5., 6., 8.],
    "palpha": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, .001, .01, .03, .05, .1, .2, .4, .6, .8, 1.],
    "gamma": [.2, .25, .28, .3, .4, .5, .6, .7, .8, 1.],
    "r": [0., 1e-6, 1e-5, 1e-4, .001, .003, .01, .03, .1, .3, 1.],
    "psi": [1e-6, .01, .05, .1, .2, .4, .6, .8, .9, .95, 1.],
    "delta_pa": [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
    "delta_le": [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
    "s1": [0., .1, .25, .5, .75, .9, 1.],
    "s2": [0., .1, .25, .5, .75, .9, 1.],
    "s3": [0., .1, .25, .5, .75, .9, 1.],
}
POSITIONS = dict(r="R", alpha_gain="AG", alpha_loss="AL", **{"lambda": "LAM"},
                 palpha="PALPHA", gamma="GAM", psi="PSI", delta_pa="DELTA",
                 delta_le="DELTA", s1="S1", s2="S2", s3="S3")
ALIASES = {"prelec_alpha": "palpha", "delta": "delta_pa", "lam": "lambda"}
_PREPARED = None


def atomic_pickle(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.pkl")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_result(path):
    with Path(path).open("rb") as handle:
        result = pickle.load(handle)
    if result.get("valuation_version") != "conditional_remaining_v1" or "spec" not in result:
        raise ValueError("Use a saved conditional Grid EM result.pkl, not a legacy EM result")
    return result


@dataclass(frozen=True)
class Target:
    cluster: int
    name: str
    position: int       # canonical vector index, or -1 for separate delta_le
    source_value: float
    original_bound: tuple
    reference_bound: tuple


def resolve_target(result, cluster, name, custom_grid=None):
    """Resolve free, linked and separate-tail parameters without changing the spec."""
    name = ALIASES.get(name, name)
    if name not in GRIDS:
        raise ValueError(f"Unknown parameter: {name}")
    layout = result["layouts"][cluster]
    indices = CANON[layout.method]
    if name == "alpha_loss" and layout.link_AL_to_AG:
        raise ValueError("alpha_loss is linked; profile alpha_gain to vary both exponents")
    if name == "delta_le" and layout.is_sepmem:
        if not layout.has_delta_le:
            raise ValueError("delta_le is fixed/unused in this component")
        position, free_index = -1, len(layout.free_positions)
        source_value = float(result["delta_les"][cluster])
    else:
        if POSITIONS[name] not in indices:
            raise ValueError(f"{name} is unavailable under {layout.method}")
        position = indices[POSITIONS[name]]
        if position not in layout.free_positions:
            raise ValueError(f"{name} is fixed/unused in this specification/component")
        free_index = layout.free_positions.index(position)
        source_value = float(result["thetas"][cluster][position])
        if name == "delta_le":
            name = "delta_pa"   # common memory: avoid duplicate profiles
    original_bound = tuple(map(float, bounds_free(layout)[free_index]))
    grid = sorted(set(map(float, custom_grid if custom_grid is not None else GRIDS[name])) | {source_value})
    if not grid or not np.isfinite(grid).all():
        raise ValueError("Profile grid must contain finite numbers")
    for value in grid:
        positive = name in {"lambda", "alpha_gain", "alpha_loss", "palpha", "gamma", "psi"}
        if (positive and value <= 0) or (not positive and value < 0):
            raise ValueError(f"Invalid {name} grid value: {value}")
        if name in {"palpha", "gamma", "psi", "delta_pa", "delta_le", "s1", "s2", "s3"} and value > 1:
            raise ValueError(f"{name} grid must not exceed 1")
    reference_bound = (min(original_bound[0], grid[0]), max(original_bound[1], grid[-1]))
    return Target(cluster, name, position, source_value, original_bound, reference_bound), grid


def profile_layouts(spec, target, value):
    layouts, identities = engine.spec_cluster_layouts(spec)
    layouts = copy.deepcopy(layouts)
    if value is not None:
        focal = layouts[target.cluster]
        if target.position == -1:
            focal.has_delta_le = False
            focal.delta_le_fixed = float(value)
        else:
            focal.free_positions.remove(target.position)
            focal.fixed_values[target.position] = float(value)
    return layouts, identities


def layout_bounds(layout, focal, target):
    bounds = list(bounds_free(layout))
    if layout is focal:
        if target.position == -1 and layout.has_delta_le:
            bounds[-1] = target.reference_bound
        elif target.position in layout.free_positions:
            bounds[layout.free_positions.index(target.position)] = target.reference_bound
    return bounds


@contextmanager
def constrained_engine(spec, layouts, identities, target):
    """Worker-local adaptation; leave shared source files/settings untouched."""
    original_layouts, original_bounds = engine.spec_cluster_layouts, engine.bounds_free
    def get_layouts(requested):
        if requested != spec:
            raise ValueError("Unexpected specification in profile worker")
        return layouts, identities
    engine.spec_cluster_layouts = get_layouts
    engine.bounds_free = lambda layout: layout_bounds(layout, layouts[target.cluster], target)
    try:
        yield
    finally:
        engine.spec_cluster_layouts, engine.bounds_free = original_layouts, original_bounds


def pack_initial(result, layouts, target):
    vectors = []
    for theta, delta, layout in zip(result["thetas"], result["delta_les"], layouts):
        vector = [theta[position] for position in layout.free_positions]
        if layout.has_delta_le:
            vector.append(theta[CANON[layout.method]["DELTA"]] if delta is None else delta)
        bounds = layout_bounds(layout, layouts[target.cluster], target)
        vectors.append(np.clip(np.asarray(vector, dtype=float),
                               [b[0] for b in bounds], [b[1] for b in bounds]))
    return dict(free_vecs=vectors, pis=np.asarray(result["pis"]).copy(), ksi=np.asarray(result["ksi"]).copy())


def likelihood(result, data):
    mu = np.column_stack([data.predict(theta, result["spec"].pw, delta)
                          for theta, delta in zip(result["thetas"], result["delta_les"])])
    row_mu = mu[data.state_index]
    if not np.isfinite(row_mu).all():
        raise FloatingPointError("Nonfinite profile predictions")
    sigma = np.asarray(result["ksi"])[data.subject_index] * data.spread
    pdf = -.5 * ((data.observed[:, None] - row_mu) / sigma[:, None]) ** 2 - np.log(sigma[:, None]) - .5 * np.log(2 * np.pi)
    log_l = np.column_stack([np.bincount(data.subject_index, weights=pdf[:, j], minlength=len(data.subjects))
                             for j in range(len(result["pis"]))])
    joint = log_l + np.log(np.maximum(result["pis"], 1e-300))[None, :]
    sums = logsumexp(joint, axis=1)
    return float(sums.sum()), np.exp(joint - sums[:, None])


def feasible_initial(spec, values, layouts, identities, data):
    """Retain a feasible likelihood floor if an EM run stalls or worsens."""
    thetas, deltas = engine._expand_all(values["free_vecs"], layouts)
    result = dict(spec=spec, thetas=thetas, delta_les=deltas, layouts=layouts,
        rp_identities=identities, free_vecs=values["free_vecs"], pis=np.maximum(values["pis"], 1e-300),
        ksi=np.clip(values["ksi"], *GlobalKsiBounds), subjects=data.subjects,
        n_obs_effective=len(data.observed), n_iter=0, converged=False,
        exit_reason="feasible_initial", optimizer_log=[], iter_log=[],
        valuation_version="conditional_remaining_v1", final_ll_change=0.)
    result["pis"] = result["pis"] / result["pis"].sum()
    result["log_likelihood"], result["resp"] = likelihood(result, data)
    return result


def init_worker(prepared):
    global _PREPARED
    _PREPARED = prepared


def fit_worker(spec, target, value, seed, initial, max_iter, tol):
    layouts, identities = profile_layouts(spec, target, value)
    packed = pack_initial(initial, layouts, target) if initial is not None else None
    floor = feasible_initial(spec, packed, layouts, identities, _PREPARED) if packed is not None else None
    with constrained_engine(spec, layouts, identities, target):
        try:
            fitted = engine.grid_em_mixture(spec, max_iter=max_iter, tol=tol, seed=seed,
                prepared=_PREPARED, initial_values=packed, verbose=False)
        except (FloatingPointError, ValueError):
            if floor is None:
                raise
            fitted = floor
    if floor is not None and floor["log_likelihood"] > fitted["log_likelihood"]:
        fitted = floor
    if value is not None:
        actual = fitted["delta_les"][target.cluster] if target.position == -1 else fitted["thetas"][target.cluster][target.position]
        if actual != value:
            raise AssertionError("Profile parameter was not held fixed")
    return fitted


def align_start(anchor, warm):
    """Align mixed warm starts only; never relabel a constrained fitted point."""
    if anchor["spec"].rp_mode != "mixed":
        return warm
    rows, columns = linear_sum_assignment(-(np.asarray(anchor["resp"]).T @ np.asarray(warm["resp"])))
    order = columns[np.argsort(rows)]
    return dict(thetas=[warm["thetas"][i] for i in order], delta_les=[warm["delta_les"][i] for i in order],
                pis=np.asarray(warm["pis"])[order], ksi=np.asarray(warm["ksi"]).copy())


def load_starts(path, anchor, maximum, window):
    starts, hashes = [("source", anchor)], {}
    summaries = sorted(anchor.get("restart_summaries", []), key=lambda row: -row["log_likelihood"])
    for row in summaries:
        if len(starts) >= maximum + 1:
            break
        if anchor["log_likelihood"] - row["log_likelihood"] > window:
            continue
        if row.get("restart_id") == anchor.get("restart_id"):
            continue
        restart_id = row.get("restart_id")
        if not restart_id:
            continue
        candidate = path.parent / "restarts" / (restart_id + ".pkl")
        if not candidate.exists():
            continue
        warm = read_result(candidate)
        if warm["spec"] != anchor["spec"] or warm["subjects"] != anchor["subjects"]:
            raise ValueError(f"Incompatible saved warm start: {candidate}")
        hashes[str(candidate.resolve())] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        starts.append((restart_id, align_start(anchor, warm)))
    return starts, hashes


def fit_candidates(pool, source, target, value, starts, seeds, args, existing=None):
    tasks = [(label, None, initial) for label, initial in starts]
    tasks.extend((f"seed_{seed}", seed, None) for seed in seeds)
    best, records = existing, []
    futures = {pool.submit(fit_worker, source["spec"], target, value, seed, initial,
                           args.max_iter, args.tol): label for label, seed, initial in tasks}
    for future in as_completed(futures):
        label = futures[future]
        try:
            fitted = future.result()
            records.append(dict(start=label, ll=fitted["log_likelihood"], converged=bool(fitted["converged"]),
                                exit_reason=fitted["exit_reason"]))
            if best is None or fitted["log_likelihood"] > best["log_likelihood"] or (
                    fitted["log_likelihood"] == best["log_likelihood"] and fitted["converged"] and not best["converged"]):
                best = fitted
                best["profile_start"] = label
        except Exception as exc:
            records.append(dict(start=label, error=f"{type(exc).__name__}: {exc}"))
    if best is None:
        raise RuntimeError(f"All profile starts failed at {value}: {records}")
    return best, records


def probability_valid(result):
    method = result["spec"].pw
    for theta in result["thetas"]:
        value = float(theta[CANON[method]["PALPHA" if method == "prelec" else "GAM"]])
        if method == "prelec":
            if value <= 0:
                return False
        elif value != 1:
            def numerator(p):
                return (value - 1) * p ** value + (1 - p) ** (value - 1) * (value + (1 - value) * p)
            if min(value, minimize_scalar(numerator, bounds=(1e-14, 1 - 1e-14), method="bounded").fun) < -1e-12:
                return False
    return True


def point_diagnostics(source, result, target):
    mapping = list(range(len(source["pis"])))
    if source["spec"].rp_mode == "mixed":
        rows, cols = linear_sum_assignment(-(np.asarray(source["resp"]).T @ np.asarray(result["resp"])))
        mapping = cols[np.argsort(rows)].tolist()
    share = float(result["pis"][target.cluster])
    focal = result["thetas"][target.cluster]
    indices = CANON[result["spec"].pw]
    parameters = {name: float(focal[indices[position]]) for name, position in POSITIONS.items()
                  if position in indices and name != "delta_le"}
    delta_le = result["delta_les"][target.cluster]
    parameters["delta_le"] = float(focal[indices["DELTA"]] if delta_le is None else delta_le)
    boundaries = []
    for j, (vector, layout) in enumerate(zip(result["free_vecs"], result["layouts"])):
        for name, value, bound in zip(param_names_free(layout), vector, bounds_free(layout)):
            if min(abs(float(value) - endpoint) for endpoint in bound) <= 1e-4:
                boundaries.append(f"cluster_{j + 1}:{name}")
    return dict(focal_share=share, low_focal_share=share < .01,
        component_shares=list(map(float, result["pis"])), focal_parameters=parameters,
        nuisance_boundaries=boundaries,
        component_mapping=mapping, component_switched=mapping[target.cluster] != target.cluster,
        probability_valid=probability_valid(result),
        optimizer_unsuccessful=sum(not entry["success"] for entry in result["optimizer_log"]),
        optimizer_rejected=sum(not entry["accepted"] for entry in result["optimizer_log"]))


def write_tables(directory, source, target, grid, points, reference):
    reference_ll = float(reference["log_likelihood"])
    best_attained = max([reference_ll] + [point["result"]["log_likelihood"] for point in points.values()])
    dominates = reference_ll >= best_attained - 1e-6
    rows = []
    for value in grid:
        point, result = points[value], points[value]["result"]
        diagnostic = point_diagnostics(source, result, target)
        ll = float(result["log_likelihood"])
        rows.append(dict(parameter=target.name, cluster=target.cluster + 1, value=value,
            source_value=target.source_value, inside_original_bounds=target.original_bound[0] <= value <= target.original_bound[1],
            log_likelihood=ll, delta_ll=best_attained - ll, reference_ll=reference_ll,
            lr_statistic=max(0., 2 * (reference_ll - ll)) if dominates else None,
            reference_dominates_grid=dominates, converged=bool(result["converged"]),
            exit_reason=result["exit_reason"], best_start=result.get("profile_start"),
            forward_ll=point["forward_ll"], reverse_gain=ll - point["forward_ll"],
            n_starts=len(point["attempts"]), n_converged=sum(r.get("converged", False) for r in point["attempts"]),
            n_failed=sum("error" in r for r in point["attempts"]), **diagnostic))
    temporary = directory / "profile.tmp.csv"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)
    temporary.replace(directory / "profile.csv")
    summary = dict(target=asdict(target), reference_ll=reference_ll,
        source_ll=float(source["log_likelihood"]), reference_gain_over_source=reference_ll - float(source["log_likelihood"]),
        reference_converged=bool(reference["converged"]), reference_dominates_grid=dominates,
        best_grid_value=max(grid, key=lambda value: points[value]["result"]["log_likelihood"]),
        incomplete_points=sum(not row["converged"] for row in rows),
        component_switch_points=sum(row["component_switched"] for row in rows),
        low_share_points=sum(row["low_focal_share"] for row in rows),
        notes=["delta_ll is relative to the best attained fit in the expanded target domain",
               "No automatic confidence interval: boundary/mixture regularity has not been established",
               "Convergence and profile shape do not prove a global optimum",
               "Original nuisance bounds apply except for the profiled parameter in its target component"], points=rows)
    atomic_json(directory / "profile.json", summary)
    return rows


def run_one(pool, source, target, grid, starts, seeds, args, output):
    directory = output / f"cluster_{target.cluster + 1}_{target.name}"
    directory.mkdir(parents=True, exist_ok=True)
    reference_path = directory / "reference.pkl"
    if reference_path.exists():
        reference = read_result(reference_path)
    else:
        reference, _ = fit_candidates(pool, source, target, None, starts, seeds, args)
        atomic_pickle(reference_path, reference)
    points, previous = {}, None
    for index, value in enumerate(grid):
        path = directory / f"point_{index:03d}.pkl"
        if path.exists():
            point = pickle.loads(path.read_bytes())
            if point["value"] != value:
                raise ValueError(f"Checkpoint grid mismatch: {path}")
        else:
            candidates = starts + [("reference", reference)]
            if previous is not None:
                candidates = candidates + [("ascending_neighbor", previous)]
            result, attempts = fit_candidates(pool, source, target, value, candidates, seeds, args)
            point = dict(value=value, result=result, attempts=attempts,
                         forward_ll=result["log_likelihood"], reverse_done=False)
            atomic_pickle(path, point)
        points[value], previous = point, point["result"]
        print(f"  {target.name}[{target.cluster + 1}]={value:.8g}: LL={previous['log_likelihood']:.8f} (ascending)", flush=True)
    previous = None
    for index in reversed(range(len(grid))):
        value, point = grid[index], points[grid[index]]
        if not point["reverse_done"]:
            if previous is not None:
                result, attempts = fit_candidates(pool, source, target, value,
                    [("descending_neighbor", previous), ("point_refinement", point["result"])], [], args,
                    existing=point["result"])
                point["result"] = result
                point["attempts"].extend(attempts)
            point["reverse_done"] = True
            atomic_pickle(directory / f"point_{index:03d}.pkl", point)
        previous = point["result"]
    final_reference = directory / "reference_final.pkl"
    if final_reference.exists():
        reference = read_result(final_reference)
    else:
        best_point = max(points.values(), key=lambda point: point["result"]["log_likelihood"])["result"]
        reference, _ = fit_candidates(pool, source, target, None,
            [("reference_refinement", reference), ("best_profile_point", best_point)], [], args, existing=reference)
        atomic_pickle(final_reference, reference)
    rows = write_tables(directory, source, target, grid, points, reference)
    print(f"  Saved {directory / 'profile.csv'}; reference LL={reference['log_likelihood']:.8f}", flush=True)
    return rows


def choose_jobs(source, args):
    clusters = set()
    for label in args.cluster:
        if label.lower() == "all":
            clusters.update(range(source["spec"].C))
        elif label.upper() in source["rp_identities"]:
            clusters.add(source["rp_identities"].index(label.upper()))
        else:
            try:
                number = int(label)
            except ValueError as exc:
                raise ValueError(f"Unknown cluster {label}; use 1-based numbers or fixed RP names") from exc
            if number < 1 or number > source["spec"].C:
                raise ValueError(f"Cluster must be between 1 and {source['spec'].C}")
            clusters.add(number - 1)
    automatic = args.parameters in (["all"], ["boundary"])
    jobs, skipped, seen = [], [], set()
    for cluster in sorted(clusters):
        for name in GRIDS if automatic else args.parameters:
            try:
                target, grid = resolve_target(source, cluster, name, args.grid)
            except ValueError as exc:
                if not automatic:
                    skipped.append(f"cluster {cluster + 1}, {name}: {exc}")
                continue
            if args.parameters == ["boundary"] and min(abs(target.source_value - b) for b in target.original_bound) > 1e-4:
                continue
            key = (cluster, target.position)
            if key not in seen:
                jobs.append((target, grid))
                seen.add(key)
    return jobs, skipped


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
    if not 0 <= args.seeds <= len(GlobalSeedsSet) or args.warm_starts < 0 or not np.isfinite(args.warm_ll_window) or args.warm_ll_window < 0:
        parser.error("Invalid restart count/window")
    if args.max_iter < 1 or args.n_workers < 1 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("max-iter, n-workers and tol must be positive")
    return args


def default_output(path, source, jobs):
    selection = "__".join(f"c{target.cluster + 1}_{target.name}" for target, _ in jobs)
    if len(selection) > 100:
        selection = "selection_" + hashlib.sha256(selection.encode()).hexdigest()[:12]
    return ROOT / "profiles" / path.parent.parent.name / source["spec"].config_id() / selection


def main(argv=None):
    args = parse_args(argv)
    path = args.result.resolve()
    source = read_result(path)
    data = engine.prepare_grid_data(get_observed_ce(export_excel=False), f.transform(GlobalLottery))
    if source["subjects"] != data.subjects or source.get("n_obs_effective", source.get("n_obs")) != len(data.observed):
        raise ValueError("Saved result and current filtered sample differ; choose a result fitted on the current sample")
    source_ll, _ = likelihood(source, data)
    if abs(source_ll - source["log_likelihood"]) > 1e-5:
        raise ValueError(f"Saved likelihood does not reproduce: gap={source_ll - source['log_likelihood']}")
    jobs, skipped = choose_jobs(source, args)
    for reason in skipped:
        print("SKIP " + reason, flush=True)
    if not jobs:
        raise ValueError("No eligible free parameters selected")
    starts, warm_hashes = load_starts(path, source, args.warm_starts, args.warm_ll_window)
    seeds = list(map(int, GlobalSeedsSet[:args.seeds]))
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
    source_files = ["profile_lambda.py", "grid_em.py", "model_spec.py", "functions.py", "Mixture.py", "GlobalSettings.py", "data.py", "main.py", "lotteries.py", "gamedata.csv"]
    protocol = dict(version="conditional_component_profile_v1", result=str(path),
        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), warm_sha256=warm_hashes,
        source_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
        spec=asdict(source["spec"]), jobs=[dict(target=asdict(target), grid=grid) for target, grid in jobs],
        seeds=seeds, max_iter=args.max_iter, tol=args.tol,
        n_subjects=len(data.subjects), n_obs=len(data.observed), warm_ll_window=args.warm_ll_window)
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["fingerprint"] = fingerprint
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Profile output has a different protocol; choose another --output-dir")
    atomic_json(manifest, protocol)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers, initializer=init_worker, initargs=(data,)) as pool:
        for target, grid in jobs:
            run_one(pool, source, target, grid, starts, seeds, args, output)
    print(f"Completed in {(time.time() - started) / 60:.1f} minutes. Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
