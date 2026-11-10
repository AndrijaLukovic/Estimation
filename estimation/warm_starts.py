"""Warm starts: project saved fits onto a model's free parameters, and find
the saved fits that make good starting points for a job."""
import copy
import dataclasses
import json

import numpy as np

from .spec import CANON, bounds_free, expand_all, spec_cluster_layouts
from .store import load_completed, load_pickle


def initial_values(result, spec, subjects=None, bounds=None):
    """Project saved parameters and align individual noise by participant ID."""
    layouts, identities = spec_cluster_layouts(spec)
    if (len(result["thetas"]) != spec.C or result["spec"].pw != spec.pw
            or getattr(result["spec"], "utility", "power") != spec.utility):
        raise ValueError("Incompatible warm start")
    vectors = []
    ix = CANON[spec.pw]
    for theta, old_delta, layout, identity in zip(result["thetas"], result["delta_les"], layouts, identities):
        theta = np.asarray(theta).copy()
        effective_le = theta[ix["DELTA"]] if old_delta is None else old_delta
        if spec.memory == "commem" and identity == "LE":
            theta[ix["DELTA"]] = effective_le
        vector = [theta[position] for position in layout.free_positions]
        if layout.has_delta_le:
            vector.append(effective_le)
        vectors.append(np.array([np.clip(value, low, high)
                                for value, (low, high) in zip(vector, bounds_free(layout, bounds))]))
    ksi = np.asarray(result["ksi"]).copy()
    if subjects is not None:
        source_subjects = result["subjects"]
        source_index = {subject: i for i, subject in enumerate(source_subjects)}
        if ksi.shape != (len(source_subjects),) or len(source_index) != len(source_subjects):
            raise ValueError("Warm start must have one noise parameter per unique participant")
        missing = [subject for subject in subjects if subject not in source_index]
        if missing:
            raise ValueError(f"Warm start is missing {len(missing)} target participants")
        ksi = ksi[[source_index[subject] for subject in subjects]]
    return dict(free_vecs=vectors, pis=result["pis"].copy(), ksi=ksi)


def equivalent_result(result, spec):
    """Relabel a fit as an equivalent specification (pure-reference models whose
    memory labels make no difference), with that specification's layout."""
    clone = copy.deepcopy(result)
    values = initial_values(result, spec)
    clone["spec"] = spec
    clone["layouts"], clone["rp_identities"] = spec_cluster_layouts(spec)
    clone["free_vecs"] = values["free_vecs"]
    clone["thetas"], clone["delta_les"] = expand_all(clone["free_vecs"], clone["layouts"])
    clone["equivalent_to"] = result["spec"].config_id()
    return clone


def split_starts(result, spec, subjects, scale=.02, bounds=None):
    """Turn a C-1 class fit into C warm starts by splitting each class in two."""
    smaller = initial_values(result, dataclasses.replace(spec, C=spec.C - 1), subjects=subjects, bounds=bounds)
    layout = spec_cluster_layouts(spec)[0][0]
    low, high = np.array(bounds_free(layout, bounds)).T
    starts = {}
    for j, vector in enumerate(smaller["free_vecs"]):
        shift = scale * (high - low) * np.random.default_rng(j).choice([-1., 1.], size=len(vector))
        halves = [np.clip(vector + shift, low, high), np.clip(vector - shift, low, high)]
        pis = np.r_[np.delete(smaller["pis"], j), [smaller["pis"][j] / 2] * 2]
        starts[f"split_C{spec.C - 1}_class{j + 1}"] = dict(
            free_vecs=[v for i, v in enumerate(smaller["free_vecs"]) if i != j] + halves,
            pis=pis, ksi=smaller["ksi"].copy())
    return starts


def job_initializations(spec, subjects, output, fingerprint, warm_dirs=(), legacy_dir=None, bounds=None):
    """Warm starts for one job (label -> initial values):

    - legacy_<id>: the original unconditional fit of the same configuration;
    - warm_<dir>: the same configuration in each warm directory (e.g. a full-sample run);
    - corrected_<id>: the two closest configurations already fitted in this run
      (a restricted model is a feasible start for a broader one, which prevents
      avoidable nested-model LL reversals);
    - Test 1: the nested Test 3 fit for C=4, and every class of the C-1 fit split
      in two for C>=2 (so C cannot end below C-1), from this run or a warm directory.

    Sources that lack target participants or do not fit the layout are skipped."""
    cid = spec.config_id()
    starts = {}

    def add(label, build):
        try:
            starts.update(build())
        except ValueError as error:
            print(f"  SKIP warm start {label}: {error}", flush=True)

    def project(label, source):
        add(label, lambda: {label: initial_values(source, spec, subjects=subjects, bounds=bounds)})

    def find(config_id):
        """Fit of config_id from this run, else from the first warm directory that has it."""
        result = load_completed(output, config_id, fingerprint)
        for warm_dir in warm_dirs:
            if result is None and (warm_dir / config_id / "result.pkl").exists():
                result = load_pickle(warm_dir / config_id / "result.pkl")
        return result

    if legacy_dir is not None:
        legacy_ids = [cid] + ([cid.replace("_commem_", "_sepmem_")] if spec.rp_mode != "mixed" else [])
        for old_id in legacy_ids:
            if (legacy_dir / old_id / "result.pkl").exists():
                project("legacy_" + old_id, load_pickle(legacy_dir / old_id / "result.pkl"))
    for warm_dir in warm_dirs:
        if (warm_dir / cid / "result.pkl").exists():
            project(f"warm_{warm_dir.name}", load_pickle(warm_dir / cid / "result.pkl"))
    candidates = []
    for path in output.glob("test*/meta.json"):
        meta = json.loads(path.read_text())
        if (meta.get("equivalent_to") or meta["config_id"] == cid or meta["pw"] != spec.pw or meta["C"] != spec.C
                or meta["test"] != spec.test or meta.get("rp_identity") != spec.rp_identity
                or meta.get("utility", "power") != spec.utility):
            continue
        score = (meta["curvature"] != spec.curvature, meta["discounting"] != spec.discounting,
                 meta["memory"] != spec.memory, meta["bic"])
        candidates.append((score, meta["config_id"]))
    for _, source_id in sorted(candidates)[:2]:
        source = load_completed(output, source_id, fingerprint)
        if source is not None:
            project("corrected_" + source_id, source)
    if spec.test == "test1" and spec.C == 4:
        test3_id = dataclasses.replace(spec, test="test3", rp_mode="fixed_4").config_id()
        nested = find(test3_id)
        if nested is not None:
            project("nested_" + test3_id, nested)
    if spec.test == "test1" and spec.C >= 2:
        smaller = find(dataclasses.replace(spec, C=spec.C - 1).config_id())
        if smaller is not None:
            add(f"split_C{spec.C - 1}", lambda: split_starts(smaller, spec, subjects, bounds=bounds))
    return starts
