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
    if len(result["thetas"]) != spec.C or result["spec"].pw != spec.pw:
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


def grid_initializations(spec, subjects, output, fingerprint, legacy_dir):
    """Warm starts for a full-grid job: the original (unconditional) fit of the
    same configuration, and the two closest already fitted configurations of
    this run (a restricted model is a feasible start for a broader one, which
    prevents avoidable nested-model LL reversals)."""
    cid = spec.config_id()
    initializations = {}
    legacy_ids = [cid]
    if spec.rp_mode != "mixed":
        legacy_ids.append(cid.replace("_commem_", "_sepmem_"))
    for old_id in legacy_ids:
        path = legacy_dir / old_id / "result.pkl"
        if path.exists():
            old = load_pickle(path)
            if not set(subjects).issubset(old["subjects"]):
                print(f"  SKIP legacy warm start {old_id}: target participants missing", flush=True)
                continue
            initializations["legacy_" + old_id] = initial_values(old, spec, subjects=subjects)
    candidates = []
    for path in output.glob("test*/meta.json"):
        meta = json.loads(path.read_text())
        if meta.get("equivalent_to") or meta["pw"] != spec.pw or meta["C"] != spec.C:
            continue
        same = meta["test"] == spec.test and meta.get("rp_identity") == spec.rp_identity
        fixed_to_mixed = spec.test == "test1" and spec.C == 4 and meta["test"] == "test3"
        if same or fixed_to_mixed:
            score = (meta["curvature"] != spec.curvature, meta["discounting"] != spec.discounting,
                     meta["memory"] != spec.memory, meta["bic"])
            candidates.append((score, meta["config_id"]))
    for _, source_id in sorted(candidates)[:2]:
        source = load_completed(output, source_id, fingerprint)
        initializations["corrected_" + source_id] = initial_values(source, spec, subjects=subjects)
    return initializations


def exclusion_initializations(spec, subjects, output, fingerprint, source_dir, warm_dirs, bounds=None):
    """Warm starts for a restricted-sample job: the full-sample fit of the same
    model, the same model in each warm directory, the nested Test 3 fit for
    Test 1 C=4, and every class of the C-1 fit split in two for Test 1 C>=2
    (so C cannot end below C-1)."""
    def find(cid):
        """Fit of cid from this run, else from the first warm directory that has it."""
        result = load_completed(output, cid, fingerprint)
        for warm_dir in warm_dirs:
            if result is None and (warm_dir / cid / "result.pkl").exists():
                result = load_pickle(warm_dir / cid / "result.pkl")
        return result

    cid = spec.config_id()
    initializations = {}
    if (source_dir / cid / "result.pkl").exists():
        initializations["full_sample_" + cid] = initial_values(
            load_pickle(source_dir / cid / "result.pkl"), spec, subjects=subjects, bounds=bounds)
    for warm_dir in warm_dirs:
        if (warm_dir / cid / "result.pkl").exists():
            initializations[f"warm_{warm_dir.name}"] = initial_values(
                load_pickle(warm_dir / cid / "result.pkl"), spec, subjects=subjects, bounds=bounds)
    if spec.test == "test1" and spec.C == 4:
        test3_id = dataclasses.replace(spec, test="test3", rp_mode="fixed_4").config_id()
        nested = find(test3_id)
        if nested is not None:
            initializations["nested_" + test3_id] = initial_values(nested, spec, subjects=subjects, bounds=bounds)
    if spec.test == "test1" and spec.C >= 2:
        smaller = find(dataclasses.replace(spec, C=spec.C - 1).config_id())
        if smaller is not None:
            initializations.update(split_starts(smaller, spec, subjects, bounds=bounds))
    return initializations
