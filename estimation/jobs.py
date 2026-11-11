"""Model configurations to estimate.

The full grid has 144 jobs: Test 1 (mixed reference points, C = 1..4),
Test 2 (one pure reference point, C = 1), Test 3 (four fixed reference-point
clusters), each crossed with the 2^4 specification factorial, all with power
utility. select_jobs picks any subset (and any Test 1 class count) in
estimation order, under one utility family.
"""
import dataclasses
from itertools import product

from .spec import ModelSpec, RP_LABELS_4, UTILITY_TAGS, all_specs_16

TESTS = ("test1", "test2", "test3")
CHOICES = dict(discounting=("exp", "qh"), curvature=("commonalpha", "sepalpha"),
               memory=("commem", "sepmem"), pw=("prelec", "tk"))


def generate_jobs():
    """(config_id, ModelSpec) for all 144 jobs: Test 1 (64) -> Test 2 (64) -> Test 3 (16)."""
    jobs = []

    # Test 1: mixed simplex RP, C in {1,2,3,4}  -> 16 * 4 = 64
    for (disc, curv, mem, pw) in all_specs_16():
        for C in (1, 2, 3, 4):
            spec = ModelSpec(test="test1", discounting=disc, curvature=curv,
                             memory=mem, pw=pw, rp_mode="mixed", C=C)
            jobs.append((spec.config_id(), spec))

    # Test 2: pure RP, C=1, one of SQ/PA/LE/FE  -> 16 * 4 = 64
    for (disc, curv, mem, pw) in all_specs_16():
        for rp in RP_LABELS_4:
            spec = ModelSpec(test="test2", discounting=disc, curvature=curv,
                             memory=mem, pw=pw, rp_mode="fixed_pure", C=1,
                             rp_identity=rp)
            jobs.append((spec.config_id(), spec))

    # Test 3: four fixed RP clusters, C=4  -> 16
    for (disc, curv, mem, pw) in all_specs_16():
        spec = ModelSpec(test="test3", discounting=disc, curvature=curv,
                         memory=mem, pw=pw, rp_mode="fixed_4", C=4)
        jobs.append((spec.config_id(), spec))

    assert len(jobs) == 144, f"expected 144 jobs, got {len(jobs)}"
    assert len(set(cid for cid, _ in jobs)) == 144, "config_id collision"
    return jobs


def spec_from_config_id(config_id):
    """Inverse of ModelSpec.config_id(), e.g. test1_exp_sepalpha_commem_tk_C5,
    test1_exp_sepalpha_commem_tk_cara_C5 (exponential utility) or
    test3_exp_sepalpha_commem_tk_cara_SQ-PA-FE-FE (Test 3 with other pinned types)."""
    parts = config_id.split("_")
    utility = "power"
    tags = {tag: name for name, tag in UTILITY_TAGS.items()}
    if len(parts) > 5 and parts[5] in tags:
        utility = tags[parts.pop(5)]
    if len(parts) not in (5, 6) or parts[0] not in TESTS:
        raise ValueError(f"Not a model config ID: {config_id}")
    for (name, allowed), value in zip(CHOICES.items(), parts[1:5]):
        if value not in allowed:
            raise ValueError(f"{config_id}: {name} must be one of {allowed}")
    base = dict(zip(["test", *CHOICES], parts[:5]), utility=utility)
    tail = parts[5] if len(parts) == 6 else None
    rp_set = tuple(tail.split("-")) if tail and "-" in tail else None
    if base["test"] == "test3" and tail is None:
        spec = ModelSpec(rp_mode="fixed_4", C=4, **base)
    elif base["test"] == "test3" and rp_set and len(rp_set) == 4 and set(rp_set) <= set(RP_LABELS_4) \
            and rp_set != tuple(RP_LABELS_4):
        spec = ModelSpec(rp_mode="fixed_4", C=4, rp_set=rp_set, **base)
    elif base["test"] == "test2" and tail in RP_LABELS_4:
        spec = ModelSpec(rp_mode="fixed_pure", C=1, rp_identity=tail, **base)
    elif base["test"] == "test1" and tail and tail[0] == "C" and tail[1:].isdigit() and int(tail[1:]) >= 1:
        spec = ModelSpec(rp_mode="mixed", C=int(tail[1:]), **base)
    else:
        raise ValueError(f"Not a model config ID: {config_id}")
    assert spec.config_id() == config_id
    return spec


def _estimation_order(job):
    """Test 3, Test 1, Test 2; the most informative specifications first, common
    memory before separate, fewer classes first (Test 3 and smaller C feed warm
    starts of Test 1 C=4 and C+1)."""
    priority = [("qh", "sepalpha", "prelec"), ("exp", "sepalpha", "prelec"),
                ("exp", "sepalpha", "tk"), ("qh", "sepalpha", "tk"),
                ("exp", "commonalpha", "prelec"), ("qh", "commonalpha", "prelec"),
                ("exp", "commonalpha", "tk"), ("qh", "commonalpha", "tk")]
    test_rank = {"test3": 0, "test1": 1, "test2": 2}
    _, spec = job
    return (test_rank[spec.test], priority.index((spec.discounting, spec.curvature, spec.pw)),
            0 if spec.memory == "commem" else 1, spec.C, spec.rp_identity or "",
            "-".join(getattr(spec, "rp_set", None) or ()))


def select_jobs(tests=TESTS, discounting=CHOICES["discounting"], curvature=CHOICES["curvature"],
                memory=CHOICES["memory"], pw=CHOICES["pw"], clusters=(1, 2, 3, 4), rps=RP_LABELS_4,
                only=None, utility="power"):
    """Chosen (config_id, ModelSpec) jobs in estimation order: the product of the
    filters with ``utility``, or the config IDs in ``only`` (which carry their
    utility). A pure-reference sepmem model is saved as an alias of its commem
    fit, so that fit is added when missing."""
    if only:
        specs = [spec_from_config_id(config_id) for config_id in only]
    else:
        specs = []
        for disc, curv, mem, weight in product(discounting, curvature, memory, pw):
            base = dict(discounting=disc, curvature=curv, memory=mem, pw=weight, utility=utility)
            if "test1" in tests:
                specs += [ModelSpec(test="test1", rp_mode="mixed", C=c, **base) for c in clusters]
            if "test2" in tests:
                specs += [ModelSpec(test="test2", rp_mode="fixed_pure", C=1, rp_identity=rp, **base) for rp in rps]
            if "test3" in tests:
                specs.append(ModelSpec(test="test3", rp_mode="fixed_4", C=4, **base))
    jobs = {spec.config_id(): spec for spec in specs}
    for spec in specs:
        if spec.memory == "sepmem" and spec.rp_mode != "mixed":
            source = dataclasses.replace(spec, memory="commem")
            jobs.setdefault(source.config_id(), source)
    return sorted(jobs.items(), key=_estimation_order)
