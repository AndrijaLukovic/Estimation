"""Model configurations to estimate.

The full grid has 144 jobs: Test 1 (mixed reference points, C = 1..4),
Test 2 (one pure reference point, C = 1), Test 3 (four fixed reference-point
clusters), each crossed with the 2^4 specification factorial.
"""
from .spec import ModelSpec, RP_LABELS_4, all_specs_16


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


def ordered_jobs(which="all"):
    """Grid jobs in estimation order: Test 3, Test 1, Test 2; within a test the
    most informative specifications first, common memory before separate."""
    priority = [("qh", "sepalpha", "prelec"), ("exp", "sepalpha", "prelec"),
                ("exp", "sepalpha", "tk"), ("qh", "sepalpha", "tk"),
                ("exp", "commonalpha", "prelec"), ("qh", "commonalpha", "prelec"),
                ("exp", "commonalpha", "tk"), ("qh", "commonalpha", "tk")]
    test_rank = {"test3": 0, "test1": 1, "test2": 2}

    def key(job):
        _, spec = job
        return (test_rank[spec.test], priority.index((spec.discounting, spec.curvature, spec.pw)),
                0 if spec.memory == "commem" else 1, spec.C, spec.rp_identity or "")
    return sorted([job for job in generate_jobs() if which == "all" or job[1].test == which], key=key)


def exclusion_jobs(pw, clusters, test3=True, test2=False):
    """exp / sepalpha / commem models for the sample-exclusion robustness check."""
    spec = dict(discounting="exp", curvature="sepalpha", memory="commem", pw=pw)
    return ([ModelSpec(test="test3", rp_mode="fixed_4", C=4, **spec)] if test3 else []) + \
           ([ModelSpec(test="test2", rp_mode="fixed_pure", C=1, rp_identity=rp, **spec) for rp in RP_LABELS_4]
            if test2 else []) + \
           [ModelSpec(test="test1", rp_mode="mixed", C=c, **spec) for c in sorted(clusters)]
