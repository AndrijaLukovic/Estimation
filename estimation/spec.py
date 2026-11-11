"""
ModelSpec: the configuration object that drives every estimation job.

Four binary model dimensions (2^4 = 16 combinations):
    discounting : "exp"     — exponential (psi pinned to 1)
                  "qh"      — quasi-hyperbolic (psi estimated)
    curvature   : "commonalpha" — alpha_loss linked to alpha_gain
                  "sepalpha"    — alpha_gain, alpha_loss estimated separately
    memory      : "commem"  — one memory-decay parameter drives R^A and R^LE
                  "sepmem"  — delta_pa (R^A) and delta_le (R^LE) estimated separately
    pw          : "prelec"  — single-parameter Prelec weighting
                  "tk"      — Tversky-Kahneman weighting

Plus the utility family, orthogonal to the factorial (one per run directory):
    utility     : "power"       — (x-R)^alpha_gain / -lambda (R-x)^alpha_loss
                  "exponential" — CARA with absolute risk-aversion coefficients
                                  alpha_gain/alpha_loss (config IDs end in _cara)

Plus a reference-point mode, orthogonal to the factorial above:
    rp_mode     : "mixed"       — a1,a2,a3 estimated per cluster (simplex), Test 1
                  "fixed_pure"  — one RP identity gets weight 1, C=1, Test 2
                  "fixed_4"     — C=4, cluster j permanently = RP identity j, Test 3
                                  (rp_set: other pinned labels, e.g. SQ-PA-FE-FE)

Per cluster, a ClusterLayout says which positions of the canonical parameter
vector are free (optimized) vs. pinned/linked; expand_free() maps the reduced
free vector back to the canonical vector (+ an optional separate delta_le).

Saved result pickles reference these classes as ``model_spec.ModelSpec`` and
``model_spec.ClusterLayout``; ``estimation/__init__.py`` keeps that name valid.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from .config import BOUNDS, CARA_BOUNDS


# ── Canonical parameter-vector layouts ───────────────────────────────────────
CANON = {
    "tk": dict(R=0, AG=1, AL=2, LAM=3, GAM=4, S1=5, S2=6, S3=7, DELTA=8, PSI=9, LEN=10),
    "prelec": dict(R=0, AG=1, AL=2, LAM=3, BETA=4, PALPHA=5, S1=6, S2=7, S3=8, DELTA=9, PSI=10, LEN=11),
}

PARAM_LABELS = {
    "tk":     ["r", "alpha_gain", "alpha_loss", "lambda", "gamma", "a1", "a2", "a3", "delta_pa", "psi"],
    "prelec": ["r", "alpha_gain", "alpha_loss", "lambda", "beta", "palpha", "a1", "a2", "a3", "delta_pa", "psi"],
}

# RP identity -> one-hot (a1, a2, a3); a4 = 1 - a1 - a2 - a3 is residual.
# SQ = status quo (a1), PA = partial adaptation / R^A (a2), LE = lagged
# expectation / R^LE (a3), FE = forward-looking / R^FE (a4, all-zero here).
RP_ONEHOT = {
    "SQ": (1.0, 0.0, 0.0),
    "PA": (0.0, 1.0, 0.0),
    "LE": (0.0, 0.0, 1.0),
    "FE": (0.0, 0.0, 0.0),
}
RP_LABELS_4 = ["SQ", "PA", "LE", "FE"]   # Test-3 cluster order

UTILITIES = ("power", "exponential")
UTILITY_TAGS = {"exponential": "cara"}   # config-ID token after pw; none for power


@dataclass(frozen=True)
class ModelSpec:
    test: str            # "test1" | "test2" | "test3"
    discounting: str      # "exp" | "qh"
    curvature: str        # "commonalpha" | "sepalpha"
    memory: str           # "commem" | "sepmem"
    pw: str               # "prelec" | "tk"
    rp_mode: str          # "mixed" | "fixed_pure" | "fixed_4"
    C: int                # number of clusters
    rp_identity: str = None   # only for rp_mode == "fixed_pure": "SQ"|"PA"|"LE"|"FE"
    utility: str = "power"    # "power" | "exponential"; pickles without it are power
    # Only for rp_mode == "fixed_4": the pinned label of each of the four types when
    # it differs from RP_LABELS_4 (a label may repeat, e.g. SQ-PA-FE-FE). None is
    # the usual Test 3; pickles without it are the usual Test 3.
    rp_set: tuple = None

    def __post_init__(self):
        if self.rp_set is None:
            return
        rp_set = tuple(self.rp_set)
        if self.rp_mode != "fixed_4" or len(rp_set) != 4 or any(rp not in RP_LABELS_4 for rp in rp_set):
            raise ValueError(f"rp_set needs rp_mode fixed_4 and four labels from {RP_LABELS_4}: {self.rp_set!r}")
        object.__setattr__(self, "rp_set", None if rp_set == tuple(RP_LABELS_4) else rp_set)

    def config_id(self):
        if self.utility not in UTILITIES:
            raise ValueError(f"Unknown utility: {self.utility!r}")
        tag = f"{self.test}_{self.discounting}_{self.curvature}_{self.memory}_{self.pw}"
        if self.utility in UTILITY_TAGS:
            tag += "_" + UTILITY_TAGS[self.utility]
        if self.rp_mode == "mixed":
            return f"{tag}_C{self.C}"
        elif self.rp_mode == "fixed_pure":
            return f"{tag}_{self.rp_identity}"
        elif self.rp_mode == "fixed_4":
            return tag if self.rp_set is None else f"{tag}_{'-'.join(self.rp_set)}"
        raise ValueError(f"Unknown rp_mode: {self.rp_mode!r}")


def all_specs_16():
    """The 16 model-specification combinations (the 2^4 factorial), as
    (discounting, curvature, memory, pw) tuples, in a fixed deterministic order."""
    return list(product(["exp", "qh"], ["commonalpha", "sepalpha"],
                         ["commem", "sepmem"], ["prelec", "tk"]))


# ── Per-cluster layout: which canonical positions are free, fixed, or linked ─

@dataclass
class ClusterLayout:
    method: str
    free_positions: list          # canonical indices, free (optimized) directly
    fixed_values: dict            # canonical index -> pinned constant
    link_AL_to_AG: bool           # if True, AL (alpha_loss) mirrors AG in expand()
    is_sepmem: bool               # True iff spec.memory == "sepmem" for this cluster
    has_delta_le: bool            # if True, an extra free scalar delta_le is appended
    delta_le_fixed: float         # value used for delta_le when sepmem but R^LE irrelevant
    utility: str = "power"        # spec.utility; selects the default curvature bounds


def _rp_relevance(rp_mode, rp_identity):
    """Return (a2_ever_positive, a3_ever_positive) for a cluster.

    mixed        : a1,a2,a3 are all freely estimated -> both may be positive.
    fixed_pure/4 : rp_identity pins one component to 1, the rest to 0.
    """
    if rp_mode == "mixed":
        return True, True
    a1, a2, a3 = RP_ONEHOT[rp_identity]
    return a2 > 0, a3 > 0


def build_cluster_layout(spec: ModelSpec, rp_identity=None):
    """
    Build the free/fixed/linked layout for one cluster.

    rp_identity : the RP label ("SQ"/"PA"/"LE"/"FE") this cluster is pinned to,
                  when spec.rp_mode in ("fixed_pure", "fixed_4"). Ignored for
                  "mixed" (a1,a2,a3 are free there).
    """
    method = spec.pw
    idx = CANON[method]

    free = []
    fixed = {}

    # ── Curvature ─────────────────────────────────────────────────────────
    free.append(idx["R"])
    free.append(idx["AG"])
    link_al = spec.curvature == "commonalpha"
    if not link_al:
        free.append(idx["AL"])
    free.append(idx["LAM"])

    # ── Probability weighting ────────────────────────────────────────────
    if method == "tk":
        free.append(idx["GAM"])
    else:
        fixed[idx["BETA"]] = 1.0     # single-parameter Prelec
        free.append(idx["PALPHA"])

    # ── Reference-point weights ──────────────────────────────────────────
    if spec.rp_mode == "mixed":
        free.extend([idx["S1"], idx["S2"], idx["S3"]])
        a2_relevant = a3_relevant = True
    else:
        a1, a2, a3 = RP_ONEHOT[rp_identity]
        fixed[idx["S1"]] = a1
        fixed[idx["S2"]] = a2
        fixed[idx["S3"]] = a3
        a2_relevant, a3_relevant = _rp_relevance(spec.rp_mode, rp_identity)

    # ── Memory decay ──────────────────────────────────────────────────────
    # delta (canonical DELTA slot) drives R^A; when memory=="sepmem" a second
    # scalar delta_le drives R^LE (appended to the free vector, not part of
    # the canonical positions — see valuation.reference_points).
    if spec.memory == "commem":
        if a2_relevant or a3_relevant:
            free.append(idx["DELTA"])
        else:
            fixed[idx["DELTA"]] = 1.0   # unidentified (no R^A/R^LE term active)
        is_sepmem = False
        has_delta_le = False
        delta_le_fixed = 1.0
    else:
        if a2_relevant:
            free.append(idx["DELTA"])
        else:
            fixed[idx["DELTA"]] = 1.0
        is_sepmem = True
        has_delta_le = a3_relevant
        delta_le_fixed = 1.0

    # ── Time discounting ──────────────────────────────────────────────────
    if spec.discounting == "qh":
        free.append(idx["PSI"])
    else:
        fixed[idx["PSI"]] = 1.0

    return ClusterLayout(
        method=method, free_positions=free, fixed_values=fixed,
        link_AL_to_AG=link_al, is_sepmem=is_sepmem, has_delta_le=has_delta_le,
        delta_le_fixed=delta_le_fixed, utility=spec.utility,
    )


def spec_cluster_layouts(spec: ModelSpec):
    """Return (layouts, rp_identities) — one entry per cluster."""
    if spec.rp_mode == "mixed":
        return ([build_cluster_layout(spec) for _ in range(spec.C)],
                [None] * spec.C)
    elif spec.rp_mode == "fixed_pure":
        assert spec.C == 1, "fixed_pure rp_mode always has C=1"
        return ([build_cluster_layout(spec, rp_identity=spec.rp_identity)],
                [spec.rp_identity])
    elif spec.rp_mode == "fixed_4":
        assert spec.C == 4, "fixed_4 rp_mode always has C=4"
        labels = list(getattr(spec, "rp_set", None) or RP_LABELS_4)
        return [build_cluster_layout(spec, rp_identity=lbl) for lbl in labels], labels
    raise ValueError(f"Unknown rp_mode: {spec.rp_mode!r}")


# ── Reduced (free) vector <-> canonical vector ────────────────────────────

def expand_free(free_vec, layout: ClusterLayout):
    """free_vec -> (theta_canonical, delta_le_or_None).

    delta_le is None whenever memory=="commem": the canonical DELTA slot then
    drives both R^A and R^LE. For "sepmem" clusters it is always an explicit
    float — either the free tail element (R^LE active) or the fixed
    placeholder (R^LE inactive; harmless since a3=0 there zeroes its
    contribution to the reference point regardless of its value).
    """
    idx = CANON[layout.method]
    theta = np.zeros(idx["LEN"])
    for pos, val in layout.fixed_values.items():
        theta[pos] = val

    n_pos = len(layout.free_positions)
    for pos, val in zip(layout.free_positions, free_vec[:n_pos]):
        theta[pos] = val

    if layout.link_AL_to_AG:
        theta[idx["AL"]] = theta[idx["AG"]]

    if not layout.is_sepmem:
        delta_le = None
    elif layout.has_delta_le:
        delta_le = float(free_vec[n_pos])
    else:
        delta_le = layout.delta_le_fixed

    return theta, delta_le


def expand_all(free_vecs, layouts):
    thetas, delta_les = [], []
    for fv, layout in zip(free_vecs, layouts):
        th, dle = expand_free(fv, layout)
        thetas.append(th)
        delta_les.append(dle)
    return thetas, delta_les


# ── Bounds, init, names, k for the reduced (free) vector ──────────────────
# ``bounds`` arguments are bound tables {method: canonical bounds}; None means
# default_bounds(layout.utility).

_BASE_NOISE = {
    "tk": (
        np.array([0.05, 0.88, 0.88, 2.25, 0.65, 0.0, 0.0, 0.0, 1.0, 1.0]),
        np.array([0.01, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    ),
    "prelec": (
        np.array([0.05, 0.88, 0.88, 2.25, 1.0, 0.65, 0.0, 0.0, 0.0, 1.0, 1.0]),
        np.array([0.01, 0.05, 0.05, 0.10, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    ),
}
_DELTA_LE_BASE_NOISE = (1.0, 0.05)   # same base/noise as the DELTA slot
_CARA_BASE_NOISE = (0.0, 0.003)      # AG and AL under exponential utility
# Random-start draws of CARA alpha: "normal" is _CARA_BASE_NOISE; "logunif" is a
# random sign times |alpha| = 10^U(log10 LOGUNIF_ALPHA_MIN, log10 bound), so starts
# spread evenly over the decades from near-linear utility to the bound.
ALPHA_INITS = ("normal", "logunif")
LOGUNIF_ALPHA_MIN = 1e-4


def default_bounds(utility="power"):
    """Default bound table of a utility family: config.BOUNDS, with both
    curvature bounds set to config.CARA_BOUNDS for exponential utility."""
    if utility == "power":
        return BOUNDS
    if utility == "exponential":
        table = BOUNDS
        for name in ("AG", "AL"):
            for side, value in zip(("low", "high"), CARA_BOUNDS):
                table = replace_bound(name, side, value, table)
        return table
    raise ValueError(f"Unknown utility: {utility!r}")


def _base_bounds(layout, bounds=None):
    return (default_bounds(layout.utility) if bounds is None else bounds)[layout.method]


def replace_bound(name, side, value, bounds=None):
    """Copy of a bound table with one canonical bound changed in every method
    that has the parameter (e.g. replace_bound("LAM", "high", 20))."""
    table = {}
    for method, base in (BOUNDS if bounds is None else bounds).items():
        base = list(base)
        if name in CANON[method]:
            position = CANON[method][name]
            low, high = base[position]
            base[position] = (value, high) if side == "low" else (low, value)
        table[method] = tuple(base)
    return table


def bounds_free(layout: ClusterLayout, bounds=None):
    """L-BFGS-B bounds for the reduced (free) vector, in free_positions order,
    with the delta_le bound (identical to the DELTA bound) appended if used."""
    base = _base_bounds(layout, bounds)
    b = [base[pos] for pos in layout.free_positions]
    if layout.has_delta_le:
        b.append(base[CANON[layout.method]["DELTA"]])
    return b


def init_free(layout: ClusterLayout, rng: np.random.Generator, bounds=None, alpha_init="normal"):
    """Draw an initial reduced vector: typical CPT values plus noise, clipped to
    bounds, restricted to free positions. s1,s2,s3 (when free/mixed) are
    sampled in a-space near 0 and converted to s-space. alpha_init="logunif"
    (exponential utility only) redraws both alphas log-uniformly (ALPHA_INITS);
    the default draw is unchanged by the option."""
    from .valuation import a_to_s   # local import: valuation imports this module

    if alpha_init not in ALPHA_INITS:
        raise ValueError(f"Unknown alpha_init {alpha_init!r}: one of {ALPHA_INITS}")
    if alpha_init != "normal" and layout.utility != "exponential":
        raise ValueError(f"alpha_init={alpha_init!r} needs exponential utility")
    idx = CANON[layout.method]
    base, noise = _BASE_NOISE[layout.method]
    if layout.utility == "exponential":
        base, noise = base.copy(), noise.copy()
        base[[idx["AG"], idx["AL"]]], noise[[idx["AG"], idx["AL"]]] = _CARA_BASE_NOISE
    lo = np.array([b[0] for b in _base_bounds(layout, bounds)], dtype=float)
    hi = np.array([b[1] for b in _base_bounds(layout, bounds)], dtype=float)
    full = np.clip(base + rng.standard_normal(len(base)) * noise, lo, hi)
    if alpha_init == "logunif":
        for name in ("AG", "AL"):
            position = idx[name]
            sign = rng.choice([-1., 1.])
            limit = hi[position] if sign > 0 else -lo[position]
            magnitude = (10 ** rng.uniform(np.log10(LOGUNIF_ALPHA_MIN), np.log10(limit))
                         if limit > LOGUNIF_ALPHA_MIN else max(limit, 0.) * rng.uniform())
            full[position] = np.clip(sign * magnitude, lo[position], hi[position])

    s_positions = {idx["S1"], idx["S2"], idx["S3"]}
    if s_positions.issubset(set(layout.free_positions)):
        full[idx["S1"]], full[idx["S2"]], full[idx["S3"]] = a_to_s(
            full[idx["S1"]], full[idx["S2"]], full[idx["S3"]]
        )

    vec = [full[pos] for pos in layout.free_positions]
    if layout.has_delta_le:
        d_lo, d_hi = _base_bounds(layout, bounds)[idx["DELTA"]]
        base_d, noise_d = _DELTA_LE_BASE_NOISE
        vec.append(float(np.clip(base_d + rng.standard_normal() * noise_d, d_lo, d_hi)))
    return np.array(vec, dtype=float)


def param_names_free(layout: ClusterLayout):
    """Human-readable names for the reduced vector, matching PARAM_LABELS
    positions plus an explicit 'delta_le' tail entry when present."""
    labels = PARAM_LABELS[layout.method]
    names = [labels[pos] for pos in layout.free_positions]
    if layout.has_delta_le:
        names.append("delta_le")
    return names


def k_free(layout: ClusterLayout):
    """Number of free structural parameters for this cluster (excludes
    mixing weights and ksi, added separately by the caller for BIC)."""
    return len(layout.free_positions) + (1 if layout.has_delta_le else 0)
