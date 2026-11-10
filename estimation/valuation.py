"""Reference points and predicted certainty equivalents of conditional lotteries.

CE and reference points are current remaining-payoff amounts; future payment
delays restart at the current decision stage.
"""
import numpy as np

from . import cpt
from .config import RSQ_MODE
from .spec import CANON

NAMES = dict(R="r", AG="alpha_gain", AL="alpha_loss", LAM="lambda", GAM="gamma",
             BETA="beta", PALPHA="palpha", S1="s1", S2="s2", S3="s3", DELTA="delta", PSI="psi")


def s_to_a(s1, s2, s3):
    """Stick-breaking: (s1,s2,s3) ∈ [0,1]^3  →  (a1,a2,a3) with a1+a2+a3 ≤ 1."""
    a1 = s1
    a2 = s2 * (1.0 - s1)
    a3 = s3 * (1.0 - s1) * (1.0 - s2)
    return a1, a2, a3


def a_to_s(a1, a2, a3):
    """Inverse stick-breaking: (a1,a2,a3) with sum ≤ 1  →  (s1,s2,s3) ∈ [0,1]^3."""
    s1 = a1
    rem1 = 1.0 - a1
    s2 = a2 / rem1 if rem1 > 1e-12 else 0.0
    rem2 = rem1 * (1.0 - s2)
    s3 = a3 / rem2 if rem2 > 1e-12 else 0.0
    return s1, s2, s3


def named_parameters(theta, method):
    """Canonical vector -> {parameter name: float}."""
    return {NAMES[k]: float(theta[pos]) for k, pos in CANON[method].items() if k != "LEN"}


def reference_points(theta, method, delta_le, inputs):
    """Composite reference point R = a1*R^SQ + a2*R^A + a3*R^LE + a4*R^FE per state.

    inputs = (E[L_0], E[L_t | path], E[L_1 | Z1], stage, Z1, Z2, Z1+Z2) arrays.
      R^A (t=1) = Z1/(δ+1),        R^A (t=2) = (δ Z1 + Z2)/(δ²+δ+1)
      R^LE(t=1) = E[L_0],          R^LE(t=2) = (δ_LE E[L_0] + E[L_1|Z1])/(δ_LE+1)
      R^FE      = E[L_t | path];   all history terms are 0 at t=0.
    delta_le None reuses δ for R^LE (common memory decay).
    """
    ix = CANON[method]
    a1, a2, a3 = s_to_a(theta[ix["S1"]], theta[ix["S2"]], theta[ix["S3"]])
    a4 = 1 - a1 - a2 - a3
    el, fe, c1, stage, z1, z2, zt = inputs
    delta = theta[ix["DELTA"]]
    ra = np.where(stage == 1, z1 / (delta + 1), np.where(
        stage == 2, (delta * z1 + z2) / (delta ** 2 + delta + 1), 0))
    dle = delta if delta_le is None else delta_le
    rle = np.where(stage == 1, el, np.where(stage == 2, (dle * el + c1) / (dle + 1), 0))
    return a1 * (zt if RSQ_MODE == "zt" else 0) + a2 * ra + a3 * rle + a4 * fe


def _weight(p, method, gamma, beta, palpha):
    """Vectorized probability weighting with exact endpoints."""
    p = np.clip(p, 0, 1)
    out = np.zeros_like(p)
    inside = (p > 0) & (p < 1)
    value = p[inside]
    if method == "tk":
        out[inside] = value ** gamma / (value ** gamma + (1 - value) ** gamma) ** (1 / gamma)
    else:
        out[inside] = np.exp(-beta * (-np.log(value)) ** palpha)
    out[p == 1] = 1
    return out


def _cara(z, a):
    """Vectorized cpt.cara: (1 - exp(-a z)) / a for z >= 0, linear at a = 0."""
    return z if a == 0 else -np.expm1(-a * z) / a


def _cara_inv(v, a):
    """Vectorized cpt.cara_inv."""
    return v if a == 0 else -np.log1p(-a * v) / a


def _cara_inv_side(value, a, weights, distance, side, side_weight, opposite):
    """Vectorized cpt.cara_ce inverse for one side of the reference point:
    1 - a*value is rebuilt from non-negative terms where it would cancel."""
    if a <= 0:
        return _cara_inv(value, a)
    kept = 1 - side_weight + np.sum(np.where(side, weights * np.exp(-a * distance), 0), axis=1) + a * opposite
    return np.where(a * value > 0.5, -np.log(kept) / a, _cara_inv(value, a))


def predict_ce(data, theta, method, delta_le=None, utility="power"):
    """Vectorized CPT certainty equivalent of every conditional state in
    ``data`` (a prepare.PreparedGridData) under one cluster's parameters.
    ``utility`` is the spec's utility family (see cpt.u)."""
    if utility not in ("power", "exponential"):
        raise ValueError(f"Unknown utility: {utility!r}")
    ix = CANON[method]
    r, ag, al, lam, psi = (theta[ix[k]] for k in ["R", "AG", "AL", "LAM", "PSI"])
    if method == "tk":
        gamma = theta[ix["GAM"]]
        beta, palpha = 1.0, 1.0
    else:
        beta, palpha = theta[ix["BETA"]], theta[ix["PALPHA"]]
        gamma = 1.0
    reference = reference_points(theta, method, delta_le, data.reference_inputs)
    delays = np.arange(data.streams.shape[2])
    discount = psi * np.exp(-r * delays)
    discount[0] = 1.0
    pv = np.sum(data.streams * discount, axis=2)
    distance = pv - reference[:, None]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gain, loss = np.maximum(distance, 0), np.maximum(-distance, 0)
        if utility == "power":
            u = np.where(distance >= 0, gain ** ag, -lam * loss ** al)
        else:
            u = np.where(distance >= 0, _cara(gain, ag), -lam * _cara(loss, al))
        # Saturated CARA utilities tie in floating point; x - R keeps their rank.
        rank = u if utility == "power" else distance
        order = np.argsort(np.where(data.path_mask, -rank, np.inf), axis=1, kind="stable")
        u_sorted = np.take_along_axis(u, order, axis=1)
        p_sorted = np.take_along_axis(data.probabilities, order, axis=1)
        mask_sorted = np.take_along_axis(data.path_mask, order, axis=1)
        u_sorted = np.where(mask_sorted, u_sorted, 0)
        left = np.cumsum(p_sorted, axis=1)
        right = np.cumsum(p_sorted[:, ::-1], axis=1)[:, ::-1]
        full_prefix = np.arange(p_sorted.shape[1])[None, :] >= mask_sorted.sum(axis=1)[:, None] - 1
        left[full_prefix] = 1.0
        right[:, 0] = 1.0
        left_before = np.column_stack([np.zeros(len(left)), left[:, :-1]])
        right_after = np.column_stack([right[:, 1:], np.zeros(len(right))])
        weight_left, weight_right = (_weight(left, method, gamma, beta, palpha),
                                     _weight(right, method, gamma, beta, palpha))
        gain_weights = weight_left - _weight(left_before, method, gamma, beta, palpha)
        loss_weights = weight_right - _weight(right_after, method, gamma, beta, palpha)
        value = np.sum(u_sorted * np.where(u_sorted > 0, gain_weights, loss_weights), axis=1)
        if utility == "power":
            ce = reference + np.where(value >= 0, np.maximum(value, 0) ** (1 / ag),
                                     -(np.maximum(-value, 0) / lam) ** (1 / al))
        else:
            gains, losses = u_sorted > 0, u_sorted < 0
            size = np.abs(np.take_along_axis(distance, order, axis=1))
            gain_value = np.sum(np.where(gains, u_sorted * gain_weights, 0), axis=1)
            loss_value = -np.sum(np.where(losses, u_sorted * loss_weights, 0), axis=1)
            # Weight of each side: gains are a prefix and losses a suffix of the
            # ranked outcomes, and the cumulative weight is exactly 1 at full mass.
            n_gains, n_losses = gains.sum(axis=1), losses.sum(axis=1)
            last_gain = np.maximum(n_gains - 1, 0)
            first_loss = np.minimum(mask_sorted.sum(axis=1) - n_losses, p_sorted.shape[1] - 1)
            gain_share = np.where(n_gains > 0, np.take_along_axis(weight_left, last_gain[:, None], axis=1)[:, 0], 0)
            loss_share = np.where(n_losses > 0, np.take_along_axis(weight_right, first_loss[:, None], axis=1)[:, 0], 0)
            ce = reference + np.where(
                value >= 0,
                _cara_inv_side(np.maximum(value, 0), ag, gain_weights, size, gains, gain_share, loss_value),
                -_cara_inv_side(np.maximum(-value, 0) / lam, al, loss_weights, size, losses, loss_share,
                                gain_value / lam))
    return ce


def scalar_ce(values, reference, lottery, method, utility="power"):
    """CE of one conditional lottery via the scalar CPT reference implementation.
    ``values`` is a named_parameters() dict. Returns (ce, evaluated lottery)."""
    evaluated = cpt.evaluation(
        {"current": lottery}, r=values["r"], R=reference, alpha=values["alpha_gain"],
        alpha_loss=values["alpha_loss"], lamb=values["lambda"], gamma=values.get("gamma", 1),
        beta=values.get("beta", 1), palpha=values.get("palpha", 1), psi=values["psi"],
        method=method, utility=utility)["current"]
    if utility == "exponential":
        ce = cpt.cara_ce(lottery, values["r"], reference, values["alpha_gain"], values["alpha_loss"],
                         values["lambda"], gamma=values.get("gamma", 1), beta=values.get("beta", 1),
                         palpha=values.get("palpha", 1), method=method, psi=values["psi"])
    else:
        ce = cpt.u_inv(evaluated["V"], reference, values["alpha_gain"], values["lambda"],
                       alpha_loss=values["alpha_loss"], utility=utility)
    return ce, evaluated
