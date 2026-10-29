"""Scalar cumulative prospect theory: the reference implementation.

Estimation uses the vectorized valuation.predict_ce; this module is the
state-by-state formula used to audit saved results and to test it.
"""
import math
from collections import defaultdict

import numpy as np


def pw(p, gamma=0.61, beta=1, alpha=1, method="tk"):
    """Probability weighting: Tversky-Kahneman (1992) or Prelec (1998)."""
    if p == 1:
        return 1
    elif p == 0:
        return 0
    else:
        if method == "tk":
            return (p ** gamma)/((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        elif method == "prelec":
            return math.exp(- beta * (- math.log(p)) ** alpha)


def rho(t, r=0.97, psi=1.0):
    """Quasi-hyperbolic discounting: weight(t=0) = 1, weight(t>=1) = psi * exp(-r*t)."""
    if t == 0:
        return 1.0

    return psi * math.exp(- r * t)


def u(x, R=0, alpha=0.88, lamb=2.25, alpha_loss=None):
    """Power utility around reference point R. ``alpha`` is the gain exponent;
    without ``alpha_loss`` both domains share it."""
    alpha_loss = alpha if alpha_loss is None else alpha_loss

    if x >= R:
        return (x - R) ** alpha

    return -lamb * (R - x) ** alpha_loss


def u_inv(y, R=0, alpha=0.88, lamb=2.25, alpha_loss=None):
    """Inverse of the domain-specific power utility. ``alpha`` is alpha_gain."""
    alpha_loss = alpha if alpha_loss is None else alpha_loss

    if y >= 0:
        return y ** (1 / alpha) + R

    return -(-y / lamb) ** (1 / alpha_loss) + R


def PV(o, r=0.97, R=0, alpha=0.88, lamb=2.25, alpha_loss=None, psi=1.0):
    """Utility of the discounted sum of an outcome stream (a list)."""
    s = 0

    for i in range(len(o)):

        s = s + rho(i, r, psi)*o[i]

    return u(s, R, alpha, lamb, alpha_loss=alpha_loss)


def dw(l, gamma=0.61, beta=1, palpha=1, method="tk"):
    """
    CPT decision weights for a lottery {outcome: probability}.
    Returns ({outcome: weight} ranked best first, [weights]).
    """
    l = dict(sorted(l.items(), reverse=True))

    pi = []

    x = list(l.keys())

    p = list(l.values())
    total_probability = math.fsum(p)
    if not math.isclose(total_probability, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Decision-weight probabilities sum to {total_probability}")

    def cumulative(start, stop):
        # The full outcome set has probability exactly one. Do not let
        # round-off (1 +/- epsilon) create a spurious distorted endpoint.
        if start == 0 and stop == len(p):
            return 1.0
        return math.fsum(p[start:stop]) / total_probability

    i = 0

    while x[i] > 0:

        if i == 0:

            pi.append(pw(cumulative(0, 1), gamma, beta, palpha, method))

        else:

            pi.append(pw(cumulative(0, i+1), gamma, beta, palpha, method) - pw(cumulative(0, i), gamma, beta, palpha, method))

        i = i + 1

        if i >= len(l):

            break

    for i in range(i, len(l)):

        if i == len(l) - 1:

            pi.append(pw(cumulative(i, len(p)), gamma, beta, palpha, method))

        else:

            pi.append(pw(cumulative(i, len(p)), gamma, beta, palpha, method) - pw(cumulative(i+1, len(p)), gamma, beta, palpha, method))

        i = i + 1

    d = {}

    for i in range(len(pi)):

        d[x[i]] = pi[i]

    return d, pi


def V(pvl, p, r=0.97, gamma=0.61, alpha=0.88, lamb=2.25, R=0, method="tk", beta=1, palpha=1):
    """
    CPT value of a lottery from the utilities of its outcome streams and their
    probabilities: rank the streams, weight them with dw(), and sum.
    """
    assert len(pvl) == len(p), "The present values and the probabilities need to be lists of the same length!"

    # Merge duplicate outcomes before weighting (CPT requires ranking unique outcomes).
    # This also avoids shape mismatches when identical outcomes exist in multiple branches.
    d = defaultdict(float)
    for x_i, p_i in zip(pvl, p):
        d[x_i] += p_i

    # dw(...) returns weights ordered by sorted outcomes, so we must dot with the
    # same ordered outcome vector (not the original unsorted/duplicated pvl list).
    dweights, _ = dw(d, gamma, beta, palpha, method)
    ranked_outcomes = np.array(list(dweights.keys()), dtype=float)
    ranked_weights = np.array(list(dweights.values()), dtype=float)

    return float(np.dot(ranked_outcomes, ranked_weights))


def evaluation(lotteries, r=0.97, R=0, alpha=0.88, lamb=2.25, gamma=0.61,
               beta=1, palpha=1, method="tk", alpha_loss=None, psi=1.0):
    """Evaluate transformed lotteries: per lottery the stream utilities ("PV"),
    the CPT value "V", and the stream utilities/probabilities as lists."""
    lotteries_v2 = {}

    l = lotteries.keys()

    for i in l:

        a = {}

        lottery = lotteries[i]

        a['name'] = lottery['name']

        outcomes = lottery['outcomes']

        n = outcomes.keys()

        b = {}

        pvl = []

        prob = []

        for j in n:

            path = outcomes[j]

            p = float(*path.keys())

            o = list(*path.values())

            pv_outcome = PV(o, r, R, alpha, lamb, alpha_loss=alpha_loss, psi=psi)

            b[j] = [p, pv_outcome]

            pvl.append(pv_outcome)

            prob.append(p)

        a["PV"] = b

        a["V"] = V(pvl,prob, r, gamma, alpha, lamb, R, method=method, beta=beta, palpha=palpha)

        a["present_values_of_streams"] = pvl

        a["probabilities_of_streams"] = prob

        lotteries_v2[i] = a

    return lotteries_v2
