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


def cara(z, a):
    """(1 - exp(-a z)) / a for z >= 0: CARA utility of a gain, or of the size of
    a loss, z = |x - R|. a > 0 is concave, a < 0 convex, a = 0 linear."""
    return z if a == 0 else -math.expm1(-a * z) / a


def cara_inv(v, a):
    """Inverse of cara(., a): the z >= 0 with cara(z, a) = v (needs a*v < 1;
    where u has saturated a*v rounds to 1, see cara_ce)."""
    return v if a == 0 else -math.log1p(-a * v) / a


def u(x, R=0, alpha=0.88, lamb=2.25, alpha_loss=None, utility="power"):
    """Utility around reference point R, separate curvature in each domain
    (without ``alpha_loss`` both domains share ``alpha``).

    power       : (x-R)^alpha for gains, -lamb (R-x)^alpha_loss for losses.
    exponential : CARA, cara(x-R, alpha) for gains, -lamb cara(R-x, alpha_loss)
                  for losses; u(R)=0, u'(R+)=1, u'(R-)=lamb, and alpha,
                  alpha_loss are absolute risk-aversion coefficients (1/£).
    """
    alpha_loss = alpha if alpha_loss is None else alpha_loss

    if utility == "power":
        if x >= R:
            return (x - R) ** alpha
        return -lamb * (R - x) ** alpha_loss
    if utility == "exponential":
        if x >= R:
            return cara(x - R, alpha)
        return -lamb * cara(R - x, alpha_loss)
    raise ValueError(f"Unknown utility: {utility!r}")


def u_inv(y, R=0, alpha=0.88, lamb=2.25, alpha_loss=None, utility="power"):
    """Inverse of u(); y >= 0 maps to gains, y < 0 to losses."""
    alpha_loss = alpha if alpha_loss is None else alpha_loss

    if utility == "power":
        if y >= 0:
            return y ** (1 / alpha) + R
        return -(-y / lamb) ** (1 / alpha_loss) + R
    if utility == "exponential":
        if y >= 0:
            return R + cara_inv(y, alpha)
        return R - cara_inv(-y / lamb, alpha_loss)
    raise ValueError(f"Unknown utility: {utility!r}")


def risk_premium(stake, alpha, alpha_loss=None, utility="power"):
    """Risk attitude implied by the utility function alone, at one stake: the
    risk premium (expected value minus certainty equivalent) of a 50/50 gamble
    between R and R+stake, and of one between R and R-stake, each as a share of
    the gamble's |expected value|. No probability weighting, and lambda cancels
    because each gamble stays on one side of R.

    Returns (gain, loss), each in (-1, 1): > 0 risk averse, 0 risk neutral,
    < 0 risk seeking (power: alpha < 1 is averse in gains and seeking in losses;
    CARA: alpha > 0 is averse in gains and seeking in losses).
    """
    options = dict(R=0, alpha=alpha, lamb=1, alpha_loss=alpha_loss, utility=utility)
    half = stake / 2
    gain = u_inv(.5 * u(stake, **options), **options)
    loss = u_inv(.5 * u(-stake, **options), **options)
    return (half - gain) / half, (-half - loss) / half


def PV(o, r=0.97, R=0, alpha=0.88, lamb=2.25, alpha_loss=None, psi=1.0, utility="power"):
    """Utility of the discounted sum of an outcome stream (a list)."""
    s = 0

    for i in range(len(o)):

        s = s + rho(i, r, psi)*o[i]

    return u(s, R, alpha, lamb, alpha_loss=alpha_loss, utility=utility)


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


def cara_ce(lottery, r, R, alpha, alpha_loss, lamb, gamma=0.61, beta=1, palpha=1,
            method="tk", psi=1.0):
    """CE of one transformed lottery under exponential utility.

    Equals u_inv(V) wherever that inverse is representable. For a > 0 with a*V
    near 1, u has saturated and 1 - a*V cancels to 0 in floating point, so it is
    rebuilt from non-negative terms over the outcomes on V's side of R (W is the
    weight of that side, z_i = |x_i - R|, V_other the opposite side's value):
        1 - a*V = (1 - W) + sum(w_i exp(-a z_i)) + a*|V_other|.
    Outcomes are ranked by x - R, which u cannot do once it has saturated.
    """
    distances = defaultdict(float)
    for path in lottery["outcomes"].values():
        p, o = float(*path.keys()), list(*path.values())
        distances[sum(rho(i, r, psi) * o[i] for i in range(len(o))) - R] += p
    weights, _ = dw(distances, gamma, beta, palpha, method)

    def side(sign, a):
        outcomes = [d for d in distances if d * sign > 0]
        share = 1.0 if len(outcomes) == len(distances) else pw(
            math.fsum(distances[d] for d in outcomes), gamma, beta, palpha, method)
        terms = [(weights[d], abs(d)) for d in outcomes]
        return share, terms, math.fsum(w * cara(z, a) for w, z in terms)

    def inverse(value, a, share, terms, opposite):
        if a <= 0 or a * value <= 0.5:
            return cara_inv(value, a)
        kept = 1 - share + math.fsum(w * math.exp(-a * z) for w, z in terms) + a * opposite
        return -math.log(kept) / a

    gain_share, gains, gain_value = side(1, alpha)
    loss_share, losses, loss_value = side(-1, alpha_loss)
    value = gain_value - lamb * loss_value
    if value >= 0:
        return R + inverse(value, alpha, gain_share, gains, lamb * loss_value)
    return R - inverse(-value / lamb, alpha_loss, loss_share, losses, gain_value / lamb)


def evaluation(lotteries, r=0.97, R=0, alpha=0.88, lamb=2.25, gamma=0.61,
               beta=1, palpha=1, method="tk", alpha_loss=None, psi=1.0, utility="power"):
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

            pv_outcome = PV(o, r, R, alpha, lamb, alpha_loss=alpha_loss, psi=psi, utility=utility)

            b[j] = [p, pv_outcome]

            pvl.append(pv_outcome)

            prob.append(p)

        a["PV"] = b

        a["V"] = V(pvl,prob, r, gamma, alpha, lamb, R, method=method, beta=beta, palpha=palpha)

        a["present_values_of_streams"] = pvl

        a["probabilities_of_streams"] = prob

        lotteries_v2[i] = a

    return lotteries_v2
