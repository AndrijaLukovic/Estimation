import numpy as np
import math
from collections import defaultdict

from lotteries import lotteries, one, lotteries_full



# Some ex ante fixed parameters

r, alpha, lamb, gamma, R, desired = 0.97, 0.88, 2.25, 0.61, 0, "lottery_3"


# Probability weighting function

def pw(p, gamma=0.61, beta=1, alpha=1, method = "tk"):


    if p == 1:

        return 1

    elif p == 0:

        return 0

    else:
        if method == "tk":
            return (p ** gamma)/((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        elif method == "prelec":
            return math.exp(- beta * (- math.log(p)) ** alpha)



# Exponential discounting

def rho(t, r=0.97):

    return math.exp(- r * t)



# Power utility function with loss aversion, with respect to a reference point R

def u(x, R=0, alpha=0.88, lamb=2.25):

    if x >= R:

        return (x - R) ** (alpha)

    else:

        return - lamb * ((-x + R) ** alpha) 
    


# Inverse of the power utility function with loss aversion with respect to a reference point R


def u_inv(y, R=0, alpha=0.88, lamb=2.25): 
    """
    The inverse of the power utility function with loss aversion with respect to a reference point R.
    """

    if y >= 0:
        return y ** (1/(alpha)) + R

    if y < 0:

        return -(-y/lamb) ** (1/alpha) + R
    


# Present value of an outcome stream, outcome stream is a list

def PV(o, r=0.97, R=0, alpha=0.88, lamb=2.25):

    s = 0

    for i in range(len(o)):

        s = s + rho(i, r)*o[i]


    return u(s, R, alpha, lamb)



# Decision weights (pi) function, takes as arguments l dictionary (keys are present values and values are probabilities) and gamma parameter

def dw(l, gamma=0.61, beta=1, palpha=1, method="tk"):

    """
    Compute CPT decision weights for a lottery given a dictionary of outcomes and their probabilities.
    Two methods applicable: Tverky and Kahneman (1992) or Prelec (1998, single param version).
    """

    l = dict(sorted(l.items(), reverse=True))

    pi = []

    x = list(l.keys())

    p = list(l.values())

    i = 0

    while x[i] > 0:

        if i == 0:

            pi.append(pw(p[i], gamma, beta, palpha, method))

        else:

            pi.append(pw(sum([p[j] for j in range(i+1)]), gamma, beta, palpha, method) - pw(sum([p[h] for h in range(i)]), gamma, beta, palpha, method))

        i = i + 1
    
        if i >= len(l):

            break

    for i in range(i, len(l)):

        if i == len(l) - 1:

            pi.append(pw(p[i], gamma, beta, palpha, method))

        else:

            pi.append(pw(sum([p[j] for j in range(i, len(l))]), gamma, beta, palpha, method) - pw(sum([p[h] for h in range(i+1, len(l))]), gamma, beta, palpha, method))

        i = i + 1


    d = {}

    for i in range(len(pi)):

        d[x[i]] = pi[i]

    return d, pi





# Value function, taking the list of present values and the list of physical proababilities as well as all the parameters

def V(pvl, p, r=0.97, gamma=0.61, alpha=0.88, lamb=2.25, R=0, method="tk", beta=1, palpha=1):
    """
    Compute the CPT value of a lottery given the present values of its outcome streams and their probabilities.
    Using the decision weights from dw(), and order the outcome streams by PV, then do the weighted sum.
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




# The function takes the original dictionary and returns a dictionary with all the outcome streams for each lottery together with the probabilities of the path
# No parametric specification necessary, purely objective probabilities and the lotteries

def _parse_payoff(label):
    return int(label.replace('£', ''))


def transform(lotteries):

    lotteries_v2 = {}

    for i, lottery in lotteries.items():

        a = {}

        a['name'] = lottery['name']

        a["spread"] = abs(lottery["max_payoff"] - lottery["min_payoff"])

        o = lottery['periods']

        last_period = max(int(k) for k in o.keys())
        last = o[str(last_period)]

        outcomes = {}

        for j, node in enumerate(last):

            p = node['abs_prob']

            # Build payoff stream: [t0=0, t1, t2, ..., terminal]
            # depth determines how many prior periods to trace back
            stream_payoffs = [0]
            if last_period >= 3:
                stream_payoffs.append(_parse_payoff(node["parent"]))
            if last_period >= 2:
                stream_payoffs.append(_parse_payoff(node["from"]))
            stream_payoffs.append(_parse_payoff(node["label"]))

            outcomes[j] = {p: stream_payoffs}

        a['outcomes'] = outcomes

        lotteries_v2[i] = a

    return lotteries_v2




def transform2(lotteries):

    lotteries_v2 = {}

    for i, lottery in lotteries.items():

        a = {}

        a['name'] = lottery['name']

        a["spread"] = abs(lottery["max_payoff"] - lottery["min_payoff"])

        o = lottery['periods']

        last_period = max(int(k) for k in o.keys())
        last = o[str(last_period)]

        outcomes = {}

        for j, node in enumerate(last):

            p = node['abs_prob']

            stream_payoffs = [0]
            if last_period >= 3:
                stream_payoffs.append(_parse_payoff(node["parent"]))
            if last_period >= 2:
                stream_payoffs.append(_parse_payoff(node["from"]))
            stream_payoffs.append(_parse_payoff(node["label"]))

            stream = {p: stream_payoffs}
            stream["label"]  = node["label"]
            stream["from"]   = node["from"]
            if last_period >= 3:
                stream["parent"] = node["parent"]

            outcomes[j] = stream

        a['outcomes'] = outcomes

        lotteries_v2[i] = a

    return lotteries_v2




def evaluation(
        r=0.97, R=0, alpha=0.88, lamb=2.25, gamma=0.61, lotteries=transform(lotteries_full),
        beta=1, palpha=1, method="tk"
        ):
    
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

            pv_outcome = PV(o, r, R, alpha, lamb)

            b[j] = [p, pv_outcome]

            pvl.append(pv_outcome)

            prob.append(p)

        a["PV"] = b

        a["V"] = V(pvl,prob, r, gamma, alpha, lamb, R, method=method, beta=beta, palpha=palpha)

        a["present_values_of_streams"] = pvl

        a["probabilities_of_streams"] = prob

        lotteries_v2[i] = a


    return lotteries_v2



# Parse a period label like "+£26" or "-£29" into an integer.
def _parse_label(label):
    return int(label.replace('£', ''))


# Cumulative discounted payoff realised up to (but not including) the CE elicitation.
# period1_label / period2_label are the realised branch labels from the data.
def realized_zt( period1_label=None, period2_label=None):
    zt = 0.0
    if period1_label is not None:
        zt += _parse_label(period1_label)
    if period2_label is not None:
        zt +=  _parse_label(period2_label)
    return zt


# Certainty equivalent given the set of parameters.
# Z_t is the cumulative discounted payoff already realised; defaults to 0 (session 1).
# Formula: ce = v^{-1}(V(L_j)) - Z_t

def ce(r=0.97, gamma=0.61, alpha=0.88, lamb=2.25, R=0, desired=desired, lotteries=transform(lotteries_full), method="tk", beta=1, palpha=1, Z_t=0):

    # Pass through all parameters so CE is computed at the current candidate point.
    evaluated_lotteries = evaluation(r=r, R=R, alpha=alpha, lamb=lamb, gamma=gamma, lotteries=lotteries, method=method, beta=beta, palpha=palpha)

    l = evaluated_lotteries[desired]

    v = l["V"]

    return u_inv(v, R, alpha, lamb) - Z_t



def ce_dict(r=0.97, gamma=0.61, alpha=0.88, lamb=2.25, R=0, lotteries=transform(lotteries_full), method="tk", beta=1, palpha=1):
    """Return v^{-1}(V(L_j)) per lottery, without Z_t (base values for session 1)."""
    evaluated_lotteries = evaluation(r=r, R=R, alpha=alpha, lamb=lamb, gamma=gamma, lotteries=lotteries, method=method, beta=beta, palpha=palpha)

    return {i: u_inv(evaluated_lotteries[i]["V"], R, alpha, lamb) for i in lotteries}


def ce_th_series(y, r=0.97, gamma=0.61, alpha=0.88, lamb=2.25, R=0, lotteries=transform(lotteries_full), method="tk", beta=1, palpha=1):
    """
    Return a Series of theoretical CEs aligned with rows of y,
    with Z_t already subtracted per row.
    round < 15  : session 1, Z_t = 0
    round == 15 : session 2, Z_t = period-1 payoff
    round == 16 : session 3, Z_t = period-1 + period-2 payoff
    """
    base = ce_dict(r, gamma, alpha, lamb, R, lotteries=lotteries, method=method, beta=beta, palpha=palpha)
    zt = y.apply(
        lambda row: realized_zt(
            period1_label=row["realized_period1_label"] if row["round_number"] >= 15 else None,
            period2_label=row["realized_period2_label"] if row["round_number"] >= 16 else None,
        ), axis=1
    )
    return y["lottery_id"].map(base) - zt



# ── Reference point candidates ────────────────────────────────────────────────

def expected_payoff(lottery_outcomes):
    """Expected total payoff of a lottery: Σ_j p_j * sum(payoffs in path j).
    lottery_outcomes: the 'outcomes' dict from transform(), where each entry is
    {j: {p: [0, z1, z2, ...]}}. stream[0] is always the t=0 placeholder (0).
    """
    total = 0.0
    for path in lottery_outcomes.values():
        p      = next(iter(path.keys()))
        stream = next(iter(path.values()))
        total += p * sum(stream[1:])   # skip the t=0 zero
    return total


def status_quo(Z_0=0.0):
    """R^SQ(t) = Z_0  (anchored to the value at the start of the sequence)."""
    return float(Z_0)


def partial_adaptation(t, Z_sequence, delta=1.0):
    """R^A(t) = Σ_{i=0}^{t} δ^{t-i} Z_i / Σ_{i=0}^{t} δ^{t-i}
    Recursive: R^A(t) = (1-α_t)*R^A(t-1) + α_t*Z_t,  R^A(0) = Z_0.
    Z_sequence : list [Z_0, Z_1, ..., Z_t]  (length t+1)
    delta      : weighting parameter δ ∈ (0,1] (1 = equal weighting)
    """
    Z = np.asarray(Z_sequence[:t + 1], dtype=float)
    if delta == 1.0:
        return float(Z.mean())
    weights = np.array([delta ** (t - i) for i in range(t + 1)])
    return float(np.dot(weights, Z) / weights.sum())


def lagged_expectation(t, EL_sequence, Z_0=0.0, delta=1.0):
    """R^LE(t) = Σ_{i=0}^{t-1} δ^{t-1-i} E[L_i] / Σ_{i=0}^{t-1} δ^{t-1-i},  R^LE(0)=Z_0
    EL_sequence : list [E[L_0], ..., E[L_{t-1}]]  (length t)
    delta       : weighting parameter δ ∈ (0,1] (1 = equal weighting)
    """
    if t == 0:
        return float(Z_0)
    EL = np.asarray(EL_sequence[:t], dtype=float)
    if delta == 1.0:
        return float(EL.mean())
    weights = np.array([delta ** (t - 1 - i) for i in range(t)])
    return float(np.dot(weights, EL) / weights.sum())


def forward_looking(EL_t):
    """R^FE(t) = E[L_t]  (expectation of the current sublottery)."""
    return float(EL_t)


def composite(a1, a2, a3, RSQ, RA, RLE, RFE):
    """R(t) = a1*R^SQ + a2*R^A + a3*R^LE + (1-a1-a2-a3)*R^FE
    The forward-looking weight is residual: a4 = 1 - a1 - a2 - a3.
    a4 is clipped at 0 so R stays well-defined even if weights sum > 1.
    """
    a4 = max(0.0, 1.0 - a1 - a2 - a3)
    return a1 * RSQ + a2 * RA + a3 * RLE + a4 * RFE


if __name__ == "__main__":

    print(ce_dict())