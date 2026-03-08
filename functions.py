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

def transform(lotteries):

    lotteries_v2 = {}

    l = lotteries.keys()

    for i in l:

        a = {}

        lottery = lotteries[i]

        a['name'] = lottery['name']

        a["spread"] = abs(lottery["max_payoff"] - lottery["min_payoff"])

        o = lottery['periods']

        last = o["3"]

        n = len(o["3"])

        outcomes = {}

        for j in range(n):

            p = last[j]['abs_prob']
    
            stream = {p: [0, int(last[j]["parent"].replace('£', '')) if '£' in last[j]["parent"] else int(last[j]["parent"]), int(last[j]["from"].replace('£', '')) if '£' in last[j]["from"] else int(last[j]["from"]), int(last[j]["label"].replace('£', '')) if '£' in last[j]["label"] else int(last[j]["label"])]}

            outcomes[j] = stream

        a['outcomes'] = outcomes

        lotteries_v2[i] = a

    return lotteries_v2




def transform2(lotteries):

    lotteries_v2 = {}

    l = lotteries.keys()

    for i in l:

        a = {}

        lottery = lotteries[i]

        a['name'] = lottery['name']

        a["spread"] = abs(lottery["max_payoff"] - lottery["min_payoff"])

        o = lottery['periods']

        last = o["3"]

        n = len(o["3"])

        outcomes = {}

        for j in range(n):

            p = last[j]['abs_prob']
    
            stream = {p: [0, int(last[j]["parent"].replace('£', '')) if '£' in last[j]["parent"] else int(last[j]["parent"]), int(last[j]["from"].replace('£', '')) if '£' in last[j]["from"] else int(last[j]["from"]), int(last[j]["label"].replace('£', '')) if '£' in last[j]["label"] else int(last[j]["label"])]}

            stream["label"] = last[j]["label"]

            stream["from"] = last[j]["from"]

            stream["parent"] = last[j]["parent"]

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



# Certainty equivalent given the set of parameters
# XG: So far, this CE function is fine. But for the second and the third period, we may need to update to ce = v^-1 - z_t.

def ce(r=0.97, gamma=0.61, alpha=0.88, lamb=2.25, R=0, desired=desired, lotteries=transform(lotteries_full), method="tk", beta=1, palpha=1):

    # Pass through all parameters so CE is computed at the current candidate point.
    evaluated_lotteries = evaluation(r=r, R=R, alpha=alpha, lamb=lamb, gamma=gamma, lotteries=lotteries, method=method, beta=beta, palpha=palpha)

    l = evaluated_lotteries[desired]

    v = l["V"]

    return u_inv(v, R, alpha, lamb)



def ce_dict(r=0.97, gamma=0.61, alpha=0.88, lamb=2.25, R=0, lotteries = transform(lotteries_full), method="tk", beta=1, palpha=1):

    evaluated_lotteries = evaluation(r=r, R=R, alpha=alpha, lamb=lamb, gamma=gamma, lotteries=lotteries, method=method, beta=beta, palpha=palpha)

    return {i: u_inv(evaluated_lotteries[i]["V"], R, alpha, lamb) for i in lotteries}


def ce_at_rounds(r=0.97, gamma=0.61, alpha=0.88, lamb=2.25, R=0, desired=desired, lotteries=transform(lotteries_full), round=1, method="tk", beta=1, palpha=1):

    if round == 1:

        return ce(r, gamma, alpha, lamb, R, desired, lotteries, method=method, beta=beta, palpha=palpha)





if __name__ == "__main__":

    print(ce_dict())