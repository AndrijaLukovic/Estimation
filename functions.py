import numpy as np
import math

from lotteries import lotteries, one, lotteries_full


r, alpha, lamb, gamma, R, desired = 0.97, 0.88, 2.25, 0.61, 0, "lottery_3"



# Probability weighting function

def pw(p, gamma):

    if p == 1:

        return 1

    elif p == 0:

        return 0

    else:
        return (p ** gamma)/((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))



# Exponential discounting

def rho(t, r):

    return math.exp(- r * t)



# Power utility function with loss aversion with respect to a reference point R

def u(x, R, alpha, lamb):

    if x >= R:

        return (x - R) ** (alpha)

    else:

        return - lamb * ((-x + R) ** alpha)
    


# Inverse of the power utility function with loss aversion with respect to a reference point R

def u_inv(y, R, alpha, lamb):

    if y >= 0:

        return y ** (1/(alpha)) + R

    if y < 0:

        return -(-y/lamb) ** (1/alpha) + R
    


# Present value of an outcome stream

def PV(o, r, R, alpha, lamb):

    # assert isinstance(o, list), "Must be a list!"

    s = 0

    for i in range(len(o)):

        s = s + rho(i, r)*o[i]


    return u(s, R, alpha, lamb)




# def dw(l, gamma):

#     l = dict(sorted(l.items(), reverse=True))

#     pi = []

#     x = list(l.keys())

#     p = list(l.values())

#     for i in range(len(x)):
    
#     return l





# Value function, taking the list of present values and the list of physical proababilities as well as all the parameters

def V(pvl, p, r, gamma, alpha, lamb, R):

    assert len(pvl) == len(p), "The present values and the probabilities need to be lists of the same length!"

    w = [pw(i, gamma) for i in p]

    #x = [u(i, R, alpha, lamb) for i in pvl]

    return float(np.dot(pvl, w).sum())




# The function takes the original dictionary and returns a dictionary with all the outcome streams for each lottery together with the probabilities of the path
# The function calculates the present value of each of the lotteries with respect to the exponential time discounting parameter

def transform(lotteries):

    lotteries_v2 = {}

    l = lotteries.keys()

    for i in l:

        a = {}

        lottery = lotteries[i]

        a['name'] = lottery['name']

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


lotteries_transformed = transform(lotteries_full)


def evaluation(r, R, alpha, lamb, gamma, desired=None, lotteries=transform(lotteries_full)):
    
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

        a["V"] = V(pvl,prob, r, gamma, alpha, lamb, R)

        a["present_values_of_streams"] = pvl

        a["probabilities_of_streams"] = prob

        lotteries_v2[i] = a

    if desired != None:

        current = lotteries_v2[desired]

        return current["V"]

    return lotteries_v2


def ce(r, gamma, alpha, lamb, R, desired=desired):

    value = (evaluation(r, R, alpha, lamb, gamma, desired = desired))

    return u_inv(value, R, alpha, lamb)


print(evaluation(r, R, alpha, lamb, gamma, desired="lottery_14"))

print(ce(r, gamma, alpha, lamb, R, desired="lottery_14"))