import numpy as np
import math

from lottery import lotteries, one, lotteries_full



#### Definitions of theoretical functions necessary for a theoretical valuation of a lottery
# (r, gamma, alpha, lambda)


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



def V(L, r, gamma, alpha, lamb, R, pi):


    # TO DO

    return np.dots(pvl, pi).sum()
    






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

        last = o[3]

        n = len(o[3])

        outcomes = {}

        for j in range(n):

            p = last[j]['abs_prob']

            stream = {p: [0, int(last[j]['parent'][0] + last[j]['parent'][2:]), int(last[j]['from'][0] + last[j]['from'][2:]), int(last[j]['label'][0] + last[j]['label'][2:])]}

            outcomes[j] = stream

        a['outcomes'] = outcomes

        lotteries_v2[i] = a

    return lotteries_v2



def evaluation(lotteries, r, R, alpha, lamb):
    
    lotteries_v2 = {}

    l = lotteries.keys()

    for i in l:

        a = {}

        lottery = lotteries[i]

        a['name'] = lottery['name']

        outcomes = lottery['outcomes']

        n = outcomes.keys()

        b = {}

        for j in n:

            path = outcomes[j]

            p = float(*path.keys())

            o = list(*path.values())

            pv_outcome = PV(o, r, R, alpha, lamb)

            b[j] = [p, pv_outcome]

        a["PV"] = b

        lotteries_v2[i] = a

    return lotteries_v2

    
        

one_transform = transform(one)

print(one_transform)

print(evaluation(one_transform, 0.97, 0, 0.88, 2.25))


t = transform(lotteries)

print(evaluation(t, 0.97, 0, 0.88, 2.25))