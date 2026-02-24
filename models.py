import numpy as np
import math
import random




### Definitions of Reference Points

# Status Quo Ante

def rsq(lottery):

    return 0



# Partial Adaptation
# The inputs are the point in time t, the discount factor delta and the sequence of the realised cumulative payoffs x (x is a list)

def rp_pa(t, delta, x):

    return sum([delta**(t-i)*x[i] for i in range(0,t+1)])/sum([delta**(t-i) for i in range(0,t+1)])





# Lagged Expectation
# The inputs need to be all the lotteries that the dm has been faced with 
#(it could be a dictionary with time index as key and lotteries at a particular point in time as values) 

def rp_le(t, delta, l):

    s = 0

    for i in range(0, t):

        p = list(l[i].keys())

        v = list(l[i].values())

        ev = sum([p[i]*v[i] for i in range(len(p))])

        s = s + (delta**(t-i)) * ev
        
    s = s / (sum([delta**(t-i) for i in range(0,t)]))

    return s


# Forward Looking

def rp_fl(t, l):

    L = l[t]





