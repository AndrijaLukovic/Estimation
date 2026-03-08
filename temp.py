from lotteries import one
from lotteries import lotteries


# Params a1, a2, a3, defining the convex linear combination of reference points

weights = [0.25, 0.25, 0.25, 0.25]

# Delta entering partial adaptation

delta = 1



def cumulative_transform(lotteries):

    """
    This function takes the original dictionary of lotteries and adds the cumulative payoff for each period and each realised branch.
    """

    d_update = {}

    for k, v in lotteries.items():

        temp = v["periods"]

        for i, outcome in temp.items():
            
            r = []

            for t in range(len(outcome)):

                j = outcome[t]

                if j["label"] == "Start":
                    
                    j["Z"+i] = 0

                    j["path"] = [0]

                    r.append(j)

                elif j["label"] != "Start" and j["from"] == "Start":

                    payoff = int(j["label"][0] + j["label"][2:])

                    j["Z"+i] = payoff

                    j["path"] = [0, int(j["label"][0] + j["label"][2:])]

                    r.append(j)

                elif j["label"] != "Start" and j["from"] != "Start" and "parent" not in j.keys():

                    payoff = int(j["label"][0] + j["label"][2:]) + int(j["from"][0] + j["from"][2:])

                    j["Z"+i] = payoff

                    j["path"] = [0, int(j["from"][0] + j["from"][2:]), int(j["label"][0] + j["label"][2:])]

                    r.append(j)

                elif j["label"] != "Start" and j["from"] != "Start" and j["parent"] != "Start":

                    payoff = int(j["label"][0] + j["label"][2:]) + int(j["from"][0] + j["from"][2:]) + int(j["parent"][0] + j["parent"][2:])

                    j["Z"+i] = payoff

                    j["path"] = [0, int(j["parent"][0] + j["parent"][2:]), int(j["from"][0] + j["from"][2:]), int(j["label"][0] + j["label"][2:])]

                    r.append(j)
                
            d_update[i] = r
            
        v["periods"] = d_update

    return lotteries


lotteries_v2 = cumulative_transform(lotteries)



def rp(period=0, delta=delta, weights=weights, label="Start", fr=None, parent=None, lottery = "lottery_1", lotteries=lotteries_v2):
    
    z = []

    d = lotteries[lottery]["periods"]

    d_period = d[str(period)]

    z = d_period["path"]


print(rp())

