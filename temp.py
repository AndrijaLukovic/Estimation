from lotteries import one, lotteries


# Params a1, a2, a3, defining the convex linear combination of reference points

weights = [0.25, 0.25, 0.25, 0.25]

# Delta entering partial adaptation

delta = 1

r = 0.97



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

print(cumulative_transform(one)["lottery_1"]["periods"])


def ev_lotteries(r=r, lotteries=lotteries_v2):

    # Calculate the expected value at time zero for all lotteries

    lotteries_new = {}

    for k, v in lotteries.items():

        dict_temp = v

        m = dict_temp["periods"]["3"]

        s = 0

        for d in m:

            path = d["path"]

            p = d["abs_prob"]

            s = s + sum(path[i]*(r**i) for i in range(len(path)))*p

        # dict_temp["EV at 0"] = s

        lotteries_new[k] = {"EV at 0":s}

    for k, v in lotteries.items():

        dict_temp = v

    return lotteries_new


print(ev_lotteries(r, lotteries=cumulative_transform(one)))


def rp(period=0, delta=delta, weights=weights, label="Start", fr=None, parent=None, lottery = "lottery_1", lotteries=lotteries_v2):
    
    z = []

    d = lotteries[lottery]["periods"]

    d_period = d[str(period)]

    # z = d_period["path"]




# print(rp())



print(cumulative_transform(one)["lottery_1"]["periods"]["3"])


def ev_by_branch(r=r, lotteries=lotteries_v2):
    """
    For each lottery, compute the discounted EV contribution and conditional EV
    for each top-level (period-1) branch, identified via the 'parent' field
    on terminal period-3 nodes.

    Returns:
        {
          lottery_key: {
            "name": ...,
            "total_ev": ...,
            "branches": {
              branch_label: {
                "branch_prob": ...,
                "ev_contribution": ...,   # weighted contribution to total EV
                "conditional_ev": ...     # EV conditional on being in this branch
              }
            }
          }
        }
    """
    results = {}

    for lottery_key, lottery_data in lotteries.items():
        periods = lottery_data["periods"]
        last_period = str(max(int(k) for k in periods.keys()))
        final_outcomes = periods[last_period]

        # Gather branch prob and EV contribution keyed by period-1 label (parent)
        branch_data = {}
        for outcome in final_outcomes:
            path = outcome["path"]
            p = outcome["abs_prob"]
            branch_label = outcome.get("parent", outcome.get("from"))

            discounted = sum(path[i] * (r ** i) for i in range(len(path))) * p

            if branch_label not in branch_data:
                branch_data[branch_label] = {"branch_prob": 0.0, "ev_contribution": 0.0}
            branch_data[branch_label]["branch_prob"] += p
            branch_data[branch_label]["ev_contribution"] += discounted

        # Compute conditional EV = ev_contribution / branch_prob
        for _, vals in branch_data.items():
            bp = vals["branch_prob"]
            vals["conditional_ev"] = vals["ev_contribution"] / bp if bp > 0 else 0.0

        results[lottery_key] = {
            "name": lottery_data["name"],
            "total_ev": sum(v["ev_contribution"] for v in branch_data.values()),
            "branches": branch_data,
        }

    return results


ev_branches = ev_by_branch(r, lotteries=cumulative_transform(lotteries))

for lottery_key, data in ev_branches.items():
    print(f"\n{lottery_key} ({data['name']})  |  Total EV: {data['total_ev']:.4f}")
    for branch_label, vals in data["branches"].items():
        print(
            f"  Branch [{branch_label}]"
            f"  prob={vals['branch_prob']:.3f}"
            f"  EV contribution={vals['ev_contribution']:.4f}"
            f"  conditional EV={vals['conditional_ev']:.4f}"
        )

