"""Lottery trees as payoff streams, and the lottery remaining after a history."""
import numpy as np


def _parse_payoff(label):
    return int(label.replace('£', ''))


def transform(lotteries):
    """Lottery trees -> {id: {name, spread, outcomes}} with outcomes
    {j: {absolute path probability: [0, z1, z2, ..., terminal]}}."""
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


def conditional_lottery(lottery, t=0, Z1=0, Z2=0):
    """Current remaining lottery, conditional on the observed payment history.

    Input paths use absolute probabilities and streams [0, z1, z2, ...].
    Keep compatible paths, normalize their probabilities, and return streams
    [0, next_payment, ...] so the next payment is one period from *now*.
    The original spread is retained as the established observation-noise scale.
    """
    if int(t) != t or t not in (0, 1, 2):
        raise ValueError(f"Unsupported decision stage: {t}")
    t = int(t)
    matches = []
    for path in lottery["outcomes"].values():
        if len(path) != 1:
            raise ValueError("conditional_lottery expects transform() paths")
        probability, stream = next(iter(path.items()))
        if probability < 0 or not np.isfinite(probability):
            raise ValueError("Invalid absolute path probability")
        if t >= 1 and (len(stream) < 2 or stream[1] != Z1):
            continue
        if t >= 2 and (len(stream) < 3 or stream[2] != Z2):
            continue
        matches.append((float(probability), [0] + list(stream[t + 1:])))
    mass = sum(probability for probability, _ in matches)
    if mass <= 0:
        raise ValueError(f"No positive-probability path for t={t}, Z1={Z1}, Z2={Z2}")
    if t == 0 and not np.isclose(mass, 1.0, atol=1e-12):
        raise ValueError(f"Full-lottery path probabilities sum to {mass}")
    if t == 0:
        return dict(lottery)
    return {**lottery, "outcomes": {
        index: {probability / mass: stream}
        for index, (probability, stream) in enumerate(matches)
    }}


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


def conditional_el(lottery_outcomes, Z1, Z2, t):
    """
    Expected payoff of the sub-lottery remaining from session t onward,
    conditional on the realised path.

    t=0 : E[z1+z2+z3]               (unconditional, full lottery)
    t=1 : E[z2+z3  | period-1 = Z1]
    t=2 : E[z3     | period-1 = Z1, period-2 = Z2]

    lottery_outcomes : 'outcomes' dict from transform(), entries {j: {p: [0,z1,z2,...]}}.
    """
    if t == 0:
        return expected_payoff(lottery_outcomes)

    matched_p, matched_r = [], []
    for path in lottery_outcomes.values():
        p      = next(iter(path.keys()))
        stream = next(iter(path.values()))   # [0, z1, z2, ...]
        if t == 1:
            if len(stream) >= 2 and stream[1] == Z1:
                matched_p.append(p)
                matched_r.append(sum(stream[2:]))
        elif t == 2:
            if len(stream) >= 3 and stream[1] == Z1 and stream[2] == Z2:
                matched_p.append(p)
                matched_r.append(stream[3] if len(stream) > 3 else 0.0)

    if not matched_p:
        return expected_payoff(lottery_outcomes)   # fallback

    denom = sum(matched_p)
    return sum(pi / denom * ri for pi, ri in zip(matched_p, matched_r))
