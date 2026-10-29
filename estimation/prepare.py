"""Observed CEs -> arrays over unique conditional states, prepared once per sample.

A state is (lottery, stage t, Z1, Z2): rounds 1-16 are session 1 (t=0),
round 17 session 2 (t=1), round 18 session 3 (t=2).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lottery_paths import conditional_el, conditional_lottery, expected_payoff
from .valuation import predict_ce


@dataclass
class PreparedGridData:
    """Fixed conditional states and row indices of one estimation sample."""
    subjects: list
    states: list               # [(state key, conditional lottery)], one per state
    reference_inputs: tuple    # per state: (E[L_0], E[L_t|path], E[L_1|Z1], t, Z1, Z2, Zt)
    streams: np.ndarray        # (state, path, delay) payoffs
    probabilities: np.ndarray  # (state, path)
    path_mask: np.ndarray      # (state, path) True for real paths
    state_index: np.ndarray    # per row
    subject_index: np.ndarray  # per row
    spread: np.ndarray         # per row
    observed: np.ndarray       # per row
    counts: np.ndarray         # rows per subject

    def predict(self, theta, method, delta_le=None):
        return predict_ce(self, theta, method, delta_le)


def _payoff(column):
    return pd.to_numeric(
        column.str.replace("£", "", regex=False), errors="coerce"
    ).fillna(0).astype(int)


def prepare_grid_data(y, lotteries):
    """y: observed CEs (data.observed_ce()); lotteries: lottery_paths.transform() output."""
    active = y[y.lottery_id.isin(lotteries)].dropna(subset=["ce_observed"])
    subjects = sorted(active.participant_label.unique())
    subj_index = {s: i for i, s in enumerate(subjects)}

    rows = active.copy().reset_index(drop=True)
    rows["spread"] = rows["lottery_id"].map(
        {lid: lotteries[lid]["spread"] for lid in lotteries}
    )
    rows["_EL"] = rows["lottery_id"].map(
        {lid: expected_payoff(lotteries[lid]["outcomes"]) for lid in lotteries}
    )
    s2 = rows["round_number"] == 17
    s3 = rows["round_number"] == 18
    rows["_t"]  = s2.astype(int) + 2 * s3.astype(int)  # 0=s1, 1=s2, 2=s3
    rows["_Z1"] = np.where(s2 | s3, _payoff(rows["realized_period1_label"]), 0)
    rows["_Z2"] = np.where(s3,      _payoff(rows["realized_period2_label"]), 0)
    rows["_Zt"] = rows["_Z1"] + rows["_Z2"]
    rows["_si"] = rows["participant_label"].map(subj_index)
    rows["_ckey"] = list(zip(
        rows["lottery_id"], rows["_t"], rows["_Z1"], rows["_Z2"]
    ))

    # Conditional expectations for R^FE and R^LE, per unique state:
    #   _EL_fe : E[L_t | path]  (t=0: unconditional; t=1: E[z2+z3|Z1]; t=2: E[z3|Z1,Z2])
    #   _EL_c1 : E[L_1 | Z1]    (what was expected entering session 2; t=2 rows only)
    el_fe_cache = {}   # (lid, t, Z1, Z2) → E[L_t | path]
    el_c1_cache = {}   # (lid, Z1)        → E[L_1 | Z1]
    for key in rows["_ckey"].unique():
        lid, t_val, z1, z2 = key
        outcomes = lotteries[lid]["outcomes"]
        el_fe_cache[key] = conditional_el(outcomes, int(z1), int(z2), int(t_val))
        if int(t_val) == 2:
            c1_key = (lid, z1)
            if c1_key not in el_c1_cache:
                el_c1_cache[c1_key] = conditional_el(outcomes, int(z1), 0, 1)

    rows["_EL_fe"] = rows["_ckey"].map(el_fe_cache)
    rows["_EL_c1"] = 0.0
    mask_t2 = rows["_t"] == 2
    rows.loc[mask_t2, "_EL_c1"] = (
        rows.loc[mask_t2]
        .apply(lambda row: el_c1_cache.get((row["lottery_id"], row["_Z1"]), 0.0), axis=1)
        .values
    )

    groups = [(key, np.asarray(idx), conditional_lottery(
        lotteries[key[0]], key[1], key[2], key[3]))
        for key, idx in rows.groupby("_ckey", sort=False).groups.items()]
    first = np.array([indices[0] for _, indices, _ in groups])
    state_index = np.empty(len(rows), dtype=int)
    for index, (_, indices, _) in enumerate(groups):
        state_index[indices] = index

    max_paths = max(len(current["outcomes"]) for _, _, current in groups)
    max_delays = max(len(stream) for _, _, current in groups
                     for path in current["outcomes"].values() for stream in path.values())
    streams = np.zeros((len(groups), max_paths, max_delays))
    probabilities = np.zeros((len(groups), max_paths))
    mask = np.zeros((len(groups), max_paths), dtype=bool)
    for state, (_, _, current) in enumerate(groups):
        for path_index, path in enumerate(current["outcomes"].values()):
            probability, stream = next(iter(path.items()))
            streams[state, path_index, :len(stream)] = stream
            probabilities[state, path_index] = probability
            mask[state, path_index] = True

    reference_inputs = tuple(
        rows[column].values.astype(float)[first]
        for column in ["_EL", "_EL_fe", "_EL_c1", "_t", "_Z1", "_Z2", "_Zt"])
    si = rows["_si"].values
    return PreparedGridData(
        subjects=subjects, states=[(key, current) for key, _, current in groups],
        reference_inputs=reference_inputs, streams=streams, probabilities=probabilities,
        path_mask=mask, state_index=state_index, subject_index=si,
        spread=rows["spread"].values.astype(float),
        observed=rows["ce_observed"].values.astype(float),
        counts=np.bincount(si, minlength=len(subjects)))
