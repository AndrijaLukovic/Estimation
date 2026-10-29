"""Observed certainty equivalents: load and clean the experiment data, filter samples."""
import numpy as np
import pandas as pd

from .config import DATA_FILE
from .prepare import prepare_grid_data

DROP_COLUMNS = [
    "session_code", "participant_code", "lottery_stake", "num_failed_attempts", "failed_too_many_1",
    "failed_too_many_2", "failed_too_many_3", "quiz1", "quiz2", "quiz3", "quiz4", "quiz5", "quiz6",
    "quiz7", "quiz8", "participant_time_started_utc", "session2_start", "session2_start_readable",
    "session3_start", "session3_start_readable", *(f"chf_{i}" for i in range(1, 21)), "selected_option",
]
OBSERVED_COLUMNS = ["participant_label", "lottery_id", "round_number", "ce_observed",
                    "realized_period1_label", "realized_period2_label", "selected_amount_effective"]


def load_cleaned(path=DATA_FILE):
    """Raw experiment rows of participants who completed the questionnaire and
    did not report the task as (very/too) complicated, with effective choices
    (refined values when present) and ce_observed = mean(selected, cutoff)."""
    data = pd.read_csv(
        path,
        sep=",",
        skipinitialspace=True,
        encoding="utf-8-sig",
        low_memory=False,
    )
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()

    data.dropna(axis=0, how="any", subset=["participant_label", "realized_period1_label"], inplace=True)

    # quiz6 is filled in one row per participant; a participant is incomplete
    # if they have no row at all where quiz6 is non-null.
    has_quiz6 = data.groupby("participant_label")["quiz6"].apply(lambda x: x.notna().any())
    incomplete_subjects = has_quiz6[~has_quiz6].index
    data = data[~data["participant_label"].isin(incomplete_subjects)]

    to_drop_subjects = data.loc[
        data["quiz6"].isin(["Very complicated", "Too complicated or extremely complicated"]),
        "participant_label",
    ].unique()
    data = data[~data["participant_label"].isin(to_drop_subjects)]

    for name in ["choice", "amount"]:
        coarse = f"selected_{name}"
        if {f"fine_{coarse}", coarse}.issubset(data.columns):
            data[f"{coarse}_effective"] = data[f"fine_{coarse}"].combine_first(data[coarse])
        elif coarse in data.columns:
            data[f"{coarse}_effective"] = data[coarse]

    if {"fine_cutoff_amount", "cutoff_amount"}.issubset(data.columns):
        data["cutoff_amount_effective"] = data["fine_cutoff_amount"].combine_first(data["cutoff_amount"])
    elif "cutoff_amount" in data.columns:
        data["cutoff_amount_effective"] = data["cutoff_amount"]

    if {"selected_amount_effective", "cutoff_amount_effective"}.issubset(data.columns):
        data["ce_observed"] = data[["selected_amount_effective", "cutoff_amount_effective"]].mean(axis=1)

    return data.drop(DROP_COLUMNS, axis=1, errors="ignore")


def observed_ce(path=DATA_FILE):
    """One row per elicited CE with its realised branch labels."""
    return load_cleaned(path)[OBSERVED_COLUMNS].copy()


def endpoint_shares(observed, lotteries):
    """Share of each participant's CEs at the top end of the choice list: the
    effective selected amount equals the best remaining payoff. List lengths
    differ across lotteries, so the row number is not used."""
    active = observed[observed.lottery_id.isin(lotteries)].dropna(subset=["ce_observed"])
    data = prepare_grid_data(observed, lotteries)
    chosen = active.selected_amount_effective.to_numpy(float)
    payoff = data.streams.sum(axis=2)
    best = np.where(data.path_mask, payoff, -np.inf).max(axis=1)[data.state_index]
    if not np.allclose(active.ce_observed.to_numpy(), data.observed):
        raise ValueError("Raw rows and prepared CE rows are misaligned")
    return pd.Series(np.isclose(chosen, best)).groupby(active.participant_label.to_numpy()).mean()


def exclude_participants(observed, participants):
    return observed[~observed.participant_label.isin(participants)]
