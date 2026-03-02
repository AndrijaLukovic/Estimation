import pandas as pd


to_drop = ["session_code", "participant_code", "participant_label", "lottery_stake", "num_failed_attempts", "failed_too_many_1", "failed_too_many_2", 
        "failed_too_many_3", "quiz1", "quiz2", "quiz3", "quiz4", "quiz5", "quiz6", "quiz7", "quiz8", "participant_time_started_utc", "session2_start",
        "session2_start_readable", "session3_start", "session3_start_readable", "chf_1", "chf_2", "chf_3", "chf_4", 'chf_5', 'chf_6', 'chf_7', 'chf_8',
        'chf_9', 'chf_10', 'chf_11', 'chf_12', 'chf_13', 'chf_14', 'chf_15', 'chf_16', 'chf_17', 'chf_18', 'chf_19', 'chf_20', "selected_option"]


def process():
# XG: i will updeate this function so that: (1) it returns the average of the selected and cutoff, (2) it returns the refined choice, if there is one.
    data = pd.read_csv("pilot.csv")

    data.dropna(axis=0, how="any", subset=["participant_label", "realized_period1_label"], inplace=True)

    # Use refined values when present; otherwise fall back to coarse values.
    if {"fine_selected_choice", "selected_choice"}.issubset(data.columns):
        data["selected_choice_effective"] = data["fine_selected_choice"].combine_first(data["selected_choice"])
    elif "selected_choice" in data.columns:
        data["selected_choice_effective"] = data["selected_choice"]

    if {"fine_selected_amount", "selected_amount"}.issubset(data.columns):
        data["selected_amount_effective"] = data["fine_selected_amount"].combine_first(data["selected_amount"])
    elif "selected_amount" in data.columns:
        data["selected_amount_effective"] = data["selected_amount"]

    if {"fine_cutoff_amount", "cutoff_amount"}.issubset(data.columns):
        data["cutoff_amount_effective"] = data["fine_cutoff_amount"].combine_first(data["cutoff_amount"])
    elif "cutoff_amount" in data.columns:
        data["cutoff_amount_effective"] = data["cutoff_amount"]

    if {"selected_amount_effective", "cutoff_amount_effective"}.issubset(data.columns):
        data["ce_observed"] = data[["selected_amount_effective", "cutoff_amount_effective"]].mean(axis=1)

    data.to_excel("pilot.xlsx", sheet_name="v1")

    data.drop(to_drop, inplace=True, axis=1, errors="ignore")

    data_period1,  data_period2, data_period3 = data[data["round_number"] < 15], data[data["round_number"] == 15], data[data["round_number"] == 16]


    with pd.ExcelWriter("pilot.xlsx", mode="a") as writer:
        data.to_excel(writer, sheet_name="v2")

        data_period1.to_excel(writer, sheet_name="period1")

        data_period2.to_excel(writer, sheet_name="period2")

        data_period3.to_excel(writer, sheet_name="period3")


    return data, data_period1, data_period2, data_period3


if __name__ == "__main__":

    data, data_period1, data_period2, data_period3 = process()


