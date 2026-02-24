import pandas as pd


to_drop = ["session_code", "participant_code", "participant_label", "lottery_stake", "num_failed_attempts", "failed_too_many_1", "failed_too_many_2", 
        "failed_too_many_3", "quiz1", "quiz2", "quiz3", "quiz4", "quiz5", "quiz6", "quiz7", "quiz8", "participant_time_started_utc", "session2_start",
        "session2_start_readable", "session3_start", "session3_start_readable", "chf_1", "chf_2", "chf_3", "chf_4", 'chf_5', 'chf_6', 'chf_7', 'chf_8',
        'chf_9', 'chf_10', 'chf_11', 'chf_12', 'chf_13', 'chf_14', 'chf_15', 'chf_16', 'chf_17', 'chf_18', 'chf_19', 'chf_20', "selected_option"]


def process():

    data = pd.read_csv("pilot.csv")

    data.dropna(axis=0, how="any", subset=["participant_label", "realized_period1_label"], inplace=True)

    data.to_excel("pilot.xlsx", sheet_name="v1")

    data.drop(to_drop, inplace=True, axis=1)

    data_period1,  data_period2, data_period3 = data[data["round_number"] < 15], data[data["round_number"] == 15], data[data["round_number"] == 16]


    with pd.ExcelWriter("pilot.xlsx", mode="a") as writer:
        data.to_excel(writer, sheet_name="v2")

        data_period1.to_excel(writer, sheet_name="period1")

        data_period2.to_excel(writer, sheet_name="period2")

        data_period3.to_excel(writer, sheet_name="period3")


    return data, data_period1, data_period2, data_period3


if __name__ == "__main__":

    data, data_period1, data_period2, data_period3 = process()


