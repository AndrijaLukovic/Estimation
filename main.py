from data import process


# Some fixed parameters
r, alpha, lamb, gamma, R, desired = 0.97, 0.88, 2.25, 0.61, 0, "lottery_3" # What does "desired" mean?


def get_observed_ce(export_excel=False):
    """
    Build observed CE choice used by MLE from period-1 responses.
    """
    data, data_period1, data_period2, data_period3 = process(export_excel=export_excel)

    if "ce_observed" in data_period1.columns:
        y = data_period1[["participant_label", "lottery_id", "ce_observed"]].copy()
    elif {"selected_amount", "cutoff_amount"}.issubset(data_period1.columns):
        y = data_period1[["participant_label", "lottery_id"]].copy()
        y["ce_observed"] = data_period1[["selected_amount", "cutoff_amount"]].mean(axis=1)
    else:
        y = data_period1[["participant_label", "lottery_id", "selected_amount"]].copy()
        y.rename(columns={"selected_amount": "ce_observed"}, inplace=True)

    return y


if __name__ == "__main__":
    y = get_observed_ce(export_excel=True)

    print(y[y["lottery_id"] == "lottery_1"]["ce_observed"])

    print(type(y))

    print(y["ce_observed"] + 3)
