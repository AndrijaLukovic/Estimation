from data import process


# Some fixed parameters
r, alpha, lamb, gamma, R, desired = 0.97, 0.88, 2.25, 0.61, 0, "lottery_3" # What does "desired" mean?


def get_observed_ce(export_excel=False):
    """
    Return participant-level observed CEs with their realised branch labels.
    CE computation lives in data.process().
    """
    data, _, _, _ = process(export_excel=export_excel)

    return data[["participant_label", "lottery_id", "round_number", "ce_observed", "realized_period1_label", "realized_period2_label"]].copy()


if __name__ == "__main__":
    y = get_observed_ce(export_excel=True)

    print(y[y["lottery_id"] == "lottery_1"]["ce_observed"])

    print(type(y))

    print(y["ce_observed"])
