import functions as f
from data import process


# Some fixed parameters
r, alpha, lamb, gamma, R, desired = 0.97, 0.88, 2.25, 0.61, 0, "lottery_3" # What does "desired" mean?


data, data_period1, data_period2, data_period3 = process()


if "ce_observed" in data_period1.columns:
    y = data_period1[["lottery_id", "ce_observed"]].copy()
elif {"selected_amount", "cutoff_amount"}.issubset(data_period1.columns):
    y = data_period1[["lottery_id"]].copy()
    y["ce_observed"] = data_period1[["selected_amount", "cutoff_amount"]].mean(axis=1)
else:
    y = data_period1[["lottery_id", "selected_amount"]].copy()
    y.rename(columns={"selected_amount": "ce_observed"}, inplace=True)


if __name__ == "__main__":

    print(y[y["lottery_id"] == "lottery_1"]["ce_observed"])

    print(type(y))

    print(y["ce_observed"] + 3)
