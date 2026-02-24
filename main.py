import functions as f
from data import process


# Some fixed parameters
r, alpha, lamb, gamma, R, desired = 0.97, 0.88, 2.25, 0.61, 0, "lottery_3"


data, data_period1, data_period2, data_period3 = process()


y = data_period1[["lottery_id","selected_amount"]].copy()


if __name__ == "__main__":

    print(y[y["lottery_id"] == "lottery_1"]["selected_amount"])