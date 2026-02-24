import functions as f
from data import data_period1


# Some fixed parameters
r, alpha, lamb, gamma, R, desired = 0.97, 0.88, 2.25, 0.61, 0, "lottery_3"



y = data_period1[["lottery_id","selected_amount"]].copy()

print(y)