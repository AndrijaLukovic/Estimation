### Gloal Settings That Affects All Processings


import lotteries
import os

GlobalMethod = "prelec"
GlobalLottery = lotteries.para_recov
GlobalCluster = 2
GlobalStarts = 8
GlobalTol = 1e-4
GlobalInterMax = 100

# Bounds for different methods

GlobalTKBounds = [
        (1e-4, 0.2), (0.5, 1.5), (1.0, 3.0), (0.2, 1.0),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    ]
GlobalPrelecBounds = [
        (1e-4, 0.2), (0.5, 1.5), (1.0, 3.0), (1.0, 1.0), (0.1, 0.8),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    ]