### Gloal Settings That Affects All Processings


import lotteries
import os

GlobalMethod = "prelec"
GlobalLottery = lotteries.lotteries_v2
GlobalCluster = 3
GlobalStarts = 8
GlobalTol = 1e-4
GlobalInterMax = 500

# Bounds for different methods

GlobalTKBounds = [
        (1e-4, 0.2), (0.5, 1.5), (1.0, 3.0), (0.2, 1.0),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    ]
GlobalPrelecBounds = [
        (1e-4, 0.2), (0.5, 1.5), (1.0, 3.0), (1.0, 1.0), (0.1, 0.8),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    ]


GlobalSeedsSet = [10, 25, 30, 45, 78, 29, 16, 23, 37, 50, 55, 60, 67, 80, 89]