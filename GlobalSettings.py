### Gloal Settings That Affects All Processings


import lotteries
import os
import numpy as np

GlobalMethod = "prelec"
GlobalLottery = lotteries.lotteries_v2

# Robustness check: low-stake only (spread < 200)
# Drops: lottery_2 (515), lottery_7 (600), cali_mix (800), cali_mix_pres (800), lottery_8 (1660)
_HIGH_STAKE_IDS = {'lottery_2', 'lottery_7', 'cali_mix', 'cali_mix_pres', 'lottery_8'}
GlobalLotteryLowStake = {k: v for k, v in lotteries.lotteries_v2.items()
                         if k not in _HIGH_STAKE_IDS}
GlobalCluster = 3
GlobalTol = 1e-4
GlobalInterMax = 1000

# Bounds for different methods

GlobalTKBounds = [
        (0.0, 0.0), (0.5, 1.5), (1.0, 3.0), (0.2, 1.0),  # r fixed at 0
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    ]
GlobalPrelecBounds = [
        (1e-6, 1e-4), (0.5, 1.5), (1, 3), (1.0, 1.0), (0.3, 1),  # r fixed at 0
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    ]


GlobalNSeeds    = 100
GlobalMasterSeed = 42
GlobalSeedsSet  = np.random.default_rng(GlobalMasterSeed).integers(0, 10000, size=GlobalNSeeds).tolist()

# Bounds for individual noise parameter ksi  (sigma_i = ksi_i * spread)
GlobalKsiBounds = (1e-4, 3)

# For two step methods
# If it is true, then in Step 2, it allows the change of clusters. If it is false, it calculates different reference points given clusters estimated from Step 1.
GlobalFixTau = False