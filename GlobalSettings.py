### Gloal Settings That Affects All Processings


import lotteries
import os
import numpy as np
# Robustness check: low-stake only (spread < 200)
# Drops: lottery_2 (515), lottery_7 (600), cali_mix (800), cali_mix_pres (800), lottery_8 (1660)
_HIGH_STAKE_IDS = {'lottery_2', 'lottery_7', 'cali_mix', 'cali_mix_pres', 'lottery_8'}
GlobalLotteryLowStake = {k: v for k, v in lotteries.lotteries_v2.items()
                         if k not in _HIGH_STAKE_IDS}

# Robustness check: full set minus the two symmetric mixed calibration lotteries
# Drops: cali_mix (+700/-100, 50/50), cali_mix_pres (+700/-100, 40/20/40)
_NO_CALI_MIX_IDS = {'cali_mix', 'cali_mix_pres', 'lottery_8'}
GlobalLotteryNoCaliMix = {k: v for k, v in lotteries.lotteries_v2.items()
                          if k not in _NO_CALI_MIX_IDS}

GlobalMethod = "prelec"
#GlobalLottery = GlobalLotteryLowStake  
GlobalLottery =  GlobalLotteryNoCaliMix #lotteries.lotteries_v2

GlobalCluster = 3
GlobalTol = 1e-4
GlobalInterMax = 1000
GlobalData = "gamedata.csv"


# Bounds for different methods
# Parameter order: [r, alpha, lamb, gamma, a1, a2, a3, delta]
GlobalTKBounds = [
        (1e-6, 0.01), (0.5, 1.5), (0.99, 3.0), (0.2, 1.0),  
        (0, 1), (0.0, 1), (0.0, 1), (0.2, 1),
    ]

# Parameter order: [r, alpha, lamb, beta, palpha, a1, a2, a3, delta]
GlobalPrelecBounds = [
        (1e-6, 0.1), (0.5, 1.5), (0.99, 3.5), (1.0, 1.0), (0.4, 1),  
        (0, 1), (0.0, 1), (0.0, 1), (0.0, 1),
    ]


GlobalNSeeds    = 1000
GlobalMasterSeed = 42
GlobalSeedsSet  = np.random.default_rng(GlobalMasterSeed).integers(0, 10000, size=GlobalNSeeds).tolist()

# Bounds for individual noise parameter ksi  (sigma_i = ksi_i * spread)
GlobalKsiBounds = (1e-4, 1.5)

# For two step methods
# If it is true, then in Step 2, it allows the change of clusters. If it is false, it calculates different reference points given clusters estimated from Step 1.
GlobalFixTau = False

# R^SQ definition for the composite reference point:
#   "zt"   : R^SQ(t) = Z_t  (accumulated realized payoffs; a1 identified from sessions 2/3)
#   "zero" : R^SQ(t) = 0    (original behaviour; a1 only shrinks the forward-looking weight)
GlobalRSQMode = "zero"