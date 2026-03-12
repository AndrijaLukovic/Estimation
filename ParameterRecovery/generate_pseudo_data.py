"""
Generate a pseudo CE dataset for parameter recovery.

Edit the TRUE_PARAMS, KSI_VALUES, METHOD, and LOTTERY blocks below,
then run:
    python generate_pseudo_data.py

All observations are assigned round_number=1 (session 1, Z_t = 0).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import functions as f
from lotteries import lotteries_full, test_lotteries


# ── TRUE STRUCTURAL PARAMETERS ────────────────────────────────────────────────
TRUE_PARAMS = {
    "r":      0.03,
    "alpha":  0.88,
    "lamb":   2.25,
    "gamma":  0.61,
    "beta":   1.0,
    "palpha": 0.65,
    "w":      1.0,    # forward-looking fraction: R_l = w * E[L_l]
    # Phase 2+: replace w with a1, a2, a3 decomposition
}

# ── KSI VALUES ────────────────────────────────────────────────────────────────
# Provide either:
#   a dict  {participant_label: ksi}   — you choose subject names
#   a list  [ksi_1, ksi_2, ...]        — subjects are named sub_1, sub_2, …

NUM_SUBJECTS  = 150
KSI_VALUES = {f"sub_{i}": np.random.normal(0.4,0.1) for i in range(1, NUM_SUBJECTS+1)}


# ── SETTINGS ──────────────────────────────────────────────────────────────────
METHOD   = "tk"            
LOTTERY  = test_lotteries #lotteries_full
ALL_SEEDS = [10,15,20,25]
SEED     = 42
_HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "pseudo_data.csv")
# ─────────────────────────────────────────────────────────────────────────────


def generate_pseudo_data(
    true_params=None,
    ksi_values=None,
    method="tk",
    lottery=None,
    seed=42,
):
    if true_params is None:
        true_params = TRUE_PARAMS
    if ksi_values is None:
        ksi_values = KSI_VALUES
    if lottery is None:
        lottery = lotteries_full

    # Normalise to dict
    if isinstance(ksi_values, (list, np.ndarray)):
        ksi_values = {f"sub_{i+1}": float(k) for i, k in enumerate(ksi_values)}

    rng = np.random.default_rng(seed)
    p   = true_params

    lotteries_t = f.transform(lottery)
    spreads     = {lid: lotteries_t[lid]["spread"] for lid in lotteries_t}

    # Phase 1: R_l = w * E[L_l] per lottery
    base_ce = {}
    for lid, lot in lotteries_t.items():
        EL_0 = f.expected_payoff(lot["outcomes"])
        R_l  = p["w"] * EL_0
        ev   = f.evaluation(r=p["r"], R=R_l, alpha=p["alpha"], lamb=p["lamb"],
                            gamma=p["gamma"], lotteries={lid: lot}, method=method,
                            beta=p["beta"], palpha=p["palpha"])
        base_ce[lid] = f.u_inv(ev[lid]["V"], R_l, p["alpha"], p["lamb"])

    rows = []
    for subj, ksi in ksi_values.items():
        for lid in lotteries_t:
            sigma = ksi * spreads[lid]
            noise = rng.normal(0, sigma)
            rows.append({
                "participant_label":      subj,
                "lottery_id":             lid,
                "round_number":           1,     
                "ce_observed":            round(base_ce[lid] + noise, 4),
                "realized_period1_label": None,
                "realized_period2_label": None,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_pseudo_data(
        true_params=TRUE_PARAMS,
        ksi_values=KSI_VALUES,
        method=METHOD,
        lottery=LOTTERY,
        seed=SEED,
    )

    # Save pseudo CE data
    df.to_csv(OUT_PATH, index=False)

    # Save true parameters to a separate CSV for reference during recovery
    params_path = OUT_PATH.replace(".csv", "_true_params.csv")
    struct_rows = [{"type": "structural", "name": k, "value": v}
                   for k, v in TRUE_PARAMS.items()]
    ksi_rows    = [{"type": "ksi", "name": subj, "value": ksi}
                   for subj, ksi in KSI_VALUES.items()]
    method_row  = [{"type": "method", "name": "method", "value": METHOD}]
    pd.DataFrame(struct_rows + ksi_rows + method_row).to_csv(params_path, index=False)

    print("=== PSEUDO DATASET GENERATED ===")
    print(f"Method       : {METHOD}")
    print(f"True params  :")
    for k, v in TRUE_PARAMS.items():
        if METHOD == "tk" and k in ("beta", "palpha"):
            continue
        if METHOD == "prelec" and k == "gamma":
            continue
        print(f"  {k:8s} = {v}")
    print(f"\nSubjects     : {df['participant_label'].nunique()}")
    print(f"Lotteries    : {df['lottery_id'].nunique()}")
    print(f"Observations : {len(df)}")
    print(f"\nKsi (true):")
    for subj, ksi in KSI_VALUES.items():
        print(f"  {subj:12s}: {ksi}")
    print(f"\nSaved to : {OUT_PATH}")
    print("\nSample rows:")
    print(df.head(6).to_string(index=False))
