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
from lotteries import lotteries_full, test_lotteries, para_recov


# ── TRUE STRUCTURAL PARAMETERS ────────────────────────────────────────────────
TRUE_PARAMS = {
    "r":      0.03,
    "alpha":  0.88,
    "lamb":   2.25,
    "gamma":  0.61,
    "beta":   1.0,
    "palpha": 0.65,
    "w":      0,    # forward-looking fraction: R_l = w * E[L_l]
    # Phase 2+: replace w with a1, a2, a3 decomposition
}

# ── KSI VALUES ────────────────────────────────────────────────────────────────
# Provide either:
#   a dict  {participant_label: ksi}   — you choose subject names
#   a list  [ksi_1, ksi_2, ...]        — subjects are named sub_1, sub_2, …

NUM_SUBJECTS  = 150
KSI_VALUES = {f"sub_{i}": max(1e-4, np.random.normal(0.19,0.106)) for i in range(1, NUM_SUBJECTS+1)}


# ── SETTINGS ──────────────────────────────────────────────────────────────────
METHOD   = "prelec"            
LOTTERY  = para_recov #lotteries_full
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


def generate_pseudo_data_multisession(
    true_params=None,
    ksi_values=None,
    method="prelec",
    lottery=None,
    seed=42,
):
    """
    Generate pseudo CE data for 3 sessions (round_numbers 1, 15, 16).

    Session 1 (round_number=1):  Z_t=0, no realized labels.
    Session 2 (round_number=15): Z_1 realized; realized_period1_label set.
    Session 3 (round_number=16): Z_1+Z_2 realized; both labels set.

    true_params must include:
        r, alpha, lamb,
        gamma (TK) OR beta/palpha (Prelec),
        a1, a2, a3, delta  — composite reference-point parameters.

    Z_1 and Z_2 are independently sampled from the lottery's path distribution
    (representing the total payoff from each session's play of the lottery).
    Labels are stored as "+£X" / "-£X" strings, matching _parse_label() format.
    """
    if true_params is None:
        true_params = TRUE_PARAMS
    if ksi_values is None:
        ksi_values = KSI_VALUES
    if lottery is None:
        lottery = para_recov

    # Normalise ksi_values to dict
    if isinstance(ksi_values, (list, np.ndarray)):
        ksi_values = {f"sub_{i+1}": float(k) for i, k in enumerate(ksi_values)}

    rng = np.random.default_rng(seed)
    p   = true_params

    # Reference-point decomposition parameters
    a1    = p.get("a1",    0.0)
    a2    = p.get("a2",    0.0)
    a3    = p.get("a3",    0.0)
    delta = p.get("delta", 1.0)

    lotteries_t = f.transform(lottery)
    spreads     = {lid: lotteries_t[lid]["spread"] for lid in lotteries_t}

    # ── Pre-compute outcome distribution per lottery for path sampling ──────────
    # Each lottery path is {p: [0, z1, z2, ...]}; total payoff = sum(stream[1:])
    lottery_dist = {}
    for lid, lot in lotteries_t.items():
        payoffs_list, probs_list = [], []
        for path_dict in lot["outcomes"].values():
            prob   = float(next(iter(path_dict.keys())))
            stream = next(iter(path_dict.values()))
            payoffs_list.append(sum(stream[1:]))   # skip the t=0 zero
            probs_list.append(prob)
        # Normalise probabilities to guard against floating-point rounding
        total = sum(probs_list)
        lottery_dist[lid] = (
            payoffs_list,
            [q / total for q in probs_list],
        )

    def _make_label(z):
        """Format a total payoff as a '+£X' / '-£X' label for _parse_label()."""
        z_int = int(round(z))
        return f"+£{z_int}" if z_int >= 0 else f"-£{abs(z_int)}"

    def _ce_for_session(lid, lot, t, Z_seq, EL_seq, Z_t):
        """
        Theoretical CE for one (lottery, session) combination.
        Builds the composite reference point from the session history,
        evaluates the lottery, and subtracts the accumulated payoff Z_t.
        """
        EL  = f.expected_payoff(lot["outcomes"])
        RSQ = f.status_quo(0.0)
        RA  = f.partial_adaptation(t, Z_seq, delta)
        RLE = f.lagged_expectation(t, EL_seq, 0.0, delta)
        RFE = f.forward_looking(EL)
        R_l = f.composite(a1, a2, a3, RSQ, RA, RLE, RFE)
        ev  = f.evaluation(
            r=p["r"], R=R_l, alpha=p["alpha"], lamb=p["lamb"],
            gamma=p.get("gamma", 0.61),
            lotteries={lid: lot}, method=method,
            beta=p.get("beta", 1.0), palpha=p.get("palpha", 1.0),
        )
        return f.u_inv(ev[lid]["V"], R_l, p["alpha"], p["lamb"]) - Z_t

    rows = []
    for subj, ksi in ksi_values.items():

        # Sample per-lottery realized payoffs for sessions 1 and 2.
        # Z_1 and Z_2 are total payoffs from two independent plays of the lottery.
        realized_z1 = {}
        realized_z2 = {}
        for lid in lotteries_t:
            payoffs_list, probs_list = lottery_dist[lid]
            idx1 = rng.choice(len(payoffs_list), p=probs_list)
            idx2 = rng.choice(len(payoffs_list), p=probs_list)
            realized_z1[lid] = payoffs_list[idx1]
            realized_z2[lid] = payoffs_list[idx2]

        for lid, lot in lotteries_t.items():
            EL     = f.expected_payoff(lot["outcomes"])
            spread = spreads[lid]
            Z1     = realized_z1[lid]
            Z2     = realized_z2[lid]
            lbl1   = _make_label(Z1)
            lbl2   = _make_label(Z2)

            # ── Session 1 (round_number = 1) ──────────────────────────────────
            # No history; only the forward-looking component is active.
            ce_s1 = _ce_for_session(
                lid, lot,
                t=0, Z_seq=[0.0], EL_seq=[], Z_t=0.0,
            )
            rows.append({
                "participant_label":      subj,
                "lottery_id":             lid,
                "round_number":           1,
                "ce_observed":            round(ce_s1 + rng.normal(0, ksi * spread), 4),
                "realized_period1_label": None,
                "realized_period2_label": None,
            })

            # ── Session 2 (round_number = 15) ─────────────────────────────────
            # Session-1 payoff Z_1 is known; reference point updated accordingly.
            ce_s2 = _ce_for_session(
                lid, lot,
                t=1, Z_seq=[0.0, float(Z1)], EL_seq=[EL], Z_t=float(Z1),
            )
            rows.append({
                "participant_label":      subj,
                "lottery_id":             lid,
                "round_number":           15,
                "ce_observed":            round(ce_s2 + rng.normal(0, ksi * spread), 4),
                "realized_period1_label": lbl1,
                "realized_period2_label": None,
            })

            # ── Session 3 (round_number = 16) ─────────────────────────────────
            # Both Z_1 and Z_2 are known; reference point uses full two-session history.
            ce_s3 = _ce_for_session(
                lid, lot,
                t=2, Z_seq=[0.0, float(Z1), float(Z2)], EL_seq=[EL, EL],
                Z_t=float(Z1) + float(Z2),
            )
            rows.append({
                "participant_label":      subj,
                "lottery_id":             lid,
                "round_number":           16,
                "ce_observed":            round(ce_s3 + rng.normal(0, ksi * spread), 4),
                "realized_period1_label": lbl1,
                "realized_period2_label": lbl2,
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
