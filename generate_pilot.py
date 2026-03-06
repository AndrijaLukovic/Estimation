"""
Augment existing pilot data by adding stochastic turbulence.

Each original subject is replicated `n_copies` times; each copy's CE values
are perturbed by Gaussian noise scaled to the lottery spread:
    ce_perturbed = ce_observed + N(0, turbulence * spread)

Usage:
    python generate_pilot.py                   # 5 copies, turbulence=0.03
    python generate_pilot.py 10 0.05           # 10 copies, turbulence=0.05
    from generate_pilot import augment_data    # use programmatically
"""
import numpy as np
import pandas as pd
import functions as f
from lotteries import lotteries_full
from main import get_observed_ce


def augment_data(
    n_copies: int = 5,
    turbulence: float = 0.03,   # noise scale as a fraction of lottery spread
    y: pd.DataFrame = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Augment the real observed CE data with stochastic turbulence.

    For each original subject, creates `n_copies` perturbed copies where:
        ce_perturbed = ce_observed + N(0, turbulence * spread)

    Parameters
    ----------
    n_copies    : number of perturbed copies per original subject
    turbulence  : noise scale as a fraction of lottery spread
    y           : observed CE DataFrame; loads from pilot.csv if None
    seed        : random seed

    Returns
    -------
    pd.DataFrame with columns [participant_label, lottery_id, ce_observed],
    containing original subjects plus all perturbed copies.
    """
    rng = np.random.default_rng(seed)

    if y is None:
        y = get_observed_ce(export_excel=False)

    lotteries = f.transform(lotteries_full)
    spreads   = {lid: lotteries[lid]["spread"] for lid in lotteries}

    copies = [y.copy()]  # start with the originals

    for k in range(n_copies):
        perturbed = y.copy()
        perturbed["participant_label"] = perturbed["participant_label"] + f"_copy{k+1}"
        perturbed["ce_observed"] = perturbed.apply(
            lambda row: row["ce_observed"] + rng.normal(
                0, turbulence * spreads.get(row["lottery_id"], 1)
            ),
            axis=1,
        ).round(4)
        copies.append(perturbed)

    return pd.concat(copies, ignore_index=True)


if __name__ == "__main__":
    import sys

    n_copies   = int(float(sys.argv[1]))   if len(sys.argv) > 1 else 5
    turbulence = float(sys.argv[2])        if len(sys.argv) > 2 else 0.03
    out_path   = "augmented_pilot.csv"

    df = augment_data(n_copies=n_copies, turbulence=turbulence)
    df.to_csv(out_path, index=False)

    n_orig = df["participant_label"].str.endswith(tuple(f"_copy{k+1}" for k in range(n_copies)))
    print(f"Copies      : {n_copies}  (turbulence={turbulence})")
    print(f"Subjects    : {df['participant_label'].nunique()}  "
          f"(original + {n_copies} perturbed copies each)")
    print(f"Observations: {len(df)}")
    print(f"Saved to    : {out_path}")
