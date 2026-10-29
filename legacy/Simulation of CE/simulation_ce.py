"""
Identification check for the reference point R.

Lotteries are designed so that CE(R_true) is achievable at a second R value,
producing two likelihood peaks — demonstrating non-identification.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import argrelmax
from functions import u, u_inv, V

np.random.seed(42)

# ── True parameters ────────────────────────────────────────────────────────────
R_true = 8
alpha  = 0.88
lamb   = 2.25
gamma  = 0.61
ksi    = 0.1  # small noise so ce_obs stays close to ce_th

# ── Lotteries confirmed to give 2 CE(R) crossings at R_true=8 ─────────────────
lottery_pool = [
    {-6: 0.20, 4: 0.30, 14: 0.30, 26: 0.20},
    {100: 0.20, 300: 0.30, 200: 0.30, 190: 0.20},
    {-20: 0.10, -5: 0.15, 3: 0.20, 11: 0.2, 22: 0.15, 35: 0.10, 15:0.1},
    {-350: 0.10, -150: 0.15, 160: 0.20, 600: 0.25, 301: 0.20, 420: 0.10},
    
    #{-350: 0.10, -150: 0.15, 169: 0.20, 601: 0.25, 310: 0.20, 420: 0.10},

    #{-3:  0.2, 10: 0.6, 28: 0.20},
    #{-6:  0.2, 19: 0.6, 37: 0.20},
    #{-10:  0.20, 10: 0.60, 30: 0.20},
    #{4:   0.30, 6: 0.2, 17: 0.5},
]

def spread(lot):
    return max(lot) - min(lot)

def ce_from_lottery(lot, R, alpha=alpha, lamb=lamb, gamma=gamma):
    pv_list   = [u(x, R, alpha, lamb) for x in lot]
    prob_list = list(lot.values())
    return u_inv(V(pv_list, prob_list, gamma=gamma, alpha=alpha, lamb=lamb, R=R), R, alpha, lamb)

R_grid = np.linspace(-50, 50, 1000)

# ── Build observations: ce_obs = ce_th clipped to CE(R) feasible range ────────
observations = []
for lot in lottery_pool:
    ce_th  = ce_from_lottery(lot, R_true)
    sigma  = ksi * spread(lot)
    ce_curve = np.array([ce_from_lottery(lot, R) for R in R_grid])
    ce_obs = np.clip(np.random.normal(ce_th, sigma), ce_curve.min(), ce_curve.max())
    observations.append((lot, ce_obs, sigma, ce_curve))
    # Count crossings: how many times does CE(R) cross ce_obs?
    crossings = np.where(np.diff(np.sign(ce_curve - ce_obs)))[0]
    print(f"  CE_th={ce_th:.2f}  CE_obs={ce_obs:.2f}  crossings at R ~ "
          f"{[round(float(R_grid[i]), 1) for i in crossings]}")

# ── Sweep R and compute cumulative joint log-likelihood ───────────────────────
loglik_cumulative = np.zeros((len(lottery_pool), len(R_grid)))

for k, (lot, ce_obs, sigma, _) in enumerate(observations):
    ll_k = np.array([
        norm.logpdf(ce_obs, loc=ce_from_lottery(lot, R), scale=sigma)
        for R in R_grid
    ])
    loglik_cumulative[k] = (loglik_cumulative[k-1] if k > 0 else 0) + ll_k

lik_cumulative = np.exp(loglik_cumulative - loglik_cumulative.max(axis=1, keepdims=True))
R_hats = [R_grid[np.argmax(lik_cumulative[k])] for k in range(len(lottery_pool))]

print("\nR_hat as lotteries are added:")
for k, rh in enumerate(R_hats):
    print(f"  {k+1} lotteries: R_hat = {rh:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
colors = ["steelblue", "darkorange", "crimson", "seagreen", "mediumpurple", "sienna"][:len(lottery_pool)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: CE(R) curves + observed CE lines + crossing markers
for i, (lot, ce_obs, sigma, ce_curve) in enumerate(observations):
    axes[0].plot(R_grid, ce_curve, color=colors[i], linewidth=1.8,
                 label=f"L{i+1}: {list(lot.keys())}")
    axes[0].axhline(ce_obs, color=colors[i], linestyle=":", alpha=0.7)
    # Mark crossings explicitly
    crossings = np.where(np.diff(np.sign(ce_curve - ce_obs)))[0]
    for idx in crossings:
        R_cross = R_grid[idx]
        axes[0].scatter(R_cross, ce_obs, color=colors[i], s=80, zorder=6,
                        edgecolors="black", linewidths=1)
        axes[0].annotate(f"R={R_cross:.1f}", xy=(R_cross, ce_obs),
                         xytext=(0, 8), textcoords="offset points",
                         fontsize=7, color=colors[i], ha="center")

axes[0].axvline(R_true, color="black", linestyle="--", linewidth=1.5, label=f"R_true={R_true}")
axes[0].set_xlabel("Reference Point R")
axes[0].set_ylabel("CE(R)")
axes[0].set_title("CE(R) curves — dots mark where CE(R) = CE_obs\n(multiple dots = non-unique identification)")
axes[0].legend(fontsize=7, ncol=2)

# Right: cumulative joint likelihood with local maxima
for k in range(len(lottery_pool)):
    axes[1].plot(R_grid, lik_cumulative[k], color=colors[k], linewidth=1.8,
                 label=f"{k+1} lottery{'s' if k>0 else ''} — R_hat={R_hats[k]:.1f}")
    global_max_idx = np.argmax(lik_cumulative[k])
    local_max_idx = argrelmax(lik_cumulative[k], order=8)[0]
    # If the global max is not among the interior local maxima (e.g. at boundary), add it
    if global_max_idx not in local_max_idx:
        local_max_idx = np.append(local_max_idx, global_max_idx)
    for idx in local_max_idx:
        is_global = (idx == global_max_idx)
        marker = "*" if is_global else "o"
        size   = 180 if is_global else 60
        label_suffix = " (global max)" if is_global else ""
        axes[1].scatter(R_grid[idx], lik_cumulative[k, idx], color=colors[k],
                        s=size, zorder=5, edgecolors="black", linewidths=1,
                        marker=marker)
        axes[1].annotate(f"{R_grid[idx]:.1f}{label_suffix}",
                         xy=(R_grid[idx], lik_cumulative[k, idx]),
                         xytext=(4, 6), textcoords="offset points",
                         fontsize=7, color=colors[k],
                         fontweight="bold" if is_global else "normal")

axes[1].axvline(R_true, color="black", linestyle="--", linewidth=1.5, label=f"R_true={R_true}")
axes[1].set_xlabel("Reference Point R")
axes[1].set_ylabel("Normalised Likelihood")
axes[1].set_title("Joint likelihood — two peaks = two candidate R values")
axes[1].legend(fontsize=7)

fig.suptitle("Lotteries designed for non-unique identification of R", fontsize=12)
plt.tight_layout()
plt.savefig("likelihood_ce_simulation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved to likelihood_ce_simulation.png")

# ── Monte Carlo: distribution of R_hat across many seeds ─────────────────────
n_seeds = 500
R_hats_mc = []

# Pre-compute CE curves once (same across seeds)
ce_curves_mc = {
    i: np.array([ce_from_lottery(lot, R) for R in R_grid])
    for i, lot in enumerate(lottery_pool)
}

for seed in range(n_seeds):
    np.random.seed(seed)
    ll = np.zeros(len(R_grid))
    for i, lot in enumerate(lottery_pool):
        ce_th = ce_from_lottery(lot, R_true)
        sigma = ksi * spread(lot)
        ce_curve = ce_curves_mc[i]
        ce_obs = np.clip(np.random.normal(ce_th, sigma), ce_curve.min(), ce_curve.max())
        ll += np.array([
            norm.logpdf(ce_obs, loc=ce_from_lottery(lot, R), scale=sigma)
            for R in R_grid
        ])
    R_hats_mc.append(R_grid[np.argmax(ll)])

R_hats_mc = np.array(R_hats_mc)
print(f"\nMonte Carlo ({n_seeds} seeds):")
print(f"  Median R_hat : {np.median(R_hats_mc):.2f}")
print(f"  Std R_hat  : {R_hats_mc.std():.2f}")
print(f"  Fraction within +/-2 of R_true: {np.mean(np.abs(R_hats_mc - R_true) < 2):.2%}")

fig2, ax = plt.subplots(figsize=(8, 4))
ax.hist(R_hats_mc, bins=60, edgecolor="black", color="steelblue", alpha=0.8)
ax.axvline(R_true, color="red", linestyle="--", linewidth=1.5, label=f"R_true={R_true}")
ax.axvline(np.median(R_hats_mc), color="darkorange", linestyle="-", linewidth=1.5,
           label=f"Median R_hat={np.median(R_hats_mc):.2f}")
ax.set_xlabel("R_hat")
ax.set_ylabel("Count")
ax.set_title(f"Monte Carlo distribution of R_hat (n={n_seeds} seeds)\n"
             f"Std={R_hats_mc.std():.2f}, ")
ax.legend()
fig2.tight_layout()
fig2.savefig("monte_carlo_R_hat.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved to monte_carlo_R_hat.png")
