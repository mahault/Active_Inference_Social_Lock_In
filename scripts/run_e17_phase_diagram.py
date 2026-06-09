"""E17: phase diagram in (R_in, c0) space.

Sweep two resource parameters at fixed r_init=4.0, fixed truth-reversal
schedule. Classify each cell by what behavior emerges:

  * BLOCKED:        phase-1 peak adaptation never exceeds 0.5
                    (resource cost prevents adaptation entirely)
  * ADAPT + REVERT: phase-1 peak > 0.5 AND phase-2 end < 0.3
                    (clean tracking of the truth schedule)
  * HYSTERESIS:     phase-1 peak > 0.5 AND phase-2 end > 0.5
                    (adapt in phase 1, locked there in phase 2)
  * PARTIAL:        intermediate cases (peak > 0.5, end in 0.3..0.5)

Run from repo root:
    python -m scripts.run_e17_phase_diagram

Output: two heatmaps (phase-1 peak, phase-2 end) plus a regime-classification
map, saved to experiments/results/E17/.
"""

from __future__ import annotations

import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from src.config import WorldConfig
from src.pomdp.gen_model import PomdpConfig
from src.pomdp.cont_step import ContConfig, run_cont


def make_cfg(R_in, c0, seed, n_steps, t_shift, t_reverse):
    return ContConfig(
        pomdp=PomdpConfig(
            n_paradigms=2, theta_vals=(0.0, 1.0), true_paradigm=1,
            x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), q_reliability=0.85,
            world=WorldConfig(
                sigma=0.5, schedule="reversal",
                theta_star_pre=0.0, theta_star_post=1.0,
                schedule_t_shift=t_shift, schedule_t_reverse=t_reverse)),
        lambda_init=1.0, eta_lambda=0.0, eps_theta=0.15, obs_x_index=4,
        n_agents=30, n_steps=n_steps, use_theta_schedule=True,
        graph_kind="watts_strogatz", mean_degree=4, social_mask=0.3,
        trust_learning=True, trust_rho=0.95, trust_alpha0=1.0, trust_beta0=1.0,
        resource_coupling=True, r_init=4.0,
        R_in=R_in, alpha_flow=0.5, delta_decay=0.0,
        c0=c0, r_min=0.1, budget_fraction=0.5, seed=seed)


def classify(peak, end_qB):
    """Classify a (peak, end) trajectory profile into a regime."""
    if peak < 0.5:
        return 0  # blocked
    if end_qB > 0.5:
        return 3  # hysteresis (locked at paradigm 1 despite truth=0)
    if end_qB > 0.3:
        return 2  # partial lock-in
    return 1      # adapt + revert (clean)


REGIME_NAMES = ["blocked", "adapt+revert", "partial", "hysteresis"]
REGIME_COLORS = ["#cccccc", "#3a7bd5", "#f5a623", "#d0021b"]


def main():
    print("=" * 72)
    print("E17 -- phase diagram: (R_in, c0) sweep")
    print("=" * 72)

    # Parameter grids
    R_in_values = [0.0, 0.01, 0.03, 0.06, 0.10]
    c0_values = [0.005, 0.015, 0.03, 0.06, 0.12]
    seeds = [0, 1, 2]
    n_steps = 500
    t_shift = 40
    t_reverse = 320

    n_R = len(R_in_values)
    n_c = len(c0_values)

    peak_grid = np.zeros((n_R, n_c))
    end_grid = np.zeros((n_R, n_c))
    peak_std = np.zeros((n_R, n_c))
    end_std = np.zeros((n_R, n_c))
    regime_grid = np.zeros((n_R, n_c), dtype=int)

    n_runs = n_R * n_c * len(seeds)
    print(f"Total runs: {n_runs} ({n_R} R_in x {n_c} c0 x {len(seeds)} seeds)")
    t0 = time.time()
    done = 0

    for i, R_in in enumerate(R_in_values):
        for j, c0 in enumerate(c0_values):
            peaks = []
            ends = []
            for s in seeds:
                cfg = make_cfg(R_in, c0, s, n_steps, t_shift, t_reverse)
                D0 = np.full((cfg.n_agents, 2), 0.5)
                out = run_cont(cfg, D_per_agent=D0)
                traj = out["mean_qB"]
                peaks.append(float(traj[t_shift:t_reverse].max()))
                ends.append(float(traj[-15:].mean()))
                done += 1
                rate = (time.time() - t0) / done
                eta = rate * (n_runs - done)
                print(f"  [{done:>3}/{n_runs}] R_in={R_in:.2f} c0={c0:.3f} "
                      f"seed={s}  peak={peaks[-1]:.3f}  end={ends[-1]:.3f}  "
                      f"({rate:.1f}s/run, ETA {eta:.0f}s)")
            peak_grid[i, j] = np.mean(peaks)
            end_grid[i, j] = np.mean(ends)
            peak_std[i, j] = np.std(peaks)
            end_std[i, j] = np.std(ends)
            regime_grid[i, j] = classify(peak_grid[i, j], end_grid[i, j])

    print("\n" + "=" * 72)
    print("Phase-1 peak adaptation (1.0 = full, 0 = no adaptation):")
    print("=" * 72)
    print(f"{'R_in/c0':>10}", *(f"{c:>8.3f}" for c in c0_values))
    for i, R_in in enumerate(R_in_values):
        print(f"{R_in:>10.2f}", *(f"{peak_grid[i,j]:>8.3f}" for j in range(n_c)))

    print("\n" + "=" * 72)
    print("Phase-2 end belief (1.0 = locked at paradigm 1, 0 = full revert):")
    print("=" * 72)
    print(f"{'R_in/c0':>10}", *(f"{c:>8.3f}" for c in c0_values))
    for i, R_in in enumerate(R_in_values):
        print(f"{R_in:>10.2f}", *(f"{end_grid[i,j]:>8.3f}" for j in range(n_c)))

    print("\n" + "=" * 72)
    print("Regime classification:")
    print("=" * 72)
    print(f"{'R_in/c0':>10}", *(f"{c:>10.3f}" for c in c0_values))
    for i, R_in in enumerate(R_in_values):
        regimes = [REGIME_NAMES[regime_grid[i, j]] for j in range(n_c)]
        print(f"{R_in:>10.2f}", *(f"{r:>10s}" for r in regimes))

    # Save summary
    os.makedirs("experiments/results/E17", exist_ok=True)
    summary = {
        "R_in_values": R_in_values,
        "c0_values": c0_values,
        "n_seeds": len(seeds),
        "phase1_peak": peak_grid.tolist(),
        "phase2_end": end_grid.tolist(),
        "phase1_peak_std": peak_std.tolist(),
        "phase2_end_std": end_std.tolist(),
        "regime_grid": regime_grid.tolist(),
        "regime_legend": REGIME_NAMES,
    }
    with open("experiments/results/E17/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ----- plot heatmaps -----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. phase-1 peak heatmap
    ax = axes[0]
    im1 = ax.imshow(peak_grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                     origin="lower")
    ax.set_xticks(range(n_c)); ax.set_xticklabels([f"{c:.3f}" for c in c0_values])
    ax.set_yticks(range(n_R)); ax.set_yticklabels([f"{r:.2f}" for r in R_in_values])
    ax.set_xlabel("cost scale c0")
    ax.set_ylabel("replenishment rate R_in")
    ax.set_title("Phase-1 peak adaptation\n(1.0 = full adaptation to truth=1)")
    for i in range(n_R):
        for j in range(n_c):
            ax.text(j, i, f"{peak_grid[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    fig.colorbar(im1, ax=ax, fraction=0.05)

    # 2. phase-2 end heatmap
    ax = axes[1]
    im2 = ax.imshow(end_grid, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto",
                     origin="lower")
    ax.set_xticks(range(n_c)); ax.set_xticklabels([f"{c:.3f}" for c in c0_values])
    ax.set_yticks(range(n_R)); ax.set_yticklabels([f"{r:.2f}" for r in R_in_values])
    ax.set_xlabel("cost scale c0")
    ax.set_ylabel("replenishment rate R_in")
    ax.set_title("Phase-2 end belief\n(high = locked at paradigm 1 despite truth=0)")
    for i in range(n_R):
        for j in range(n_c):
            ax.text(j, i, f"{end_grid[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    fig.colorbar(im2, ax=ax, fraction=0.05)

    # 3. regime classification
    ax = axes[2]
    cmap = ListedColormap(REGIME_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 4, 1), cmap.N)
    im3 = ax.imshow(regime_grid, cmap=cmap, norm=norm, aspect="auto",
                     origin="lower")
    ax.set_xticks(range(n_c)); ax.set_xticklabels([f"{c:.3f}" for c in c0_values])
    ax.set_yticks(range(n_R)); ax.set_yticklabels([f"{r:.2f}" for r in R_in_values])
    ax.set_xlabel("cost scale c0")
    ax.set_ylabel("replenishment rate R_in")
    ax.set_title("Regime classification")
    for i in range(n_R):
        for j in range(n_c):
            ax.text(j, i, REGIME_NAMES[regime_grid[i, j]],
                    ha="center", va="center", color="white", fontsize=8,
                    fontweight="bold")
    cbar = fig.colorbar(im3, ax=ax, fraction=0.05, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(REGIME_NAMES)

    fig.suptitle("E17: phase diagram in (R_in, c0) at fixed r_init=4.0, "
                  "truth schedule 0->1->0", fontsize=12)
    fig.tight_layout()
    fig.savefig("experiments/results/E17/phase_diagram.png", dpi=120)
    plt.close(fig)
    print("\nSaved figure to experiments/results/E17/phase_diagram.png")


if __name__ == "__main__":
    main()
