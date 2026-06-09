"""Compare E9 (state-tilt motivated update) vs E20 (evidence-gate).

Both are (q_reliability x lambda_tilt) phase-diagram sweeps with identical
config except the motivated update mode. The question: does the social-
coupling phase boundary (visible at lambda=0) SURVIVE motivated reasoning?

  - state_tilt (E9):    boundary washes out at any lambda>0 (flat ~0.70)
  - evidence_gate (E20): boundary should PERSIST -- motivated reasoning
                         resists disconfirmation but doesn't impose a fixed
                         attractor, so social lock-in still operates.

Builds side-by-side heatmaps and prints both grids.

    python -m scripts.analyze_e9_vs_e20
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt


def load_grid(path):
    with open(path) as f:
        recs = json.load(f)
    qrs = sorted({r["sweep"]["q_reliability"] for r in recs})
    lams = sorted({r["sweep"]["lambda_tilt"] for r in recs})
    grid = np.full((len(qrs), len(lams)), np.nan)
    for i, qr in enumerate(qrs):
        for j, lam in enumerate(lams):
            vals = [r["final_mean_qB"] for r in recs
                    if r["sweep"]["q_reliability"] == qr
                    and r["sweep"]["lambda_tilt"] == lam]
            if vals:
                grid[i, j] = float(np.mean(vals))
    return qrs, lams, grid


def print_grid(name, qrs, lams, grid):
    print(f"\n{name}")
    print("qr\\lam " + "  ".join(f"{l:>6.1f}" for l in lams))
    for i, qr in enumerate(qrs):
        print(f"{qr:>5.2f} " + "  ".join(f"{grid[i,j]:>6.3f}" for j in range(len(lams))))


def boundary_strength(qrs, lams, grid):
    """Range of mean_qB across q_reliability, per lambda column. Large range =
    social-coupling boundary intact; ~0 = washed out (utility-determined)."""
    return {lams[j]: float(np.nanmax(grid[:, j]) - np.nanmin(grid[:, j]))
            for j in range(len(lams))}


def main():
    qrs9, lams9, g9 = load_grid("experiments/results/E9/results.json")
    qrs20, lams20, g20 = load_grid("experiments/results/E20/results.json")

    print_grid("E9  (state_tilt) mean_qB", qrs9, lams9, g9)
    print_grid("E20 (evidence_gate) mean_qB", qrs20, lams20, g20)

    print("\nSocial-coupling boundary strength (max-min over q_reliability), per lambda:")
    b9 = boundary_strength(qrs9, lams9, g9)
    b20 = boundary_strength(qrs20, lams20, g20)
    print(f"{'lambda':>8} {'E9 state_tilt':>15} {'E20 evidence_gate':>18}")
    for lam in sorted(set(lams9) & set(lams20)):
        print(f"{lam:>8.1f} {b9.get(lam, float('nan')):>15.3f} {b20.get(lam, float('nan')):>18.3f}")
    print("\nLarge boundary strength at lambda>0 = phase boundary SURVIVES motivated reasoning.")

    # side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (title, qrs, lams, grid) in zip(
            axes,
            [("E9: state-tilt (reward on belief state)", qrs9, lams9, g9),
             ("E20: evidence-gate (cost of mind-change)", qrs20, lams20, g20)]):
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                       origin="lower")
        ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:.1f}" for l in lams])
        ax.set_yticks(range(len(qrs))); ax.set_yticklabels([f"{q:.2f}" for q in qrs])
        ax.set_xlabel("lambda_tilt (motivated strength)")
        ax.set_ylabel("q_reliability (social coupling)")
        ax.set_title(title)
        for i in range(len(qrs)):
            for j in range(len(lams)):
                ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.05, label="mean q(truth)")
    fig.suptitle("Motivated update: state-tilt washes out the phase boundary; "
                 "evidence-gate preserves it", fontsize=12)
    fig.tight_layout()
    fig.savefig("experiments/results/E20/e9_vs_e20_comparison.png", dpi=120)
    plt.close(fig)
    print("\nSaved figure to experiments/results/E20/e9_vs_e20_comparison.png")


if __name__ == "__main__":
    main()
