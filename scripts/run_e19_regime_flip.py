"""E19: map the trust x resource interaction sign-flip across regimes.

E14 (cont substrate, severe starvation) found trust x resource SUPER-additive
for lock-in. E18 (structural, correct majority intact) found it SUB-additive
(trust corrective). The hypothesis: the sign is governed by whether the
social channel carries signal or noise, and the cleanest knob for that is
the reliability of the correct majority.

This experiment sweeps the WRONG-BLOC FRACTION from a small minority (reliable
correct majority -> trust corrective) up to a majority (no reliable signal ->
trust amplifies). At each fraction it runs the 2x2 (trust x resource) and
computes the interaction. Prediction: the interaction crosses from negative
(sub-additive, trust-dominated) to positive (super-additive, lock-in) as the
correct majority is lost (wrong_frac passes ~0.5).

gamma is held OFF throughout (so conviction/u_source has no effect; this
isolates the trust x resource pair exactly as in E14).

Readout: POPULATION mean q(truth) at end (robust when bloc sizes vary).
Interaction convention (as E14/E18):
    additive = base - (delta_trust + delta_resource)
    interaction = additive - observed_both
    > 0 super-additive (more lock-in than sum), < 0 sub-additive.

Run from repo root:
    python -m scripts.run_e19_regime_flip
"""

from __future__ import annotations

import os
import json

import numpy as np
import matplotlib.pyplot as plt

from scripts.run_e18_cross_channel import make_cfg, setup_blocs
from src.pomdp.structural_step import run_structural


def section(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def main():
    section("E19 -- trust x resource interaction vs wrong-bloc fraction")
    print("gamma OFF throughout (isolates trust x resource). Structural substrate.")
    print("Readout: population mean q(truth=paradigm 1) at end.")
    print("Prediction: interaction crosses 0 (sub -> super additive) as the")
    print("correct majority is lost (wrong_frac passes ~0.5).")

    N = 30
    K = 2
    n_dep = 3
    edge_prec = 0.8
    n_steps = 200
    seeds = [0, 1, 2, 3, 4]
    wrong_fracs = [0.2, 0.35, 0.5, 0.65, 0.8]

    def run_cell(wrong_frac, trust, resource, seed):
        cfg = make_cfg(0.0, trust, resource, seed, N, n_steps, n_dep, edge_prec)
        D0, u_src, is_wrong = setup_blocs(
            N, K, n_dep, wrong_frac, 0, 1, 0.95, 1.0, seed)
        out = run_structural(cfg, D_per_agent=D0, u_source_per_agent=u_src)
        # population mean q(truth) at end
        return float(out["final_q"][:, 1].mean())

    results = {}
    interactions = []
    bases = []
    both_obs = []
    n_runs = len(wrong_fracs) * 4 * len(seeds)
    done = 0
    for wf in wrong_fracs:
        cell = {}
        for trust in (False, True):
            for resource in (False, True):
                vals = []
                for s in seeds:
                    vals.append(run_cell(wf, trust, resource, s))
                    done += 1
                cell[(trust, resource)] = vals
                print(f"  [{done}/{n_runs}] wf={wf:.2f} trust={'on ' if trust else 'off'} "
                      f"rsrc={'on ' if resource else 'off'} -> "
                      f"q(truth)={np.mean(vals):.3f}")
        results[wf] = cell

        base = float(np.mean(cell[(False, False)]))
        d_trust = base - float(np.mean(cell[(True, False)]))
        d_rsrc = base - float(np.mean(cell[(False, True)]))
        observed = float(np.mean(cell[(True, True)]))
        additive = base - (d_trust + d_rsrc)
        interaction = additive - observed
        interactions.append(interaction)
        bases.append(base)
        both_obs.append(observed)
        print(f"    -> base={base:.3f} d_trust={d_trust:+.3f} d_rsrc={d_rsrc:+.3f} "
              f"additive={additive:.3f} observed={observed:.3f} "
              f"INTERACTION={interaction:+.3f}")

    section("Summary: interaction vs wrong-bloc fraction")
    print(f"{'wrong_frac':>10} {'base':>7} {'both_obs':>9} {'interaction':>12} {'sign':>6}")
    for wf, b, o, it in zip(wrong_fracs, bases, both_obs, interactions):
        sign = "super" if it > 0.02 else ("sub" if it < -0.02 else "~0")
        print(f"{wf:>10.2f} {b:>7.3f} {o:>9.3f} {it:>+12.3f} {sign:>6}")

    summary = {
        "wrong_fracs": wrong_fracs,
        "interactions": interactions,
        "bases": bases,
        "both_observed": both_obs,
        "cube": {f"{wf}": {f"t{int(t)}_r{int(r)}": results[wf][(t, r)]
                            for t in (False, True) for r in (False, True)}
                 for wf in wrong_fracs},
    }
    os.makedirs("experiments/results/E19", exist_ok=True)
    with open("experiments/results/E19/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # left: interaction vs wrong_frac
    colors = ["#16a085" if it < 0 else "#d0021b" for it in interactions]
    ax1.bar([f"{wf:.2f}" for wf in wrong_fracs], interactions, color=colors)
    ax1.axhline(0, color="black", lw=1)
    ax1.axvline(2.0, color="gray", linestyle="--", alpha=0.5)  # ~0.5 majority loss
    ax1.set_xlabel("wrong-bloc fraction (0.5 = majority lost)")
    ax1.set_ylabel("trust x resource interaction")
    ax1.set_title("Interaction sign flip\n(green=sub-additive/corrective, red=super-additive/lock-in)")
    ax1.grid(True, alpha=0.3, axis="y")

    # right: the four 2x2 corners across wrong_frac
    for (t, r), style, lab in [
        ((False, False), "k-", "neither"),
        ((True, False), "C0-", "trust only"),
        ((False, True), "C1-", "resource only"),
        ((True, True), "C3--", "both")]:
        ys = [float(np.mean(results[wf][(t, r)])) for wf in wrong_fracs]
        ax2.plot(wrong_fracs, ys, style, lw=2, marker="o", label=lab)
    ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("wrong-bloc fraction")
    ax2.set_ylabel("population q(truth) at end")
    ax2.set_title("The four 2x2 corners across regime")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("E19: trust x resource interaction flips sign as the correct "
                 "majority is lost", fontsize=12)
    fig.tight_layout()
    fig.savefig("experiments/results/E19/regime_flip.png", dpi=120)
    plt.close(fig)
    print("\nSaved figure to experiments/results/E19/regime_flip.png")


if __name__ == "__main__":
    main()
