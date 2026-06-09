"""Run three focused empirical investigations:

  E13: trust ablation with heterogeneous initial beliefs
       (does trust learning isolate the wrong-paradigm bloc?)
  E14: trust x resource interaction matrix
       (do channels combine additively or multiplicatively?)
  E15: tuned hysteresis under truth reversal
       (does resource drain create path-dependence?)

All three call run_cont / run_simple directly with custom D_per_agent so we
can set up heterogeneous priors and isolate channel effects. Saves PNG
figures into experiments/results/E13|E14|E15 and prints summary tables.

Run from repo root:
    python -m scripts.run_e13_e14_e15
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from src.config import WorldConfig
from src.pomdp.gen_model import PomdpConfig
from src.pomdp.cont_step import ContConfig, run_cont
from src.pomdp.simple_step import SimpleConfig, run_simple


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def make_world(schedule="step", t_shift=80, t_reverse=None, ramp_T=80):
    return WorldConfig(
        sigma=0.5,
        schedule=schedule,
        theta_star_pre=0.0,
        theta_star_post=1.0,
        schedule_t_shift=t_shift,
        schedule_t_reverse=t_reverse,
        schedule_ramp_T=ramp_T,
    )


def heterogeneous_D(N: int, K: int, frac_wrong: float, wrong_paradigm: int,
                    right_paradigm: int, certainty: float, seed: int):
    """N agents: frac_wrong heavily believe wrong_paradigm, rest right_paradigm.
    certainty in [0.5, 1.0]: concentration on the favored paradigm."""
    rng = np.random.RandomState(seed)
    is_wrong = rng.rand(N) < frac_wrong
    D = np.full((N, K), (1 - certainty) / (K - 1))
    D[is_wrong, wrong_paradigm] = certainty
    D[~is_wrong, right_paradigm] = certainty
    return D, is_wrong


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------
# E13 -- trust ablation with heterogeneous priors
# ----------------------------------------------------------------------

def run_e13():
    section("E13 -- Trust ablation under heterogeneous priors")
    print("Setup: 30% of agents start with P(paradigm 0) = 0.95 (WRONG);")
    print("       70% start with P(paradigm 1) = 0.95 (RIGHT). Truth = paradigm 1.")
    print("       Cont substrate, lambda=1.0 (no tempering), q_reliability=0.75.")
    print("       Compare trust_learning=False vs trust_learning=True.")

    N = 40
    K = 2
    n_steps = 150
    seeds = [0, 1, 2, 3, 4]
    true_p = 1

    base_pomdp = PomdpConfig(
        n_paradigms=K, theta_vals=(0.0, 1.0), true_paradigm=true_p,
        x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), q_reliability=0.75,
        world=make_world(schedule="step", t_shift=0))

    results = {False: [], True: []}
    wrong_finals = {False: [], True: []}
    right_finals = {False: [], True: []}

    for trust in (False, True):
        for s in seeds:
            cfg = ContConfig(
                pomdp=base_pomdp, lambda_init=1.0, eta_lambda=0.0,
                eps_theta=0.02, obs_x_index=4,
                n_agents=N, n_steps=n_steps, use_theta_schedule=False,
                graph_kind="watts_strogatz", mean_degree=4,
                social_mask=1.0,
                trust_learning=trust, trust_rho=0.95,
                trust_alpha0=1.0, trust_beta0=1.0,
                resource_coupling=False, seed=s)
            D0, is_wrong = heterogeneous_D(N, K, 0.30, 0, 1, 0.95, s)
            out = run_cont(cfg, D_per_agent=D0)
            results[trust].append(out["mean_qB"])
            # Per-agent final belief in true paradigm
            wrong_finals[trust].extend(out["final_q"][is_wrong, true_p].tolist())
            right_finals[trust].extend(out["final_q"][~is_wrong, true_p].tolist())

    # Summarize
    print(f"\n{'condition':<20} {'mean_qB_final':>15} {'wrong_bloc_final':>18} {'right_bloc_final':>18}")
    for trust in (False, True):
        traj = np.array(results[trust])
        wf = np.array(wrong_finals[trust])
        rf = np.array(right_finals[trust])
        print(f"  trust_learning={str(trust):<6} {traj[:, -1].mean():>15.3f}"
              f" {wf.mean():>18.3f} {rf.mean():>18.3f}")

    # Plot trajectories
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, trust in zip(axes, (False, True)):
        traj = np.array(results[trust])
        m, s = traj.mean(0), traj.std(0)
        t = np.arange(n_steps)
        ax.plot(t, m, color="C0" if not trust else "C3", lw=2)
        ax.fill_between(t, m - s, m + s, alpha=0.2,
                         color="C0" if not trust else "C3")
        ax.axhline(0.7, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax.axhline(0.3, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax.set_title(f"trust_learning = {trust}")
        ax.set_xlabel("step")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean q(paradigm=1)  [truth=1]")
    fig.suptitle("E13: trust ablation under 30/70 heterogeneous prior", fontsize=12)
    fig.tight_layout()
    os.makedirs("experiments/results/E13", exist_ok=True)
    fig.savefig("experiments/results/E13/trust_ablation.png", dpi=120)
    plt.close(fig)

    # Per-agent histogram
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, trust in zip(axes, (False, True)):
        wf = np.array(wrong_finals[trust])
        rf = np.array(right_finals[trust])
        ax.hist(wf, bins=20, range=(0, 1), alpha=0.6, label="wrong-bloc agents",
                color="C3")
        ax.hist(rf, bins=20, range=(0, 1), alpha=0.6, label="right-bloc agents",
                color="C0")
        ax.set_title(f"trust_learning = {trust}")
        ax.set_xlabel("final q(paradigm=1)")
        ax.legend(loc="upper center")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("agent count")
    fig.suptitle("E13: final per-agent belief by bloc and condition")
    fig.tight_layout()
    fig.savefig("experiments/results/E13/per_agent_histogram.png", dpi=120)
    plt.close(fig)

    summary = {
        "trust_off_mean_qB": float(np.array(results[False])[:, -1].mean()),
        "trust_on_mean_qB": float(np.array(results[True])[:, -1].mean()),
        "trust_off_wrong_bloc_final": float(np.array(wrong_finals[False]).mean()),
        "trust_on_wrong_bloc_final": float(np.array(wrong_finals[True]).mean()),
        "trust_off_right_bloc_final": float(np.array(right_finals[False]).mean()),
        "trust_on_right_bloc_final": float(np.array(right_finals[True]).mean()),
    }
    summary["trust_marginal_on_wrong_bloc"] = (
        summary["trust_off_wrong_bloc_final"] - summary["trust_on_wrong_bloc_final"])
    print(f"\nTrust marginal on WRONG-bloc convergence: "
          f"{summary['trust_marginal_on_wrong_bloc']:+.3f}")
    print("  Positive = trust learning prevents wrong-bloc convergence (lock-in).")

    with open("experiments/results/E13/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ----------------------------------------------------------------------
# E14 -- trust x resource interaction matrix
# ----------------------------------------------------------------------

def run_e14():
    section("E14 -- Trust x Resource interaction matrix (2x2)")
    print("Setup: same heterogeneous 30/70 prior as E13. Sweep trust and resource.")
    print("       n=30, q_reliability=0.70, 5 seeds.")

    N = 30
    K = 2
    n_steps = 200
    seeds = [0, 1, 2, 3, 4]
    true_p = 1

    base_pomdp = PomdpConfig(
        n_paradigms=K, theta_vals=(0.0, 1.0), true_paradigm=true_p,
        x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), q_reliability=0.70,
        world=make_world(schedule="step", t_shift=0))

    cells = [(False, False), (False, True), (True, False), (True, True)]
    means: dict[tuple, list[float]] = {}
    trajectories: dict[tuple, list[np.ndarray]] = {}

    for (trust, rsrc) in cells:
        finals = []
        trajs = []
        for s in seeds:
            cfg = ContConfig(
                pomdp=base_pomdp, lambda_init=1.0, eta_lambda=0.0,
                eps_theta=0.02, obs_x_index=4,
                n_agents=N, n_steps=n_steps, use_theta_schedule=False,
                graph_kind="watts_strogatz", mean_degree=4,
                social_mask=1.0,
                trust_learning=trust, trust_rho=0.95,
                trust_alpha0=1.0, trust_beta0=1.0,
                resource_coupling=rsrc, r_init=1.0, R_in=0.05,
                alpha_flow=0.4, delta_decay=0.04, c0=0.12, r_min=0.1,
                budget_fraction=0.5, seed=s)
            D0, is_wrong = heterogeneous_D(N, K, 0.30, 0, 1, 0.95, s)
            out = run_cont(cfg, D_per_agent=D0)
            finals.append(out["final_q"][is_wrong, true_p].mean())   # wrong bloc convergence
            trajs.append(out["mean_qB"])
        means[(trust, rsrc)] = finals
        trajectories[(trust, rsrc)] = trajs

    print(f"\n{'trust':<6} {'rsrc':<6} {'wrong-bloc final q(truth)':>30}")
    summary = {}
    for c in cells:
        arr = np.array(means[c])
        print(f"  {str(c[0]):<6} {str(c[1]):<6} {arr.mean():>20.3f} +- {arr.std():.3f}  (n={len(arr)})")
        summary[f"trust={c[0]}_rsrc={c[1]}"] = {
            "mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr)}

    # Compute additive prediction vs observed
    base = np.array(means[(False, False)]).mean()
    only_trust = np.array(means[(True, False)]).mean()
    only_rsrc = np.array(means[(False, True)]).mean()
    both = np.array(means[(True, True)]).mean()
    delta_trust = base - only_trust   # capture amount due to trust alone
    delta_rsrc = base - only_rsrc
    additive_predicted_both = base - (delta_trust + delta_rsrc)
    interaction = additive_predicted_both - both   # positive = SUPER-additive (more lock-in than sum)

    print(f"\nbaseline (both off) wrong-bloc convergence:      {base:.3f}")
    print(f"only trust on:                                   {only_trust:.3f}  (delta = -{delta_trust:.3f})")
    print(f"only rsrc on:                                    {only_rsrc:.3f}  (delta = -{delta_rsrc:.3f})")
    print(f"BOTH on, observed:                               {both:.3f}")
    print(f"BOTH on, additive prediction:                    {additive_predicted_both:.3f}")
    print(f"INTERACTION (additive_predicted - observed):     {interaction:+.3f}")
    if interaction > 0.05:
        print("  -> SUPER-ADDITIVE: trust + resources amplify each other (stronger lock-in than sum)")
    elif interaction < -0.05:
        print("  -> SUB-ADDITIVE: trust + resources interfere (weaker lock-in than sum)")
    else:
        print("  -> APPROXIMATELY ADDITIVE: channels contribute independently")

    summary["interaction"] = float(interaction)
    summary["base"] = float(base)
    summary["only_trust"] = float(only_trust)
    summary["only_rsrc"] = float(only_rsrc)
    summary["both"] = float(both)

    # 2x2 heatmap
    cell_means = np.array([[np.array(means[(False, False)]).mean(),
                             np.array(means[(False, True)]).mean()],
                            [np.array(means[(True, False)]).mean(),
                             np.array(means[(True, True)]).mean()]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cell_means, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["resource OFF", "resource ON"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["trust OFF", "trust ON"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cell_means[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")
    ax.set_title("E14: wrong-bloc convergence to truth\n(1.0 = full conversion, 0.0 = stuck)")
    fig.colorbar(im, ax=ax, label="wrong-bloc final q(truth)")
    fig.tight_layout()
    os.makedirs("experiments/results/E14", exist_ok=True)
    fig.savefig("experiments/results/E14/interaction_heatmap.png", dpi=120)
    plt.close(fig)

    # Trajectories
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {False: "C0", True: "C3"}
    styles = {False: "-", True: "--"}
    for (trust, rsrc), trajs in trajectories.items():
        arr = np.array(trajs)
        m = arr.mean(0)
        ax.plot(np.arange(n_steps), m, color=colors[trust], linestyle=styles[rsrc],
                lw=2, label=f"trust={trust}, rsrc={rsrc}")
    ax.axhline(0.7, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("population mean q(paradigm=1)")
    ax.set_title("E14: trust x resource interaction (population-mean trajectories)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("experiments/results/E14/trajectories.png", dpi=120)
    plt.close(fig)

    with open("experiments/results/E14/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ----------------------------------------------------------------------
# E15 -- tuned hysteresis under truth reversal
# ----------------------------------------------------------------------

def run_e15():
    section("E15 -- Resource dose-response (no schedule change)")
    print("Setup: same heterogeneous 30/70 prior as E14, truth=paradigm 1 fixed.")
    print("       Sweep c0 in {0.0 (off), 0.04, 0.08, 0.12, 0.20, 0.30, 0.50}.")
    print("       Trust learning ON. Measures wrong-bloc convergence vs cost.")
    print("       Goal: smooth phase-transition curve from 'full adaptation' to")
    print("       'blocked' as resource cost increases.")

    N = 30
    K = 2
    n_steps = 200
    seeds = [0, 1, 2, 3, 4]
    true_p = 1
    c0_values = [0.0, 0.04, 0.08, 0.12, 0.20, 0.30, 0.50]

    base_pomdp = PomdpConfig(
        n_paradigms=K, theta_vals=(0.0, 1.0), true_paradigm=true_p,
        x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), q_reliability=0.70,
        world=make_world(schedule="step", t_shift=0))

    trajectories: dict[float, list[np.ndarray]] = {}
    wrong_finals: dict[float, list[float]] = {}
    right_finals: dict[float, list[float]] = {}
    final_r: dict[float, list[float]] = {}

    for c0 in c0_values:
        trajs = []
        wf = []
        rf = []
        fr = []
        for s in seeds:
            cfg = ContConfig(
                pomdp=base_pomdp, lambda_init=1.0, eta_lambda=0.0,
                eps_theta=0.02, obs_x_index=4,
                n_agents=N, n_steps=n_steps, use_theta_schedule=False,
                graph_kind="watts_strogatz", mean_degree=4,
                social_mask=1.0,
                trust_learning=True, trust_rho=0.95,
                trust_alpha0=1.0, trust_beta0=1.0,
                resource_coupling=(c0 > 0.0), r_init=1.0, R_in=0.05,
                alpha_flow=0.4, delta_decay=0.04, c0=max(c0, 0.01), r_min=0.1,
                budget_fraction=0.5, seed=s)
            D0, is_wrong = heterogeneous_D(N, K, 0.30, 0, 1, 0.95, s)
            out = run_cont(cfg, D_per_agent=D0)
            trajs.append(out["mean_qB"])
            wf.append(float(out["final_q"][is_wrong, true_p].mean()))
            rf.append(float(out["final_q"][~is_wrong, true_p].mean()))
            if c0 > 0 and "final_r" in out:
                fr.append(float(out["final_r"].mean()))
            else:
                fr.append(1.0)
        trajectories[c0] = trajs
        wrong_finals[c0] = wf
        right_finals[c0] = rf
        final_r[c0] = fr

    print(f"\n{'c0':>6} {'wrong-bloc q(truth)':>22} {'right-bloc q(truth)':>22} {'final r':>10}")
    summary = {}
    for c0 in c0_values:
        w = float(np.mean(wrong_finals[c0]))
        r_ = float(np.mean(right_finals[c0]))
        r_final = float(np.mean(final_r[c0]))
        print(f"  {c0:>4.2f}  {w:>20.3f}  {r_:>20.3f}  {r_final:>10.3f}")
        summary[f"c0={c0}"] = {"wrong_final": w, "right_final": r_, "final_r": r_final}

    # Plot: dose-response curve
    fig, ax = plt.subplots(figsize=(8, 5))
    wfs = [np.mean(wrong_finals[c0]) for c0 in c0_values]
    rfs = [np.mean(right_finals[c0]) for c0 in c0_values]
    wfs_std = [np.std(wrong_finals[c0]) for c0 in c0_values]
    ax.errorbar(c0_values, wfs, yerr=wfs_std, fmt="o-", lw=2,
                color="C3", label="WRONG-bloc final q(truth)", capsize=4)
    ax.errorbar(c0_values, rfs, yerr=[np.std(right_finals[c0]) for c0 in c0_values],
                fmt="s-", lw=2, color="C0", label="RIGHT-bloc final q(truth)", capsize=4)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("resource cost scale c0 (0 = off)")
    ax.set_ylabel("final q(paradigm=1) (truth=1)")
    ax.set_title("E15: dose-response curve -- resource cost vs paradigm adaptation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    os.makedirs("experiments/results/E15", exist_ok=True)
    fig.savefig("experiments/results/E15/dose_response.png", dpi=120)
    plt.close(fig)

    # Plot: trajectories
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(c0_values)))
    for c0, col in zip(c0_values, cmap):
        traj = np.array(trajectories[c0])
        m = traj.mean(0)
        label = f"c0={c0:.2f}" + (" (off)" if c0 == 0.0 else "")
        ax.plot(np.arange(n_steps), m, color=col, lw=2, label=label)
    ax.axhline(0.7, color="gray", linestyle=":", alpha=0.5, label="full convergence ~0.7 (30% wrong stays)")
    ax.axhline(1.0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("step")
    ax.set_ylabel("population mean q(paradigm=1)")
    ax.set_title("E15: population trajectory across resource cost c0")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("experiments/results/E15/trajectories_by_c0.png", dpi=120)
    plt.close(fig)

    with open("experiments/results/E15/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    s13 = run_e13()
    s14 = run_e14()
    s15 = run_e15()
    section("Combined summary")
    print(json.dumps({"E13": s13, "E14": s14, "E15": s15}, indent=2))
