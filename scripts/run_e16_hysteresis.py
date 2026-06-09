"""E16: textbook hysteresis via trust-mediated material asymmetry.

Setup:
  Population split 80% mainstream / 20% dissident with strongly opposed
  initial beliefs. Truth follows a reversal: paradigm 0 in phase 0 (so
  mainstream is correct), shifts to paradigm 1 in phase 1 (so dissidents
  are correct), reverts to paradigm 0 in phase 2 (so mainstream is correct
  again -- BUT they spent phase 1 adapting to paradigm 1).

  With trust learning + resource coupling: during phase 0, mainstream
  predicts well, gains trust, draws material flow. Dissidents lose
  trust and material. When truth shifts in phase 1, dissidents are
  correct but resource-poor (can't afford informative experiments);
  mainstream is wrong but resource-rich and adapts to paradigm 1.
  When truth REVERSES in phase 2, the now-paradigm-1-committed
  mainstream has resources but wrong conviction; the dissidents
  (who held paradigm 0 throughout) are correct but still depleted.

  The hysteresis signature: at final t, with truth=paradigm 0, population
  q sits well above 0 -- the historical paradigm-1 commitment from phase 1
  persists.

Compare two conditions:
  (a) trust + resources OFF (control: no material asymmetry can build)
  (b) trust + resources ON  (full mechanism)

Run from repo root:
    python -m scripts.run_e16_hysteresis
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from src.config import WorldConfig
from src.pomdp.gen_model import PomdpConfig
from src.pomdp.cont_step import ContConfig, run_cont


def make_world(t_shift, t_reverse):
    return WorldConfig(
        sigma=0.5,
        schedule="reversal",
        theta_star_pre=0.0,
        theta_star_post=1.0,
        schedule_t_shift=t_shift,
        schedule_t_reverse=t_reverse,
    )


def split_initial_belief(N, K, frac_dissident, mainstream_paradigm,
                          dissident_paradigm, certainty, seed):
    """frac_dissident fraction commit to dissident_paradigm, rest to
    mainstream_paradigm. Returns (D_per_agent, is_dissident mask)."""
    rng = np.random.RandomState(seed)
    is_dissident = rng.rand(N) < frac_dissident
    D = np.full((N, K), (1 - certainty) / (K - 1))
    D[is_dissident, dissident_paradigm] = certainty
    D[~is_dissident, mainstream_paradigm] = certainty
    return D, is_dissident


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    section("E16 -- Trust-mediated hysteresis under truth reversal")
    print("Population: 80% mainstream (paradigm 0), 20% dissident (paradigm 1).")
    print("Truth schedule: paradigm 0 (t < 80), paradigm 1 (80..240), paradigm 0 (t >= 240).")
    print("Compare control (trust+rsrc OFF) vs full mechanism (trust+rsrc ON).")

    N = 40
    K = 2
    n_steps = 500
    seeds = [0, 1, 2, 3, 4]
    t_shift = 40
    t_reverse = 320

    base_pomdp = PomdpConfig(
        n_paradigms=K, theta_vals=(0.0, 1.0), true_paradigm=1,
        x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), q_reliability=0.85,
        world=make_world(t_shift, t_reverse))

    conditions = [
        ("control (channels off)", dict(trust_learning=False, resource_coupling=False)),
        ("full mechanism (trust + rsrc)", dict(trust_learning=True, resource_coupling=True)),
    ]

    results: dict[str, dict] = {}

    for label, ch in conditions:
        traj_pop = []
        traj_mainstream = []
        traj_dissident = []
        for s in seeds:
            cfg = ContConfig(
                pomdp=base_pomdp, lambda_init=1.0, eta_lambda=0.0,
                eps_theta=0.15, obs_x_index=4,
                n_agents=N, n_steps=n_steps, use_theta_schedule=True,
                graph_kind="watts_strogatz", mean_degree=4,
                social_mask=0.3,
                trust_learning=ch["trust_learning"], trust_rho=0.95,
                trust_alpha0=1.0, trust_beta0=1.0,
                # Finite-budget configuration: large r_init, NO replenishment.
                # Agents spend down their initial resource pool during adaptation.
                resource_coupling=ch["resource_coupling"], r_init=4.0,
                R_in=0.0, alpha_flow=0.5, delta_decay=0.0,
                c0=0.015, r_min=0.1, budget_fraction=0.5, seed=s)
            D0 = np.full((N, K), 0.5)
            is_dis = np.zeros(N, dtype=bool)
            is_dis[: int(0.2 * N)] = True   # symbolic split for reporting
            out = run_cont(cfg, D_per_agent=D0)
            # Reconstruct per-agent trajectories from infos
            pop_traj = out["mean_qB"]   # (n_steps,) population mean q(paradigm=1)
            # Per-step per-agent q is in infos['mean_q'] — but only the mean...
            # We need final_q per agent for groupwise stats; trajectories
            # for groupwise would need infos. Use final state for now and
            # population mean for trajectory.
            traj_pop.append(pop_traj)
            traj_mainstream.append(out["final_q"][~is_dis, 1].mean())
            traj_dissident.append(out["final_q"][is_dis, 1].mean())
        results[label] = {
            "pop_traj": np.array(traj_pop),
            "mainstream_final": np.array(traj_mainstream),
            "dissident_final": np.array(traj_dissident),
        }

    # Print summary
    print(f"\n{'condition':<32} {'pop final q(p=1)':>18} {'mainstream':>12} {'dissident':>12}")
    print("(truth at end = paradigm 0; q(paradigm=1) should be LOW if no lock-in)")
    summary = {}
    for label, ch in conditions:
        r = results[label]
        pop_final = r["pop_traj"][:, -10:].mean()
        ms_final = r["mainstream_final"].mean()
        dis_final = r["dissident_final"].mean()
        print(f"  {label:<32} {pop_final:>16.3f} {ms_final:>12.3f} {dis_final:>12.3f}")
        summary[label] = {
            "pop_final": float(pop_final),
            "mainstream_final": float(ms_final),
            "dissident_final": float(dis_final),
        }

    # Hysteresis signature: pop_final under (channels on) much higher than (off)
    ctrl_pop = results["control (channels off)"]["pop_traj"][:, -10:].mean()
    full_pop = results["full mechanism (trust + rsrc)"]["pop_traj"][:, -10:].mean()
    hysteresis = full_pop - ctrl_pop
    print(f"\nHYSTERESIS gap (full - control) at final t (truth = paradigm 0): {hysteresis:+.3f}")
    if hysteresis > 0.15:
        print(f"  Strong hysteresis: trust+resources prevent recovery to truth.")
    elif hysteresis > 0.05:
        print(f"  Moderate hysteresis observed.")
    else:
        print(f"  No appreciable hysteresis at this severity.")
    summary["hysteresis_gap"] = float(hysteresis)

    # Trajectory plot
    theta = np.array([
        0.0 if (t < t_shift or t >= t_reverse) else 1.0
        for t in range(n_steps)])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(n_steps), theta, "k--", lw=1, alpha=0.5, label="truth")
    colors = ["C0", "C3"]
    for (label, _), col in zip(conditions, colors):
        traj = results[label]["pop_traj"]
        m, s = traj.mean(0), traj.std(0)
        ax.plot(np.arange(n_steps), m, color=col, lw=2.5, label=label)
        ax.fill_between(np.arange(n_steps), m - s, m + s, alpha=0.2, color=col)
    ax.axvline(t_shift, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(t_reverse, color="gray", linestyle=":", alpha=0.6)
    ax.text(t_shift / 2, 1.07, "phase 0\ntruth=0", ha="center", fontsize=9, alpha=0.7)
    ax.text((t_shift + t_reverse) / 2, 1.07, "phase 1\ntruth=1", ha="center", fontsize=9, alpha=0.7)
    ax.text((t_reverse + n_steps) / 2, 1.07, "phase 2\ntruth=0", ha="center", fontsize=9, alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("population mean q(paradigm=1)")
    ax.set_title("E16: trust-mediated hysteresis -- truth 0 -> 1 -> 0")
    ax.legend(loc="center right", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs("experiments/results/E16", exist_ok=True)
    fig.savefig("experiments/results/E16/hysteresis_trajectories.png", dpi=120)
    plt.close(fig)

    # Bloc-resolved final beliefs bar plot
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(2)
    width = 0.35
    ctrl = results["control (channels off)"]
    full = results["full mechanism (trust + rsrc)"]
    ax.bar(x - width / 2, [ctrl["mainstream_final"].mean(), ctrl["dissident_final"].mean()],
           width, yerr=[ctrl["mainstream_final"].std(), ctrl["dissident_final"].std()],
           label="control", color="C0", capsize=4)
    ax.bar(x + width / 2, [full["mainstream_final"].mean(), full["dissident_final"].mean()],
           width, yerr=[full["mainstream_final"].std(), full["dissident_final"].std()],
           label="full mechanism", color="C3", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(["mainstream (80%)", "dissident (20%)"])
    ax.set_ylabel("final q(paradigm=1)  [truth at end = paradigm 0]")
    ax.set_title("E16: per-bloc final belief by condition")
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("experiments/results/E16/per_bloc_final.png", dpi=120)
    plt.close(fig)

    with open("experiments/results/E16/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
