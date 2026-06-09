"""E18: full 2x2x2 cross-channel interaction (gamma x trust x resource).

All three lock-in channels live on the structural substrate:
  - gamma_strength      (epistemic): conviction-driven evidence attenuation
  - trust_learning      (social):    Gamma-conjugate trust reweighting
  - resource_coupling   (material):  Fisher-info cost barrier on experiments

E14 showed trust x resource is super-additive (+0.31) on the cont
substrate. This experiment tests whether ALL THREE pairs interact, using
one 2x2x2 factorial on a single substrate so the readouts are comparable.

Setup mirrors E14: heterogeneous initial belief (30% committed to the
WRONG paradigm), truth = the OTHER paradigm, readout = wrong-bloc final
q(truth). Low readout = locked in. The wrong bloc also carries conviction
on its (wrong) paradigm's dependents, so gamma has an asymmetry to bite on
and silences the wrong bloc's disconfirming evidence.

lambda_tilt = 0 and motivated = False: the ONLY effect of conviction is
through gamma's evidence attenuation, not a direct posterior tilt. This
isolates each channel to its precision-modulating role.

Interaction convention (same as E14):
    additive_prediction = base - (delta_A + delta_B)
    interaction = additive_prediction - observed_both
  interaction > 0  => super-additive (more lock-in than the sum of parts)
  interaction ~ 0  => independent channels
  interaction < 0  => sub-additive (channels interfere)

Run from repo root:
    python -m scripts.run_e18_cross_channel
"""

from __future__ import annotations

import os
import json
import itertools

import numpy as np
import matplotlib.pyplot as plt

from src.pomdp.gen_model import PomdpConfig
from src.pomdp.simple_step import SimpleConfig
from src.pomdp.motivated_step import MotivatedConfig
from src.pomdp.structural_step import StructuralConfig, run_structural


def setup_blocs(N, K, n_dep, frac_wrong, wrong_p, right_p,
                certainty, u_mag, seed):
    """Heterogeneous initial belief + conviction source.

    Returns (D_per_agent (N,K), u_source (N,K_int), is_wrong (N,)).
    Wrong bloc: initial belief on wrong_p AND conviction on wrong_p's
    dependent nodes (so gamma silences their disconfirming evidence).
    Right bloc: initial belief on right_p, no conviction (evidence flows).
    """
    rng = np.random.RandomState(seed)
    is_wrong = rng.rand(N) < frac_wrong
    K_int = K * (1 + n_dep)

    D = np.full((N, K), (1.0 - certainty) / (K - 1))
    D[is_wrong, wrong_p] = certainty
    D[~is_wrong, right_p] = certainty

    u = np.zeros((N, K_int))
    block_start = K + wrong_p * n_dep
    block_end = block_start + n_dep
    u[is_wrong, block_start:block_end] = u_mag
    return D, u, is_wrong


def make_cfg(gamma, trust, resource, seed, N, n_steps, n_dep, edge_prec):
    simple = SimpleConfig(
        pomdp=PomdpConfig(
            n_paradigms=2, theta_vals=(0.0, 1.0), true_paradigm=1,
            x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), q_reliability=0.70),
        n_context=2, eps_crisis=0.05, eps_resolve=0.30, D_c_normal=0.9,
        alpha_tl=1.0, eps_theta=0.02, obs_x_index=4,
        n_agents=N, n_steps=n_steps, use_theta_schedule=False,
        graph_kind="watts_strogatz", mean_degree=4, social_mask=1.0,
        trust_learning=trust, trust_rho=0.95, trust_alpha0=1.0, trust_beta0=1.0,
        resource_coupling=resource, r_init=1.0, R_in=0.05, alpha_flow=0.4,
        delta_decay=0.04, c0=0.12, r_min=0.1, budget_fraction=0.5, seed=seed)
    mot = MotivatedConfig(simple=simple, lambda_tilt=0.0, motivated=False)
    return StructuralConfig(motivated=mot, n_dependents=n_dep,
                            edge_precision=edge_prec, gamma_strength=gamma)


def section(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def main():
    section("E18 -- 2x2x2 cross-channel interaction (gamma x trust x resource)")
    print("Structural substrate. 30% wrong-bloc (committed + convicted), truth=p1.")
    print("Readout: wrong-bloc final q(truth). Low = locked in.")

    N = 30
    K = 2
    n_dep = 3
    edge_prec = 0.8
    n_steps = 200
    seeds = [0, 1, 2, 3, 4]
    GAMMA_HI = 2.0
    u_mag = 1.0   # conviction magnitude -> asymmetry ~ n_dep*edge_prec*u_mag = 2.4

    # cube[gamma][trust][resource] = list of wrong-bloc finals over seeds
    cube = {}
    levels = [False, True]
    n_cells = 2 * 2 * 2
    done = 0
    for g_on, t_on, r_on in itertools.product(levels, levels, levels):
        gamma = GAMMA_HI if g_on else 0.0
        finals = []
        for s in seeds:
            cfg = make_cfg(gamma, t_on, r_on, s, N, n_steps, n_dep, edge_prec)
            D0, u_src, is_wrong = setup_blocs(
                N, K, n_dep, 0.30, 0, 1, 0.95, u_mag, s)
            out = run_structural(cfg, D_per_agent=D0, u_source_per_agent=u_src)
            finals.append(float(out["final_q"][is_wrong, 1].mean()))
        cube[(g_on, t_on, r_on)] = finals
        done += 1
        print(f"  [{done}/{n_cells}] gamma={'on ' if g_on else 'off'} "
              f"trust={'on ' if t_on else 'off'} rsrc={'on ' if r_on else 'off'} "
              f"-> wrong-bloc q(truth) = {np.mean(finals):.3f} +- {np.std(finals):.3f}")

    def m(g, t, r):
        return float(np.mean(cube[(g, t, r)]))

    base = m(False, False, False)

    section("Cube (wrong-bloc convergence to truth; 1.0=converged, 0=locked)")
    print(f"{'gamma':>6} {'trust':>6} {'rsrc':>6} {'mean':>8} {'std':>7}")
    for g_on, t_on, r_on in itertools.product(levels, levels, levels):
        vals = np.array(cube[(g_on, t_on, r_on)])
        print(f"{str(g_on):>6} {str(t_on):>6} {str(r_on):>6} "
              f"{vals.mean():>8.3f} {vals.std():>7.3f}")

    # Main effects (each channel alone, others off): lock-in = base - convergence
    d_gamma = base - m(True, False, False)
    d_trust = base - m(False, True, False)
    d_rsrc = base - m(False, False, True)

    section("Main effects (lock-in induced by each channel alone)")
    print(f"  baseline (all off) wrong-bloc convergence: {base:.3f}")
    print(f"  gamma alone:     delta = {d_gamma:+.3f}")
    print(f"  trust alone:     delta = {d_trust:+.3f}")
    print(f"  resource alone:  delta = {d_rsrc:+.3f}")

    # Pairwise interactions (third channel held OFF)
    def pair_interaction(dA, dB, both_on_key):
        additive = base - (dA + dB)
        observed = m(*both_on_key)
        return additive - observed, additive, observed

    section("Pairwise interactions (third channel OFF)")
    print("  interaction > 0 => super-additive (more lock-in than sum of parts)")
    int_gt, add_gt, obs_gt = pair_interaction(d_gamma, d_trust, (True, True, False))
    int_gr, add_gr, obs_gr = pair_interaction(d_gamma, d_rsrc, (True, False, True))
    int_tr, add_tr, obs_tr = pair_interaction(d_trust, d_rsrc, (False, True, True))
    print(f"  gamma x trust:     additive={add_gt:.3f}  observed={obs_gt:.3f}  interaction={int_gt:+.3f}")
    print(f"  gamma x resource:  additive={add_gr:.3f}  observed={obs_gr:.3f}  interaction={int_gr:+.3f}")
    print(f"  trust x resource:  additive={add_tr:.3f}  observed={obs_tr:.3f}  interaction={int_tr:+.3f}")

    # Three-way
    additive_3 = base - (d_gamma + d_trust + d_rsrc)
    observed_3 = m(True, True, True)
    int_3 = additive_3 - observed_3
    section("Three-way (all channels on)")
    print(f"  additive prediction = {additive_3:.3f}")
    print(f"  observed all-on     = {observed_3:.3f}")
    print(f"  three-way deviation = {int_3:+.3f}")

    summary = {
        "cube": {f"g{int(g)}_t{int(t)}_r{int(r)}": cube[(g, t, r)]
                 for g, t, r in itertools.product(levels, levels, levels)},
        "base": base,
        "main_effects": {"gamma": d_gamma, "trust": d_trust, "resource": d_rsrc},
        "pairwise_interactions": {
            "gamma_x_trust": int_gt,
            "gamma_x_resource": int_gr,
            "trust_x_resource": int_tr},
        "three_way_deviation": int_3,
    }
    os.makedirs("experiments/results/E18", exist_ok=True)
    with open("experiments/results/E18/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure: main effects + pairwise interactions bar chart ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.bar(["gamma\n(epistemic)", "trust\n(social)", "resource\n(material)"],
            [d_gamma, d_trust, d_rsrc],
            color=["#d0021b", "#3a7bd5", "#f5a623"])
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_ylabel("lock-in induced (base - convergence)")
    ax1.set_title("Main effects: each channel alone")
    ax1.grid(True, alpha=0.3, axis="y")

    pairs = ["gamma x\ntrust", "gamma x\nresource", "trust x\nresource"]
    ints = [int_gt, int_gr, int_tr]
    colors = ["#9b59b6", "#e67e22", "#16a085"]
    ax2.bar(pairs, ints, color=colors)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("interaction (additive pred - observed)")
    ax2.set_title("Pairwise interactions (>0 = super-additive)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("E18: cross-channel interactions on the structural substrate",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("experiments/results/E18/cross_channel_interactions.png", dpi=120)
    plt.close(fig)
    print("\nSaved figure to experiments/results/E18/cross_channel_interactions.png")


if __name__ == "__main__":
    main()
