"""End-to-end demonstration of BMR in the multi-agent simulation loop.

The previous demo (demo_structural_phlogiston.py) showed the BMR utilities
working on a static phlogiston net with offline Gaussian conjugate inference.
This demo shows the SAME machinery running inside the running multi-agent
substrate: each agent maintains per-edge Gaussian posteriors, updated each
step by soft-gated conjugate inference, with Savage-Dickey BMR pruning the
unsupported edges every cfg.bmr_period steps. T_i, kappa_i, U_i are
recomputed live from the surviving structure.

Run from repo root:
    python -m scripts.demo_bmr_in_loop

What you should see:
  * False-paradigm edges (paradigm 0 in the default setup) start at low
    initial precision (cross_paradigm_init=0.05). They accumulate zero-mean
    evidence and get pruned by BMR over the first few periods.
  * True-paradigm edges (paradigm 1 = oxygen) accumulate real evidence
    and survive every BMR pass.
  * Population mean q_B (paradigm-1 belief) rises as the structural learner
    converges.
  * Paradigm-root conservatism kappa stays high for the true paradigm and
    drops for the false paradigm as its edges are pruned.
"""

from __future__ import annotations

import numpy as np

from src.pomdp.gen_model import PomdpConfig
from src.pomdp.simple_step import SimpleConfig
from src.pomdp.motivated_step import MotivatedConfig
from src.pomdp.structural_step import StructuralConfig, run_structural


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    # ------------------------------------------------------------------
    # Configure: hierarchical-context substrate + structural extension +
    # BMR-in-loop with per-dependent observations.
    # ------------------------------------------------------------------
    cfg = StructuralConfig(
        motivated=MotivatedConfig(
            simple=SimpleConfig(
                pomdp=PomdpConfig(
                    x_grid=(0.1, 0.3, 0.5, 0.8, 1.0),
                    true_paradigm=1,           # paradigm 1 = "oxygen" is true
                    q_reliability=0.75),
                n_agents=30,
                n_steps=200,
                obs_x_index=4,
                seed=2,
            ),
            lambda_tilt=0.0,                    # no value tilt -- isolate BMR effect
            motivated=False,
        ),
        n_dependents=3,
        edge_precision=0.7,                     # initial own-paradigm edge precision
        cross_paradigm_init=0.05,               # tentative cross-paradigm edges
        gamma_strength=0.0,
        # BMR-in-loop
        bmr_in_loop=True,
        dep_noise_prec=4.0,                     # observation precision on dependents
        edge_tau_prior=0.05,                    # Gaussian prior precision per edge
        bmr_period=25,                          # apply BMR every 25 steps
        bmr_threshold=0.0,                      # log BF > 0 -> prune
    )

    section("Configuration")
    K = cfg.motivated.simple.pomdp.n_paradigms
    D = K * cfg.n_dependents
    print(f"N agents: {cfg.motivated.simple.n_agents}")
    print(f"n_steps:  {cfg.motivated.simple.n_steps}")
    print(f"K paradigms: {K}  |  n_dependents per paradigm: {cfg.n_dependents}  |  D total: {D}")
    print(f"True paradigm: {cfg.motivated.simple.pomdp.true_paradigm}")
    print(f"Initial own-paradigm edges: {cfg.edge_precision}")
    print(f"Initial cross-paradigm edges (hypothesis): {cfg.cross_paradigm_init}")
    print(f"BMR period: {cfg.bmr_period} steps  |  log BF threshold: {cfg.bmr_threshold}")

    # ------------------------------------------------------------------
    # Run with BMR enabled
    # ------------------------------------------------------------------
    section("Run with BMR-in-loop ENABLED")
    out_bmr = run_structural(cfg)

    pruned_traj = out_bmr["pruned_fraction"]
    print("BMR prune events (cumulative fraction of edges zeroed):")
    print(f"  step  20: {pruned_traj[20]:.3f}")
    print(f"  step  50: {pruned_traj[50]:.3f}")
    print(f"  step 100: {pruned_traj[100]:.3f}")
    print(f"  step 150: {pruned_traj[150]:.3f}")
    print(f"  step 199: {pruned_traj[199]:.3f}")

    alive = out_bmr["final_edge_alive"]                # (N, K, D)
    mu = out_bmr["final_mu_edge"]                      # (N, K, D)
    true_p = cfg.motivated.simple.pomdp.true_paradigm
    false_p = 1 - true_p
    n_dep = cfg.n_dependents

    print("\nPer-paradigm edge survival rates (averaged over agents):")
    p_own_lo, p_own_hi = true_p * n_dep, (true_p + 1) * n_dep
    f_own_lo, f_own_hi = false_p * n_dep, (false_p + 1) * n_dep
    print(f"  TRUE  paradigm {true_p} -> own block (deps {p_own_lo}..{p_own_hi - 1}):  "
          f"alive fraction = {alive[:, true_p, p_own_lo:p_own_hi].mean():.3f}")
    print(f"  TRUE  paradigm {true_p} -> false block (deps {f_own_lo}..{f_own_hi - 1}): "
          f"alive fraction = {alive[:, true_p, f_own_lo:f_own_hi].mean():.3f}")
    print(f"  FALSE paradigm {false_p} -> own block (deps {f_own_lo}..{f_own_hi - 1}):  "
          f"alive fraction = {alive[:, false_p, f_own_lo:f_own_hi].mean():.3f}")
    print(f"  FALSE paradigm {false_p} -> true block (deps {p_own_lo}..{p_own_hi - 1}): "
          f"alive fraction = {alive[:, false_p, p_own_lo:p_own_hi].mean():.3f}")

    print("\nPer-paradigm posterior edge means (averaged over agents and dependents):")
    print(f"  TRUE  paradigm {true_p} -> own block:  mean mu = {mu[:, true_p, p_own_lo:p_own_hi].mean():+.3f}  (~ A_true_dep)")
    print(f"  FALSE paradigm {false_p} -> any block:  mean mu = {mu[:, false_p, :].mean():+.3f}  (should be ~ 0)")

    print("\nParadigm-root conservatism kappa = T * 1 (1 + sum descendant precisions):")
    kappa = out_bmr["final_kappa"]                     # (N, K_int)
    print(f"  kappa[TRUE  paradigm {true_p}]:  {kappa[:, true_p].mean():.3f}")
    print(f"  kappa[FALSE paradigm {false_p}]: {kappa[:, false_p].mean():.3f}  (should be lower)")

    print("\nPopulation belief trajectory in the true paradigm:")
    qB = out_bmr["mean_qB"]
    print(f"  step  0:  mean q(theta=truth) = {qB[0]:.3f}")
    print(f"  step 50:  mean q(theta=truth) = {qB[50]:.3f}")
    print(f"  step 100: mean q(theta=truth) = {qB[100]:.3f}")
    print(f"  step 199: mean q(theta=truth) = {qB[199]:.3f}")

    # ------------------------------------------------------------------
    # Control: same config but with BMR disabled (compare structure)
    # ------------------------------------------------------------------
    section("Control: bmr_period beyond run length (no pruning)")
    cfg_ctrl = StructuralConfig(
        motivated=cfg.motivated,
        n_dependents=cfg.n_dependents,
        edge_precision=cfg.edge_precision,
        cross_paradigm_init=cfg.cross_paradigm_init,
        gamma_strength=cfg.gamma_strength,
        bmr_in_loop=True,                          # still update edges...
        dep_noise_prec=cfg.dep_noise_prec,
        edge_tau_prior=cfg.edge_tau_prior,
        bmr_period=99999,                          # ...but never prune
        bmr_threshold=cfg.bmr_threshold,
    )
    out_ctrl = run_structural(cfg_ctrl)
    print(f"FALSE paradigm kappa with pruning OFF: {out_ctrl['final_kappa'][:, false_p].mean():.3f}")
    print(f"FALSE paradigm kappa with pruning ON:  {kappa[:, false_p].mean():.3f}")
    print(f"Conservatism drop from BMR: {(out_ctrl['final_kappa'][:, false_p].mean() - kappa[:, false_p].mean()):.3f}")

    # ------------------------------------------------------------------
    section("Summary")
    print("BMR-in-loop is wired into the multi-agent simulation:")
    print("  1. World samples per-dependent observations from A_true_dep")
    print("  2. Each agent does soft-gated Gaussian conjugate edge inference")
    print("     (gate on edge (p, d) = q_i(theta=p) -- the principled E-step")
    print("     responsibility for a latent-categorical-parent linear-Gaussian model)")
    print("  3. Every bmr_period steps: Savage-Dickey log Bayes factor per edge")
    print("  4. Persistent sticky-alive mask keeps pruned edges pruned")
    print("  5. T_i, kappa_i, U_i are recomputed live from surviving structure")
    print("All inference closed-form. No heuristics in the inference path.")


if __name__ == "__main__":
    main()
