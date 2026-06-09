"""Benchmark the cont substrate on whatever JAX backend is active.

Runs a fixed set of cont simulations and reports:
  - JAX backend + device
  - compile (first-run) time
  - steady-state per-run time (after compile)
  - total wall time for the batch

Run on CPU (Windows) and GPU (WSL) and compare. Identical workload both
times so the numbers are directly comparable.

    python -m scripts.benchmark_backend
"""

from __future__ import annotations

import time
import numpy as np
import jax

from src.config import WorldConfig
from src.pomdp.gen_model import PomdpConfig
from src.pomdp.cont_step import ContConfig, run_cont


def make_cfg(seed, n_agents, n_steps):
    return ContConfig(
        pomdp=PomdpConfig(
            n_paradigms=2, theta_vals=(0.0, 1.0), true_paradigm=1,
            x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), q_reliability=0.85,
            world=WorldConfig(
                sigma=0.5, schedule="reversal",
                theta_star_pre=0.0, theta_star_post=1.0,
                schedule_t_shift=40, schedule_t_reverse=320)),
        lambda_init=1.0, eta_lambda=0.0, eps_theta=0.15, obs_x_index=4,
        n_agents=n_agents, n_steps=n_steps, use_theta_schedule=True,
        graph_kind="watts_strogatz", mean_degree=4, social_mask=0.3,
        trust_learning=True, trust_rho=0.95, trust_alpha0=1.0, trust_beta0=1.0,
        resource_coupling=True, r_init=4.0,
        R_in=0.03, alpha_flow=0.5, delta_decay=0.0,
        c0=0.02, r_min=0.1, budget_fraction=0.5, seed=seed)


def main():
    N_AGENTS = 30
    N_STEPS = 500
    N_RUNS = 10

    print("=" * 64)
    print("Backend benchmark")
    print("=" * 64)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices:     {jax.devices()}")
    print(f"Workload:    {N_RUNS} runs, N={N_AGENTS} agents, T={N_STEPS} steps")
    print("-" * 64)

    # First run (includes compile)
    t0 = time.time()
    out = run_cont(make_cfg(0, N_AGENTS, N_STEPS),
                   D_per_agent=np.full((N_AGENTS, 2), 0.5))
    _ = float(out["mean_qB"][-1])  # force materialization
    t_first = time.time() - t0
    print(f"First run (compile + exec):  {t_first:7.3f} s")

    # Steady-state runs
    times = []
    for s in range(1, N_RUNS):
        t0 = time.time()
        out = run_cont(make_cfg(s, N_AGENTS, N_STEPS),
                       D_per_agent=np.full((N_AGENTS, 2), 0.5))
        _ = float(out["mean_qB"][-1])
        times.append(time.time() - t0)

    times = np.array(times)
    print(f"Steady-state per run:        {times.mean():7.3f} s "
          f"(min {times.min():.3f}, max {times.max():.3f})")
    print(f"Total wall ({N_RUNS} runs):       "
          f"{t_first + times.sum():7.3f} s")
    print("=" * 64)


if __name__ == "__main__":
    main()
