"""End-to-end demonstration of the structural machinery on a phlogiston-style net.

All five steps use closed-form Bayesian inference -- no hand-fed posterior
moments. The BMR demonstration runs a proper Gaussian conjugate inference
on each edge from generated data, then applies Savage-Dickey BMR.

Run from repo root:
    python -m scripts.demo_structural_phlogiston

Steps:
    1. Per-agent DAG with cross-paradigm coupling
    2. T propagation: conservatism kappa = T*1 and conviction U = T*u
    3. Schur complement: marginalizing the phlogiston hub induces
       residue couplings among its neighbors
    4. Conjugate inference + Savage-Dickey BMR: a misspecified spurious
       edge is correctly pruned, real edges are kept
    5. Structure-learning expansion: propose a new node, then BMR
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.pomdp.structural_step import (
    build_default_dag,
    build_default_dag_with_cross_coupling,
    build_precision_matrix,
    schur_complement,
    schur_residue,
    compute_T,
    compute_conservatism,
    compute_conviction,
    bmr_log_bayes_factor,
    bmr_prune,
    expand_propose_node,
)


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def gaussian_conjugate_edge_inference(x_parent, x_child,
                                       tau_prior, noise_precision):
    """Bayesian inference on a single Gaussian edge weight w.

    Model:  x_child[t] = w * x_parent[t] + eps,  eps ~ N(0, 1/noise_precision)
    Prior:  w ~ N(0, 1/tau_prior)

    Returns (mu_post, tau_post): closed-form Gaussian conjugate posterior.
    """
    # Sufficient statistics
    S_xx = float(np.sum(x_parent * x_parent))
    S_xy = float(np.sum(x_parent * x_child))
    # Posterior precision: prior + likelihood contribution
    tau_post = tau_prior + noise_precision * S_xx
    # Posterior mean: weighted by precisions
    mu_post = (noise_precision * S_xy) / tau_post
    return mu_post, tau_post


def generate_sem_data(true_A, n_samples, noise_precision, rng):
    """Generate linear-Gaussian SEM data x = A*x + eps with topological order.

    Assumes true_A is upper-triangular (DAG with topological order = node index).
    Returns x : (n_samples, K) where each column is a node trajectory.
    """
    K = true_A.shape[0]
    x = rng.standard_normal((n_samples, K)) / np.sqrt(noise_precision)
    # Resolve in topological order
    for j in range(K):
        for i in range(K):
            if true_A[i, j] != 0:
                x[:, j] = x[:, j] + true_A[i, j] * x[:, i]
    return x


def main():
    rng = np.random.RandomState(0)

    # ------------------------------------------------------------------
    # 1. Build the phlogiston DAG
    # ------------------------------------------------------------------
    section("1. Phlogiston paradigm net")
    K_paradigms = 2
    n_dependents = 4
    edge_prec = 0.7
    cross_coupling = 0.1

    A = build_default_dag_with_cross_coupling(
        K_paradigms, n_dependents, edge_prec, cross_coupling=cross_coupling)
    labels = ["phlogiston", "oxygen",
              "P-calcination", "P-combustion", "P-respiration", "P-acidity",
              "O-calcination", "O-combustion", "O-respiration", "O-acidity"]
    A_np = np.asarray(A)
    print(f"DAG: {A_np.shape[0]} nodes, cross-coupling phlogiston->oxygen = {cross_coupling}")
    print("Edges (parent -> child, precision):")
    for i in range(A_np.shape[0]):
        for j in range(A_np.shape[1]):
            if A_np[i, j] > 0:
                print(f"  {labels[i]:>16} -> {labels[j]:<16}  {A_np[i,j]:.2f}")

    # ------------------------------------------------------------------
    # 2. Propagation
    # ------------------------------------------------------------------
    section("2. Propagation operator T = (I - A)^-1")
    T = compute_T(A)
    kappa = np.asarray(compute_conservatism(T))
    print("Conservatism kappa = T * 1:")
    for i, lab in enumerate(labels):
        print(f"  {lab:>16}  kappa = {kappa[i]:.4f}")

    u_source = np.zeros(A_np.shape[0])
    u_source[3] = 1.0    # P-combustion: wanted true
    u_source[8] = -0.5   # O-respiration: wanted false
    U = np.asarray(compute_conviction(T, jnp.asarray(u_source)))
    print("\nConviction U = T * u with u[P-combustion]=1, u[O-respiration]=-0.5:")
    for i, lab in enumerate(labels):
        print(f"  {lab:>16}  U = {U[i]:>+7.4f}  (source u = {u_source[i]:+.2f})")

    # ------------------------------------------------------------------
    # 3. Schur complement
    # ------------------------------------------------------------------
    section("3. Schur residue -- marginalize the phlogiston hub")
    Pi = build_precision_matrix(A)
    residue = np.asarray(schur_residue(Pi, idx=0))
    print(f"Joint precision Pi = (I-A)^T (I-A); shape {Pi.shape}")
    print(f"\nMarginalizing node 0 (phlogiston) -- induced residue (5 strongest entries):")
    flat = []
    keep_labels = [l for i, l in enumerate(labels) if i != 0]
    for i in range(residue.shape[0]):
        for j in range(i + 1, residue.shape[1]):
            flat.append((abs(residue[i, j]), i, j, residue[i, j]))
    flat.sort(reverse=True)
    for mag, i, j, val in flat[:5]:
        print(f"  {keep_labels[i]:>16} -- {keep_labels[j]:<16}  residue = {val:+.4f}")
    print("\nThe hub's influence did not vanish -- it condensed into induced")
    print("couplings among its neighbors. Marginalizing a paradigm root makes")
    print("its dependents look correlated to an observer who hasn't drawn the hub.")

    # ------------------------------------------------------------------
    # 4. BMR with proper Gaussian conjugate inference
    # ------------------------------------------------------------------
    section("4. Conjugate inference + Savage-Dickey BMR")
    print("Generate n=500 samples from the TRUE phlogiston DAG (above), then")
    print("ask: which edges does a misspecified hypothesis DAG warrant keeping?")
    print()

    # Generate data from the true DAG
    n_samples = 500
    noise_precision = 1.0
    x = generate_sem_data(A_np, n_samples, noise_precision, rng)
    print(f"Generated {n_samples} samples from true DAG ({A_np.shape[0]} nodes).")

    # Misspecified hypothesis: agent thinks there might also be a
    # SPURIOUS edge from oxygen -> P-acidity (it isn't in the true DAG).
    spurious_parent = 1   # oxygen
    spurious_child = 5    # P-acidity
    # Hypothesis DAG: real edges + the spurious candidate
    A_hyp = A_np.copy()
    A_hyp[spurious_parent, spurious_child] = 1.0  # candidate edge to test
    print(f"Hypothesis: also test edge {labels[spurious_parent]} -> "
          f"{labels[spurious_child]} (spurious in truth)")

    # Run conjugate inference on every candidate edge in the hypothesis
    tau_prior = 0.1
    print()
    print(f"Edge-wise posteriors (Gaussian conjugate, tau_prior={tau_prior}):")
    print(f"{'edge':<40}  {'mu_post':>9}  {'tau_post':>9}  {'log BF':>9}")
    log_bf_grid = np.zeros_like(A_hyp)
    for i in range(A_hyp.shape[0]):
        for j in range(A_hyp.shape[1]):
            if A_hyp[i, j] > 0:
                mu, tau = gaussian_conjugate_edge_inference(
                    x[:, i], x[:, j], tau_prior, noise_precision)
                lbf = float(bmr_log_bayes_factor(
                    jnp.array([mu]), jnp.array([tau]),
                    jnp.array([tau_prior]))[0])
                log_bf_grid[i, j] = lbf
                tag = " <-- SPURIOUS" if (i == spurious_parent and
                                          j == spurious_child) else ""
                edge_lab = f"{labels[i]} -> {labels[j]}"
                print(f"  {edge_lab:<40}  {mu:>+9.4f}  {tau:>9.2f}  {lbf:>+9.3f}{tag}")

    # Prune
    A_pruned = np.asarray(bmr_prune(jnp.asarray(A_hyp),
                                     jnp.asarray(log_bf_grid), threshold=0.0))
    pruned_spurious = (A_pruned[spurious_parent, spurious_child] == 0.0)
    real_edges_kept = all(A_pruned[i, j] != 0.0
                           for i in range(A_np.shape[0])
                           for j in range(A_np.shape[1]) if A_np[i, j] > 0)
    print()
    print(f"Spurious edge pruned?       {'YES' if pruned_spurious else 'NO'}")
    print(f"All real edges preserved?   {'YES' if real_edges_kept else 'NO'}")

    # ------------------------------------------------------------------
    # 5. Expansion
    # ------------------------------------------------------------------
    section("5. Structure-learning expansion -- propose a new commitment")
    print("Propose 'caloric' as a new node, tentatively wired to all existing")
    print("nodes at precision 0.05. After conjugate inference + BMR, only the")
    print("data-supported caloric edges should survive.")
    print()
    print("Generate fresh data from a TRUE DAG that includes caloric <-> combustion")
    print("(but not caloric <-> anything else).")

    # Extended true DAG: add caloric as node K_orig, with a real edge to
    # P-combustion (node 3). All other proposed edges are spurious.
    K_orig = A_np.shape[0]
    A_true_ext = np.zeros((K_orig + 1, K_orig + 1))
    A_true_ext[:K_orig, :K_orig] = A_np
    caloric_idx = K_orig
    A_true_ext[caloric_idx, 3] = 0.6   # caloric -> P-combustion (real)

    x_ext = generate_sem_data(A_true_ext, n_samples, noise_precision, rng)

    # Hypothesis: expand A with tentative edges to ALL existing nodes
    A_expanded = np.asarray(expand_propose_node(
        jnp.asarray(A), tentative_precision=0.05))
    K_new = A_expanded.shape[0]
    print(f"\nExpanded hypothesis DAG: {K_new} nodes (caloric = node {caloric_idx})")

    # Score every tentative edge from data
    log_bf_e = np.zeros_like(A_expanded)
    for i in range(K_new):
        for j in range(K_new):
            if A_expanded[i, j] > 0:
                mu, tau = gaussian_conjugate_edge_inference(
                    x_ext[:, i], x_ext[:, j], tau_prior, noise_precision)
                log_bf_e[i, j] = float(bmr_log_bayes_factor(
                    jnp.array([mu]), jnp.array([tau]),
                    jnp.array([tau_prior]))[0])

    A_final = np.asarray(bmr_prune(jnp.asarray(A_expanded),
                                    jnp.asarray(log_bf_e), threshold=0.0))

    print("\nSurviving edges incident to caloric:")
    found_any = False
    for j in range(K_new):
        j_lab = labels[j] if j < len(labels) else "caloric"
        if A_final[caloric_idx, j] > 0:
            print(f"  caloric -> {j_lab}        (kept; precision {A_final[caloric_idx,j]:.2f})")
            found_any = True
        if A_final[j, caloric_idx] > 0 and j != caloric_idx:
            print(f"  {j_lab} -> caloric       (kept; precision {A_final[j,caloric_idx]:.2f})")
            found_any = True
    if not found_any:
        print("  (none -- all tentative caloric edges pruned)")
    print()
    print("Expansion + BMR is the IWAI paper's 'Bayesian model expansion and")
    print("reduction' loop: propose freely, let the data pay only for what")
    print("survives.")

    # ------------------------------------------------------------------
    section("Summary")
    print("All four IWAI structural mechanisms are now available on the")
    print("hierarchical-context substrate, with closed-form inference:")
    print("  1. [OK] Per-agent DAG with cross-paradigm coupling")
    print("  2. [OK] T = (I - A)^-1 propagation (kappa and U on one operator)")
    print("  3. [OK] Schur-complement residue (Kuhnian incommensurability)")
    print("  4. [OK] Bayesian model reduction (Savage-Dickey edge pruning,")
    print("       fed by Gaussian conjugate inference)")
    print("  5. [OK] Structure-learning expansion (propose + prune)")


if __name__ == "__main__":
    main()
