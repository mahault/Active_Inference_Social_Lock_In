"""Structural extension: per-agent dependency network + T propagation.

Adds the IWAI paper's structural machinery to the hierarchical-context
substrate without rewriting the inference loop. Each agent carries:

- An internal DAG A_i over K_int commitments (K_paradigms paradigm nodes
  + n_dependents subsidiary commitment nodes per paradigm). A_i has
  paradigm nodes as roots; each paradigm is the parent of its subsidiary
  commitments. By default the DAG is generated automatically; users can
  override with A_per_agent.
- The propagation operator T_i = (I − A_i)^{-1}. For a DAG, A_i is nilpotent,
  so the Neumann series terminates: T_i = sum_{k≥0} A_i^k.
- A source utility vector u_i over the K_int nodes. The propagated
  conviction is U_i = T_i · u_i; the per-paradigm slice
  U_i[:K_paradigms] is what enters the value-tilted update.
- A propagated conservatism vector kappa_i = T_i · 1.

Two new mechanisms layered on top of motivated_step:

1. **Propagated conviction tilt.** The value-tilted posterior uses
   U_i[:K_paradigms] (propagated through structure) instead of the raw
   per-paradigm utility. A paradigm that supports many cherished
   subsidiary commitments inherits their conviction.

2. **Endogenous γ.** Per-agent evidence weight
       w_i = exp(-gamma_strength · conviction_asymmetry_i)
   attenuates the world log-likelihood for agents with strong paradigm
   convictions. This is the formal counterpart of the IWAI paper's
   γ-crossover: high conviction silences disconfirming evidence at the
   precision level, so the susceptibility of the posterior to
   disconfirming observations collapses.

In-loop structure inference (cfg.bmr_in_loop=True):
The simulation can additionally generate per-dependent-node observations
sampled from the world's TRUE linear-Gaussian SEM (default: only the true
paradigm has non-zero edges to its own block of dependents; override via
A_true_dep). Each agent then runs SOFT-GATED Gaussian conjugate inference
on each edge in their hypothesis DAG: the gate on edge (p, d) is the
agent's current posterior q_i(theta = p), so an edge only learns when
the agent puts mass on paradigm p. Every cfg.bmr_period steps, the
Savage-Dickey log Bayes factor is computed in closed form per edge; edges
with positive log BF are zeroed out, the prune persists via a sticky
alive-mask, and T_i, kappa_i, U_i are recomputed from the live edge means.
This makes the multi-agent substrate a real Bayesian structure-learner:
spurious cross-paradigm edges are pruned as data accumulates, and the
propagated conviction field U evolves with the surviving structure.

BMR, expansion, and Schur are also available as one-off operations for
analysis scripts (see scripts/demo_structural_phlogiston.py for a closed-
form conjugate-inference end-to-end demonstration on a 10-node net).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from src.pomdp import agent_pop
from src.pomdp.gen_model import build_generative_model
from src.pomdp.motivated_step import (
    MotivatedConfig,
    MotivatedState,
    apply_value_tilt,
    init_motivated,
)
from src.pomdp.simple_step import (
    EPS,
    SimpleState,
    affordable_experiment,
    build_joint_A_world,
    build_joint_A_social,
    build_B_c,
    build_trust,
    categorical_surprisal,
    flow_from_trust,
    flow_step,
    h1,
    inflow_share,
    marginalize_context,
    marginalize_theta,
    readout_trust_mixture,
    sample_observations,
    transition_joint,
    trust_update,
)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass(frozen=True)
class StructuralConfig:
    motivated: MotivatedConfig = field(default_factory=MotivatedConfig)
    n_dependents: int = 3          # subsidiary commitments per paradigm
    edge_precision: float = 0.8    # coupling strength on each paradigm->child edge
    gamma_strength: float = 0.0    # 0 = no endogenous gamma; > 0 enables silencing
    u_source_dependents: float = 0.5  # default per-dependent utility magnitude

    # ------------------------------------------------------------------
    # BMR-in-loop (Bayesian model reduction over per-agent DAGs).
    # When enabled, the world generates per-dependent-node observations
    # x_d ~ N(A_true_dep[theta_true, d], 1/dep_noise_prec). Each agent
    # maintains a per-edge Gaussian posterior over A_i[p, d], updated
    # via soft-gated conjugate inference with gate q_i(theta=p). Every
    # bmr_period steps, edges with positive Savage-Dickey log Bayes
    # factor are pruned; T_i, kappa_i, U_i are recomputed from the live
    # mu_post values. This makes the substrate a true structure-learner.
    # ------------------------------------------------------------------
    bmr_in_loop: bool = False
    dep_noise_prec: float = 1.0      # noise precision on dependent observations
    edge_tau_prior: float = 0.1      # Gaussian prior precision per edge weight
    cross_paradigm_init: float = 0.05  # initial precision on hypothesis-only cross-paradigm edges
    bmr_period: int = 20             # apply BMR every this many steps (skip t=0)
    bmr_threshold: float = 0.0       # log BF > threshold -> prune the edge


# ------------------------------------------------------------------
# DAG and propagation operator
# ------------------------------------------------------------------

def build_default_dag(K_paradigms: int, n_dependents: int,
                      edge_precision: float) -> jnp.ndarray:
    """Generate a default per-agent DAG.

    Nodes 0..K_paradigms-1 are paradigm roots. Nodes K_paradigms..K_int-1
    are subsidiary commitments, each a child of exactly one paradigm.
    Convention: A[parent, child] = edge_precision (IWAI paper §3.1, eq. 6).
    Then T = (I − A)^{-1} has T[i, w] = total path weight from i to w,
    so κ_i = (T · 1)_i = 1 + sum of i's descendant precisions, and
    U_i = (T · u)_i = u_i + propagated utility from descendants.
    """
    K_int = K_paradigms * (1 + n_dependents)
    A = jnp.zeros((K_int, K_int))
    for p in range(K_paradigms):
        for d in range(n_dependents):
            child = K_paradigms + p * n_dependents + d
            A = A.at[p, child].set(edge_precision)
    return A


def build_default_dag_with_cross_coupling(K_paradigms: int, n_dependents: int,
                                          edge_precision: float,
                                          cross_coupling: float = 0.0
                                          ) -> jnp.ndarray:
    """Extends `build_default_dag` with optional directed cross-paradigm edges.

    For each ordered pair of paradigms (p, q) with p < q, adds A[p, q] =
    cross_coupling. This creates an upper-triangular cross-paradigm structure
    (still a DAG) where downstream paradigms inherit some upstream influence.
    Setting cross_coupling = 0 recovers the original disconnected paradigms.
    """
    A = build_default_dag(K_paradigms, n_dependents, edge_precision)
    if cross_coupling > 0.0:
        for p in range(K_paradigms):
            for q in range(p + 1, K_paradigms):
                A = A.at[p, q].set(cross_coupling)
    return A


# ------------------------------------------------------------------
# Schur complement
# ------------------------------------------------------------------

def build_precision_matrix(A: jnp.ndarray,
                           D: jnp.ndarray | None = None) -> jnp.ndarray:
    """Joint precision matrix Pi = (I - A)^T · D · (I - A) for a linear-Gaussian
    structural equation model x = A·x + noise with noise precision diagonal D.

    Defaults D = I (unit noise precision per node). Pi is symmetric positive
    semi-definite. Off-diagonal entries encode conditional dependencies.
    """
    K_int = A.shape[0]
    if D is None:
        D = jnp.eye(K_int)
    elif D.ndim == 1:
        D = jnp.diag(D)
    M = jnp.eye(K_int) - A
    return M.T @ D @ M


def schur_complement(Pi: jnp.ndarray, idx: int) -> jnp.ndarray:
    """Schur complement: marginalize out node `idx` from precision matrix Pi.

        Pi' = Pi[~idx, ~idx] - Pi[~idx, idx] · (1/Pi[idx, idx]) · Pi[idx, ~idx]

    The residue (the second term) is the "carry-over" coupling that the
    marginalization deposits on idx's neighbors. The IWAI paper calls this
    the incommensurability term: marginalizing a hidden hub induces explicit
    couplings among its dependents.

    Args:
        Pi: (K, K) precision matrix
        idx: index of node to marginalize out

    Returns:
        Pi_reduced: (K-1, K-1) precision matrix over the remaining nodes
    """
    K = Pi.shape[0]
    keep = [i for i in range(K) if i != idx]
    keep_idx = jnp.asarray(keep)
    P_kk = Pi[jnp.ix_(keep_idx, keep_idx)]
    P_kv = Pi[keep_idx, idx]
    P_vv = Pi[idx, idx]
    return P_kk - jnp.outer(P_kv, P_kv) / (P_vv + 1e-12)


def schur_residue(Pi: jnp.ndarray, idx: int) -> jnp.ndarray:
    """The induced-coupling residue alone: Pi[~idx, idx] · (1/Pi[idx, idx]) · Pi[idx, ~idx]."""
    K = Pi.shape[0]
    keep = [i for i in range(K) if i != idx]
    keep_idx = jnp.asarray(keep)
    P_kv = Pi[keep_idx, idx]
    P_vv = Pi[idx, idx]
    return jnp.outer(P_kv, P_kv) / (P_vv + 1e-12)


# ------------------------------------------------------------------
# Bayesian model reduction — edge pruning via Savage-Dickey
# ------------------------------------------------------------------

def bmr_log_bayes_factor(mu_post: jnp.ndarray, tau_post: jnp.ndarray,
                         tau_prior: jnp.ndarray) -> jnp.ndarray:
    """Log Bayes factor for setting a Gaussian edge weight to zero (pruning it).

    For a single edge weight w with:
      prior:     w ~ N(0, 1/tau_prior)
      posterior: w ~ N(mu_post, 1/tau_post)
    the Savage-Dickey density ratio at w = 0 is the ratio of posterior to
    prior densities at zero:
        BF = p(w=0 | data) / p(w=0 | prior)
           = sqrt(tau_post / tau_prior) · exp(-0.5 · mu_post^2 · tau_post)

    LOG BF > 0 → prune the edge (data does not support a nonzero weight).
    LOG BF < 0 → keep the edge (posterior concentrates away from zero).

    Operates element-wise on arrays of edge weights/precisions.
    """
    log_ratio_norm = 0.5 * jnp.log((tau_post + EPS) / (tau_prior + EPS))
    log_density_term = -0.5 * mu_post * mu_post * tau_post
    return log_ratio_norm + log_density_term


def bmr_prune(A: jnp.ndarray, log_bf: jnp.ndarray,
              threshold: float = 0.0) -> jnp.ndarray:
    """Zero out edges in A where log BF exceeds threshold.

    A: (K, K) adjacency. log_bf: same shape, log Bayes factor per (parent, child) cell.
    Returns the pruned adjacency.
    """
    keep_mask = (log_bf <= threshold).astype(A.dtype)
    return A * keep_mask


# ------------------------------------------------------------------
# Structure-learning expansion — propose new node with tentative edges
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Per-edge soft-gated Gaussian conjugate inference
# ------------------------------------------------------------------

def sample_dependent_observations(A_true_dep: jnp.ndarray,
                                  true_paradigm: int,
                                  key: jax.Array,
                                  N: int,
                                  noise_prec: float) -> jnp.ndarray:
    """World samples per-agent observations on each dependent commitment.

    Model: x_d | theta_true ~ N(A_true_dep[theta_true, d], 1/noise_prec).
    A_true_dep: (K_paradigms, D_total) -- true linear-Gaussian edge weights
        from each paradigm to each dependent. Only row [theta_true] is the
        actual generative root; other rows are alternative-paradigm beliefs
        that the truth happens to not realize. Returns (N, D_total).
    """
    mu_d = A_true_dep[true_paradigm]                   # (D_total,)
    D_total = mu_d.shape[0]
    keys = jax.random.split(key, N)
    sigma = 1.0 / jnp.sqrt(noise_prec)
    eps = jax.vmap(lambda k: jax.random.normal(k, (D_total,)))(keys) * sigma
    return mu_d[None, :] + eps                          # (N, D_total)


def update_edge_posteriors(S_xx: jnp.ndarray, S_xy: jnp.ndarray,
                           q_theta: jnp.ndarray, x_dep: jnp.ndarray,
                           noise_prec: float, tau_prior: float
                           ) -> tuple[jnp.ndarray, jnp.ndarray,
                                       jnp.ndarray, jnp.ndarray]:
    """Soft-gated conjugate update for per-edge Gaussian posteriors.

    Model (per agent, per edge (p, d)):
        x_d | A, theta=p ~ N(A[p,d], 1/noise_prec)
    Latent parent is the paradigm indicator I[theta=p]. Under soft
    assignment, the E-step responsibility on edge (p, d) is r_pd = q(theta=p).

    Sufficient stats accumulate with the responsibility:
        S_xx[p,d] += r_pd      (since I[theta=p]^2 = I[theta=p])
        S_xy[p,d] += r_pd * x_d
    Posterior (with prior A[p,d] ~ N(0, 1/tau_prior)):
        tau_post[p,d] = tau_prior + noise_prec * S_xx[p,d]
        mu_post[p,d]  = (noise_prec * S_xy[p,d]) / tau_post[p,d]

    Args:
        S_xx, S_xy:  (N, K, D) running sufficient stats
        q_theta:     (N, K)    current per-agent paradigm posterior
        x_dep:       (N, D)    per-agent dependent observations
        noise_prec:  scalar    likelihood precision
        tau_prior:   scalar    Gaussian prior precision on edge weights

    Returns: (S_xx_new, S_xy_new, mu_post, tau_post), all (N, K, D).
    """
    r = q_theta[:, :, None]                            # (N, K, 1)
    delta_S_xx = r                                     # broadcasts over D
    delta_S_xy = r * x_dep[:, None, :]                 # (N, K, D)
    S_xx_new = S_xx + delta_S_xx
    S_xy_new = S_xy + delta_S_xy
    tau_post = tau_prior + noise_prec * S_xx_new
    mu_post = (noise_prec * S_xy_new) / (tau_post + EPS)
    return S_xx_new, S_xy_new, mu_post, tau_post


def build_A_per_agent_from_edges(mu_edge: jnp.ndarray,
                                 K_paradigms: int) -> jnp.ndarray:
    """Assemble per-agent K_int x K_int DAG from (N, K, D) edge means.

    Edges live in the (paradigm, dependent) block: A[p, K+d] = mu_edge[p, d].
    Paradigm-paradigm and dependent-dependent blocks are zero. K_int = K + D.
    """
    N, K, D = mu_edge.shape
    K_int = K + D
    A = jnp.zeros((N, K_int, K_int), dtype=mu_edge.dtype)
    A = A.at[:, :K, K:].set(mu_edge)
    return A


def expand_propose_node(A: jnp.ndarray,
                        tentative_precision: float = 0.1) -> jnp.ndarray:
    """Propose a new node by appending a row and column to A.

    The new node (index K) is wired to every existing node with tentative
    bidirectional edges at low precision (tentative_precision). After BMR,
    most of these will be pruned; the surviving edges are the data-supported
    structural revision.

    Returns: (K+1, K+1) expanded adjacency.
    """
    K = A.shape[0]
    new_A = jnp.zeros((K + 1, K + 1), dtype=A.dtype)
    new_A = new_A.at[:K, :K].set(A)
    # Outgoing tentative edges from new node to existing nodes
    new_A = new_A.at[K, :K].set(tentative_precision)
    # Incoming tentative edges from existing nodes to new node
    new_A = new_A.at[:K, K].set(tentative_precision)
    return new_A


def compute_T(A: jnp.ndarray) -> jnp.ndarray:
    """T = (I - A)^{-1}. For a nilpotent DAG, this is a finite series.

    We use direct linear solve for numerical safety; A is small (K_int ~ 8).
    """
    K_int = A.shape[0]
    I = jnp.eye(K_int)
    return jnp.linalg.solve(I - A, I)


def compute_conservatism(T: jnp.ndarray) -> jnp.ndarray:
    """κ = T · 1. Each entry is 1 + (precision-weighted descendant mass)."""
    K_int = T.shape[0]
    return T @ jnp.ones(K_int)


def compute_conviction(T: jnp.ndarray, u_source: jnp.ndarray) -> jnp.ndarray:
    """U = T · u. The propagated utility field; per-node propagated value."""
    return T @ u_source


# ------------------------------------------------------------------
# Endogenous γ via evidence attenuation
# ------------------------------------------------------------------

def conviction_asymmetry(U_paradigm: jnp.ndarray) -> jnp.ndarray:
    """Per-agent conviction strength: max - min over paradigm nodes."""
    return U_paradigm.max(axis=-1) - U_paradigm.min(axis=-1)


def evidence_weight(asymmetry: jnp.ndarray, gamma_strength: float) -> jnp.ndarray:
    """w_i = exp(-gamma_strength · conviction_asymmetry_i). High conviction
    → low evidence weight. Returns shape (N,) values in (0, 1]."""
    return jnp.exp(-gamma_strength * asymmetry)


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

@dataclass(frozen=True)
class StructuralState:
    motivated: MotivatedState
    T_per_agent: jax.Array       # (N, K_int, K_int)
    u_source: jax.Array          # (N, K_int)
    kappa: jax.Array             # (N, K_int)
    U_propagated: jax.Array      # (N, K_int)
    # BMR-in-loop state (None when cfg.bmr_in_loop is False)
    S_xx_edge: jax.Array | None = None     # (N, K, D) sufficient stats
    S_xy_edge: jax.Array | None = None     # (N, K, D)
    mu_edge: jax.Array | None = None       # (N, K, D) current posterior means
    tau_edge: jax.Array | None = None      # (N, K, D) current posterior precisions
    edge_alive: jax.Array | None = None    # (N, K, D) 0/1 mask -- 1 = not pruned


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def init_structural(cfg: StructuralConfig,
                    D_per_agent: np.ndarray | None = None,
                    u_source_per_agent: np.ndarray | None = None,
                    A_per_agent: np.ndarray | None = None) -> StructuralState:
    K = cfg.motivated.simple.pomdp.n_paradigms
    D = K * cfg.n_dependents             # total dependents
    K_int = K + D
    N = cfg.motivated.simple.n_agents

    if A_per_agent is None:
        if cfg.bmr_in_loop:
            # Seed the hypothesis DAG with BOTH own-paradigm edges
            # (high precision) and cross-paradigm edges (low precision).
            # BMR will then test which edges survive the data.
            mu0 = jnp.zeros((N, K, D), dtype=jnp.float32)
            for p in range(K):
                for d in range(cfg.n_dependents):
                    d_idx = p * cfg.n_dependents + d   # this paradigm's own dependent
                    mu0 = mu0.at[:, p, d_idx].set(cfg.edge_precision)
                # cross edges: every other paradigm gets a low init precision
                for d_idx in range(D):
                    if (d_idx // cfg.n_dependents) != p:
                        mu0 = mu0.at[:, p, d_idx].set(cfg.cross_paradigm_init)
            A_batch = build_A_per_agent_from_edges(mu0, K)
        else:
            A_default = build_default_dag(K, cfg.n_dependents, cfg.edge_precision)
            A_batch = jnp.broadcast_to(A_default, (N, K_int, K_int))
    else:
        A_batch = jnp.asarray(A_per_agent, dtype=float)
        assert A_batch.shape == (N, K_int, K_int)

    T_batch = jax.vmap(compute_T)(A_batch)               # (N, K_int, K_int)

    if u_source_per_agent is None:
        u_source = jnp.zeros((N, K_int))
    else:
        u_source = jnp.asarray(u_source_per_agent, dtype=float)
        assert u_source.shape == (N, K_int)

    kappa = jax.vmap(compute_conservatism)(T_batch)
    U_prop = jax.vmap(compute_conviction)(T_batch, u_source)

    U_paradigm = U_prop[:, :K]
    base_mot_state = init_motivated(cfg.motivated, D_per_agent,
                                    U_per_agent=np.asarray(U_paradigm))

    # BMR-in-loop edge state
    S_xx0 = S_xy0 = mu0_edge = tau0_edge = alive0 = None
    if cfg.bmr_in_loop:
        # Initial sufficient stats: zero. mu = 0, tau = tau_prior. We seed
        # the working A_batch (above) with high own / low cross precisions
        # to give propagation a sensible starting field, but the EDGE
        # POSTERIORS start at the proper Gaussian prior N(0, 1/tau_prior).
        # Subsequent observations move them.
        S_xx0 = jnp.zeros((N, K, D))
        S_xy0 = jnp.zeros((N, K, D))
        mu0_edge = jnp.zeros((N, K, D))
        tau0_edge = jnp.full((N, K, D), cfg.edge_tau_prior)
        alive0 = jnp.ones((N, K, D))

    return StructuralState(motivated=base_mot_state, T_per_agent=T_batch,
                           u_source=u_source, kappa=kappa,
                           U_propagated=U_prop,
                           S_xx_edge=S_xx0, S_xy_edge=S_xy0,
                           mu_edge=mu0_edge, tau_edge=tau0_edge,
                           edge_alive=alive0)


# ------------------------------------------------------------------
# Single step
# ------------------------------------------------------------------

def structural_step(state: StructuralState,
                    gm_joint: dict,
                    A_world_orig: jnp.ndarray,
                    T_trust: jnp.ndarray,
                    adj: jnp.ndarray,
                    cfg: StructuralConfig,
                    social_mask: jnp.ndarray,
                    t: int,
                    A_true_dep: jnp.ndarray | None = None
                    ) -> tuple[StructuralState, jnp.ndarray, dict]:
    """One step with structural-aware inference.

    With cfg.bmr_in_loop=False (default): existing behavior. Computes per-agent
    evidence attenuation from propagated conviction, runs hierarchical-context
    inference with attenuated world LL, applies value tilt using propagated
    per-paradigm conviction.

    With cfg.bmr_in_loop=True: additionally samples per-dependent observations
    from the world's true SEM (A_true_dep), updates per-edge Gaussian posteriors
    via soft-gated conjugate inference, and (every cfg.bmr_period steps) prunes
    edges with positive Savage-Dickey log Bayes factor. T_i, kappa_i, U_i are
    recomputed from the live edge means after each pruning.
    """

    K = cfg.motivated.simple.pomdp.n_paradigms
    C = cfg.motivated.simple.n_context
    N = state.motivated.simple.q_joint.shape[0]
    key_obs, key_next = jax.random.split(state.motivated.simple.key)

    # 1. TRANSITION via B_c
    B_c = gm_joint["B_c"]
    q_prior = transition_joint(state.motivated.simple.q_joint, B_c, K, C,
                               eps_theta=cfg.motivated.simple.eps_theta)

    # 2. CHOOSE EXPERIMENT
    if cfg.motivated.simple.resource_coupling and state.motivated.simple.r is not None:
        actions = affordable_experiment(state.motivated.simple.r,
                                        cfg.motivated.simple.pomdp.x_grid,
                                        cfg.motivated.simple)
    else:
        actions = jnp.full((N,), cfg.motivated.simple.obs_x_index,
                           dtype=jnp.int32)

    # 3. OBSERVE
    o_world, theta_star_t = sample_observations(
        A_world_orig, cfg.motivated.simple, t, key_obs, N)

    # 4. EMIT + ROUTE
    q_theta = marginalize_theta(q_prior, K, C)
    o_social = readout_trust_mixture(q_theta, T_trust)

    # 5. ENDOGENOUS γ: per-agent evidence attenuation from conviction
    U_paradigm = state.U_propagated[:, :K]                  # (N, K)
    asym = conviction_asymmetry(U_paradigm)                 # (N,)
    w_evidence = evidence_weight(asym, cfg.gamma_strength)  # (N,)

    # 6. INFER with attenuated world likelihood. We scale the world LL
    #    contribution by w_evidence per agent. Identity:
    #         log q_target = log_prior + w · ll_world + ll_social
    #                      = w · log q_full + (1-w) · log q_no_world  (+ const)
    #    where q_no_world is the posterior using a zero-observation vector
    #    so that ll_world contributes exactly zero. The constants drop out
    #    of the softmax renormalization. This is mathematically equivalent
    #    to scaling p(o|theta,a) by an exponent of w_evidence per agent.
    actions_arr = actions
    mask_social = social_mask

    # Standard inference (full evidence weight)
    q_full = agent_pop.infer_state_batch(
        q_prior, gm_joint["A_world_joint"], o_world, actions_arr,
        gm_joint["A_social_joint"], o_social, mask_social)

    # Inference with zero world evidence: o_world=0 -> ll_world = 0.
    # (Uniform observations would leave a state-dependent entropy term.)
    o_world_blank = jnp.zeros_like(o_world)
    q_no_world = agent_pop.infer_state_batch(
        q_prior, gm_joint["A_world_joint"], o_world_blank, actions_arr,
        gm_joint["A_social_joint"], o_social, mask_social)

    # Log-blend by per-agent evidence weight; softmax renormalizes the
    # constant offsets away, leaving the principled scaled-likelihood form.
    w_b = w_evidence[:, None]                              # (N, 1)
    log_full = jnp.log(q_full + EPS)
    log_no = jnp.log(q_no_world + EPS)
    log_blend = w_b * log_full + (1.0 - w_b) * log_no
    log_blend = log_blend - log_blend.max(axis=1, keepdims=True)
    e = jnp.exp(log_blend)
    q_joint_post = e / (e.sum(axis=1, keepdims=True) + EPS)

    # 7. VALUE TILT using propagated conviction (motivated update)
    if cfg.motivated.motivated and cfg.motivated.lambda_tilt > 0:
        q_joint_post = apply_value_tilt(
            q_joint_post, state.motivated.U, cfg.motivated.lambda_tilt, K, C)

    # 8. TRUST LEARNING
    alpha_new, beta_new, T_new = (state.motivated.simple.alpha_t,
                                   state.motivated.simple.beta_t, T_trust)
    if cfg.motivated.simple.trust_learning and state.motivated.simple.alpha_t is not None:
        q_theta_post = marginalize_theta(q_joint_post, K, C)
        epsilon = categorical_surprisal(
            q_theta_post, A_world_orig, o_world,
            cfg.motivated.simple.obs_x_index, adj)
        alpha_new, beta_new, T_new = trust_update(
            state.motivated.simple.alpha_t, state.motivated.simple.beta_t,
            epsilon, adj, cfg.motivated.simple.trust_rho)

    # 9. RESOURCE FLOW
    r_new = state.motivated.simple.r
    if cfg.motivated.simple.resource_coupling and state.motivated.simple.r is not None:
        W = jnp.asarray(flow_from_trust(np.asarray(T_trust)))
        eta = jnp.asarray(inflow_share(np.asarray(T_trust)))
        x_chosen = jnp.array(cfg.motivated.simple.pomdp.x_grid)[actions]
        h1_vals = h1(x_chosen, cfg.motivated.simple.pomdp.world)
        fisher = h1_vals ** 2 / (cfg.motivated.simple.pomdp.world.sigma ** 2)
        c_x = cfg.motivated.simple.c0 * fisher / (
            state.motivated.simple.r - cfg.motivated.simple.r_min + EPS)
        r_new = jnp.asarray(flow_step(
            np.asarray(state.motivated.simple.r), np.asarray(W),
            np.asarray(eta), np.asarray(c_x),
            cfg.motivated.simple.R_in, cfg.motivated.simple.alpha_flow,
            cfg.motivated.simple.delta_decay))
        r_new = jnp.maximum(r_new, cfg.motivated.simple.r_min)

    # ------------------------------------------------------------------
    # 10. BMR-IN-LOOP: per-edge conjugate update + periodic pruning
    # ------------------------------------------------------------------
    T_per_agent_new = state.T_per_agent
    kappa_new = state.kappa
    U_prop_new = state.U_propagated
    S_xx_new = state.S_xx_edge
    S_xy_new = state.S_xy_edge
    mu_e_new = state.mu_edge
    tau_e_new = state.tau_edge
    alive_new = state.edge_alive
    bmr_info: dict = {}

    if cfg.bmr_in_loop and state.S_xx_edge is not None and A_true_dep is not None:
        D_total = state.S_xx_edge.shape[2]
        # 10a. World samples per-dependent observations from the true SEM.
        key_dep, key_next = jax.random.split(key_next)
        x_dep = sample_dependent_observations(
            A_true_dep, cfg.motivated.simple.pomdp.true_paradigm,
            key_dep, N, cfg.dep_noise_prec)            # (N, D_total)

        # 10b. Soft-gated conjugate update on edge posteriors. Gate is the
        # POSTERIOR paradigm belief (after world+social inference this step).
        q_theta_after = marginalize_theta(q_joint_post, K, C)   # (N, K)
        S_xx_new, S_xy_new, mu_e_new, tau_e_new = update_edge_posteriors(
            state.S_xx_edge, state.S_xy_edge,
            q_theta_after, x_dep,
            cfg.dep_noise_prec, cfg.edge_tau_prior)

        # 10c. Apply the persistent edge_alive mask (so pruned edges stay pruned).
        mu_e_new = mu_e_new * state.edge_alive

        # 10d. Periodic Savage-Dickey BMR. Skip t=0 (degenerate posterior).
        do_bmr = (cfg.bmr_period > 0) and (t > 0) and (t % cfg.bmr_period == 0)
        if do_bmr:
            log_bf = bmr_log_bayes_factor(
                mu_e_new, tau_e_new,
                jnp.full_like(tau_e_new, cfg.edge_tau_prior))
            keep = (log_bf <= cfg.bmr_threshold).astype(state.edge_alive.dtype)
            alive_new = state.edge_alive * keep
            mu_e_new = mu_e_new * alive_new
            bmr_info["mean_log_bf"] = float(np.asarray(log_bf).mean())
            bmr_info["pruned_fraction"] = float(1.0 - np.asarray(alive_new).mean())
        else:
            alive_new = state.edge_alive

        # 10e. Rebuild T_i, kappa_i, U_i from the live edge means.
        A_new = build_A_per_agent_from_edges(mu_e_new, K)        # (N, K_int, K_int)
        T_per_agent_new = jax.vmap(compute_T)(A_new)
        kappa_new = jax.vmap(compute_conservatism)(T_per_agent_new)
        U_prop_new = jax.vmap(compute_conviction)(T_per_agent_new, state.u_source)

    new_simple = SimpleState(q_joint=q_joint_post, key=key_next,
                             alpha_t=alpha_new, beta_t=beta_new, r=r_new)
    # When BMR is active and structure has changed, also refresh the
    # motivated state's U (per-paradigm slice) so the next step's value tilt
    # uses the updated propagated conviction.
    if cfg.bmr_in_loop and state.S_xx_edge is not None:
        U_paradigm_new = U_prop_new[:, :K]
        new_motivated = MotivatedState(simple=new_simple, U=U_paradigm_new)
    else:
        new_motivated = MotivatedState(simple=new_simple, U=state.motivated.U)

    new_state = StructuralState(motivated=new_motivated,
                                T_per_agent=T_per_agent_new,
                                u_source=state.u_source,
                                kappa=kappa_new,
                                U_propagated=U_prop_new,
                                S_xx_edge=S_xx_new, S_xy_edge=S_xy_new,
                                mu_edge=mu_e_new, tau_edge=tau_e_new,
                                edge_alive=alive_new)

    q_theta_post = marginalize_theta(q_joint_post, K, C)
    q_c_post = marginalize_context(q_joint_post, K, C)
    info = {
        "mean_q": np.asarray(q_theta_post.mean(axis=0)),
        "mean_q_c": np.asarray(q_c_post.mean(axis=0)),
        "mean_evidence_weight": float(np.asarray(w_evidence).mean()),
        "mean_conviction_asymmetry": float(np.asarray(asym).mean()),
        "theta_star_t": theta_star_t,
    }
    info.update(bmr_info)
    return new_state, T_new, info


# ------------------------------------------------------------------
# Full run
# ------------------------------------------------------------------

def default_A_true_dep(K_paradigms: int, n_dependents: int,
                       true_paradigm: int, edge_precision: float
                       ) -> jnp.ndarray:
    """World-side true edge weights when no override is given.

    Convention: only the TRUE paradigm has nonzero outgoing edges to its
    own block of dependents. Other paradigms (false hypotheses) have
    A_true_dep[p, :] = 0 -- so the world generates data ONLY from the
    true paradigm's dependents. Edges from false paradigms in any agent's
    hypothesis DAG will accumulate zero-mean evidence and be pruned by BMR.

    Returns: (K_paradigms, K_paradigms * n_dependents).
    """
    D_total = K_paradigms * n_dependents
    A_true = jnp.zeros((K_paradigms, D_total))
    own_block = slice(true_paradigm * n_dependents,
                      (true_paradigm + 1) * n_dependents)
    A_true = A_true.at[true_paradigm, own_block].set(edge_precision)
    return A_true


def run_structural(cfg: StructuralConfig,
                   D_per_agent: np.ndarray | None = None,
                   u_source_per_agent: np.ndarray | None = None,
                   A_per_agent: np.ndarray | None = None,
                   social_mask_per_agent: np.ndarray | None = None,
                   A_true_dep: np.ndarray | None = None) -> dict:
    K = cfg.motivated.simple.pomdp.n_paradigms
    C = cfg.motivated.simple.n_context

    gm_orig = build_generative_model(cfg.motivated.simple.pomdp)
    A_world_orig = gm_orig["A_world"]

    gm_joint = {
        "A_world_joint": build_joint_A_world(A_world_orig, K, C,
                                             cfg.motivated.simple.alpha_tl),
        "A_social_joint": build_joint_A_social(gm_orig["A_social"], K, C),
        "B_c": build_B_c(cfg.motivated.simple.eps_crisis,
                          cfg.motivated.simple.eps_resolve, C),
    }

    T_trust = build_trust(cfg.motivated.simple.pomdp,
                           cfg.motivated.simple.n_agents,
                           mean_degree=cfg.motivated.simple.mean_degree,
                           kind=cfg.motivated.simple.graph_kind,
                           rewiring_p=cfg.motivated.simple.rewiring_p,
                           seed=cfg.motivated.simple.seed)
    adj = (T_trust > 0).astype(float)
    state = init_structural(cfg, D_per_agent, u_source_per_agent, A_per_agent)

    if social_mask_per_agent is not None:
        s_mask = jnp.asarray(social_mask_per_agent, dtype=float)
    else:
        s_mask = jnp.full((cfg.motivated.simple.n_agents,),
                          cfg.motivated.simple.social_mask, dtype=float)

    true_paradigm_idx = cfg.motivated.simple.pomdp.true_paradigm
    n_steps = cfg.motivated.simple.n_steps
    mean_qB = np.empty(n_steps)
    occ_B = np.empty(n_steps)
    mean_w_evidence = np.empty(n_steps)
    mean_asymmetry = np.empty(n_steps)
    theta_star_trace = np.empty(n_steps)
    pruned_fraction = np.zeros(n_steps) if cfg.bmr_in_loop else None
    infos: list[dict] = []

    # World-side dependent SEM. Only used when bmr_in_loop is on.
    A_true_dep_j: jnp.ndarray | None = None
    if cfg.bmr_in_loop:
        if A_true_dep is None:
            A_true_dep_j = default_A_true_dep(
                K, cfg.n_dependents, true_paradigm_idx, cfg.edge_precision)
        else:
            A_true_dep_j = jnp.asarray(A_true_dep, dtype=float)

    for t in range(n_steps):
        state, T_trust, info = structural_step(
            state, gm_joint, A_world_orig, T_trust, adj, cfg, s_mask, t,
            A_true_dep=A_true_dep_j)
        q_theta = np.asarray(marginalize_theta(state.motivated.simple.q_joint,
                                                K, C))
        mean_qB[t] = info["mean_q"][true_paradigm_idx]
        occ_B[t] = float(np.mean(q_theta[:, true_paradigm_idx] > 0.5))
        mean_w_evidence[t] = info["mean_evidence_weight"]
        mean_asymmetry[t] = info["mean_conviction_asymmetry"]
        theta_star_trace[t] = info["theta_star_t"]
        if pruned_fraction is not None and "pruned_fraction" in info:
            pruned_fraction[t] = info["pruned_fraction"]
        elif pruned_fraction is not None and t > 0:
            pruned_fraction[t] = pruned_fraction[t - 1]
        infos.append(info)

    return {
        "mean_qB": mean_qB,
        "occ_B": occ_B,
        "mean_evidence_weight": mean_w_evidence,
        "mean_conviction_asymmetry": mean_asymmetry,
        "final_q": np.asarray(marginalize_theta(state.motivated.simple.q_joint,
                                                 K, C)),
        "final_U_propagated": np.asarray(state.U_propagated),
        "final_kappa": np.asarray(state.kappa),
        "infos": infos,
        "T_trust": np.asarray(T_trust),
        "theta_star_trace": theta_star_trace,
        "pruned_fraction": pruned_fraction,
        "final_mu_edge": (np.asarray(state.mu_edge)
                          if state.mu_edge is not None else None),
        "final_tau_edge": (np.asarray(state.tau_edge)
                           if state.tau_edge is not None else None),
        "final_edge_alive": (np.asarray(state.edge_alive)
                             if state.edge_alive is not None else None),
    }
