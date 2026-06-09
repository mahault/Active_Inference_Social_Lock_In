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

What this still does NOT do (out of scope here):
- Bayesian model reduction over the DAG (no edge pruning)
- Structure-learning expansion (no proposing new nodes)
- The Schur-complement residue analysis of marginalized hubs
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
    edge_precision: float = 0.8    # coupling strength on each paradigm→child edge
    gamma_strength: float = 0.0    # 0 = no endogenous γ; > 0 enables silencing
    u_source_dependents: float = 0.5  # default per-dependent utility magnitude


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


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def init_structural(cfg: StructuralConfig,
                    D_per_agent: np.ndarray | None = None,
                    u_source_per_agent: np.ndarray | None = None,
                    A_per_agent: np.ndarray | None = None) -> StructuralState:
    K = cfg.motivated.simple.pomdp.n_paradigms
    K_int = K * (1 + cfg.n_dependents)
    N = cfg.motivated.simple.n_agents

    if A_per_agent is None:
        A_default = build_default_dag(K, cfg.n_dependents, cfg.edge_precision)
        A_batch = jnp.broadcast_to(A_default, (N, K_int, K_int))
    else:
        A_batch = jnp.asarray(A_per_agent, dtype=float)
        assert A_batch.shape == (N, K_int, K_int)

    T_batch = jax.vmap(compute_T)(A_batch)               # (N, K_int, K_int)

    if u_source_per_agent is None:
        # Default: zero source utility on dependents and paradigms
        u_source = jnp.zeros((N, K_int))
    else:
        u_source = jnp.asarray(u_source_per_agent, dtype=float)
        assert u_source.shape == (N, K_int)

    kappa = jax.vmap(compute_conservatism)(T_batch)      # (N, K_int)
    U_prop = jax.vmap(compute_conviction)(T_batch, u_source)  # (N, K_int)

    # Seed the motivated state's per-paradigm U with the propagated values
    # at the paradigm slice (so the value tilt uses propagated conviction)
    U_paradigm = U_prop[:, :K]                            # (N, K)
    base_mot_state = init_motivated(cfg.motivated, D_per_agent,
                                    U_per_agent=np.asarray(U_paradigm))

    return StructuralState(motivated=base_mot_state, T_per_agent=T_batch,
                           u_source=u_source, kappa=kappa,
                           U_propagated=U_prop)


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
                    t: int) -> tuple[StructuralState, jnp.ndarray, dict]:
    """One step with structural-aware inference. Computes per-agent evidence
    attenuation from propagated conviction, then runs hierarchical-context
    inference with attenuated world LL, then applies value tilt using the
    propagated per-paradigm conviction."""

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

    # 6. INFER (with attenuated world LL via the social_mask analog on
    #    the world channel — we scale the world likelihood by w_evidence
    #    per agent by computing the inference twice and blending).
    actions_arr = actions
    mask_social = social_mask

    # Standard inference (full evidence weight)
    q_full = agent_pop.infer_state_batch(
        q_prior, gm_joint["A_world_joint"], o_world, actions_arr,
        gm_joint["A_social_joint"], o_social, mask_social)

    # Inference with zero world evidence (only social + prior)
    o_world_blank = jnp.ones_like(o_world) / o_world.shape[1]
    q_no_world = agent_pop.infer_state_batch(
        q_prior, gm_joint["A_world_joint"], o_world_blank, actions_arr,
        gm_joint["A_social_joint"], o_social, mask_social)

    # Blend by per-agent evidence weight: w=1 → full evidence; w=0 → no evidence
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

    new_simple = SimpleState(q_joint=q_joint_post, key=key_next,
                             alpha_t=alpha_new, beta_t=beta_new, r=r_new)
    new_motivated = MotivatedState(simple=new_simple, U=state.motivated.U)
    new_state = StructuralState(motivated=new_motivated,
                                T_per_agent=state.T_per_agent,
                                u_source=state.u_source,
                                kappa=state.kappa,
                                U_propagated=state.U_propagated)

    q_theta_post = marginalize_theta(q_joint_post, K, C)
    q_c_post = marginalize_context(q_joint_post, K, C)
    info = {
        "mean_q": np.asarray(q_theta_post.mean(axis=0)),
        "mean_q_c": np.asarray(q_c_post.mean(axis=0)),
        "mean_evidence_weight": float(np.asarray(w_evidence).mean()),
        "mean_conviction_asymmetry": float(np.asarray(asym).mean()),
        "theta_star_t": theta_star_t,
    }
    return new_state, T_new, info


# ------------------------------------------------------------------
# Full run
# ------------------------------------------------------------------

def run_structural(cfg: StructuralConfig,
                   D_per_agent: np.ndarray | None = None,
                   u_source_per_agent: np.ndarray | None = None,
                   A_per_agent: np.ndarray | None = None,
                   social_mask_per_agent: np.ndarray | None = None) -> dict:
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
    infos: list[dict] = []

    for t in range(n_steps):
        state, T_trust, info = structural_step(
            state, gm_joint, A_world_orig, T_trust, adj, cfg, s_mask, t)
        q_theta = np.asarray(marginalize_theta(state.motivated.simple.q_joint,
                                                K, C))
        mean_qB[t] = info["mean_q"][true_paradigm_idx]
        occ_B[t] = float(np.mean(q_theta[:, true_paradigm_idx] > 0.5))
        mean_w_evidence[t] = info["mean_evidence_weight"]
        mean_asymmetry[t] = info["mean_conviction_asymmetry"]
        theta_star_trace[t] = info["theta_star_t"]
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
    }
