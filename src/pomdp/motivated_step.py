"""Motivated update extension on top of the hierarchical-context substrate.

Wraps `simple_step` and adds the value-tilted posterior of Hyland & Albarracin
(2025) Eq. 13 at the multi-agent level:

    q_lambda(theta, c) ∝ q_post(theta, c) · exp(lambda_tilt · U(theta))

where U is a per-agent intrinsic utility over paradigms (not over contexts).
The tilt is applied AFTER the Bayesian update so the inference stays honest
and the value field appears as a separate scalar tilt on the marginal posterior
— the cleanest separation between epistemic update and motivated reasoning.

What this DOES:
  - Per-agent utility U_i(theta) (heterogeneous values across agents)
  - Tempered/tilted posterior update after standard categorical Bayes
  - Reuses everything from simple_step: multi-level context, B_c transition,
    trust learning, resource coupling, paradigm leak

What this DOES NOT do (deferred):
  - Per-agent dependency network with propagation operator T = (I − A)^-1
  - Conviction U = T · u propagating through structured commitments

The deferred piece is the IWAI paper's specific mechanism; the present module
implements the multi-agent lift of Hyland & Albarracin 2025 on a flat
paradigm representation (cf. Albarracin 2022 epistemic communities) with
proper hierarchical-context theory-ladenness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from src.pomdp.simple_step import (
    EPS,
    SimpleConfig,
    SimpleState,
    build_joint_A_world,
    build_joint_A_social,
    build_B_c,
    build_trust,
    init_simple,
    marginalize_context,
    marginalize_theta,
    readout_trust_mixture,
    simple_step,
)
from src.pomdp.gen_model import build_generative_model


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass(frozen=True)
class MotivatedConfig:
    """Wraps a SimpleConfig and adds motivated-update parameters.

    Two update modes (see apply_value_tilt vs apply_motivated_gate):

      "state_tilt"   q_lambda(theta,c) proportional to q_post(theta,c) *
                     exp(lambda * U(theta)). A reward on the belief STATE.
                     This makes utility a fixed sigma(lambda*U) attractor:
                     applied every step it compounds and swamps evidence and
                     social coupling, so the social-coupling phase boundary
                     washes out (see E9). Kept for comparison.

      "evidence_gate"  The faithful "cost of mind-change" reading (Hyland &
                     Albarracin 2025; theory_brief line 19): resistance to
                     revising belief AWAY from a valued paradigm. Disconfirming
                     log-odds movement against paradigm theta is attenuated by
                     w(theta) = exp(-lambda * relu(U(theta))); confirming
                     movement flows at full strength. This is a precision gate
                     on disconfirming evidence -- the flat-substrate analogue
                     of structural_step's endogenous gamma -- and it does NOT
                     saturate to a fixed attractor, so motivated reasoning,
                     evidence, and social coupling genuinely compete.
    """
    simple: SimpleConfig = field(default_factory=SimpleConfig)
    lambda_tilt: float = 1.0      # strength of the value tilt / gate
    motivated: bool = True        # set False to recover plain simple_step behavior
    update_mode: str = "state_tilt"   # "state_tilt" | "evidence_gate"


# ------------------------------------------------------------------
# Value tilt
# ------------------------------------------------------------------

def apply_value_tilt(q_joint: jnp.ndarray, U_per_agent: jnp.ndarray,
                     lambda_tilt: float, K: int, C: int) -> jnp.ndarray:
    """Apply per-agent value tilt to the joint posterior.

    U_per_agent : (N, K) intrinsic utility over paradigms (NOT context).
    The tilt factors over c because U only depends on theta:
        q_tilted(theta, c) = (1/Z_i) · q_joint(theta, c) · exp(lambda_tilt · U_i(theta))
    """
    # Expand U_per_agent (N, K) to (N, K*C) by repeating across context.
    # Indexing: q_joint[i, theta*C + c]; we want U_full[i, theta*C + c] = U[i, theta].
    N = q_joint.shape[0]
    U_block = jnp.repeat(U_per_agent[:, :, None], C, axis=2)   # (N, K, C)
    U_full = U_block.reshape(N, K * C)                          # (N, K*C)

    log_q = jnp.log(q_joint + EPS) + lambda_tilt * U_full
    # softmax over the joint dimension to renormalize
    log_q = log_q - jnp.max(log_q, axis=1, keepdims=True)
    e = jnp.exp(log_q)
    return e / (e.sum(axis=1, keepdims=True) + EPS)


def apply_motivated_gate(q_prior_joint: jnp.ndarray, q_post_joint: jnp.ndarray,
                         U_per_agent: jnp.ndarray, lambda_tilt: float,
                         K: int, C: int) -> jnp.ndarray:
    """Cost-of-mind-change update: resist revising belief away from valued paradigms.

    This is the faithful reading of the variational cost of mind-change
    (Hyland & Albarracin 2025): the agent resists log-odds movement that
    DECREASES belief in a paradigm it values, while accepting confirming
    movement at full strength. It is asymmetric (only disconfirming moves are
    gated), so unlike the state tilt it does not impose a fixed attractor.

    Per agent i, per paradigm theta, on the theta-marginal:
        delta(theta)   = log m_post(theta) - log m_prior(theta)   # this step's move
        w(theta)       = exp(-lambda * relu(U_i(theta)))          # in (0, 1]
        delta_gated    = delta * w   where delta < 0, else delta  # gate decreases only
        m_gated(theta) proportional to m_prior(theta) * exp(delta_gated)

    The context conditional q(c | theta) from the honest posterior is preserved;
    only the paradigm marginal's movement is motivatedly reweighted, so epistemic
    inference over context is untouched.

    q_prior_joint : (N, K*C) belief carried INTO this step (the reference the
                    agent resists moving from).
    q_post_joint  : (N, K*C) honest Bayesian posterior after this step.
    U_per_agent   : (N, K) intrinsic utility over paradigms.
    """
    N = q_prior_joint.shape[0]
    m_prior = q_prior_joint.reshape(N, K, C).sum(axis=2)        # (N, K)
    post_kc = q_post_joint.reshape(N, K, C)                     # (N, K, C)
    m_post = post_kc.sum(axis=2)                                # (N, K)

    log_delta = jnp.log(m_post + EPS) - jnp.log(m_prior + EPS)  # (N, K)
    w = jnp.exp(-lambda_tilt * jnp.maximum(U_per_agent, 0.0))   # (N, K) in (0,1]
    # gate only the disconfirming (negative) moves
    gated = jnp.where(log_delta < 0.0, log_delta * w, log_delta)

    log_m = jnp.log(m_prior + EPS) + gated                     # (N, K)
    log_m = log_m - jnp.max(log_m, axis=1, keepdims=True)
    m_gated = jnp.exp(log_m)
    m_gated = m_gated / (m_gated.sum(axis=1, keepdims=True) + EPS)   # (N, K)

    # rebuild joint preserving the honest context conditional q(c | theta)
    cond_c = post_kc / (m_post[:, :, None] + EPS)               # (N, K, C)
    q_joint = (m_gated[:, :, None] * cond_c).reshape(N, K * C)
    return q_joint / (q_joint.sum(axis=1, keepdims=True) + EPS)


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

@dataclass(frozen=True)
class MotivatedState:
    """Wraps SimpleState; carries U as static or evolving per-agent utility."""
    simple: SimpleState
    U: jax.Array        # (N, K) per-agent utility over paradigms


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def init_motivated(cfg: MotivatedConfig,
                   D_per_agent: np.ndarray | None = None,
                   U_per_agent: np.ndarray | None = None) -> MotivatedState:
    s = init_simple(cfg.simple, D_per_agent)
    K = cfg.simple.pomdp.n_paradigms
    if U_per_agent is None:
        # Default: zero utility (no preference) — equivalent to no motivated update
        U = jnp.zeros((cfg.simple.n_agents, K))
    else:
        U = jnp.asarray(U_per_agent, dtype=float)
    return MotivatedState(simple=s, U=U)


# ------------------------------------------------------------------
# Single step (wraps simple_step, applies value tilt)
# ------------------------------------------------------------------

def motivated_step(state: MotivatedState,
                   gm_joint: dict,
                   A_world_orig: jnp.ndarray,
                   T: jnp.ndarray,
                   adj: jnp.ndarray,
                   cfg: MotivatedConfig,
                   social_mask: jnp.ndarray,
                   t: int) -> tuple[MotivatedState, jnp.ndarray, dict]:
    # Reference prior (carried INTO this step) — needed by the gate mode.
    q_prior_joint = state.simple.q_joint

    # 1. Standard hierarchical-context update
    new_simple, T_new, info = simple_step(
        state.simple, gm_joint, A_world_orig, T, adj,
        cfg.simple, social_mask, t)

    # 2. Apply the motivated update (if motivated mode is on)
    K = cfg.simple.pomdp.n_paradigms
    C = cfg.simple.n_context
    if cfg.motivated and cfg.lambda_tilt > 0:
        if cfg.update_mode == "evidence_gate":
            q_mot = apply_motivated_gate(q_prior_joint, new_simple.q_joint,
                                         state.U, cfg.lambda_tilt, K, C)
        else:  # "state_tilt" (default, legacy)
            q_mot = apply_value_tilt(new_simple.q_joint, state.U,
                                     cfg.lambda_tilt, K, C)
        new_simple = SimpleState(
            q_joint=q_mot, key=new_simple.key,
            alpha_t=new_simple.alpha_t, beta_t=new_simple.beta_t,
            r=new_simple.r,
        )
        # Update the info dict to reflect the motivated marginal
        q_theta_post = marginalize_theta(q_mot, K, C)
        info["mean_q"] = np.asarray(q_theta_post.mean(axis=0))
        info["mean_q_c"] = np.asarray(marginalize_context(q_mot, K, C).mean(axis=0))

    new_state = MotivatedState(simple=new_simple, U=state.U)
    return new_state, T_new, info


# ------------------------------------------------------------------
# Full run
# ------------------------------------------------------------------

def run_motivated(cfg: MotivatedConfig,
                  D_per_agent: np.ndarray | None = None,
                  U_per_agent: np.ndarray | None = None,
                  social_mask_per_agent: np.ndarray | None = None) -> dict:
    K = cfg.simple.pomdp.n_paradigms
    C = cfg.simple.n_context

    gm_orig = build_generative_model(cfg.simple.pomdp)
    A_world_orig = gm_orig["A_world"]

    gm_joint = {
        "A_world_joint": build_joint_A_world(A_world_orig, K, C, cfg.simple.alpha_tl),
        "A_social_joint": build_joint_A_social(gm_orig["A_social"], K, C),
        "B_c": build_B_c(cfg.simple.eps_crisis, cfg.simple.eps_resolve, C),
    }

    T = build_trust(cfg.simple.pomdp, cfg.simple.n_agents,
                    mean_degree=cfg.simple.mean_degree,
                    kind=cfg.simple.graph_kind,
                    rewiring_p=cfg.simple.rewiring_p,
                    seed=cfg.simple.seed)
    adj = (T > 0).astype(float)
    state = init_motivated(cfg, D_per_agent, U_per_agent)

    if social_mask_per_agent is not None:
        s_mask = jnp.asarray(social_mask_per_agent, dtype=float)
    else:
        s_mask = jnp.full((cfg.simple.n_agents,), cfg.simple.social_mask, dtype=float)

    true_paradigm_idx = cfg.simple.pomdp.true_paradigm
    n_steps = cfg.simple.n_steps
    mean_qB = np.empty(n_steps)
    occ_B = np.empty(n_steps)
    mean_c_normal = np.empty(n_steps)
    theta_star_trace = np.empty(n_steps)
    infos: list[dict] = []

    for t in range(n_steps):
        state, T, info = motivated_step(
            state, gm_joint, A_world_orig, T, adj, cfg, s_mask, t)
        q_theta = np.asarray(marginalize_theta(state.simple.q_joint, K, C))
        mean_qB[t] = info["mean_q"][true_paradigm_idx]
        occ_B[t] = float(np.mean(q_theta[:, true_paradigm_idx] > 0.5))
        mean_c_normal[t] = info["mean_q_c"][0]
        theta_star_trace[t] = info["theta_star_t"]
        infos.append(info)

    return {
        "mean_qB": mean_qB,
        "occ_B": occ_B,
        "mean_c_normal": mean_c_normal,
        "final_q": np.asarray(marginalize_theta(state.simple.q_joint, K, C)),
        "final_q_joint": np.asarray(state.simple.q_joint),
        "final_U": np.asarray(state.U),
        "infos": infos,
        "T": np.asarray(T),
        "theta_star_trace": theta_star_trace,
    }
