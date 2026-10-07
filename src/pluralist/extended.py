"""Endogenous two-channel precision for the pluralist Gaussian model.

Extends David Hyland's linear-Gaussian pluralist model (``src/pluralist/model.py``,
copied from github.com/dkhyland/epistemic-pluralism) with three switchable
mechanisms, plus a monist variant that runs through the same code path:

1. Endogenous world-channel precision on the discriminating channel M:
       a_iM = a0_iM * w(conviction_i) * g(r_i)
   * w = exp(-gamma * conviction_i), the form of
     ``src/pomdp/structural_step.evidence_weight``.
   * g is a budget gate mirroring ``src/pomdp/simple_step.affordable_experiment``:
     observing M at attention a carries Fisher information I_F = a / sigma^2 and
     costs c = c0 * I_F / (r - r_min); the agent takes the largest attention it
     can afford under c <= budget_fraction * r. Resources follow
     r <- max((1 - delta) r + R_in * N * eta - c, r_min), eta = normalised
     incoming social weight (``src/resource.inflow_share``); uniform under the
     doubly-stochastic all-to-all W.
2. Learned social trust: Gamma-conjugate update of
   ``src/pomdp/simple_step.trust_update`` with surprisal
   eps_ij = -log p_j(o_i), the Gaussian-mixture predictive density of agent j's
   entering (pre-communication) belief evaluated at agent i's fresh observation,
   with i's attention weighting (j is judged only on what i actually looked at).
   Option ``trust_surprisal="excess"`` drops the j-independent Gaussian
   normalising constant (which otherwise dominates eps and flattens trust).
   The row-stochastic trust matrix T replaces W in BOTH support sharing and
   belief fusion.
3. Support forgetting: L <- rho_L * [(1 - alpha) L + alpha S L] + scores.

Monist agents: K = 1 with the pooled reference of ``model.gaussian_reference``
(see ``monist_config``). Support sharing is then a no-op exactly as in the
original step.

With gamma = 0, resources = False, trust = False and rho_L = 1, ``ext_step``
executes the same arithmetic, in the same order, as ``model.step`` and consumes
the same random numbers (tested in tests/test_pluralist_mechanisms.py).

All functions are pure JAX with explicit PRNG keys; configs and state are frozen
dataclasses. No randomness is added by the extensions, so observation streams
are paired across all arms that share seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from .model import Config, gaussian_reference, initial_state, predictive_score

EPS = 1e-12
M_ROW = 2  # Row of H that measures mass change M (the discriminating channel).


@dataclass(frozen=True)
class ExtConfig:
    """Base Hyland config plus the three mechanisms. Defaults = all off."""

    base: Config
    # 1a. conviction gate on M attention (gamma = 0 -> off)
    gamma: float = 0.0
    conviction_mode: str = "predictive"  # "predictive" | "support"
    # 1b. resource gate on M attention
    resources: bool = False
    c0: float = 0.1
    r_min: float = 0.1
    r_init: float = 1.0
    R_in: float = 0.1
    delta: float = 0.05
    budget_fraction: float = 0.5
    # 2. learned trust
    trust: bool = False
    trust_rho: float = 0.95
    trust_alpha0: float = 1.0
    trust_beta0: float = 1.0
    # "full": eps = -log p_j(o_i), exactly the categorical_surprisal analogue.
    # "excess": subtract the j-independent noise constant 0.5*3*log(2 pi sigma^2),
    #           so eps >= 0 measures only belief-dependent misprediction.
    trust_surprisal: str = "full"
    # 3. support forgetting (1.0 -> off, original model)
    rho_L: float = 1.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ExtState:
    P: jax.Array        # (N, K, 4, 4)
    h: jax.Array        # (N, K, 4)
    L: jax.Array        # (N, K)
    key: jax.Array      # PRNG key
    r: jax.Array        # (N,) resources
    trust_a: jax.Array  # (N, N) Gamma shape
    trust_b: jax.Array  # (N, N) Gamma rate


def monist_config(cfg: Config) -> Config:
    """K = 1 pooled Gaussian, consistent with model.gaussian_reference."""
    P0, h0 = gaussian_reference(cfg.P0, cfg.h0)
    return replace(cfg, P0=P0, h0=h0)


def ext_initial_state(ecfg: ExtConfig, n: int, seeds) -> ExtState:
    P, h, L, key = initial_state(ecfg.base, n, seeds)
    s = len(seeds)
    return ExtState(
        P=P, h=h, L=L, key=key,
        r=jnp.full((s, n), ecfg.r_init),
        trust_a=jnp.full((s, n, n), ecfg.trust_alpha0),
        trust_b=jnp.full((s, n, n), ecfg.trust_beta0),
    )


# ------------------------------------------------------------------
# Readouts
# ------------------------------------------------------------------

def prob_mass_positive(P, h, L):
    """P(M > 0) under the agent's mixture belief over the latent M coordinate.

    Shapes: P (N,K,4,4), h (N,K,4), L (N,K) -> (N,). Theory-free readout
    available for both pluralists (mixture) and monists (K = 1).
    """
    q = jax.nn.softmax(L, axis=-1)
    mean = jnp.linalg.solve(P, h[..., None])[..., 0][..., 3]
    var = jnp.linalg.inv(P)[..., 3, 3]
    return jnp.einsum("nk,nk->n", q, norm.cdf(mean / jnp.sqrt(var)))


def conviction(P, h, L, mode: str):
    """Per-agent conviction in [0, 1].

    "support":    max_k q_k - min_k q_k (structural_step.conviction_asymmetry
                  applied to q = softmax(L)); identically 0 for monists.
    "predictive": |2 P(M>0) - 1|, commitment about the discriminating
                  quantity; defined identically for pluralists and monists.
    """
    if mode == "support":
        q = jax.nn.softmax(L, axis=-1)
        return q.max(-1) - q.min(-1)
    if mode == "predictive":
        return jnp.abs(2.0 * prob_mass_positive(P, h, L) - 1.0)
    raise ValueError(f"unknown conviction_mode {mode!r}")


def trust_matrix(trust_a, trust_b, adj):
    """Row-normalised Gamma mean, as in simple_step.trust_update."""
    gamma = (trust_a / (trust_b + EPS)) * adj
    return gamma / (gamma.sum(axis=1, keepdims=True) + EPS)


def neighbour_surprisal(P, h, L, H, observations, sigma, attention):
    """eps[i, j] = -log sum_k q_jk N(o_i; j's k-th predictive, i's attention)."""
    q_log = jax.nn.log_softmax(L, axis=-1)  # (N, K)

    def for_observer(o_i, a_i):
        s = jax.vmap(jax.vmap(
            lambda p, v: predictive_score(p, v, H, o_i, sigma, a_i)))(P, h)  # (N, K)
        return -logsumexp(q_log + s, axis=-1)                               # (N,)

    return jax.vmap(for_observer)(observations, attention)


def affordable_attention(r, desired, ecfg: ExtConfig, sigma):
    """Largest a <= desired with c0 * (a/sigma^2) / (r - r_min) <= bf * r."""
    slack = jnp.maximum(r - ecfg.r_min, 0.0)
    a_max = ecfg.budget_fraction * r * slack * sigma**2 / ecfg.c0
    a = jnp.minimum(desired, a_max)
    cost = ecfg.c0 * (a / sigma**2) / jnp.maximum(slack, EPS)
    return a, cost


# ------------------------------------------------------------------
# Step
# ------------------------------------------------------------------

def ext_step(state: ExtState, signal, ecfg: ExtConfig, W, attention, trust_attention=None):
    """One round. Mirrors model.step; extra mechanisms are Python-level switches.

    Order: social matrix S (W or learned T) -> support sharing (+forgetting
    later) -> belief fusion -> conviction/resource gate on M attention (from the
    post-communication prior) -> observe, score, deposit, anchor (original) ->
    trust update from entering beliefs -> resource update.
    """
    cfg = ecfg.base
    P_in, h_in, L_in = state.P, state.h, state.L
    n = W.shape[0]
    adj = (W > 0).astype(W.dtype)
    S = trust_matrix(state.trust_a, state.trust_b, adj) if ecfg.trust else W

    P, h, L, key = P_in, h_in, L_in, state.key
    if cfg.P0.shape[0] > 1:
        L = (1.0 - cfg.alpha) * L + cfg.alpha * (S @ L)
    if cfg.fuse:
        P = jnp.einsum("ij,jkab->ikab", S, P)
        h = jnp.einsum("ij,jkd->ikd", S, h)

    # ---- endogenous M-channel attention --------------------------------
    gated = ecfg.gamma != 0.0 or ecfg.resources
    r_new, cost = state.r, jnp.zeros(n)
    if gated:
        a_m = attention[:, M_ROW]
        if ecfg.gamma != 0.0:
            a_m = a_m * jnp.exp(-ecfg.gamma * conviction(P, h, L, ecfg.conviction_mode))
        if ecfg.resources:
            a_m, cost = affordable_attention(state.r, a_m, ecfg, cfg.sigma)
        attention = attention.at[:, M_ROW].set(a_m)

    # ---- world channel (identical to model.step) ------------------------
    key, observation_key = jax.random.split(key)
    agent_keys = jax.random.split(observation_key, len(W))

    def observe(Pk, hk, agent_key, weights):
        o = cfg.H @ signal + jax.random.normal(agent_key, (3,)) * cfg.sigma
        scores = jax.vmap(
            lambda p, q: predictive_score(p, q, cfg.H, o, cfg.sigma, weights)
        )(Pk, hk)
        root = jnp.sqrt(jnp.clip(weights, 0.0, None))
        H, weighted_o = root[:, None] * cfg.H, root * o
        inv_var = 1.0 / cfg.sigma**2
        return scores, (H.T @ H) * inv_var, (H.T @ weighted_o) * inv_var, o

    scores, J, j, obs = jax.vmap(observe)(P, h, agent_keys, attention)
    P = P + J[:, None, :, :]
    h = h + j[:, None, :]
    P = cfg.P0[None] + cfg.omega * (P - cfg.P0[None])
    h = cfg.h0[None] + cfg.omega * (h - cfg.h0[None])
    if ecfg.rho_L != 1.0:
        L = ecfg.rho_L * L
    L = L + scores

    # ---- social channel: learn trust from neighbours' predictions -------
    trust_a, trust_b = state.trust_a, state.trust_b
    if ecfg.trust:
        # trust_attention (N,3): channels on which i judges neighbours; default = i's
        # own (post-gate) attention this round.
        judged = attention if trust_attention is None else trust_attention
        eps = neighbour_surprisal(P_in, h_in, L_in, cfg.H, obs, cfg.sigma, judged)
        if ecfg.trust_surprisal == "excess":
            eps = eps - 0.5 * cfg.H.shape[0] * jnp.log(2 * jnp.pi * cfg.sigma**2)
        elif ecfg.trust_surprisal != "full":
            raise ValueError(f"unknown trust_surprisal {ecfg.trust_surprisal!r}")
        trust_a = ecfg.trust_rho * trust_a + adj
        trust_b = ecfg.trust_rho * trust_b + eps * adj

    # ---- resources -------------------------------------------------------
    if ecfg.resources:
        incoming = S.sum(axis=0)
        eta = incoming / (incoming.sum() + EPS)
        r_new = jnp.maximum((1.0 - ecfg.delta) * state.r
                            + ecfg.R_in * n * eta - cost, ecfg.r_min)

    new = ExtState(P=P, h=h, L=L, key=key, r=r_new, trust_a=trust_a, trust_b=trust_b)
    return new, attention, S


def ext_rollout(ecfg: ExtConfig, W, attention, signals, seeds, initial=None,
                observers=None, trust_attention=None):
    """Seeded batch (jit + vmap over seeds). Returns (final ExtState, history).

    History arrays have shape (seeds, steps, agents) unless noted:
      oxygen (q of last candidate; ==1 for monists), mixture (.., 3),
      p_mass_pos = P(M>0), attention_M, r, conviction,
      trust_on_observers = sum_{j in observers} S_ij (S used that round).
    """
    W, attention, signals = map(jnp.asarray, (W, attention, signals))
    if trust_attention is not None:
        trust_attention = jnp.asarray(trust_attention)
    n = W.shape[0]
    entering = ext_initial_state(ecfg, n, seeds) if initial is None else initial
    obs_mask = np.zeros(n)
    if observers is not None:
        obs_mask[list(observers)] = 1.0
    obs_mask = jnp.asarray(obs_mask)

    def one(state):
        def update(carry, signal):
            new, used_attention, S = ext_step(carry, signal, ecfg, W, attention, trust_attention)
            q = jax.nn.softmax(new.L, axis=-1)
            means = jnp.linalg.solve(new.P, new.h[..., None])[..., 0][..., 1:]
            mixture = jnp.einsum("nk,nkd->nd", q, means)
            out = {
                "oxygen": q[:, -1],
                "mixture": mixture,
                "p_mass_pos": prob_mass_positive(new.P, new.h, new.L),
                "attention_M": used_attention[:, M_ROW],
                "r": new.r,
                "conviction": conviction(new.P, new.h, new.L, ecfg.conviction_mode),
                "trust_on_observers": S @ obs_mask,
            }
            return new, out

        return jax.lax.scan(update, state, signals)

    final, history = jax.jit(jax.vmap(one))(entering)
    history = {k: np.asarray(v) for k, v in history.items()}
    if not all(np.isfinite(v).all() for v in history.values()):
        raise FloatingPointError("Non-finite simulation readout")
    return final, history


# ------------------------------------------------------------------
# Figure-4-style shift protocol
# ------------------------------------------------------------------

def ungated(ecfg: ExtConfig) -> ExtConfig:
    """Same config with both world-channel gates switched off."""
    return replace(ecfg, gamma=0.0, resources=False)


def run_prehistory(ecfg: ExtConfig, seeds, steps: int = 600, shift: int = 80,
                   n: int = 24, n_full: int = 4, phase2_seed_offset: int = 1000,
                   trust_attention=None):
    """Rounds 0..shift-1 (everyone attention 1, M = -1); returns the branch
    state (RNG reset to seed + offset) and the pre-shift history."""
    from .experiments import signal_path, social_matrix

    W, signals = social_matrix(n), signal_path(steps, shift)
    state, before = ext_rollout(ecfg, W, np.ones((n, 3)), signals[:shift], seeds,
                                observers=list(range(n_full)),
                                trust_attention=trust_attention)
    keys = jnp.stack([jax.random.PRNGKey(int(s) + phase2_seed_offset) for s in seeds])
    return replace(state, key=keys), before


def run_post_shift(ecfg: ExtConfig, state, seeds, steps: int = 600, shift: int = 80,
                   n: int = 24, n_full: int = 4, restricted_attention: float = 0.01,
                   trust_attention=None):
    """Rounds shift..steps-1 (M = +1; agents n_full.. at base attention
    ``restricted_attention`` on G and M) from a branch state."""
    from .experiments import signal_path, social_matrix

    W, signals = social_matrix(n), signal_path(steps, shift)
    attention = np.ones((n, 3))
    attention[n_full:, 1:] = restricted_attention
    return ext_rollout(ecfg, W, attention, signals[shift:], seeds, initial=state,
                       observers=list(range(n_full)), trust_attention=trust_attention)


def shift_protocol(ecfg: ExtConfig, seeds, steps: int = 600, shift: int = 80,
                   n: int = 24, n_full: int = 4, restricted_attention: float = 0.01,
                   phase2_seed_offset: int = 1000, gates_after_shift: bool = False,
                   trust_attention=None):
    """Hyland Fig. 4 protocol with the arm's mechanisms.

    Rounds 0..shift-1: everyone has base attention 1, signal M = -1.
    Round shift onward: M = +1; agents n_full.. have base attention
    ``restricted_attention`` on G and M (agents 0..n_full-1 keep 1); RNG reset
    to seed + phase2_seed_offset (as in experiments.network). Endogenous gates
    multiply the base M attention; with ``gates_after_shift`` they are off
    before the shift (trust and forgetting stay on throughout).
    """
    pre_cfg = ungated(ecfg) if gates_after_shift else ecfg
    state, before = run_prehistory(pre_cfg, seeds, steps, shift, n, n_full,
                                   phase2_seed_offset, trust_attention)
    _, after = run_post_shift(ecfg, state, seeds, steps, shift, n, n_full,
                              restricted_attention, trust_attention)
    return {k: np.concatenate((before[k], after[k]), axis=1) for k in before}


def monist_moment_matched(cfg: Config, weights=(0.5, 0.5)) -> Config:
    """K = 1 reference = moment-matched (M-projection) Gaussian of the
    pluralist's prior mixture over the observables (C, G, M), weighted by the
    pluralist's prior support. Same dummy latent coordinate (precision 4,
    independent) as model.gaussian_reference. Unlike the log-linear pool,
    this keeps the between-theory spread in M as variance, so the monist's
    anchor on M is as weak as the pluralist's prior predictive.
    """
    P, h = np.asarray(cfg.P0), np.asarray(cfg.h0)
    w = np.asarray(weights, float)
    cov = np.linalg.inv(P)[:, 1:, 1:]
    mean = np.linalg.solve(P, h[..., None])[..., 0][:, 1:]
    m = np.einsum("k,kd->d", w, mean)
    dev = mean - m
    S = np.einsum("k,kij->ij", w, cov) + np.einsum("k,ki,kj->ij", w, dev, dev)
    prec = np.linalg.inv(S)
    P0, h0 = np.zeros((1, 4, 4)), np.zeros((1, 4))
    P0[0, 0, 0] = 4.0
    P0[0, 1:, 1:] = prec
    h0[0, 1:] = prec @ m
    return replace(cfg, P0=jnp.asarray(P0), h0=jnp.asarray(h0))
