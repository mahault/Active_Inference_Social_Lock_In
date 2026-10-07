"""Mixed communities: pluralists, committed monists and neutral monists together.

Representation (padded K = 2)
-----------------------------
Every agent carries two candidate slots (P, h) and log-support L, plus its
OWN anchoring reference (P0_i, h0_i):

* pluralist        - slots = the two theories (Hyland's P0, h0); L evolves.
* committed monist - both slots = the single Gaussian of
                     ``extended.monist_moment_matched(cfg, (1-q, q))`` with q the
                     pluralist community's pre-shift oxygen support (the
                     "incumbent" monist of P1b).
* neutral monist   - both slots = ``monist_moment_matched(cfg)`` (prior weights
                     .5/.5).

A monist's two slots are identical at all times (identical references, messages,
evidence and scores), so its L entries stay equal, q = (.5, .5), and every
mixture readout (0.5 x + 0.5 x = x, exact in floating point) equals the single
Gaussian. This avoids ragged per-agent K while keeping one vmapped code path.

Cross-type communication rule
-----------------------------
Following Hyland's principle that only the shared observables (C, G, M) can be
compared across models (the hidden coordinate means different things in
different models):

* Same class (pluralist->pluralist, monist->monist): the original full-joint
  per-slot fusion P_ik <- sum_j W_ij P_jk (unchanged; for monists the hidden
  coordinate is an independent dummy, so this already only moves observables).
* Cross class: the sender communicates its belief about the OBSERVABLES only,
  as one Gaussian: the moment-matched (M-projection) observable marginal of its
  mixture, N(m_j, S_j) with m_j = sum_k q_jk mu_jk,
  S_j = sum_k q_jk (Sigma_jk + (mu_jk - m_j)(mu_jk - m_j)^T). (For a monist this
  is just its Gaussian's observable marginal.) Receiver i reads it into each of
  its slots k through its OWN conditional of the hidden coordinate given the
  observables, p_ik(latent | obs): the message is the joint
  p_ik(latent | obs) N(obs; m_j, S_j), whose information form is
      P_msg = [[P_ll, P_lo], [P_ol, P_ol P_ll^-1 P_lo + S_j^-1]]
      h_msg = [h_l, P_ol P_ll^-1 h_l + S_j^-1 m_j]
  with (P_ll, P_lo, h_l) taken from the receiver's entering slot. Its observable
  marginal is exactly the sender's; its conditional is exactly the receiver's.
  The receiver then fuses messages with the usual weights W_ij.
* Support sharing: only pluralists hold support over the theory set, so only
  pluralists send or receive it: L_i <- (1-alpha) L_i + alpha sum_{j plural}
  W~_ij L_j for pluralist i, with W~ = W restricted to pluralist columns and
  row-renormalised. Monists neither send nor receive support (their L is inert).

A pure community uses exactly the original operations (tested to reproduce
``extended.shift_protocol`` runs). Learned trust is not supported here (fixed W).

World-channel gates take traced parameters (gamma, c0) so a whole gamma x c0
grid runs as one jit + vmap; gamma = 0 multiplies by exp(0) = 1 and c0 = 0 sets
the affordable attention to +inf and the cost to 0, i.e. both gates are exactly
off.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from .extended import EPS, M_ROW, ExtConfig, conviction, monist_moment_matched, prob_mass_positive
from .model import Config, predictive_score

PLURALIST, COMMITTED, NEUTRAL = 0, 1, 2
TYPE_NAMES = {PLURALIST: "pluralist", COMMITTED: "committed", NEUTRAL: "neutral"}


@dataclass(frozen=True)
class MixedConfig:
    """Community spec. ``types`` is a tuple of per-agent type codes.

    ``ext`` supplies resource constants (r_min, r_init, R_in, delta,
    budget_fraction) and rho_L; its gamma/c0/resources/trust fields are
    ignored (gates are traced parameters; trust unsupported).
    """

    base: Config
    types: tuple
    committed_q: float = 0.0  # pluralist pre-shift oxygen support for the committed anchor
    ext: ExtConfig | None = None
    # Message bandwidth (P1g, after Tanaka 2026 Quantized Simplex Gossip):
    # fusion = "rule": same-type full joint + cross-type observable messages (P1c);
    #          "observables": every pair communicates the sender's moment-matched
    #          observable marginal, read through the receiver's conditional.
    # message_m = 0: exact message mean; m > 0: the message mean is the empirical
    #          mean of m samples from the sender's belief over (C, G, M) (requires
    #          fusion = "observables"); the message keeps the sender's covariance.
    # support_m = 0: pluralists share log-support L; m > 0: they share the log of
    #          smoothed frequencies of m draws of a theory index from softmax(L)
    #          ((count + .5) / (m + .5 K)).
    fusion: str = "rule"
    message_m: int = 0
    support_m: int = 0

    @property
    def n(self):
        return len(self.types)


def references(mcfg: MixedConfig):
    """Per-agent anchoring references (N, K, 4, 4), (N, K, 4).

    ``committed_q``: a float q (K = 2: weights (1-q, q), as in P1b-P1e) or a
    length-K weight vector (the pluralist community's pre-shift support).
    Neutral monists use uniform weights 1/K (= (.5, .5) for K = 2)."""
    base = mcfg.base
    K = int(np.asarray(base.P0).shape[0])   # monists are padded to the pluralists' K
    q = mcfg.committed_q
    w_committed = (1.0 - q, q) if np.ndim(q) == 0 else tuple(float(x) for x in q)
    refs = {
        PLURALIST: (np.asarray(base.P0), np.asarray(base.h0)),
        COMMITTED: monist_moment_matched(base, w_committed),
        NEUTRAL: monist_moment_matched(base, (1.0 / K,) * K),
    }
    P0, h0 = [], []
    for t in mcfg.types:
        if t == PLURALIST:
            P, h = refs[t]
        else:
            c = refs[t]
            P = np.repeat(np.asarray(c.P0), K, axis=0)
            h = np.repeat(np.asarray(c.h0), K, axis=0)
        P0.append(P)
        h0.append(h)
    return jnp.asarray(np.stack(P0)), jnp.asarray(np.stack(h0))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MixedState:
    P: jax.Array    # (N, 2, 4, 4)
    h: jax.Array    # (N, 2, 4)
    L: jax.Array    # (N, 2)
    key: jax.Array
    r: jax.Array    # (N,)


def mixed_initial_state(mcfg: MixedConfig, seeds) -> MixedState:
    P0, h0 = references(mcfg)
    s, n = len(seeds), mcfg.n
    return MixedState(
        P=jnp.broadcast_to(P0, (s,) + P0.shape),
        h=jnp.broadcast_to(h0, (s,) + h0.shape),
        L=jnp.zeros((s, n, P0.shape[1])),
        key=jnp.stack([jax.random.PRNGKey(int(x)) for x in seeds]),
        r=jnp.full((s, n), mcfg.ext.r_init),
    )


def observable_marginal_moment_matched(P, h, L):
    """(N,3) mean and (N,3,3) covariance of each agent's mixture over (C,G,M)."""
    q = jax.nn.softmax(L, axis=-1)
    cov = jnp.linalg.inv(P)[..., 1:, 1:]
    mean = jnp.linalg.solve(P, h[..., None])[..., 0][..., 1:]
    m = jnp.einsum("nk,nkd->nd", q, mean)
    dev = mean - m[:, None, :]
    S = jnp.einsum("nk,nkab->nab", q, cov) + jnp.einsum("nk,nka,nkb->nab", q, dev, dev)
    return m, S


def social_operators(types, W):
    """Static numpy operators for the communication rule."""
    types = np.asarray(types)
    plural = types == PLURALIST
    cls = np.where(plural, 0, 1)
    same = (cls[:, None] == cls[None, :]).astype(float)
    pure = bool(same.all())
    if plural.all():
        Wsupport = W
    else:
        Wp = W * plural[None, :]
        rows = Wp.sum(1, keepdims=True)
        Wsupport = np.where(rows > 0, Wp / np.where(rows > 0, rows, 1.0), 0.0)
    return {"pure": pure, "plural": plural, "any_plural": bool(plural.any()),
            "all_plural": bool(plural.all()), "W_same": W * same, "W_cross": W * (1 - same),
            "W_support": Wsupport}


def sample_message_mean(P, h, L, key, m):
    """Empirical mean of m samples of the observables from each agent's mixture."""
    q = jax.nn.softmax(L, axis=-1)
    cov = jnp.linalg.inv(P)[..., 1:, 1:]
    mean = jnp.linalg.solve(P, h[..., None])[..., 0][..., 1:]
    chol = jnp.linalg.cholesky(0.5 * (cov + jnp.swapaxes(cov, -1, -2)))
    k1, k2 = jax.random.split(key)
    n = P.shape[0]
    comp = jax.random.categorical(k1, jnp.log(q)[:, None, :], axis=-1, shape=(n, m))   # (N,m)
    z = jax.random.normal(k2, (n, m, 3))
    mu = jnp.take_along_axis(mean, comp[..., None], axis=1)                             # (N,m,3)
    Lc = jnp.take_along_axis(chol, comp[..., None, None], axis=1)                       # (N,m,3,3)
    x = mu + jnp.einsum("nmab,nmb->nma", Lc, z)
    return x.mean(1)


def sample_support(L, key, m):
    """log of smoothed frequencies of m theory draws from softmax(L): (N,K)."""
    K = L.shape[-1]
    draws = jax.random.categorical(key, L[:, None, :], axis=-1, shape=(L.shape[0], m))
    counts = jax.nn.one_hot(draws, K).sum(1)
    return jnp.log((counts + 0.5) / (m + 0.5 * K))


def fuse(P, h, L, W, ops, msg_mean=None):
    """Belief fusion under the cross-type rule (see module docstring).
    ``msg_mean`` (N,3) replaces each sender's exact observable mean (sampled messages)."""
    if ops["pure"]:
        return (jnp.einsum("ij,jkab->ikab", W, P), jnp.einsum("ij,jkd->ikd", W, h))
    Ws, Wc = jnp.asarray(ops["W_same"]), jnp.asarray(ops["W_cross"])
    P_same = jnp.einsum("ij,jkab->ikab", Ws, P)
    h_same = jnp.einsum("ij,jkd->ikd", Ws, h)
    m, S = observable_marginal_moment_matched(P, h, L)
    if msg_mean is not None:
        m = msg_mean
    Lam = jnp.linalg.inv(S)                         # (N,3,3) sender precision
    eta = jnp.einsum("nab,nb->na", Lam, m)
    Lam_in = jnp.einsum("ij,jab->iab", Wc, Lam)     # received, summed with weights
    eta_in = jnp.einsum("ij,ja->ia", Wc, eta)
    wc = Wc.sum(1)                                  # (N,) total cross weight
    Pll = P[..., 0, 0]                              # (N,K) receiver's conditional
    Plo = P[..., 0, 1:]                             # (N,K,3)
    hl = h[..., 0]                                  # (N,K)
    A = jnp.zeros_like(P)
    A = A.at[..., 0, 0].set(Pll)
    A = A.at[..., 0, 1:].set(Plo)
    A = A.at[..., 1:, 0].set(Plo)
    A = A.at[..., 1:, 1:].set(jnp.einsum("nka,nkb->nkab", Plo, Plo) / Pll[..., None, None])
    a = jnp.concatenate([hl[..., None], Plo * (hl / Pll)[..., None]], axis=-1)
    P_cross = wc[:, None, None, None] * A
    P_cross = P_cross.at[..., 1:, 1:].add(Lam_in[:, None])
    h_cross = wc[:, None, None] * a
    h_cross = h_cross.at[..., 1:].add(eta_in[:, None])
    return P_same + P_cross, h_same + h_cross


def mixed_step(state: MixedState, signal, mcfg: MixedConfig, W, attention, ops,
               gamma, c0, P0, h0):
    cfg, ecfg = mcfg.base, mcfg.ext
    n = W.shape[0]
    P, h, L, key = state.P, state.h, state.L, state.key

    sampled = mcfg.message_m > 0 or mcfg.support_m > 0
    if sampled:
        k_msg, k_sup = jax.random.split(jax.random.fold_in(state.key, 7_000_001))
    if ops["any_plural"]:
        L_send = sample_support(L, k_sup, mcfg.support_m) if mcfg.support_m > 0 else L
        shared = (1.0 - cfg.alpha) * L + cfg.alpha * (jnp.asarray(ops["W_support"]) @ L_send)
        L = shared if ops["all_plural"] else jnp.where(jnp.asarray(ops["plural"])[:, None], shared, L)
    if mcfg.fusion == "rule":
        if mcfg.message_m > 0:
            raise ValueError("sampled messages require fusion='observables'")
        P, h = fuse(P, h, state.L, W, ops)
    elif mcfg.fusion == "observables":
        obs_ops = dict(ops, pure=False, W_same=np.zeros_like(ops["W_same"]), W_cross=ops["W_same"] + ops["W_cross"])
        msg = (sample_message_mean(P, h, state.L, k_msg, mcfg.message_m) if mcfg.message_m > 0 else None)
        P, h = fuse(P, h, state.L, W, obs_ops, msg)
    else:
        raise ValueError(mcfg.fusion)

    # gates (traced; exactly off at gamma = 0, c0 = 0)
    a_m = attention[:, M_ROW] * jnp.exp(-gamma * conviction(P, h, L, "predictive"))
    slack = jnp.maximum(state.r - ecfg.r_min, 0.0)
    on = c0 > 0
    a_max = jnp.where(on, ecfg.budget_fraction * state.r * slack * cfg.sigma**2
                      / jnp.where(on, c0, 1.0), jnp.inf)
    a_m = jnp.minimum(a_m, a_max)
    cost = jnp.where(on, c0 * (a_m / cfg.sigma**2) / jnp.maximum(slack, EPS), 0.0)
    attention = attention.at[:, M_ROW].set(a_m)

    key, observation_key = jax.random.split(key)
    agent_keys = jax.random.split(observation_key, n)

    def observe(Pk, hk, agent_key, weights):
        o = cfg.H @ signal + jax.random.normal(agent_key, (3,)) * cfg.sigma
        scores = jax.vmap(
            lambda p, q: predictive_score(p, q, cfg.H, o, cfg.sigma, weights)
        )(Pk, hk)
        root = jnp.sqrt(jnp.clip(weights, 0.0, None))
        H, weighted_o = root[:, None] * cfg.H, root * o
        inv_var = 1.0 / cfg.sigma**2
        return scores, (H.T @ H) * inv_var, (H.T @ weighted_o) * inv_var

    scores, J, j = jax.vmap(observe)(P, h, agent_keys, attention)
    P = P + J[:, None, :, :]
    h = h + j[:, None, :]
    P = P0 + cfg.omega * (P - P0)
    h = h0 + cfg.omega * (h - h0)
    if ecfg.rho_L != 1.0:
        L = ecfg.rho_L * L
    L = L + scores

    incoming = W.sum(axis=0)
    eta = incoming / (incoming.sum() + EPS)
    r_upd = jnp.maximum((1.0 - ecfg.delta) * state.r + ecfg.R_in * n * eta - cost, ecfg.r_min)
    r = jnp.where(on, r_upd, state.r)
    return MixedState(P=P, h=h, L=L, key=key, r=r), attention


def mixed_rollout(mcfg: MixedConfig, W, attention, signals, seeds, gammas, c0s, initial=None):
    """Batch over grid points x seeds. ``gammas``/``c0s`` are (G,) arrays; the
    batch is the outer product grid-point x seed, so histories have shape
    (G, S, T, N). ``initial``: MixedState with leading (G, S) axes, or None."""
    W_np = np.asarray(W)
    ops = social_operators(mcfg.types, W_np)
    W, attention, signals = map(jnp.asarray, (W_np, attention, signals))
    P0, h0 = references(mcfg)
    gammas, c0s = jnp.asarray(gammas, float), jnp.asarray(c0s, float)
    G = gammas.shape[0]
    if initial is None:
        s0 = mixed_initial_state(mcfg, seeds)
        initial = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (G,) + x.shape), s0)

    def one(state, gamma, c0):
        def update(carry, signal):
            new, used = mixed_step(carry, signal, mcfg, W, attention, ops, gamma, c0, P0, h0)
            q = jax.nn.softmax(new.L, axis=-1)
            return new, {"p_mass_pos": prob_mass_positive(new.P, new.h, new.L),
                         "attention_M": used[:, M_ROW], "oxygen": q[:, -1],
                         "mean_M": jnp.einsum("nk,nk->n", q, jnp.linalg.solve(
                             new.P, new.h[..., None])[..., 0][..., 3])}
        return jax.lax.scan(update, state, signals)

    over_seeds = jax.vmap(one, in_axes=(0, None, None))
    final, hist = jax.jit(jax.vmap(over_seeds, in_axes=(0, 0, 0)))(initial, gammas, c0s)
    hist = {k: np.asarray(v) for k, v in hist.items()}
    if not all(np.isfinite(v).all() for v in hist.values()):
        raise FloatingPointError("Non-finite simulation readout")
    return final, hist


def mixed_shift_protocol(mcfg: MixedConfig, seeds, gammas, c0s, steps: int = 600,
                         shift: int = 80, n_full: int = 4, restricted_attention: float = 0.01,
                         phase2_seed_offset: int = 1000, signals=None):
    """Fig-4 protocol, gates on from the shift only. One ungated prehistory is
    shared by all grid points. Returns histories (G, S, T, N). ``signals``
    (steps, 4) overrides the single-flip world (worlds.py)."""
    from .experiments import signal_path, social_matrix

    n = mcfg.n
    W = social_matrix(n)
    signals = signal_path(steps, shift) if signals is None else np.asarray(signals)
    state, before = mixed_rollout(mcfg, W, np.ones((n, 3)), signals[:shift], seeds, [0.0], [0.0])
    keys = jnp.stack([jax.random.PRNGKey(int(s) + phase2_seed_offset) for s in seeds])
    state = replace(state, key=keys[None])
    G = len(gammas)
    state = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[0], (G,) + x.shape[1:]), state)
    attention = np.ones((n, 3))
    attention[n_full:, 1:] = restricted_attention
    _, after = mixed_rollout(mcfg, W, attention, signals[shift:], seeds, gammas, c0s, initial=state)
    return {k: np.concatenate((np.broadcast_to(before[k], (G,) + before[k].shape[1:]), after[k]), axis=2)
            for k in before}
