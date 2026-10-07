"""Delayed public verification of forecasts as the social-trust channel (P1d).

Design D1 of notes/trust_models_lit_review_2026-10-05.md (after Diaconescu et
al. 2014 delayed verification; Sobel 1985; Merdes et al.), on top of the mixed
communities of ``mixed.py``.

Each round t every agent commits a forecast of the discriminating outcome M
from its post-communication prior: the predictive density of a noisy M reading,
p_j(y) = sum_k q_jk N(y; mu_jk, Var_jk(M) + sigma^2) (mixture for pluralists,
one Gaussian for monists). A public outcome y*_t = M_t + sigma * xi is drawn
from a key folded off the round's observation key (so no existing random
stream changes) and is revealed at round t + D to the "seers".

Trust rules (``rule``):
  "fixed"    : W (Hyland).
  "relative" : seer i accumulates l_ij <- lam l_ij + s_j at each reveal,
               s_j = log p_j(y*) - log pbar(y*), pbar = population-average
               (linear-pool) forecast; trust w_ij ∝ W_ij exp(kappa l_ij), rows
               normalised (self-loop kept). Non-seers keep W.
               Any reference common to all j cancels in the row normalisation;
               the population pool is used so l_ij reads as j's log Bayes
               factor against the crowd and is identical for every seer.
  "gamma"    : the Gamma-conjugate accuracy rule (simple_step.trust_update) fed
               the same revealed outcome: a <- rho a + 1, b <- rho b + eps_j,
               eps_j = -log p_j(y*) - .5 log(2 pi sigma^2) (excess surprisal,
               >= 0); w_ij ∝ W_ij a_ij / b_ij.
  "agree"    : expectation-based (O'Connor & Weatherall): w_ij ∝ W_ij
               max(0, 1 - m |p_i - p_j|), p = P(M > 0) of the entering beliefs.
               Uses no outcomes.
Trust replaces W in support sharing (restricted to pluralists, row
renormalised, as in mixed.py) and in belief fusion (mixed.fuse, cross-type rule
unchanged), and in the resource inflow shares.

Modes: "trust" - y* only updates trust; "trust+belief" - at the reveal, y* also
enters each seer's evidence as one extra reading of M at the seer's current
(post-gate) M attention (``reveal_precision="attention"``, P1d) or at full
precision 1/sigma^2 (``"full"``, P1e), both as precision/information and in the support
score (scored under the same post-communication prior as that round's direct
reading).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from .extended import EPS, M_ROW, conviction, prob_mass_positive
from .mixed import MixedConfig, fuse, mixed_initial_state, references, social_operators
from .model import predictive_score

REVEAL_FOLD = 1_000_003


@dataclass(frozen=True)
class VerifyConfig:
    mixed: MixedConfig
    rule: str = "fixed"          # fixed | relative | gamma | agree
    mode: str = "trust"          # trust | trust+belief
    delay: int | None = None     # None = outcomes never revealed
    kappa: float = 1.0
    lam: float = 0.98
    gamma_rho: float = 0.95
    agree_m: float = 2.0
    # "attention": y* enters at the seer's current M attention (P1d);
    # "full": a public replication is seen clearly, precision 1/sigma^2 (P1e).
    reveal_precision: str = "attention"
    # Only every ``reveal_every``-th round's outcome is ever made public
    # (sparse replication events); 1 = every round (P1d/P1e main).
    reveal_every: int = 1
    # Public outcome noise as a multiple of the sensor noise sigma (P1f); the
    # forecast scoring and the evidence precision use the same noise.
    reveal_noise: float = 1.0

    @property
    def reveals(self):
        return self.delay is not None

    @property
    def trust_is_W(self):
        return (self.rule == "fixed" or (self.rule == "relative" and self.kappa == 0.0)
                or (self.rule in ("relative", "gamma") and not self.reveals))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VState:
    P: jax.Array
    h: jax.Array
    L: jax.Array
    key: jax.Array
    r: jax.Array
    ell: jax.Array     # (N,N) accumulated relative log scores
    ta: jax.Array      # (N,N) Gamma shape
    tb: jax.Array      # (N,N) Gamma rate
    bq: jax.Array      # (D,N,2) forecast weights
    bm: jax.Array      # (D,N,2) forecast means
    bv: jax.Array      # (D,N,2) forecast variances
    by: jax.Array      # (D,) outcomes awaiting reveal
    t: jax.Array       # round counter


def initial_state(vcfg: VerifyConfig, seeds) -> VState:
    s0 = mixed_initial_state(vcfg.mixed, seeds)
    S, n, D = len(seeds), vcfg.mixed.n, max(vcfg.delay or 1, 1)
    z = jnp.zeros((S, D, n, 2))
    return VState(P=s0.P, h=s0.h, L=s0.L, key=s0.key, r=s0.r,
                  ell=jnp.zeros((S, n, n)), ta=jnp.ones((S, n, n)), tb=jnp.ones((S, n, n)),
                  bq=z + 0.5, bm=z, bv=z + 1.0, by=jnp.zeros((S, D)), t=jnp.zeros((S,), jnp.int32))


def forecast(P, h, L, sigma):
    """Mixture forecast of a noisy M reading: weights, means, variances (N,2)."""
    q = jax.nn.softmax(L, axis=-1)
    mean = jnp.linalg.solve(P, h[..., None])[..., 0][..., 3]
    var = jnp.linalg.inv(P)[..., 3, 3] + sigma**2
    return q, mean, var


def log_forecast_density(q, m, v, y):
    lp = -0.5 * (jnp.log(2 * jnp.pi * v) + (y - m) ** 2 / v)
    return logsumexp(jnp.log(q) + lp, axis=-1)          # (N,)


def row_normalise(A):
    return A / (A.sum(axis=1, keepdims=True) + EPS)


def trust_matrix(vcfg, state, W, p_in):
    if vcfg.trust_is_W:
        return W
    if vcfg.rule == "relative":
        z = vcfg.kappa * state.ell
        z = z - jnp.max(jnp.where(W > 0, z, -jnp.inf), axis=1, keepdims=True)
        return row_normalise(W * jnp.exp(z))
    if vcfg.rule == "gamma":
        return row_normalise(W * state.ta / state.tb)
    if vcfg.rule == "agree":
        d = jnp.abs(p_in[:, None] - p_in[None, :])
        return row_normalise(W * jnp.maximum(0.0, 1.0 - vcfg.agree_m * d))
    raise ValueError(vcfg.rule)


def verified_step(state: VState, signal, vcfg: VerifyConfig, W, attention, ops,
                  gamma, c0, P0, h0, seer):
    mcfg = vcfg.mixed
    cfg, ecfg = mcfg.base, mcfg.ext
    n = W.shape[0]
    P, h, L, key = state.P, state.h, state.L, state.key
    p_in = prob_mass_positive(P, h, L)
    S = trust_matrix(vcfg, state, W, p_in)

    # support sharing among pluralists, then fusion, with trust S
    if ops["any_plural"]:
        if ops["all_plural"]:
            Ssup = S
        elif vcfg.trust_is_W:
            Ssup = jnp.asarray(ops["W_support"])
        else:
            Sp = S * jnp.asarray(ops["plural"], S.dtype)[None, :]
            Ssup = Sp / (Sp.sum(1, keepdims=True) + EPS)
        shared = (1.0 - cfg.alpha) * L + cfg.alpha * (Ssup @ L)
        L = shared if ops["all_plural"] else jnp.where(jnp.asarray(ops["plural"])[:, None], shared, L)
    if vcfg.trust_is_W:
        fops = ops
    else:
        same = jnp.asarray(ops["W_same"] > 0, S.dtype) if not ops["pure"] else None
        fops = dict(ops) if ops["pure"] else dict(ops, W_same=S * same, W_cross=S * (1 - same))
    P, h = fuse(P, h, state.L, S, fops)

    # world-channel gates (as mixed.py)
    a_m = attention[:, M_ROW] * jnp.exp(-gamma * conviction(P, h, L, "predictive"))
    slack = jnp.maximum(state.r - ecfg.r_min, 0.0)
    on = c0 > 0
    a_max = jnp.where(on, ecfg.budget_fraction * state.r * slack * cfg.sigma**2
                      / jnp.where(on, c0, 1.0), jnp.inf)
    a_m = jnp.minimum(a_m, a_max)
    cost = jnp.where(on, c0 * (a_m / cfg.sigma**2) / jnp.maximum(slack, EPS), 0.0)
    attention = attention.at[:, M_ROW].set(a_m)

    sig_rev = vcfg.reveal_noise * cfg.sigma
    fq, fm, fv = forecast(P, h, L, sig_rev)

    key, observation_key = jax.random.split(key)
    agent_keys = jax.random.split(observation_key, n)
    y_now = signal[3] + sig_rev * jax.random.normal(jax.random.fold_in(observation_key, REVEAL_FOLD))

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
    P_prior, h_prior = P, h
    P = P + J[:, None, :, :]
    h = h + j[:, None, :]

    ell, ta, tb = state.ell, state.ta, state.tb
    bq, bm, bv, by = state.bq, state.bm, state.bv, state.by
    if vcfg.reveals:
        D = vcfg.delay
        idx = state.t % D
        valid = state.t >= D
        if vcfg.reveal_every != 1:
            valid = valid & ((state.t - D) % vcfg.reveal_every == 0)
        y_old = by[idx]
        lp = log_forecast_density(bq[idx], bm[idx], bv[idx], y_old)     # (N,)
        upd = valid & (seer > 0)                                        # (N,) receivers
        if vcfg.rule == "relative":
            s = lp - (logsumexp(lp) - jnp.log(n))
            ell = jnp.where(upd[:, None], vcfg.lam * ell + s[None, :], ell)
        elif vcfg.rule == "gamma":
            eps = -lp - 0.5 * jnp.log(2 * jnp.pi * sig_rev**2)
            ta = jnp.where(upd[:, None], vcfg.gamma_rho * ta + 1.0, ta)
            tb = jnp.where(upd[:, None], vcfg.gamma_rho * tb + eps[None, :], tb)
        if vcfg.mode == "trust+belief":
            if vcfg.reveal_precision == "full":
                a_rev = jnp.where(upd, 1.0, 0.0)
            elif vcfg.reveal_precision == "attention":
                a_rev = jnp.where(upd, attention[:, M_ROW], 0.0)           # (N,)
            else:
                raise ValueError(vcfg.reveal_precision)
            Hm = jnp.eye(4)[3:4]

            def rev_score(Pk, hk, a):
                return jax.vmap(lambda p, q: predictive_score(
                    p, q, Hm, jnp.array([y_old]), sig_rev, jnp.array([a])))(Pk, hk)

            scores = scores + jnp.where(upd[:, None], jax.vmap(rev_score)(P_prior, h_prior, a_rev), 0.0)
            P = P.at[..., 3, 3].add((a_rev / sig_rev**2)[:, None])
            h = h.at[..., 3].add((a_rev * y_old / sig_rev**2)[:, None])
        elif vcfg.mode != "trust":
            raise ValueError(vcfg.mode)
        bq, bm, bv, by = bq.at[idx].set(fq), bm.at[idx].set(fm), bv.at[idx].set(fv), by.at[idx].set(y_now)

    P = P0 + cfg.omega * (P - P0)
    h = h0 + cfg.omega * (h - h0)
    if ecfg.rho_L != 1.0:
        L = ecfg.rho_L * L
    L = L + scores

    incoming = S.sum(axis=0)
    eta = incoming / (incoming.sum() + EPS)
    r_upd = jnp.maximum((1.0 - ecfg.delta) * state.r + ecfg.R_in * n * eta - cost, ecfg.r_min)
    r = jnp.where(on, r_upd, state.r)
    new = VState(P=P, h=h, L=L, key=key, r=r, ell=ell, ta=ta, tb=tb,
                 bq=bq, bm=bm, bv=bv, by=by, t=state.t + 1)
    return new, S


def verified_rollout(vcfg: VerifyConfig, W, attention, signals, seeds, gammas, c0s, seer_masks,
                     initial=None):
    """Batch (F seer sets) x (G grid points) x (S seeds). Histories (F,G,S,T).

    Readouts per round: p_comm (mean P(M>0)), frac_correct (agents on the
    right side of .5 given the current sign of M), trust_correct (mean over
    agents of the trust share on currently-correct agents), and the same for
    seers only (trust_correct_seers)."""
    W_np = np.asarray(W)
    ops = social_operators(vcfg.mixed.types, W_np)
    W, attention, signals = map(jnp.asarray, (W_np, attention, signals))
    P0, h0 = references(vcfg.mixed)
    gammas, c0s = jnp.asarray(gammas, float), jnp.asarray(c0s, float)
    seer_masks = jnp.asarray(seer_masks, float)
    F, G = seer_masks.shape[0], gammas.shape[0]
    if initial is None:
        s0 = initial_state(vcfg, seeds)
        initial = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (F, G) + x.shape), s0)

    def one(state, gamma, c0, seer):
        def update(carry, signal):
            new, S = verified_step(carry, signal, vcfg, W, attention, ops, gamma, c0, P0, h0, seer)
            p = prob_mass_positive(new.P, new.h, new.L)
            correct = jnp.where(signal[3] > 0, p > 0.5, p < 0.5).astype(p.dtype)
            share = S @ correct
            ns = jnp.maximum(seer.sum(), 1.0)
            return new, {"p_comm": p.mean(), "frac_correct": correct.mean(),
                         "trust_correct": share.mean(),
                         "trust_correct_seers": (share * seer).sum() / ns}
        return jax.lax.scan(update, state, signals)

    f_s = jax.vmap(one, in_axes=(0, None, None, None))
    f_gs = jax.vmap(f_s, in_axes=(0, 0, 0, None))
    f_fgs = jax.vmap(f_gs, in_axes=(0, None, None, 0))
    final, hist = jax.jit(f_fgs)(initial, gammas, c0s, seer_masks)
    hist = {k: np.asarray(v) for k, v in hist.items()}
    if not all(np.isfinite(v).all() for v in hist.values()):
        raise FloatingPointError("Non-finite simulation readout")
    return final, hist


def verified_shift_protocol(vcfg: VerifyConfig, seeds, gammas, c0s, seer_masks, steps: int = 600,
                            shift: int = 80, n_full: int = 4, restricted_attention: float = 0.01,
                            phase2_seed_offset: int = 1000, signals=None):
    """Fig-4 protocol; gates from the shift; trust and reveals run throughout.
    Returns (final state, histories (F,G,S,T), final per-agent P(M>0) (F,G,S,N)).
    ``signals`` (steps, 4) overrides the single-flip world (worlds.py)."""
    from .experiments import signal_path, social_matrix

    n = vcfg.mixed.n
    W = social_matrix(n)
    signals = signal_path(steps, shift) if signals is None else np.asarray(signals)
    state, before = verified_rollout(vcfg, W, np.ones((n, 3)), signals[:shift], seeds,
                                     [0.0], [0.0], seer_masks)
    keys = jnp.stack([jax.random.PRNGKey(int(s) + phase2_seed_offset) for s in seeds])
    G = len(gammas)
    state = replace(state, key=jnp.broadcast_to(keys, state.key.shape))
    state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[:, :1], (x.shape[0], G) + x.shape[2:]), state)
    attention = np.ones((n, 3))
    attention[n_full:, 1:] = restricted_attention
    final, after = verified_rollout(vcfg, W, attention, signals[shift:], seeds, gammas, c0s,
                                    seer_masks, initial=state)
    hist = {k: np.concatenate((np.broadcast_to(before[k], after[k].shape[:2] + before[k].shape[2:]),
                               after[k]), axis=-1) for k in before}
    p_final = np.asarray(jax.vmap(jax.vmap(jax.vmap(prob_mass_positive)))(final.P, final.h, final.L))
    return final, hist, p_final
