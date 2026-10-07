"""World variants, rival-theory priors and outcome-based metrics for P1f.

Signals have the experiments.signal_path layout: (steps, 4) rows
(hidden, C, G, M) with C = G = 1 throughout; only M changes.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from .model import Config


# ------------------------------------------------------------------
# Worlds
# ------------------------------------------------------------------

def multi_flip_signals(steps: int = 680, first: int = 80, period: int = 150):
    """M = -1 before ``first``; then the sign flips every ``period`` rounds,
    starting at +1 (flips at 80, 230, 380, 530 for the defaults)."""
    s = np.broadcast_to([0.0, 1.0, 1.0, -1.0], (steps, 4)).copy()
    t = np.arange(steps)
    seg = (t - first) // period
    s[:, 3] = np.where(t < first, -1.0, np.where(seg % 2 == 0, 1.0, -1.0))
    return s


def drift_signals(steps: int = 600, first: int = 80, half_period: int = 260):
    """M = -1 before ``first``; then M(t) = -cos(pi (t - first) / half_period):
    a slow drift from -1 to +1 (t = first + half_period) and back."""
    s = np.broadcast_to([0.0, 1.0, 1.0, -1.0], (steps, 4)).copy()
    t = np.arange(steps)
    s[:, 3] = np.where(t < first, -1.0, -np.cos(np.pi * (t - first) / half_period))
    return s


# ------------------------------------------------------------------
# Rival theories
# ------------------------------------------------------------------

def theory_priors(m_values, structures):
    """Hyland's DAG compilation for arbitrary theory menus.

    ``structures``: per theory "phlogiston" (G <- C, M <- -0.8 latent) or
    "oxygen" (G <- latent, M <- +0.8 latent); ``m_values``: each theory's mean
    of M. Means of (latent, C, G) stay (0, 1, 1); residual variances as in
    model.candidate_priors. With m_values (-1, 1) and structures
    ("phlogiston", "oxygen") this reproduces model.candidate_priors exactly.
    """
    s = jnp.array([0.25, 0.5, 0.5, 0.5])
    P, h = [], []
    for m, kind in zip(m_values, structures):
        B = np.zeros((4, 4))
        if kind == "phlogiston":
            B[0, 1], B[2, 1], B[3, 0] = 0.8, 0.8, -0.8
        elif kind == "oxygen":
            B[0, 1], B[2, 0], B[3, 0] = 0.8, 0.8, 0.8
        else:
            raise ValueError(kind)
        means = np.array([0.0, 1.0, 1.0, float(m)])
        b = (np.eye(4) - B) @ means
        A = jnp.eye(4) - jnp.asarray(B)
        AtD = A.T * (1.0 / s)[None, :]
        P.append(AtD @ A)
        h.append(AtD @ jnp.asarray(b))
    return jnp.stack(P), jnp.stack(h)


RIVALS = {
    "original": ((-1.0, 1.0), ("phlogiston", "oxygen")),
    "rival_M+3": ((-1.0, 3.0), ("phlogiston", "oxygen")),      # right sign, wrong magnitude
    "rival_M-3": ((-1.0, -3.0), ("phlogiston", "oxygen")),     # wrong sign: both theories wrong
    "K3_-1_+1_+3": ((-1.0, 1.0, 3.0), ("phlogiston", "oxygen", "oxygen")),  # one of three right
}


def rival_config(cfg: Config, name: str) -> Config:
    m, kinds = RIVALS[name]
    P0, h0 = theory_priors(m, kinds)
    return replace(cfg, P0=P0, h0=h0)


# ------------------------------------------------------------------
# Outcome-based metrics
# ------------------------------------------------------------------

def outcome_metrics(p_final, p_comm_trace, shift: int = 80, truth_positive: bool = True,
                    reach: float = 0.75, eps: float = 1e-6):
    """Metrics against the true sign of M at the end of the run.

    p_final      : (..., S, N) per-agent final P(M>0)
    p_comm_trace : (..., S, T) community-mean P(M>0) per round
    Returns dict of (..., S) arrays:
      frac_agents_reach  fraction of agents with P(truth) >= reach
      community_reach    1 if the community mean P(truth) >= reach
      log_score          mean over agents of log P(truth)   (<= 0; 0 is perfect)
      brier              mean over agents of (1 - P(truth))^2
      t_reach            first post-shift round with community P(truth) >= reach
                         (censored at the horizon; see t_reach_censored)
    """
    p_true = p_final if truth_positive else 1.0 - p_final
    c_true = p_comm_trace if truth_positive else 1.0 - p_comm_trace
    post = c_true[..., shift:] >= reach
    horizon = post.shape[-1]
    t_reach = np.where(post.any(-1), post.argmax(-1), horizon).astype(float)
    return {
        "frac_agents_reach": (p_true >= reach).mean(-1),
        "community_reach": (p_true.mean(-1) >= reach).astype(float),
        "log_score": np.log(np.clip(p_true, eps, 1.0)).mean(-1),
        "brier": ((1.0 - p_true) ** 2).mean(-1),
        "t_reach": t_reach,
        "t_reach_censored": (~post.any(-1)).astype(float),
    }


def tracking_metrics(p_comm_trace, signals, start: int = 80):
    """Changing-world accuracy of the community-mean P(M>0) over rounds >= start
    against the CURRENT sign of M: mean Brier, mean fraction of rounds on the
    right side of .5, and mean log score. Shapes (..., S)."""
    m = np.asarray(signals)[start:, 3]
    p = p_comm_trace[..., start:]
    p_true = np.where(m > 0, p, 1.0 - p)
    return {"track_brier": ((1 - p_true) ** 2).mean(-1),
            "track_right_side": (p_true > 0.5).mean(-1),
            "track_log_score": np.log(np.clip(p_true, 1e-6, 1.0)).mean(-1)}
