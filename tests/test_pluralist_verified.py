"""Contracts for src/pluralist/verified.py (delayed public verification -> trust)."""

import numpy as np
import pytest

from tests.pluralist_x64 import import_pluralist, restore, x64_only_here  # noqa: F401

model, experiments, X = import_pluralist()
from src.pluralist import mixed as M  # noqa: E402
from src.pluralist import verified as V  # noqa: E402

BASE = model.Config(*model.candidate_priors())
TYPES = (0, 0, 1, 1) + (0,) * 10 + (1,) * 10
MC = M.MixedConfig(BASE, TYPES, 0.003, X.ExtConfig(BASE))
restore()

SEEDS = [0, 1]
G, C = [0.0, 3.0], [0.0, 0.3]


def _setup(steps=40, shift=15):
    W = experiments.social_matrix(24)
    att = np.ones((24, 3))
    att[4:, 1:] = 0.01
    seer = np.zeros((2, 24))
    seer[0, 4:10] = 1
    seer[1] = 1
    return W, att, experiments.signal_path(steps, shift), seer


def _mixed_final(W, att, sig):
    final, _ = M.mixed_rollout(MC, W, att, sig, SEEDS, G, C)
    return final


@pytest.mark.parametrize("vcfg", [
    V.VerifyConfig(MC, "relative", "trust", None),            # never revealed
    V.VerifyConfig(MC, "relative", "trust", 5, kappa=0.0),     # kappa = 0
    V.VerifyConfig(MC, "gamma", "trust", None),               # gamma rule, never revealed
    V.VerifyConfig(MC, "fixed", "trust", 5),                  # mode (a), fixed trust, reveals on
])
def test_reduces_exactly_to_fixed_W_mixed_run(vcfg):
    W, att, sig, seer = _setup()
    ref = _mixed_final(W, att, sig)
    final, _ = V.verified_rollout(vcfg, W, att, sig, SEEDS, G, C, seer)
    for f in range(seer.shape[0]):
        for name in ("P", "h", "L", "r"):
            np.testing.assert_array_equal(np.asarray(getattr(final, name))[f], np.asarray(getattr(ref, name)))


def test_relative_rule_with_zero_scores_is_W():
    W = experiments.social_matrix(24)
    vcfg = V.VerifyConfig(MC, "relative", "trust", 5)
    st = V.initial_state(vcfg, [0])
    st0 = X.jax.tree_util.tree_map(lambda x: x[0], st)
    S = np.asarray(V.trust_matrix(vcfg, st0, X.jnp.asarray(W), X.jnp.zeros(24)))
    np.testing.assert_allclose(S, W, atol=1e-15)


@pytest.mark.parametrize("rule", ["relative", "gamma", "agree"])
def test_trust_rows_stochastic(rule):
    W, att, sig, seer = _setup(steps=60, shift=20)
    vcfg = V.VerifyConfig(MC, rule, "trust", 3)
    final, hist = V.verified_rollout(vcfg, W, att, sig, SEEDS, G, C, seer)
    st = X.jax.tree_util.tree_map(lambda x: x[1, 0, 0], final)
    p = X.prob_mass_positive(st.P, st.h, st.L)
    S = np.asarray(V.trust_matrix(vcfg, st, X.jnp.asarray(W), p))
    np.testing.assert_allclose(S.sum(1), 1.0, atol=1e-12)
    assert (S >= 0).all() and (np.diag(S) > 0).all()
    assert ((hist["trust_correct"] >= -1e-12) & (hist["trust_correct"] <= 1 + 1e-12)).all()


def test_mode_b_changes_beliefs_and_mode_a_does_not():
    W, att, sig, seer = _setup()
    ref = _mixed_final(W, att, sig)
    fa, _ = V.verified_rollout(V.VerifyConfig(MC, "fixed", "trust", 2), W, att, sig, SEEDS, G, C, seer)
    fb, _ = V.verified_rollout(V.VerifyConfig(MC, "fixed", "trust+belief", 2), W, att, sig, SEEDS, G, C, seer)
    np.testing.assert_array_equal(np.asarray(fa.P)[0], np.asarray(ref.P))
    assert np.abs(np.asarray(fb.P)[0] - np.asarray(ref.P)).max() > 1e-6


def test_relative_trust_favours_accurate_forecasters():
    # Pluralist community, open world, everyone sees outcomes after 1 round:
    # after the shift the 4 observers forecast better; seers' trust share on
    # correct agents exceeds the fraction of correct agents.
    mc = M.MixedConfig(BASE, (0,) * 24, 0.003, X.ExtConfig(BASE))
    seer = np.ones((1, 24))
    _, hist, _ = V.verified_shift_protocol(V.VerifyConfig(mc, "relative", "trust", 1, kappa=1.0),
                                           [0, 1], [0.0], [0.0], seer, steps=140)
    post = slice(82, 140)
    excess = hist["trust_correct_seers"][..., post] - hist["frac_correct"][..., post]
    assert excess.mean() > 0.0


def test_full_precision_reveal_exceeds_attention_weighted_for_restricted_seers():
    W, att, sig, seer = _setup()
    fa, _ = V.verified_rollout(V.VerifyConfig(MC, "fixed", "trust+belief", 2), W, att, sig, SEEDS, G, C, seer)
    ff, _ = V.verified_rollout(V.VerifyConfig(MC, "fixed", "trust+belief", 2, reveal_precision="full"),
                               W, att, sig, SEEDS, G, C, seer)
    ref = _mixed_final(W, att, sig)
    # restricted seers (4-9, seer set 0) gain more M precision under full reveals
    gain_full = np.asarray(ff.P)[0][..., 4:10, :, 3, 3] - np.asarray(ref.P)[..., 4:10, :, 3, 3]
    gain_att = np.asarray(fa.P)[0][..., 4:10, :, 3, 3] - np.asarray(ref.P)[..., 4:10, :, 3, 3]
    assert gain_full.mean() > 5 * gain_att.mean() > 0


def test_sparse_reveals_between_none_and_every_round():
    W, att, sig, seer = _setup()
    ref = _mixed_final(W, att, sig)
    kw = dict(rule="fixed", mode="trust+belief", delay=2, reveal_precision="full")
    f1, _ = V.verified_rollout(V.VerifyConfig(MC, **kw), W, att, sig, SEEDS, G, C, seer)
    f5, _ = V.verified_rollout(V.VerifyConfig(MC, reveal_every=5, **kw), W, att, sig, SEEDS, G, C, seer)
    g1 = (np.asarray(f1.P)[0] - np.asarray(ref.P))[..., 4:10, :, 3, 3].mean()
    g5 = (np.asarray(f5.P)[0] - np.asarray(ref.P))[..., 4:10, :, 3, 3].mean()
    assert g1 > g5 > 0
