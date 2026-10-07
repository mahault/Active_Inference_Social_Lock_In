"""Contracts for src/pluralist/mixed.py (mixed pluralist / committed / neutral communities)."""

import numpy as np
import pytest

from tests.pluralist_x64 import import_pluralist, restore, x64_only_here  # noqa: F401

model, experiments, X = import_pluralist()
from src.pluralist import mixed as M  # noqa: E402

BASE = model.Config(*model.candidate_priors())
EXT = X.ExtConfig(BASE)
Q = 0.003
restore()

GRID = [(0.0, 0.0), (3.0, 0.0), (0.0, 0.3), (6.0, 1.0)]


def _ext_kw(g, c):
    kw = {"gamma": g}
    if c > 0:
        kw.update(resources=True, c0=c)
    return kw


@pytest.mark.parametrize("t,rho", [(M.PLURALIST, 1.0), (M.PLURALIST, 0.95),
                                   (M.COMMITTED, 1.0), (M.NEUTRAL, 1.0)])
def test_pure_community_reproduces_pure_type_runs_exactly(t, rho):
    ref = {M.PLURALIST: BASE,
           M.COMMITTED: X.monist_moment_matched(BASE, (1 - Q, Q)),
           M.NEUTRAL: X.monist_moment_matched(BASE)}[t]
    mc = M.MixedConfig(BASE, (t,) * 24, Q, X.ExtConfig(BASE, rho_L=rho))
    seeds = [0, 1]
    out = M.mixed_shift_protocol(mc, seeds, [g for g, _ in GRID], [c for _, c in GRID], steps=120)
    for i, (g, c) in enumerate(GRID):
        r = X.shift_protocol(X.ExtConfig(ref, rho_L=rho, **_ext_kw(g, c)), seeds, steps=120,
                             gates_after_shift=True)
        for k in ("p_mass_pos", "attention_M"):
            np.testing.assert_array_equal(out[k][i], r[k])


def test_cross_message_has_sender_marginal_and_receiver_conditional():
    # Agent 0 pluralist, agent 1 neutral monist; each listens only to the other.
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    mc = M.MixedConfig(BASE, (M.PLURALIST, M.NEUTRAL), Q, EXT)
    P0, h0 = M.references(mc)
    P, h = np.array(P0), np.array(h0)
    P[0] += np.diag([0.3, 0.1, 0.2, 0.5])           # make beliefs non-trivial
    h[0] += np.array([[0.1, 0.2, -0.3, 0.4], [0.0, 0.1, 0.1, -0.2]])
    L = np.array([[0.7, -0.4], [0.0, 0.0]])
    ops = M.social_operators(mc.types, W)
    Pn, hn = (np.asarray(x) for x in M.fuse(X.jnp.asarray(P), X.jnp.asarray(h), X.jnp.asarray(L), W, ops))
    m, S = (np.asarray(x) for x in M.observable_marginal_moment_matched(X.jnp.asarray(P), X.jnp.asarray(h), X.jnp.asarray(L)))
    for i, j in ((0, 1), (1, 0)):
        for k in range(2):
            cov = np.linalg.inv(Pn[i, k])
            mean = cov @ hn[i, k]
            np.testing.assert_allclose(cov[1:, 1:], S[j], atol=1e-10)      # sender's marginal
            np.testing.assert_allclose(mean[1:], m[j], atol=1e-10)
            np.testing.assert_allclose(Pn[i, k, 0], P[i, k, 0], atol=1e-12)  # receiver's conditional
            np.testing.assert_allclose(hn[i, k, 0], h[i, k, 0], atol=1e-12)
    # A monist's message is its own Gaussian's marginal (no mixture spread).
    np.testing.assert_allclose(S[1], np.linalg.inv(P[1, 0])[1:, 1:], atol=1e-12)


@pytest.fixture(scope="module")
def mixed_run():
    X.jax.config.update("jax_enable_x64", True)
    try:
        types = (M.PLURALIST,) * 2 + (M.COMMITTED,) * 1 + (M.NEUTRAL,) * 1 \
            + (M.PLURALIST,) * 6 + (M.COMMITTED,) * 7 + (M.NEUTRAL,) * 7
        mc = M.MixedConfig(BASE, types, Q, X.ExtConfig(BASE, rho_L=0.95))
        W = experiments.social_matrix(24)
        sig = experiments.signal_path(60, 30)
        att = np.ones((24, 3))
        att[4:, 1:] = 0.01
        final, hist = M.mixed_rollout(mc, W, att, sig, [0, 1], [0.0, 3.0], [0.0, 0.3])
        return types, final, hist
    finally:
        restore()


def test_monist_slots_stay_identical_and_support_inert(mixed_run):
    types, final, hist = mixed_run
    mono = np.asarray(types) != M.PLURALIST
    P, h, L = (np.asarray(x) for x in (final.P, final.h, final.L))
    np.testing.assert_array_equal(P[:, :, mono, 0], P[:, :, mono, 1])
    np.testing.assert_array_equal(h[:, :, mono, 0], h[:, :, mono, 1])
    np.testing.assert_array_equal(L[:, :, mono, 0], L[:, :, mono, 1])
    np.testing.assert_array_equal(hist["oxygen"][..., mono], 0.5)
    # pluralists' support moved
    assert np.abs(L[:, :, ~mono, 1] - L[:, :, ~mono, 0]).max() > 0.1


def test_fused_precisions_positive_definite_and_readouts_bounded(mixed_run):
    _, final, hist = mixed_run
    eig = np.linalg.eigvalsh(np.asarray(final.P))
    assert (eig > 0).all()
    assert ((hist["p_mass_pos"] >= 0) & (hist["p_mass_pos"] <= 1)).all()
    assert (hist["attention_M"] >= 0).all() and (hist["attention_M"] <= 1 + 1e-15).all()
    assert (np.asarray(final.r) >= EXT.r_min - 1e-15).all()


def test_support_operator_restricted_to_pluralists():
    types = (M.PLURALIST, M.COMMITTED, M.PLURALIST, M.NEUTRAL)
    W = experiments.social_matrix(4)
    ops = M.social_operators(types, W)
    Ws = ops["W_support"]
    np.testing.assert_allclose(Ws.sum(1), 1.0, atol=1e-15)
    np.testing.assert_array_equal(Ws[:, [1, 3]], 0.0)
    assert not ops["pure"]
    np.testing.assert_array_equal(ops["W_same"] + ops["W_cross"], W)
    pure = M.social_operators((M.PLURALIST,) * 4, W)
    assert pure["pure"] and pure["W_support"] is W
