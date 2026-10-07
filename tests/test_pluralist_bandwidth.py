"""Contracts for the P1g message-bandwidth options in src/pluralist/mixed.py."""

import numpy as np

from tests.pluralist_x64 import import_pluralist, restore, x64_only_here  # noqa: F401

model, experiments, X = import_pluralist()
from src.pluralist import mixed as M  # noqa: E402

BASE = model.Config(*model.candidate_priors())
EXT = X.ExtConfig(BASE)
restore()


def _closed_run(mc, seeds, steps=60):
    n = mc.n
    W = experiments.social_matrix(n)
    sig = np.broadcast_to([0.0, 1.0, 1.0, 1.0], (steps, 4))
    _, hist = M.mixed_rollout(mc, W, np.zeros((n, 3)), sig, seeds, [0.0], [0.0])
    return hist


def test_observables_fusion_equals_rule_for_monists():
    for t in (M.COMMITTED, M.NEUTRAL):
        a = M.MixedConfig(BASE, (t,) * 24, 0.003, EXT)
        b = M.MixedConfig(BASE, (t,) * 24, 0.003, EXT, fusion="observables")
        ha = M.mixed_shift_protocol(a, [0, 1], [0.0, 3.0], [0.0, 0.3], steps=100)
        hb = M.mixed_shift_protocol(b, [0, 1], [0.0, 3.0], [0.0, 0.3], steps=100)
        np.testing.assert_allclose(ha["p_mass_pos"], hb["p_mass_pos"], atol=1e-10)


def test_sampling_does_not_touch_observation_stream_and_converges():
    # The observation noise is drawn from the main key stream, sampling from a
    # folded key: with an enormous m the run approaches the exact-moment run.
    exact = M.MixedConfig(BASE, (M.NEUTRAL,) * 24, 0.003, EXT, fusion="observables")
    big = M.MixedConfig(BASE, (M.NEUTRAL,) * 24, 0.003, EXT, fusion="observables", message_m=4000)
    ha = M.mixed_shift_protocol(exact, [0], [0.0], [0.0], steps=90)
    hb = M.mixed_shift_protocol(big, [0], [0.0], [0.0], steps=90)
    assert np.abs(ha["p_mass_pos"] - hb["p_mass_pos"]).max() < 0.03


def test_sample_support_unbiased_in_log_odds_direction():
    L = X.jnp.array([[0.0, 2.0]] * 4000)
    ls = np.asarray(M.sample_support(L, X.jax.random.PRNGKey(0), 50))
    q = np.exp(ls) / np.exp(ls).sum(-1, keepdims=True)
    assert abs(q[:, 1].mean() - 1 / (1 + np.exp(-2.0))) < 0.02


def test_closed_world_symmetry_exact_without_sampling_broken_with():
    mc0 = M.MixedConfig(BASE, (M.PLURALIST,) * 8, 0.0, EXT)
    h0 = _closed_run(mc0, [0, 1])
    np.testing.assert_allclose(h0["oxygen"], 0.5, atol=1e-12)
    mc1 = M.MixedConfig(BASE, (M.PLURALIST,) * 8, 0.0, EXT, fusion="observables", message_m=1, support_m=1)
    h1 = _closed_run(mc1, [0, 1])
    assert np.abs(h1["oxygen"][..., -1, :] - 0.5).max() > 1e-3
    assert np.isfinite(h1["p_mass_pos"]).all()


def test_sampled_message_requires_observables_fusion():
    mc = M.MixedConfig(BASE, (M.PLURALIST,) * 8, 0.0, EXT, message_m=3)
    try:
        _closed_run(mc, [0], steps=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
