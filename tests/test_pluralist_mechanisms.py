"""Contracts for src/pluralist/extended.py (endogenous precision, learned trust,
support forgetting, monist variant) on top of David Hyland's pluralist model."""

from dataclasses import replace

import numpy as np
import pytest

from tests.pluralist_x64 import import_pluralist, restore, x64_only_here  # noqa: F401

model, experiments, X = import_pluralist()
CFG = model.Config(*model.candidate_priors())  # float64, built while x64 is on
restore()
SEEDS = [3, 11]


def _setup(steps=25, shift=10, n=8):
    W = experiments.social_matrix(n, 0.3)
    attention = np.ones((n, 3))
    attention[n // 2:, 1:] = 0.2
    return W, attention, experiments.signal_path(steps, shift)


def _assert_same(original, ext):
    (f0, h0), (f1, h1) = original, ext
    for k in ("oxygen", "mixture"):
        np.testing.assert_array_equal(h0[k], h1[k])
    for a, b in zip(f0, (f1.P, f1.h, f1.L, f1.key)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fuse,alpha", [(True, 0.02), (False, 0.02), (True, 0.0)])
def test_all_off_is_bitwise_original_pluralist(fuse, alpha):
    cfg = replace(CFG, fuse=fuse, alpha=alpha)
    W, att, sig = _setup()
    _assert_same(model.rollout(cfg, W, att, sig, SEEDS),
                 X.ext_rollout(X.ExtConfig(cfg), W, att, sig, SEEDS))


def test_all_off_is_bitwise_original_monist():
    mono = X.monist_config(CFG)
    P0, h0 = model.gaussian_reference(CFG.P0, CFG.h0)
    np.testing.assert_array_equal(mono.P0, P0)
    np.testing.assert_array_equal(mono.h0, h0)
    W, att, sig = _setup()
    _assert_same(model.rollout(mono, W, att, sig, SEEDS),
                 X.ext_rollout(X.ExtConfig(mono), W, att, sig, SEEDS))


@pytest.mark.parametrize("off", [
    dict(gamma=0.0, conviction_mode="support"),          # conviction gate off
    dict(resources=False, c0=5.0, r_init=0.2),           # resource gate off
    dict(trust=False, trust_rho=0.5, trust_beta0=7.0),   # trust off
    dict(rho_L=1.0),                                     # forgetting off
])
def test_each_mechanism_off_recovers_original(off):
    W, att, sig = _setup()
    _assert_same(model.rollout(CFG, W, att, sig, SEEDS),
                 X.ext_rollout(X.ExtConfig(CFG, **off), W, att, sig, SEEDS))


@pytest.mark.parametrize("on", [dict(gamma=3.0), dict(resources=True, c0=0.1),
                                dict(trust=True), dict(rho_L=0.9)])
def test_each_mechanism_on_changes_dynamics(on):
    W, att, sig = _setup()
    f0, _ = model.rollout(CFG, W, att, sig, SEEDS)
    f1, _ = X.ext_rollout(X.ExtConfig(CFG, **on), W, att, sig, SEEDS)
    assert np.max(np.abs(np.asarray(f0[2]) - np.asarray(f1.L))) > 1e-6
    np.testing.assert_array_equal(f0[3], f1.key)  # no extra randomness consumed


def test_shift_protocol_all_off_reproduces_figure4_N11():
    seeds, steps = [0, 1], 90
    ref = experiments.network(CFG, seeds, steps)
    out = X.shift_protocol(X.ExtConfig(CFG), seeds, steps=steps)
    for g, ids in (("A", slice(0, 4)), ("B", slice(4, 24))):
        np.testing.assert_allclose(out["oxygen"][:, :, ids].mean(2), ref["N11__q_" + g],
                                   atol=1e-14, rtol=0)


def test_reference_fixture_through_extended_path():
    import json
    from pathlib import Path
    fixture = json.loads(Path(__file__).with_name("pluralist_reference.json").read_text())
    out = X.shift_protocol(X.ExtConfig(CFG), [fixture["seed"]], steps=fixture["steps"])
    exp = fixture["conditions"]["N11"]
    np.testing.assert_allclose(out["oxygen"][0, 79:84, :4].mean(1), exp["q_A"], atol=2e-11, rtol=1e-10)
    np.testing.assert_allclose(out["oxygen"][0, 79:84, 4:].mean(1), exp["q_B"], atol=2e-11, rtol=1e-10)


ALL_ON = X.ExtConfig(CFG, gamma=3.0, resources=True, c0=0.1, trust=True, rho_L=0.95)


@pytest.fixture(scope="module")
def all_on_runs():
    # Module-scoped fixtures run before the function-scoped x64 switch.
    X.jax.config.update("jax_enable_x64", True)
    try:
        return {name: X.shift_protocol(ecfg, [0, 1, 2], steps=160)
            for name, ecfg in (("plural", ALL_ON),
                               ("monist", replace(ALL_ON, base=X.monist_config(CFG))),
                               ("plural_excess", replace(ALL_ON, trust_surprisal="excess")))}
    finally:
        restore()


def test_trust_rows_stochastic_and_nonnegative():
    W, att, sig = _setup()
    for mode in ("full", "excess"):
        final, hist = X.ext_rollout(X.ExtConfig(CFG, trust=True, trust_surprisal=mode),
                                    W, att, sig, SEEDS, observers=[0, 1])
        adj = (W > 0).astype(float)
        T = np.asarray(X.jax.vmap(lambda a, b: X.trust_matrix(a, b, adj))(final.trust_a, final.trust_b))
        np.testing.assert_allclose(T.sum(-1), 1.0, atol=1e-12)
        assert (T >= 0).all() and (np.asarray(final.trust_b) > 0).all()
        assert ((hist["trust_on_observers"] >= 0) & (hist["trust_on_observers"] <= 1 + 1e-12)).all()


def test_attention_bounded_and_resources_nonnegative(all_on_runs):
    for out in all_on_runs.values():
        a = out["attention_M"]
        base = np.ones_like(a)
        base[:, 80:, 4:] = 0.01
        assert (a >= 0).all() and (a <= base + 1e-15).all()
        assert (out["r"] >= ALL_ON.r_min - 1e-15).all() and (out["r"] >= 0).all()
        assert ((out["conviction"] >= 0) & (out["conviction"] <= 1)).all()
        assert ((out["p_mass_pos"] >= 0) & (out["p_mass_pos"] <= 1)).all()


def test_affordable_attention_respects_budget():
    r = np.linspace(0.05, 3.0, 50)
    a, cost = X.affordable_attention(r, np.ones_like(r), ALL_ON, 1.0)
    a, cost = np.asarray(a), np.asarray(cost)
    assert ((a >= 0) & (a <= 1)).all()
    assert (cost <= ALL_ON.budget_fraction * r + 1e-12).all()
    assert np.all(np.diff(a) >= -1e-15)          # richer agents afford more
    assert (a[r <= ALL_ON.r_min] == 0).all()      # at/below r_min: cannot observe


def test_conviction_gate_monotone_in_gamma():
    means = []
    for g in (0.0, 1.0, 4.0):
        out = X.shift_protocol(X.ExtConfig(CFG, gamma=g), [0, 1], steps=100)
        means.append(out["attention_M"][:, 40:80].mean())
    assert means[0] == 1.0 and means[0] > means[1] > means[2]


def test_forgetting_bounds_support():
    W, att, sig = _setup(steps=200, shift=None)
    f0, _ = model.rollout(CFG, W, np.ones_like(att), sig, SEEDS)
    f1, _ = X.ext_rollout(X.ExtConfig(CFG, rho_L=0.9), W, np.ones_like(att), sig, SEEDS)
    spread0 = np.abs(np.diff(np.asarray(f0[2]), axis=-1)).mean()
    spread1 = np.abs(np.diff(np.asarray(f1.L), axis=-1)).mean()
    assert spread1 < spread0


def test_observers_learn_to_trust_each_other(all_on_runs):
    # Full observers see M after the shift and score each other's predictions
    # better than the restricted majority's: trust mass on observers rises
    # above the uniform 4/24 for observer rows.
    for name in ("plural", "plural_excess"):
        t = all_on_runs[name]["trust_on_observers"][:, 100:160, :4].mean()
        assert t > 4 / 24


def test_moment_matched_monist_matches_mixture_moments():
    mono = X.monist_moment_matched(CFG)
    P, h = np.asarray(CFG.P0), np.asarray(CFG.h0)
    cov = np.linalg.inv(P)[:, 1:, 1:]
    mean = np.linalg.solve(P, h[..., None])[..., 0][:, 1:]
    m = mean.mean(0)
    S = cov.mean(0) + np.einsum("ki,kj->ij", mean - m, mean - m) / 2
    Pm, hm = np.asarray(mono.P0[0]), np.asarray(mono.h0[0])
    np.testing.assert_allclose(np.linalg.inv(Pm)[1:, 1:], S, atol=1e-12)
    np.testing.assert_allclose(np.linalg.solve(Pm, hm)[1:], m, atol=1e-12)
    np.testing.assert_allclose(m, [1.0, 1.0, 0.0], atol=1e-12)


def test_gates_after_shift_share_prehistory_with_ungated_run():
    seeds = [0, 1]
    gated = X.ExtConfig(CFG, gamma=6.0, resources=True, c0=0.3, rho_L=0.95)
    a = X.shift_protocol(gated, seeds, steps=100, gates_after_shift=True)
    b = X.shift_protocol(X.ungated(gated), seeds, steps=100)
    for k in a:
        np.testing.assert_array_equal(a[k][:, :80], b[k][:, :80])
    assert np.abs(a["attention_M"][:, 80:, :4] - 1).max() > 0.5


def test_trust_attention_default_is_own_attention():
    W, att, sig = _setup()
    ecfg = X.ExtConfig(CFG, trust=True, trust_surprisal="excess", gamma=2.0)
    f0, h0 = X.ext_rollout(ecfg, W, att, sig, SEEDS)
    f1, h1 = X.ext_rollout(ecfg, W, att, sig, SEEDS, trust_attention=None)
    np.testing.assert_array_equal(f0.trust_b, f1.trust_b)
    cg = np.ones_like(att); cg[:, 2] = 0.0
    f2, _ = X.ext_rollout(ecfg, W, att, sig, SEEDS, trust_attention=cg)
    assert np.abs(np.asarray(f2.trust_b) - np.asarray(f0.trust_b)).max() > 1e-6
