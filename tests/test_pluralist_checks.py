"""Contracts for the P1f additions: worlds.py, K>2 mixed communities, reveal noise."""

import numpy as np
import pytest

from tests.pluralist_x64 import import_pluralist, restore, x64_only_here  # noqa: F401

model, experiments, X = import_pluralist()
from src.pluralist import mixed as M  # noqa: E402
from src.pluralist import verified as V  # noqa: E402
from src.pluralist import worlds as Wd  # noqa: E402

BASE = model.Config(*model.candidate_priors())
restore()


def test_theory_priors_reproduce_candidate_priors_exactly():
    P, h = Wd.theory_priors((-1.0, 1.0), ("phlogiston", "oxygen"))
    np.testing.assert_array_equal(np.asarray(P), np.asarray(BASE.P0))
    np.testing.assert_array_equal(np.asarray(h), np.asarray(BASE.h0))


@pytest.mark.parametrize("name", list(Wd.RIVALS))
def test_rival_means(name):
    c = Wd.rival_config(BASE, name)
    mean = np.linalg.solve(np.asarray(c.P0), np.asarray(c.h0)[..., None])[..., 0]
    np.testing.assert_allclose(mean[:, 3], Wd.RIVALS[name][0], atol=1e-12)
    np.testing.assert_allclose(mean[:, 1:3], 1.0, atol=1e-12)


def test_worlds():
    s = Wd.multi_flip_signals()
    assert s.shape == (680, 4)
    np.testing.assert_array_equal(s[[0, 79, 80, 229, 230, 379, 380, 530, 679], 3],
                                  [-1, -1, 1, 1, -1, -1, 1, -1, -1])
    np.testing.assert_array_equal(s[:, :3], np.broadcast_to([0, 1, 1], (680, 3)))
    d = Wd.drift_signals()
    np.testing.assert_allclose(d[[79, 80, 340], 3], [-1, -1, 1], atol=1e-12)
    assert np.all(np.abs(d[:, 3]) <= 1)


def test_custom_signals_reproduce_default_protocols():
    seeds = [0, 1]
    sig = experiments.signal_path(120, 80)
    mc = M.MixedConfig(BASE, (0, 0, 1, 1) + (0,) * 10 + (2,) * 10, 0.003, X.ExtConfig(BASE))
    a = M.mixed_shift_protocol(mc, seeds, [0.0, 3.0], [0.0, 0.3], steps=120)
    b = M.mixed_shift_protocol(mc, seeds, [0.0, 3.0], [0.0, 0.3], steps=120, signals=sig)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    seer = np.zeros((1, 24))
    seer[0, 4:] = 1
    vc = V.VerifyConfig(mc, "relative", "trust+belief", 5, reveal_precision="full", reveal_every=3)
    _, h1, p1 = V.verified_shift_protocol(vc, seeds, [0.0], [0.0], seer, steps=120)
    _, h2, p2 = V.verified_shift_protocol(vc, seeds, [0.0], [0.0], seer, steps=120, signals=sig)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(h1["p_comm"], h2["p_comm"])


def test_reveal_noise_one_is_identity_and_three_weakens_reveals():
    seeds = [0]
    mc = M.MixedConfig(BASE, (0,) * 24, 0.003, X.ExtConfig(BASE))
    seer = np.zeros((1, 24))
    seer[0, 4:] = 1
    kw = dict(rule="fixed", mode="trust+belief", delay=2, reveal_precision="full")
    f0, _, _ = V.verified_shift_protocol(V.VerifyConfig(mc, **kw), seeds, [0.0], [0.0], seer, steps=100)
    f1, _, _ = V.verified_shift_protocol(V.VerifyConfig(mc, reveal_noise=1.0, **kw), seeds, [0.0], [0.0],
                                         seer, steps=100)
    f3, _, _ = V.verified_shift_protocol(V.VerifyConfig(mc, reveal_noise=3.0, **kw), seeds, [0.0], [0.0],
                                         seer, steps=100)
    np.testing.assert_array_equal(np.asarray(f0.P), np.asarray(f1.P))
    prec1 = np.asarray(f1.P)[..., 4:, :, 3, 3].mean()
    prec3 = np.asarray(f3.P)[..., 4:, :, 3, 3].mean()
    assert prec1 > prec3


def test_k3_communities_and_padding():
    base3 = Wd.rival_config(BASE, "K3_-1_+1_+3")
    types = (0, 1, 2, 0) + (0,) * 6 + (1,) * 7 + (2,) * 7
    mc = M.MixedConfig(base3, types, (0.99, 0.005, 0.005), X.ExtConfig(base3))
    P0, h0 = M.references(mc)
    assert P0.shape == (24, 3, 4, 4)
    for i, t in enumerate(types):
        if t != M.PLURALIST:
            np.testing.assert_array_equal(P0[i, 0], P0[i, 1])
            np.testing.assert_array_equal(P0[i, 0], P0[i, 2])
    # neutral = uniform moment match
    neutral = X.monist_moment_matched(base3, (1 / 3,) * 3)
    np.testing.assert_array_equal(P0[2, 0], neutral.P0[0])
    out = M.mixed_shift_protocol(mc, [0, 1], [0.0, 3.0], [0.0, 0.3], steps=110)
    assert np.isfinite(out["mean_M"]).all()
    assert ((out["p_mass_pos"] >= 0) & (out["p_mass_pos"] <= 1)).all()
    # K=2 default neutral weights are exactly (.5, .5)
    mc2 = M.MixedConfig(BASE, (2,) * 24, 0.003, X.ExtConfig(BASE))
    np.testing.assert_array_equal(M.references(mc2)[0][0, 0], X.monist_moment_matched(BASE).P0[0])


def test_outcome_and_tracking_metrics():
    p_final = np.array([[0.9, 0.8, 0.2, 0.76]])         # (S=1, N=4)
    trace = np.concatenate([np.zeros((1, 80)), np.linspace(0.5, 0.9, 20)[None]], axis=1)
    m = Wd.outcome_metrics(p_final, trace, shift=80)
    assert m["frac_agents_reach"][0] == 0.75
    assert m["community_reach"][0] == 0.0               # mean 0.665
    np.testing.assert_allclose(m["brier"][0], np.mean((1 - p_final) ** 2))
    assert m["t_reach"][0] == np.argmax(np.linspace(0.5, 0.9, 20) >= 0.75)
    sig = experiments.signal_path(100, 80)
    tr = Wd.tracking_metrics(trace, sig, start=80)
    assert 0 < tr["track_right_side"][0] <= 1
