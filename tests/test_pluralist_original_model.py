# Ported from David Hyland's epistemic-pluralism tests/test_model.py (commit 6772b24).
# Only imports and the reference-fixture path changed.
"""Independent checks of the equations and random-stream conventions.

The reference calculation below conditions Gaussians in covariance form, using
NumPy rather than the production information-form update. Small numerical
fixtures check probabilities on both sides of the signal change.
"""
from pathlib import Path
from dataclasses import replace
import json
import sys
import unittest

import numpy as np


from tests.pluralist_x64 import import_pluralist, restore, x64_only_here  # noqa: F401

model, experiments, _ = import_pluralist()
restore()


def covariance_update(P0, h0, P, h, support, W, observation, attention,
                      sigma, retention, share_beliefs, share_support):
    """NumPy/Kalman reference: communicate, score, condition, then anchor."""
    n, candidates, dim = h.shape
    H = np.eye(dim)[-3:]
    if share_beliefs:
        prior_P = np.stack([sum(W[i, j] * P[j] for j in range(n))
                            for i in range(n)])
        prior_h = np.stack([sum(W[i, j] * h[j] for j in range(n))
                            for i in range(n)])
    else:
        prior_P, prior_h = P.copy(), h.copy()
    out_P, out_h, scores = np.empty_like(P), np.empty_like(h), np.empty_like(support)
    for agent in range(n):
        Hw = np.sqrt(attention[agent])[:, None] * H
        measured = np.sqrt(attention[agent]) * observation[agent]
        for candidate in range(candidates):
            cov = np.linalg.inv(prior_P[agent, candidate])
            mean = cov @ prior_h[agent, candidate]
            predictive_cov = Hw @ cov @ Hw.T + sigma**2 * np.eye(3)
            residual = measured - Hw @ mean
            scores[agent, candidate] = -0.5 * (
                3 * np.log(2 * np.pi) + np.linalg.slogdet(predictive_cov)[1]
                + residual @ np.linalg.solve(predictive_cov, residual)
            )
            gain = np.linalg.solve(predictive_cov, Hw @ cov).T
            learned_mean = mean + gain @ residual
            learned_cov = cov - gain @ Hw @ cov
            learned_P = np.linalg.inv(learned_cov)
            learned_h = learned_P @ learned_mean
            # A geometric density pool interpolates natural parameters.
            out_P[agent, candidate] = ((1 - retention) * P0[candidate]
                                        + retention * learned_P)
            out_h[agent, candidate] = ((1 - retention) * h0[candidate]
                                        + retention * learned_h)
    out_support = ((1 - share_support) * support
                   + share_support * W @ support + scores)
    return out_P, out_h, out_support


class ModelEquations(unittest.TestCase):
    def test_candidate_covariance_and_marginal_gaussian_reference(self):
        P, h, H = model.candidate_priors()
        # Derived directly from the fixed structural equations in Appendix B.1.
        expected_cov = np.array([
            [[.5, .4, -.32], [.4, .82, -.256], [-.32, -.256, .8648]],
            [[.5, .32, .32], [.32, .8648, .3648], [.32, .3648, .8648]],
        ])
        expected_mean = np.array([[0., 1., 1., -1.], [0., 1., 1., 1.]])
        np.testing.assert_allclose(np.linalg.inv(P)[:, 1:, 1:], expected_cov,
                                   atol=2e-14, rtol=0)
        np.testing.assert_allclose(np.linalg.solve(P, np.asarray(h)[..., None])[..., 0],
                                   expected_mean, atol=2e-14, rtol=0)
        np.testing.assert_array_equal(H, np.eye(4)[1:])
        pooled_P, pooled_h = model.gaussian_reference(P, h)
        marginal_P = np.linalg.inv(expected_cov)
        expected_P = marginal_P.mean(0)
        expected_h = np.mean([marginal_P[k] @ expected_mean[k, 1:]
                              for k in range(2)], axis=0)
        np.testing.assert_allclose(pooled_P[0, 1:, 1:], expected_P, atol=2e-14, rtol=0)
        np.testing.assert_allclose(pooled_h[0, 1:], expected_h, atol=2e-14, rtol=0)
        # Pooling joint precisions before marginalizing gives a different model.
        self.assertGreater(np.max(np.abs(expected_P - np.asarray(P).mean(0)[1:, 1:])), .1)
        np.testing.assert_array_equal(pooled_P[0, 0, 1:], 0.)
        self.assertEqual(float(pooled_h[0, 0]), 0.)

    def test_update_and_prelearning_score_against_covariance_form(self):
        base = model.Config(*model.candidate_priors(), sigma=.7)
        P, h, support, key = (np.asarray(x[0]).copy()
                              for x in model.initial_state(base, 4, [137]))
        # Heterogeneous entering beliefs make communication observable in one step.
        P += np.arange(4)[:, None, None, None] * .07 * np.eye(4)
        h += np.arange(h.size).reshape(h.shape) * .012
        support += np.array([[1., -2.], [-.5, 2.], [3., .1], [.3, -1.]])
        W = experiments.social_matrix(4, .15)
        attention = np.array([[.2, 0., 1.], [0., 1., 0.],
                              [1., .01, .3], [0., 0., 0.]])
        signal = np.array([0., 1., 1., 1.])
        next_key, observation_key = model.jax.random.split(key)
        agent_keys = model.jax.random.split(observation_key, 4)
        # Follow the documented RNG contract without calling production sampling.
        observation = np.stack([
            signal[1:] + base.sigma * np.asarray(model.jax.random.normal(k, (3,)))
            for k in agent_keys
        ])
        for retention, fuse, alpha in ((0., True, .02), (.3, True, .02), (1., False, 0.)):
            with self.subTest(retention=retention, fuse=fuse, alpha=alpha):
                cfg = replace(base, omega=retention, fuse=fuse, alpha=alpha)
                expected = covariance_update(
                    np.asarray(cfg.P0), np.asarray(cfg.h0), P, h, support,
                    W, observation, attention, cfg.sigma, retention, fuse, alpha)
                actual = model.step(tuple(map(model.jnp.asarray, (P, h, support, key))),
                                    signal, cfg, model.jnp.asarray(W), model.jnp.asarray(attention))
                for observed, reference in zip(actual[:3], expected):
                    np.testing.assert_allclose(observed, reference, atol=2e-12, rtol=0)
                np.testing.assert_array_equal(actual[3], next_key)
                if retention == 0:
                    np.testing.assert_array_equal(actual[0], np.broadcast_to(cfg.P0, P.shape))
                    np.testing.assert_allclose(actual[1], np.broadcast_to(cfg.h0, h.shape),
                                               atol=2e-14, rtol=0)

    def test_unobserved_channels_cannot_discriminate_candidates(self):
        P, h, H = model.candidate_priors()
        attention = model.jnp.array([0., 1., 0.])
        for candidate in range(2):
            a = model.predictive_score(P[candidate], h[candidate], H,
                                       model.jnp.array([1., 1., 1.]), .5, attention)
            b = model.predictive_score(P[candidate], h[candidate], H,
                                       model.jnp.array([1e8, 1., -1e8]), .5, attention)
            self.assertEqual(float(a), float(b))
        scores = [model.predictive_score(P[k], h[k], H, model.jnp.ones(3),
                                         .5, model.jnp.zeros(3)) for k in range(2)]
        self.assertEqual(float(scores[0]), float(scores[1]))

    def test_checkpoint_continuation_and_paired_observation_stream(self):
        cfg = model.Config(*model.candidate_priors())
        W, attention = experiments.social_matrix(4), np.ones((4, 3))
        signals, seeds = experiments.signal_path(7, 3), [17, 19]
        whole_state, whole = model.rollout(cfg, W, attention, signals, seeds)
        first_state, first = model.rollout(cfg, W, attention, signals[:3], seeds)
        end_state, second = model.rollout(cfg, W, attention, signals[3:], seeds,
                                         initial=first_state)
        for name in whole:
            np.testing.assert_allclose(whole[name], np.concatenate((first[name], second[name]), axis=1),
                                       atol=2e-13, rtol=0)
        for observed, reference in zip(end_state, whole_state):
            np.testing.assert_allclose(observed, reference, atol=2e-13, rtol=0)
        private_state, _ = model.rollout(replace(cfg, alpha=0.), W, attention, signals, seeds)
        # Support exchange cannot modify conditional learning or consume RNG.
        for index in (0, 1, 3):
            np.testing.assert_array_equal(private_state[index], whole_state[index])
        self.assertGreater(np.max(np.abs(np.asarray(private_state[2]) - whole_state[2])), 1e-5)

    def test_network_branches_share_prehistory_and_phase_reset(self):
        fixture = json.loads(Path(__file__).with_name("pluralist_reference.json").read_text())
        cfg = model.Config(*model.candidate_priors())
        observed = experiments.network(cfg, [fixture["seed"]], fixture["steps"])
        for condition, expected in fixture["conditions"].items():
            for group in ("A", "B"):
                series = observed[condition + "__q_" + group]
                # State 80 is pre-change. States 81..84 use seed+1000 draws.
                np.testing.assert_allclose(series[0, 79:84], expected["q_" + group],
                                           atol=2e-11, rtol=1e-10)
                np.testing.assert_array_equal(series[:, :80], observed["N11__q_" + group][:, :80])


if __name__ == "__main__":
    unittest.main()
