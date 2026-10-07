"""Observations are generated at the experiment each agent ran, and trust
scores neighbours at that same experiment."""

import jax
import jax.numpy as jnp
import numpy as np

from src.pomdp.simple_step import (SimpleConfig, categorical_surprisal,
                                   sample_observations)


def _two_experiment_world(n_o=2):
    # experiment 0 is uninformative, experiment 1 is fully informative
    A = np.zeros((n_o, 2, 2))
    A[:, :, 0] = 0.5
    A[:, 0, 1] = [1.0, 0.0]
    A[:, 1, 1] = [0.0, 1.0]
    return jnp.asarray(A)


def test_observations_follow_chosen_experiment():
    cfg = SimpleConfig()
    A = _two_experiment_world()
    A_full = jnp.zeros((cfg.pomdp.n_o, 2, 2)).at[:2].set(A)
    N = 400
    actions = jnp.ones((N,), dtype=jnp.int32)          # everyone runs experiment 1
    o, _ = sample_observations(A_full, cfg, 0, jax.random.PRNGKey(0), N, actions)
    o_idx = np.asarray(o.argmax(axis=1))
    true_k = cfg.pomdp.true_paradigm
    # experiment 1 is deterministic: every agent sees outcome == true paradigm
    assert np.all(o_idx == true_k)


def test_default_actions_keep_old_behaviour():
    cfg = SimpleConfig()
    A = jnp.ones((cfg.pomdp.n_o, 2, 3)) / cfg.pomdp.n_o
    o1, _ = sample_observations(A, cfg, 0, jax.random.PRNGKey(1), 8)
    o2, _ = sample_observations(A, cfg, 0, jax.random.PRNGKey(1), 8,
                                jnp.full((8,), cfg.obs_x_index, dtype=jnp.int32))
    assert np.array_equal(np.asarray(o1), np.asarray(o2))


def test_surprisal_scored_at_observer_experiment():
    A = _two_experiment_world()
    q = jnp.array([[1.0, 0.0], [0.0, 1.0]])            # agent 0 says theta0, agent 1 theta1
    o = jnp.array([[1.0, 0.0], [1.0, 0.0]])            # both observed outcome 0
    adj = jnp.ones((2, 2))
    actions = jnp.array([1, 0], dtype=jnp.int32)       # agent 0 ran the informative one
    eps = np.asarray(categorical_surprisal(q, A, o, actions, adj))
    # row 0 (observer ran exp 1): neighbour 1 predicted the wrong outcome -> large
    assert eps[0, 1] > 10 * max(eps[0, 0], 1e-6)
    # row 1 (observer ran exp 0, uninformative): both neighbours equally surprised
    assert np.isclose(eps[1, 0], eps[1, 1])
    # scalar action still works and matches the per-agent form
    eps_s = np.asarray(categorical_surprisal(q, A, o, 1, adj))
    eps_v = np.asarray(categorical_surprisal(q, A, o, jnp.array([1, 1]), adj))
    assert np.allclose(eps_s, eps_v)
