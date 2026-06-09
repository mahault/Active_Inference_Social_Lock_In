"""Tests for the motivated-update extension on the hierarchical-context substrate."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.pomdp.gen_model import PomdpConfig
from src.pomdp.simple_step import SimpleConfig
from src.pomdp.motivated_step import (
    MotivatedConfig,
    apply_value_tilt,
    apply_motivated_gate,
    run_motivated,
)


def _pomdp(**kw):
    base = dict(x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), true_paradigm=1,
                q_reliability=0.85)
    base.update(kw)
    return PomdpConfig(**base)


def _simple_cfg(**kw):
    pomdp_kw = kw.pop("pomdp_kw", {})
    base = dict(pomdp=_pomdp(**pomdp_kw), n_agents=20, n_steps=40, seed=7,
                obs_x_index=4)
    base.update(kw)
    return SimpleConfig(**base)


def _cfg(**kw):
    simple_kw = kw.pop("simple_kw", {})
    base = dict(simple=_simple_cfg(**simple_kw), lambda_tilt=1.0, motivated=True)
    base.update(kw)
    return MotivatedConfig(**base)


# ======================================================================
# Algebra
# ======================================================================

class TestTilt:

    def test_zero_lambda_is_identity(self):
        q = jnp.array([[0.25, 0.25, 0.25, 0.25]])      # (N=1, K*C=4)
        U = jnp.array([[1.0, -1.0]])                    # (N=1, K=2)
        out = apply_value_tilt(q, U, lambda_tilt=0.0, K=2, C=2)
        np.testing.assert_allclose(np.asarray(out), np.asarray(q), atol=1e-6)

    def test_tilt_preserves_normalization(self):
        q = jnp.array([[0.1, 0.2, 0.3, 0.4]])
        U = jnp.array([[2.0, -1.0]])
        for lam in [0.5, 1.0, 3.0]:
            out = apply_value_tilt(q, U, lam, K=2, C=2)
            assert abs(float(out.sum()) - 1.0) < 1e-6

    def test_positive_utility_shifts_marginal(self):
        """U[theta=1] > U[theta=0] should make q(theta=1) larger after tilt."""
        # q_joint indexing: [theta*C + c]. For K=2, C=2 → 4 entries.
        # Equal initial mass on theta=0 and theta=1 (each splits across 2 contexts):
        q = jnp.array([[0.25, 0.25, 0.25, 0.25]])
        U = jnp.array([[0.0, 1.0]])     # prefer paradigm 1
        out = apply_value_tilt(q, U, lambda_tilt=2.0, K=2, C=2)
        # Marginal over context: q(theta=0) = out[0]+out[1]; q(theta=1) = out[2]+out[3]
        m0 = float(out[0, 0] + out[0, 1])
        m1 = float(out[0, 2] + out[0, 3])
        assert m1 > m0
        assert abs((m0 + m1) - 1.0) < 1e-6

    def test_tilt_factorizes_over_context(self):
        """U only depends on theta, so within a theta block, context ratio is
        preserved by the tilt."""
        q = jnp.array([[0.1, 0.3, 0.2, 0.4]])    # theta=0: 0.1,0.3 | theta=1: 0.2,0.4
        U = jnp.array([[1.5, -0.5]])
        out = apply_value_tilt(q, U, lambda_tilt=1.0, K=2, C=2)
        # Original c-ratio within theta=0: 0.3/0.1 = 3.0; theta=1: 0.4/0.2 = 2.0
        r0_in = 0.3 / 0.1
        r1_in = 0.4 / 0.2
        r0_out = float(out[0, 1]) / float(out[0, 0])
        r1_out = float(out[0, 3]) / float(out[0, 2])
        assert abs(r0_out - r0_in) < 1e-6
        assert abs(r1_out - r1_in) < 1e-6


# ======================================================================
# Run-level behavior
# ======================================================================

class TestRun:

    def test_zero_utility_matches_simple(self):
        """U=0 + lambda_tilt=1 should produce the same trajectory as plain
        simple_step (motivated tilt is identity at zero U)."""
        from src.pomdp.simple_step import run_simple
        scfg = _simple_cfg(n_steps=30, seed=11)
        mcfg = MotivatedConfig(simple=scfg, lambda_tilt=1.0, motivated=True)
        out_plain = run_simple(scfg)
        out_mot = run_motivated(mcfg)  # U=0 by default
        np.testing.assert_allclose(
            out_plain["mean_qB"], out_mot["mean_qB"], atol=1e-5)

    def test_tilt_toward_truth_speeds_convergence(self):
        """A pro-truth utility tilt should produce stronger belief in the
        true paradigm relative to a no-tilt run on the same seeded substrate."""
        # Use a low-discriminability regime so the tilt's effect is visible
        scfg = _simple_cfg(n_steps=60, seed=11, obs_x_index=2,
                           eps_crisis=0.10, eps_resolve=0.30)
        N, K = scfg.n_agents, scfg.pomdp.n_paradigms
        # Half the agents are tilted toward truth (paradigm 1), half toward
        # wrong (paradigm 0). With true_paradigm=1, the pro-truth tilt should
        # produce higher final mean_qB than the anti-truth tilt.
        U_pro = np.tile([0.0, 1.0], (N, 1))
        U_anti = np.tile([1.0, 0.0], (N, 1))
        mcfg = MotivatedConfig(simple=scfg, lambda_tilt=2.0, motivated=True)
        out_pro = run_motivated(mcfg, U_per_agent=U_pro)
        out_anti = run_motivated(mcfg, U_per_agent=U_anti)
        assert out_pro["mean_qB"][-1] > out_anti["mean_qB"][-1]

    def test_motivated_false_matches_zero_lambda(self):
        scfg = _simple_cfg(n_steps=20, seed=3)
        N = scfg.n_agents
        U = np.tile([1.0, -1.0], (N, 1))
        cfg_off = MotivatedConfig(simple=scfg, lambda_tilt=2.0, motivated=False)
        cfg_zero = MotivatedConfig(simple=scfg, lambda_tilt=0.0, motivated=True)
        out_off = run_motivated(cfg_off, U_per_agent=U)
        out_zero = run_motivated(cfg_zero, U_per_agent=U)
        np.testing.assert_allclose(
            out_off["mean_qB"], out_zero["mean_qB"], atol=1e-6)


# ======================================================================
# Integration
# ======================================================================

class TestIntegration:

    def test_shapes(self):
        scfg = _simple_cfg(n_steps=20, n_agents=10)
        cfg = MotivatedConfig(simple=scfg, lambda_tilt=1.0, motivated=True)
        out = run_motivated(cfg)
        assert out["mean_qB"].shape == (20,)
        assert out["final_q"].shape == (10, 2)
        assert out["final_q_joint"].shape == (10, 4)
        assert out["final_U"].shape == (10, 2)

    def test_deterministic(self):
        scfg = _simple_cfg(n_steps=15, seed=99)
        cfg = MotivatedConfig(simple=scfg, lambda_tilt=1.0, motivated=True)
        N = scfg.n_agents
        U = np.tile([0.3, -0.3], (N, 1))
        a = run_motivated(cfg, U_per_agent=U)
        b = run_motivated(cfg, U_per_agent=U)
        np.testing.assert_array_equal(a["mean_qB"], b["mean_qB"])

    def test_resource_coupling_compatible(self):
        scfg = _simple_cfg(n_steps=20, n_agents=10, resource_coupling=True)
        cfg = MotivatedConfig(simple=scfg, lambda_tilt=1.0, motivated=True)
        out = run_motivated(cfg)
        assert out["mean_qB"].shape == (20,)
        # Should not crash; trajectory should be finite
        assert np.all(np.isfinite(out["mean_qB"]))


# ======================================================================
# Evidence-gate mode (cost-of-mind-change)
# ======================================================================

class TestEvidenceGate:

    def test_lambda_zero_is_identity(self):
        """At lambda=0 the gate returns the honest posterior unchanged."""
        q_prior = jnp.array([[0.25, 0.25, 0.25, 0.25]])
        q_post = jnp.array([[0.15, 0.15, 0.35, 0.35]])
        U = jnp.array([[2.0, 0.0]])
        out = apply_motivated_gate(q_prior, q_post, U, 0.0, K=2, C=2)
        np.testing.assert_allclose(np.asarray(out), np.asarray(q_post), atol=1e-5)

    def test_confirming_evidence_flows_freely(self):
        """Evidence that CONFIRMS a valued paradigm is not gated, any lambda."""
        q_prior = jnp.array([[0.25, 0.25, 0.25, 0.25]])
        # honest move increases belief in theta=0 (the valued one)
        q_post = jnp.array([[0.35, 0.35, 0.15, 0.15]])
        U = jnp.array([[2.0, 0.0]])
        out = apply_motivated_gate(q_prior, q_post, U, 3.0, K=2, C=2)
        np.testing.assert_allclose(np.asarray(out), np.asarray(q_post), atol=1e-5)

    def test_disconfirming_evidence_is_resisted(self):
        """Disconfirming a valued paradigm is partially resisted: the gated
        belief sits BETWEEN the honest posterior and the prior, monotonically
        in lambda."""
        q_prior = jnp.array([[0.25, 0.25, 0.25, 0.25]])   # theta-marginal [0.5,0.5]
        q_post = jnp.array([[0.15, 0.15, 0.35, 0.35]])    # honest [0.3,0.7], drops theta0
        U = jnp.array([[2.0, 0.0]])                        # values theta0

        def theta0(q):
            return float(q.reshape(1, 2, 2).sum(2)[0, 0])

        prev = theta0(q_post)   # 0.3, the honest (unresisted) value
        for lam in [0.5, 1.0, 3.0]:
            g = apply_motivated_gate(q_prior, q_post, U, lam, K=2, C=2)
            v = theta0(g)
            # resisted upward toward the prior (0.5), but not past it
            assert prev <= v <= 0.5 + 1e-6, (lam, v)
            prev = v

    def test_gate_preserves_context_conditional(self):
        """The gate only reweights the paradigm marginal; q(c|theta) is kept."""
        q_prior = jnp.array([[0.25, 0.25, 0.25, 0.25]])
        # within theta=1 block, context split 0.2/0.8
        q_post = jnp.array([[0.15, 0.15, 0.10, 0.60]])
        U = jnp.array([[2.0, 0.0]])
        out = np.asarray(apply_motivated_gate(q_prior, q_post, U, 2.0, K=2, C=2))[0]
        # context conditional within theta=1: post was 0.10:0.60 -> 1:6
        ratio_post = 0.10 / 0.60
        ratio_out = out[2] / out[3]
        np.testing.assert_allclose(ratio_out, ratio_post, rtol=1e-4)

    def test_run_with_gate_mode(self):
        """End-to-end run in evidence_gate mode produces finite trajectory."""
        scfg = _simple_cfg(n_steps=20, n_agents=10)
        cfg = MotivatedConfig(simple=scfg, lambda_tilt=2.0, motivated=True,
                              update_mode="evidence_gate")
        N = scfg.n_agents
        U = np.tile([1.0, 0.0], (N, 1))   # all value the wrong paradigm
        out = run_motivated(cfg, U_per_agent=U)
        assert out["mean_qB"].shape == (20,)
        assert np.all(np.isfinite(out["mean_qB"]))
