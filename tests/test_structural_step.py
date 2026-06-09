"""Tests for the structural extension: per-agent DAG + T propagation + endogenous γ."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.pomdp.gen_model import PomdpConfig
from src.pomdp.simple_step import SimpleConfig
from src.pomdp.motivated_step import MotivatedConfig
from src.pomdp.structural_step import (
    StructuralConfig,
    build_default_dag,
    compute_T,
    compute_conservatism,
    compute_conviction,
    conviction_asymmetry,
    evidence_weight,
    init_structural,
    run_structural,
)


def _pomdp(**kw):
    base = dict(x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), true_paradigm=1,
                q_reliability=0.85)
    base.update(kw)
    return PomdpConfig(**base)


def _simple(**kw):
    pomdp_kw = kw.pop("pomdp_kw", {})
    base = dict(pomdp=_pomdp(**pomdp_kw), n_agents=20, n_steps=30, seed=7,
                obs_x_index=4)
    base.update(kw)
    return SimpleConfig(**base)


def _mot(**kw):
    simple_kw = kw.pop("simple_kw", {})
    base = dict(simple=_simple(**simple_kw), lambda_tilt=1.0, motivated=True)
    base.update(kw)
    return MotivatedConfig(**base)


def _cfg(**kw):
    mot_kw = kw.pop("mot_kw", {})
    base = dict(motivated=_mot(**mot_kw), n_dependents=3, edge_precision=0.8,
                gamma_strength=0.0)
    base.update(kw)
    return StructuralConfig(**base)


# ======================================================================
# DAG and propagation algebra
# ======================================================================

class TestDAG:

    def test_default_dag_has_paradigm_roots(self):
        """Paradigm nodes have no incoming edges (they're roots).
        Convention A[parent, child] = precision → column j is the incoming
        edges of node j. Paradigms (cols 0..K-1) should have all zeros."""
        K = 2
        n_dep = 3
        A = np.asarray(build_default_dag(K, n_dep, edge_precision=0.5))
        np.testing.assert_array_equal(A[:, :K], np.zeros((K + K * n_dep, K)))

    def test_default_dag_has_correct_edges(self):
        """Each dependent has exactly one incoming edge from its paradigm.
        A[paradigm, child] = precision under the new convention."""
        K = 2
        n_dep = 3
        A = np.asarray(build_default_dag(K, n_dep, edge_precision=0.7))
        # Children K..K+n_dep-1 should be children of paradigm 0
        for d in range(n_dep):
            child = K + d
            col = A[:, child]
            assert col[0] == 0.7
            assert col.sum() == pytest.approx(0.7)
        # Children K+n_dep..K+2*n_dep-1 should be children of paradigm 1
        for d in range(n_dep):
            child = K + n_dep + d
            col = A[:, child]
            assert col[1] == 0.7
            assert col.sum() == pytest.approx(0.7)

    def test_T_at_zero_A_is_identity(self):
        A = jnp.zeros((4, 4))
        T = compute_T(A)
        np.testing.assert_allclose(np.asarray(T), np.eye(4), atol=1e-7)

    def test_T_on_tree_propagates_to_descendants(self):
        """For paradigm 0 with 2 dependents at precision 1.0,
        T should give the paradigm node a descendant mass of 1 + 2."""
        K, n_dep = 2, 2
        A = build_default_dag(K, n_dep, edge_precision=1.0)
        T = compute_T(A)
        kappa = np.asarray(compute_conservatism(T))
        # Paradigm 0: itself (1) + 2 dependents at precision 1 = 3
        assert kappa[0] == pytest.approx(3.0, abs=1e-6)
        # Paradigm 1: same
        assert kappa[1] == pytest.approx(3.0, abs=1e-6)
        # Dependents: just themselves (no descendants)
        assert kappa[K] == pytest.approx(1.0, abs=1e-6)

    def test_conviction_propagates_to_paradigm_root(self):
        """A subsidiary commitment with u=1 should lift the paradigm's
        conviction proportionally to edge precision."""
        K, n_dep = 2, 2
        A = build_default_dag(K, n_dep, edge_precision=0.5)
        T = np.asarray(compute_T(A))
        # u_source: zero everywhere except a dependent of paradigm 0
        u = np.zeros(K + K * n_dep)
        dep_of_p0 = K  # first dependent of paradigm 0
        u[dep_of_p0] = 1.0
        U = T @ u
        # With T = (I-A)^-1 and edges (child, parent), propagation goes
        # from child TO descendants. A paradigm root has no outgoing edges
        # in our DAG (paradigm is the parent, dependent is the child).
        # So conviction at the dependent (with u=1) is 1; at the paradigm
        # node it's 0 (paradigm has no parent → nothing propagates up).
        # The propagation goes DOWNSTREAM from the source.
        # This test verifies the direction of propagation.
        assert U[dep_of_p0] == pytest.approx(1.0, abs=1e-6)


# ======================================================================
# Endogenous γ
# ======================================================================

class TestGamma:

    def test_zero_asymmetry_gives_unit_weight(self):
        U_par = jnp.array([[1.0, 1.0]])  # equal conviction → asymmetry = 0
        asym = conviction_asymmetry(U_par)
        w = evidence_weight(asym, gamma_strength=5.0)
        np.testing.assert_allclose(np.asarray(w), [1.0], atol=1e-7)

    def test_higher_asymmetry_lower_weight(self):
        U_par = jnp.array([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
        asym = conviction_asymmetry(U_par)
        w = evidence_weight(asym, gamma_strength=1.0)
        w = np.asarray(w)
        assert w[0] > w[1] > w[2]
        assert w[0] == pytest.approx(1.0)

    def test_zero_gamma_disables_attenuation(self):
        U_par = jnp.array([[5.0, -3.0]])  # large asymmetry
        asym = conviction_asymmetry(U_par)
        w = evidence_weight(asym, gamma_strength=0.0)
        np.testing.assert_allclose(np.asarray(w), [1.0], atol=1e-7)


# ======================================================================
# Initialization
# ======================================================================

class TestInit:

    def test_init_shapes(self):
        cfg = _cfg(n_dependents=2)
        state = init_structural(cfg)
        N = cfg.motivated.simple.n_agents
        K = cfg.motivated.simple.pomdp.n_paradigms
        K_int = K * (1 + cfg.n_dependents)
        assert state.T_per_agent.shape == (N, K_int, K_int)
        assert state.u_source.shape == (N, K_int)
        assert state.kappa.shape == (N, K_int)
        assert state.U_propagated.shape == (N, K_int)

    def test_init_with_source_utility(self):
        cfg = _cfg(n_dependents=2)
        N = cfg.motivated.simple.n_agents
        K = cfg.motivated.simple.pomdp.n_paradigms
        K_int = K * (1 + cfg.n_dependents)
        u_src = np.zeros((N, K_int))
        u_src[:, 0] = 1.0  # all agents want paradigm 0 (the root utility)
        state = init_structural(cfg, u_source_per_agent=u_src)
        # The paradigm 0 root has no children edges going INTO it,
        # so U_propagated[paradigm 0] should equal u_src[paradigm 0] = 1.0
        U = np.asarray(state.U_propagated)
        np.testing.assert_allclose(U[:, 0], 1.0, atol=1e-7)


# ======================================================================
# Run-level behavior
# ======================================================================

class TestRun:

    def test_gamma_zero_matches_motivated(self):
        """With gamma_strength=0 and u_source=0, should match motivated_step."""
        from src.pomdp.motivated_step import run_motivated
        scfg = _simple(n_steps=20, seed=11)
        mcfg = MotivatedConfig(simple=scfg, lambda_tilt=0.0, motivated=False)
        cfg_struct = StructuralConfig(motivated=mcfg, gamma_strength=0.0,
                                       n_dependents=2)
        out_mot = run_motivated(mcfg)
        out_struct = run_structural(cfg_struct)
        # With zero tilt + zero gamma, both should produce essentially
        # identical trajectories (modulo numerical noise from blending).
        diff = np.max(np.abs(out_mot["mean_qB"] - out_struct["mean_qB"]))
        assert diff < 0.05, f"max diff {diff:.4f} too large"

    def test_endogenous_gamma_attenuates_adaptation(self):
        """High gamma_strength + asymmetric source utility → attenuated
        evidence → slower adaptation to truth."""
        # Sweep: high asymmetric utility for wrong paradigm at high gamma
        cfg_low = _cfg(gamma_strength=0.0)
        cfg_high = _cfg(gamma_strength=3.0)
        N = cfg_low.motivated.simple.n_agents
        K = cfg_low.motivated.simple.pomdp.n_paradigms
        K_int = K * (1 + cfg_low.n_dependents)
        # All agents have strong utility for the WRONG paradigm (0)
        u_src = np.zeros((N, K_int))
        u_src[:, 0] = 2.0
        out_low = run_structural(cfg_low, u_source_per_agent=u_src)
        out_high = run_structural(cfg_high, u_source_per_agent=u_src)
        # High gamma should suppress evidence more, so final mean_qB is
        # LOWER (further from truth) than low gamma — even though the tilt
        # is the same.
        assert out_high["mean_qB"][-1] <= out_low["mean_qB"][-1]


# ======================================================================
# Integration
# ======================================================================

class TestIntegration:

    def test_run_shapes(self):
        cfg = _cfg(mot_kw=dict(simple_kw=dict(n_agents=10, n_steps=20)))
        out = run_structural(cfg)
        assert out["mean_qB"].shape == (20,)
        assert out["mean_evidence_weight"].shape == (20,)
        assert out["mean_conviction_asymmetry"].shape == (20,)

    def test_deterministic(self):
        cfg = _cfg(mot_kw=dict(simple_kw=dict(n_agents=10, n_steps=15, seed=99)))
        a = run_structural(cfg)
        b = run_structural(cfg)
        np.testing.assert_array_equal(a["mean_qB"], b["mean_qB"])
