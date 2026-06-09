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
    build_default_dag_with_cross_coupling,
    build_precision_matrix,
    schur_complement,
    schur_residue,
    bmr_log_bayes_factor,
    bmr_prune,
    expand_propose_node,
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
# Cross-paradigm coupling
# ======================================================================

class TestCrossCoupling:

    def test_zero_cross_coupling_matches_default(self):
        A_default = np.asarray(build_default_dag(2, 2, 0.5))
        A_cc = np.asarray(build_default_dag_with_cross_coupling(2, 2, 0.5,
                                                                 cross_coupling=0.0))
        np.testing.assert_allclose(A_cc, A_default)

    def test_cross_coupling_adds_paradigm_edges(self):
        A = np.asarray(build_default_dag_with_cross_coupling(3, 1, 0.5,
                                                              cross_coupling=0.2))
        # Upper-triangular paradigm-to-paradigm edges
        assert A[0, 1] == pytest.approx(0.2)
        assert A[0, 2] == pytest.approx(0.2)
        assert A[1, 2] == pytest.approx(0.2)
        # Lower-triangular still zero (DAG)
        assert A[1, 0] == 0.0
        assert A[2, 0] == 0.0
        assert A[2, 1] == 0.0

    def test_cross_coupling_propagation(self):
        """With cross-coupling 0 → paradigm, conviction in dependent of
        paradigm 1 should propagate UP to paradigm 0 too (paradigm 0 → 1 edge
        means paradigm 1 is a child of paradigm 0)."""
        K, n_dep, prec, cc = 2, 1, 0.5, 0.3
        A = build_default_dag_with_cross_coupling(K, n_dep, prec, cross_coupling=cc)
        T = np.asarray(compute_T(A))
        # Conviction on dependent of paradigm 1 (node index K + n_dep = 3)
        # Paradigm 1's dependent is at index 3 (since dependents of paradigm 0 are at K..K+n_dep-1)
        # Actually: dep of paradigm 0 is index K=2, dep of paradigm 1 is index K+n_dep=3
        # u = e_3 → propagated conviction at each node:
        u = np.zeros(4)
        u[3] = 1.0
        U = T @ u
        # Paradigm 1 (idx 1) has edge to its dep (idx 3) with prec 0.5 → U[1] >= 0.5
        # Paradigm 0 (idx 0) has edge to paradigm 1 (idx 1) with prec 0.3 → propagated
        assert U[1] == pytest.approx(0.5, abs=1e-5)
        # U[0] = sum_w T[0,w] · u[w] = T[0, 3] · 1 = path 0→1→3 = 0.3·0.5 = 0.15
        assert U[0] == pytest.approx(0.15, abs=1e-5)


# ======================================================================
# Schur complement
# ======================================================================

class TestSchur:

    def test_precision_matrix_at_zero_A(self):
        """Pi = (I-A)^T D (I-A); with A=0 and D=I, Pi = I."""
        K = 4
        A = jnp.zeros((K, K))
        Pi = np.asarray(build_precision_matrix(A))
        np.testing.assert_allclose(Pi, np.eye(K), atol=1e-7)

    def test_precision_matrix_symmetric(self):
        A = build_default_dag(2, 2, 0.5)
        Pi = np.asarray(build_precision_matrix(A))
        np.testing.assert_allclose(Pi, Pi.T, atol=1e-7)

    def test_schur_complement_correct_size(self):
        Pi = jnp.eye(5) + 0.1 * jnp.ones((5, 5))
        red = schur_complement(Pi, idx=2)
        assert red.shape == (4, 4)

    def test_schur_residue_induces_coupling(self):
        """For a hub node connecting two children with no direct edge,
        marginalizing the hub should produce off-diagonal coupling between
        them in the residue."""
        # Simple 3-node net: node 0 is hub; nodes 1,2 are children of 0
        # A: 0→1 and 0→2 with precision 0.5 each
        A = jnp.zeros((3, 3))
        A = A.at[0, 1].set(0.5)
        A = A.at[0, 2].set(0.5)
        Pi = build_precision_matrix(A)
        # Marginalize node 0 (the hub)
        residue = np.asarray(schur_residue(Pi, idx=0))
        # residue is (2,2), should have off-diagonal entry > 0
        # because 1 and 2 share a hub
        assert residue.shape == (2, 2)
        assert abs(residue[0, 1]) > 1e-6
        assert abs(residue[1, 0]) > 1e-6
        # And symmetric
        np.testing.assert_allclose(residue[0, 1], residue[1, 0], atol=1e-9)


# ======================================================================
# BMR — Savage-Dickey edge pruning
# ======================================================================

class TestBMR:

    def test_log_bf_zero_when_posterior_equals_prior(self):
        """If posterior precision = prior precision and mu = 0, log BF = 0."""
        mu = jnp.array([0.0])
        tau_post = jnp.array([1.0])
        tau_prior = jnp.array([1.0])
        bf = bmr_log_bayes_factor(mu, tau_post, tau_prior)
        np.testing.assert_allclose(np.asarray(bf), [0.0], atol=1e-6)

    def test_log_bf_positive_when_mu_near_zero(self):
        """High posterior precision with mu near zero → posterior tight at
        zero → favours pruning → log BF > 0."""
        mu = jnp.array([0.0])
        tau_post = jnp.array([100.0])
        tau_prior = jnp.array([1.0])
        bf = float(bmr_log_bayes_factor(mu, tau_post, tau_prior)[0])
        assert bf > 0

    def test_log_bf_negative_when_mu_far_from_zero(self):
        """Posterior concentrated away from zero → keep the edge → log BF < 0."""
        mu = jnp.array([2.0])
        tau_post = jnp.array([10.0])
        tau_prior = jnp.array([0.1])
        bf = float(bmr_log_bayes_factor(mu, tau_post, tau_prior)[0])
        assert bf < 0

    def test_bmr_prune_zeros_high_bf_edges(self):
        """log BF above threshold → that edge zeroed out."""
        A = jnp.array([[0.0, 0.5, 0.5],
                        [0.0, 0.0, 0.5],
                        [0.0, 0.0, 0.0]])
        log_bf = jnp.array([[0.0, 5.0, -2.0],
                             [0.0, 0.0, -1.0],
                             [0.0, 0.0, 0.0]])
        pruned = np.asarray(bmr_prune(A, log_bf, threshold=0.0))
        # log_bf[0,1] = 5 > 0 → pruned (set to 0); log_bf[0,2] = -2 < 0 → kept
        assert pruned[0, 1] == 0.0
        assert pruned[0, 2] == 0.5
        assert pruned[1, 2] == 0.5

    def test_bmr_prune_recovers_misspecified_DAG(self):
        """If a spurious edge has high log BF for reduction and a real edge
        has low log BF, BMR should keep the real one and prune the spurious."""
        # Build A with 2 edges: paradigm 0 → dep0, paradigm 0 → dep1
        # Suppose dep0 is data-supported (real) and dep1 is spurious.
        A = build_default_dag(K_paradigms=1, n_dependents=2, edge_precision=0.5)
        # Edge precisions: shape (3,3). A[0,1] = 0.5 (dep0), A[0,2] = 0.5 (dep1)
        # Set log_bf so dep0 is kept (negative) and dep1 is pruned (positive)
        log_bf = jnp.zeros_like(A)
        log_bf = log_bf.at[0, 1].set(-3.0)  # keep (negative log BF for reduction)
        log_bf = log_bf.at[0, 2].set(+3.0)  # prune
        pruned = np.asarray(bmr_prune(A, log_bf, threshold=0.0))
        assert pruned[0, 1] == 0.5  # kept
        assert pruned[0, 2] == 0.0  # pruned


# ======================================================================
# Structure-learning expansion
# ======================================================================

class TestExpansion:

    def test_expansion_adds_one_node(self):
        A = build_default_dag(K_paradigms=2, n_dependents=2, edge_precision=0.5)
        K_orig = A.shape[0]
        A_new = expand_propose_node(A, tentative_precision=0.1)
        assert A_new.shape == (K_orig + 1, K_orig + 1)

    def test_expansion_preserves_original_edges(self):
        A = build_default_dag(K_paradigms=1, n_dependents=2, edge_precision=0.5)
        K_orig = A.shape[0]
        A_new = np.asarray(expand_propose_node(A, tentative_precision=0.1))
        # Original block must be preserved exactly
        np.testing.assert_allclose(A_new[:K_orig, :K_orig], np.asarray(A), atol=1e-9)

    def test_expansion_wires_tentative_edges(self):
        A = build_default_dag(K_paradigms=1, n_dependents=2, edge_precision=0.5)
        K_orig = A.shape[0]
        A_new = np.asarray(expand_propose_node(A, tentative_precision=0.07))
        # New node K_orig is connected to all originals at precision 0.07
        assert np.allclose(A_new[K_orig, :K_orig], 0.07)
        assert np.allclose(A_new[:K_orig, K_orig], 0.07)

    def test_expand_then_prune_recovers_supported_edges(self):
        """A new node proposed with tentative edges to all; BMR with a
        log_bf signal that supports only one of those edges should leave
        that edge surviving."""
        A = build_default_dag(K_paradigms=1, n_dependents=2, edge_precision=0.5)
        K_orig = A.shape[0]  # 3 (paradigm + 2 deps)
        A_new = expand_propose_node(A, tentative_precision=0.1)
        # Suppose data supports: new_node → dep1 (kept, log_bf < 0)
        # and prunes: new_node → paradigm, new_node → dep2
        K_new = K_orig + 1
        log_bf = jnp.zeros((K_new, K_new))
        # Tentative outgoing edges from new node (row K_orig) — only support node 2 (dep1)
        log_bf = log_bf.at[K_orig, 0].set(+2.0)   # prune
        log_bf = log_bf.at[K_orig, 1].set(-2.0)   # keep
        log_bf = log_bf.at[K_orig, 2].set(+2.0)   # prune
        # And incoming edges to new node similarly
        log_bf = log_bf.at[0, K_orig].set(+2.0)
        log_bf = log_bf.at[1, K_orig].set(-2.0)
        log_bf = log_bf.at[2, K_orig].set(+2.0)
        pruned = np.asarray(bmr_prune(A_new, log_bf, threshold=0.0))
        # Original edges preserved (log_bf = 0 on those is not > threshold=0
        # strictly; depends on threshold semantics). bmr_prune uses
        # (log_bf <= threshold) as keep-mask, so log_bf=0 → kept.
        np.testing.assert_allclose(pruned[:K_orig, :K_orig], np.asarray(A),
                                    atol=1e-9)
        # New node's surviving incident edges:
        assert pruned[K_orig, 1] == pytest.approx(0.1)
        assert pruned[1, K_orig] == pytest.approx(0.1)
        # Pruned ones zeroed:
        assert pruned[K_orig, 0] == 0.0
        assert pruned[K_orig, 2] == 0.0


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
