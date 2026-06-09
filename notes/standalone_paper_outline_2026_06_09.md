# Standalone paper outline — Mahault's solo or first-author paper

Provisional working title:

> **Principled Theory-Ladenness and Motivated Belief Revision in Multi-Agent
> Active Inference: A Hierarchical-Context Model of Paradigm Lock-In**

Target venue: **AAAI 2027** (Jul deadline), with **AAMAS** (Oct) as a fallback.
David explicitly suggested both venues in the May call as natural homes for
this line of work.

This outline is a self-contained paper that uses Mahault's substrate
(`src/pomdp/simple_step.py`, `cont_step.py`, `motivated_step.py`) and her
results (E1-E9). It does NOT depend on inclusion in the IWAI 2026 paper —
even if Jonas declines, this is publishable.

---

## What the paper claims

A principled active-inference model of paradigm dynamics in scientific
communities, with three contributions:

1. **Principled hierarchical context.** Theory-ladenness is formalized as
   a latent context variable c ∈ {0, …, C−1} where the agent's
   observation likelihood is a graded average of paradigm-specific
   likelihoods. The joint state (theta, c) is inferred by standard
   categorical Bayes; the context posterior carries forward through a
   tridiagonal transition B_c. No heuristic precision tweaks, no
   hand-tuned weights — just a multi-level extension of the binary
   context model that turns the "lock-in via theory-ladenness" intuition
   into proper AIF.

2. **Multi-agent lift of Hyland & Albarracin 2025.** The variational
   cost of mind-change Eq. 13 is extended from a single agent to a
   community via per-agent intrinsic utility U_i(theta) and a value-
   tilted posterior q_lambda(theta, c) ∝ q_post(theta, c) ·
   exp(lambda · U_i(theta)). The tilt acts on the marginal over
   paradigm and preserves the context posterior, so motivated reasoning
   and epistemic inference are kept formally separate.

3. **Phase diagram of lock-in.** Sweeping social coupling, paradigm
   inertia, lambda, and the fraction of motivated agents, the model
   produces a quantitative regime map of when the community adapts,
   when it locks in, and when adaptation is partial. Resource gating
   (cost of informative experiments scaled by per-agent endowment)
   produces an additional irreversibility regime.

## Why a separate paper

- The IWAI 2026 paper (Hallgren/Hyland/Albarracin) uses a different
  substrate: Bayes-net of commitments + BMR + propagation operator
  T = (I − A)⁻¹ over a structured world. Its contribution is structure
  *learning* via expansion/reduction over a hidden dependency graph.

- This paper's substrate is a flat paradigm representation with a
  latent context, no structure learning, no BMR. The phenomenology is
  similar (graded lock-in, motivated persistence, resource gating) but
  the mechanism differs. Two distinct contributions; cleaner to
  publish them separately than to entangle them.

- David's note: the IWAI paper's appendix can credit this work as
  "the hierarchical-context substrate developed by the second author
  in parallel work" with a forward reference to this paper.

## Section structure (LNCS / AAAI-style, ~10-14 pages)

### §1 Introduction

Hook with the Kuhnian phenomenology — communities lag, lock in, eventually
reorganize — and the gap in scalar network-epistemology models that the
IWAI 2026 paper also names. Compress the literature review here; main
contribution claim by paragraph 2.

Cite:
- Albarracin et al. 2022 (Entropy 24:4) — flat AIF lock-in precedent
- Hyland & Albarracin 2025 (arXiv 2509.17957) — single-agent variational
  cost of mind-change, extended here
- Zollman 2010, O'Connor & Weatherall 2019 — network epistemology
- PolyGraphs (Ball et al. 2024) — the scalar-credence framing this
  contests

### §2 Background and motivation

- Kuhn's theory-ladenness as the empirical anchor
- Why scalar credences over a fixed menu miss the structural feature
- Why hierarchical context is the minimal principled formalization
- Forward reference to the IWAI paper for the structure-learning angle

Cite:
- Kuhn 1962
- Holmes 2000 — rival research programs reading
- Boantza & Gal 2011 — load-bearing role of phlogiston
- Blumenthal & Ladyman 2017 — late-phlogistic theory proliferation
- Thagard 1989, 1990 — ECHO as conceptual-change precedent (with the
  formalism distinction noted)

### §3 The model

- Per-agent generative model with joint state (theta, c)
- A_world (per-paradigm) and A_context (graded blend)
- B = B_theta ⊗ B_c with B_theta = (1 − eps_theta)·I + eps_theta·unif
  and tridiagonal B_c (eps_crisis, eps_resolve)
- Social channel: trust-weighted mixture of neighbors' marginal posteriors
- Trust learning via Gamma-conjugate update on categorical surprisal
- Per-agent utility U_i(theta) and value-tilted posterior
- Resource flow: trust-derived W and η, per-agent r, cost barrier

Reference: `src/pomdp/simple_step.py`, `src/pomdp/motivated_step.py`.

### §4 Experiments

Drawn from your E1–E9:

- **E1**: stationary, no social — individual-level theory-ladenness baseline
- **E2/E7**: discrete shift, (q_reliability × eps_resolve) phase diagram
- **E3**: slow drift, hysteresis
- **E4/E5**: continuous-lambda variants (tempered update only, no
  hierarchical context)
- **E6/E8**: resource ablation
- **E9 (NEW)**: motivated-update phase diagram — direct multi-agent
  realization of Hyland & Albarracin 2025

Headline figure: the E7 phase diagram (diagonal boundary in
(q_reliability, eps_resolve) at C=5, 560 seeded runs, graded
transition zone). Secondary: E9 showing how lambda interacts with
the boundary.

### §5 Comparison to the structural / BMR account (Hallgren/Hyland/Albarracin)

A short positioning section that:

- Acknowledges the parallel IWAI 2026 work
- Names what each substrate captures: hierarchical context (this
  paper) vs. structured dependency network + BMR (IWAI paper)
- Argues the two are complementary: the lock-in *phenomenology* is
  substrate-robust (shown here on the simpler substrate), while the
  specific endogenous-γ *mechanism* (conviction propagating through
  the dependency network) is a feature of the structured model
- A pointer to the IWAI 2026 paper for readers who want the
  structure-learning treatment

### §6 Discussion

- Why hierarchical context is the right minimal formalization
- Limits: no model expansion (no Bayesian Model Reduction), so the
  paper does not address "discovering an unconceived dimension"
- Future work: combining hierarchical context with BMR; learning the
  per-agent dependency network

### §7 Related work

Network epistemology, motivated reasoning, AIF social cognition, theory
acquisition, attitude networks. Use the citation discipline from
`notes/framing_research_2026_06_05.md`.

---

## Asset inventory — what you already have

| Asset | Location | Ready? |
|-------|----------|--------|
| Hierarchical context model | `src/pomdp/simple_step.py` | ✓ Done, 18 tests |
| Continuous-lambda model | `src/pomdp/cont_step.py` | ✓ Done, 14 tests |
| Motivated-update extension | `src/pomdp/motivated_step.py` | ✓ Just built, 10 tests |
| Sweep runner | `experiments/run_experiment.py` | ✓ supports motivated model |
| E1-E8 results | `experiments/results/` | ✓ Done |
| E9 results | running (background `bem6p6bu9`) | Running |
| Background research / lit review | `notes/framing_research_2026_06_05.md` | ✓ Substantial |
| §§1-2 prose | `notes/sections_1_2_draft_2026_06_05.tex` | ✓ Adaptable |
| Bib entries | inline in §§1-2 draft + PR #2 | ✓ Verified |
| Headline figures | E7 phase diagram + E8 ablation table | ✓ JSON, plots TBD |
| §3 model spec | needs writing | ✗ ~2 days |
| §4 experiments writeup | partial (notes/experiment_results_*.md) | ~50% |
| §5 positioning vs IWAI | needs writing | ✗ ~1 day |
| §6 discussion | needs writing | ✗ ~0.5 day |
| §7 related work | adapt from §§1-2 + notes | ✗ ~1 day |

Realistic timeline: **3-4 weeks of focused part-time work** gets you a
submission-ready draft. AAAI Jul deadline is comfortably reachable.

---

## What's still missing for full standalone-paper independence

Two things that, if added, would close any open gap with the IWAI paper:

1. **Per-agent dependency network.** Optional extension: each agent
   carries a small (5-10 node) Bayes net of commitments with its own
   adjacency A_agent. Conservatism κ_i = (I−A_agent)⁻¹·1 and conviction
   U_i = (I−A_agent)⁻¹·u_i propagate through structure exactly as in
   the IWAI paper, but per-agent rather than community-level. This is
   the same machinery you'd implement to validate the IWAI paper's
   mechanism on the simpler substrate. Cost: ~1.5-2 days of code +
   tests + one new experiment.

2. **Lite BMR.** A reduce-only Bayesian model reduction step that
   prunes edges in the per-agent A_agent based on the Savage-Dickey
   ratio. Mirrors the IWAI paper's reduction half. Cost: ~1 day.

Both are post-IWAI-deadline work. Decision point: do them, and the
standalone paper has a near-superset of the IWAI paper's mechanisms;
skip them, and the paper is honestly a different (simpler, flatter)
substrate. Either is publishable.

---

## Recommended posture for tonight's meeting

1. State the appendix as fine — it's exactly what David offered
2. Mention you'll add motivated update to your substrate anyway
   (it's already done as of today — `src/pomdp/motivated_step.py`)
3. Mention the E9 result will land tonight/tomorrow regardless
4. Don't tip your hand about the standalone paper — leave that for
   after the IWAI submission lands

After Jun 12, decide whether to grow the standalone paper. The
material is already there.
