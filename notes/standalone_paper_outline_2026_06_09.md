# Standalone paper outline — Mahault's solo or first-author paper

> **NOTE (2026-06-09 revision):** This outline has been substantially
> revised after the E11-E17 empirical work and the BMR-in-loop /
> structural extension. The original thesis ("principled hierarchical
> context as theory-ladenness") is now demoted to the *vehicle*; the new
> headline is the **multi-channel / precision-collapse** account that
> *connects* this work to the IWAI 2026 structural-BMR paper rather than
> merely sitting beside it. See the new "Unifying framework" section.

Provisional working title (revised):

> **One Currency, Three Routes: Precision Collapse as a Unified Account of
> Paradigm Lock-In in Multi-Agent Active Inference**

(Old title, kept for reference: *Principled Theory-Ladenness and Motivated
Belief Revision in Multi-Agent Active Inference: A Hierarchical-Context
Model of Paradigm Lock-In*.)

Target venue: **AAAI 2027** (Jul deadline), with **AAMAS** (Oct) as a fallback.
David explicitly suggested both venues in the May call as natural homes for
this line of work.

This outline is a self-contained paper that uses Mahault's substrates
(`src/pomdp/simple_step.py`, `cont_step.py`, `motivated_step.py`,
`structural_step.py`) and results (E1-E17). It does NOT depend on inclusion
in the IWAI 2026 paper — but, importantly, the revised framing now *contains*
the IWAI mechanism as one of its three channels, so the two papers connect
through a single theoretical spine rather than competing.

---

## Unifying framework: precision is the common currency

The central theoretical claim that connects this paper to the IWAI 2026
structural-BMR paper:

> **Paradigm lock-in is what happens when the effective precision on
> disconfirming evidence collapses. There is one currency — precision-
> weighting in the free energy principle — and at least three distinct
> mechanisms that each drive it toward zero. They are not rival accounts
> of lock-in; they are three routes to the same precision collapse.**

In active inference, prediction errors that drive belief updating are
*precision-weighted*: precision sets the gain on the evidence channel.
High precision amplifies disconfirming evidence (the community updates);
low precision attenuates it (the community ignores it and locks in). This
is standard FEP (Friston; precision as inverse variance / attentional gain;
see the precision-weighting literature below). The contribution here is to
show that **three structurally different social/material/epistemic
mechanisms all reduce to lowering this one quantity**, and to map the
phase boundaries each produces.

### The three channels, as precision modulators

| Channel | Mechanism in our model | What it does to precision | Whose contribution |
|---|---|---|---|
| **Epistemic / structural** | Conviction `U = T·u` propagated through the per-agent dependency network; endogenous `gamma` attenuates the world likelihood | Directly multiplies the evidence weight `w = exp(-gamma · conviction_asymmetry)` — precision on disconfirming evidence is gated by how much cherished structure depends on the paradigm | IWAI 2026 (Hallgren/Hyland/Albarracin); reproduced here in `structural_step.py` + E10b |
| **Social / trust** | Gamma-conjugate trust learning reweights neighbours' messages by predictive accuracy | Reallocates precision *across sources* — confirming neighbours gain gain, disconfirming sources are muted. Under uninformative observation it amplifies noise into false consensus | Mahault (E13, E14) |
| **Material / resource** | Resource flow + Fisher-info cost barrier `c = c0·I_F(x)/(r - r_min)` gates which experiments an agent can afford | Caps the precision an agent can *purchase* — depleted agents are forced to low-Fisher (low-precision) observations | Mahault (E12, E15, E16, E17) |

The empirical payoff (E14): because all three act on the same quantity,
they **interact**. Trust is a near-no-op alone (E13) but is super-additive
with the resource channel (E14, interaction +0.31) — precisely because when
resources starve the evidence channel, trust's reallocation of precision
across sources has nothing truthful left to lock onto and amplifies noise.
A single-currency account predicts this interaction; three-separate-
mechanisms accounts do not.

### Why this connects rather than competes with the IWAI paper

- The IWAI paper's `gamma`-crossover is *one column* of this table — the
  epistemic route. We reproduce it (`structural_step.py`, E10b) on the
  same substrate, so the connection is demonstrated, not asserted.
- The trust and resource channels are precision modulators the IWAI paper
  does not model. They are not alternatives to its mechanism; they are
  additional routes to the same collapse, and the paper shows they compose.
- So the two papers share a spine: IWAI develops the structural route in
  depth (BMR, structure learning, the propagation operator); this paper
  develops the *unifying precision account* and the social + material
  routes, with the structural route reproduced as the bridge.

### The material channel has an economics home: path dependence

The resource channel is not ad hoc — it is the active-inference image of
**increasing-returns lock-in** from the economics of path dependence:

- The cost barrier `c = c0·I_F(x)/(r - r_min)` is an *increasing-returns
  switching cost*: the more depleted an agent, the more prohibitive the
  informative experiment that would change its mind. This is exactly the
  positive-feedback / self-reinforcing mechanism in Arthur's increasing-
  returns economics and David's QWERTY lock-in.
- Our **hysteresis** result (E16: adapt in phase 1, locked in phase 2; E17
  phase diagram) is the formal signature of *path dependence* — the same
  point in truth-space supports different stable population states
  depending on the trajectory taken to reach it. David's central claim
  ("the choice is governed by history, not by what is optimal apart from
  history") is what E16 demonstrates mechanistically in an AIF community.

This gives the paper an interdisciplinary hook: it renders a classical
economics-of-science / path-dependence intuition as precision dynamics in
a Bayesian community, and shows when it produces hysteresis.

### Connecting literature (verify each before citing — see citation note)

Precision-weighting / FEP spine:
- Friston et al. 2024, "Designing ecosystems of intelligence from first
  principles," *Collective Intelligence* (Mahault is a co-author) —
  collective AIF, shared generative models. VERIFIED real (sagepub
  10.1177/26339137231222481).
- Hyland & Albarracin 2025, "On the Variational Costs of Changing Our
  Minds," arXiv 2509.17957 — single-agent variational cost of mind-change;
  the precision-cost backbone. Already in the IWAI bib.
- Standard precision-as-gain references (Friston; Parr & Friston on
  precision and attention) — pick canonical ones, verify.

Social / collective:
- Ramstead, Veissière & Kirmayer, "Cultural Affordances: Scaffolding Local
  Worlds Through Shared Intentionality and Regimes of Attention"
  (PhilPapers RAMCAS-2) — cultural niche construction, shared attention.
- "Federated inference and belief sharing" (PMC11139662) and "Belief
  sharing: a blessing or a curse" (arXiv 2407.02465) — when shared belief
  helps vs. produces collective error. Directly relevant to the trust
  channel's noise-amplification finding.
- NOTE: arXiv 2104.01066 ("An active inference model of collective
  intelligence") was MISATTRIBUTED earlier in this project (it is
  Kaufmann/Gupta/Taylor 2021, NOT Albarracin 2022). Do not cite as
  Albarracin without re-verifying authorship.

Resource-rationality:
- Lieder & Griffiths 2020, "Resource-rational analysis," *Behavioral and
  Brain Sciences* — accuracy-vs-cognitive-cost tradeoff; grounds the
  resource channel in bounded cognition. VERIFIED real.

Economics of path dependence (the material-channel home):
- W. Brian Arthur 1994, *Increasing Returns and Path Dependence in the
  Economy* (Univ. Michigan Press). VERIFIED real.
- Paul A. David 1985, "Clio and the Economics of QWERTY," *American
  Economic Review* 75(2):332-337. VERIFIED real.

Network epistemology (already in the bib): Zollman 2010, O'Connor &
Weatherall 2019, Ball et al. (PolyGraphs) 2024.

**Citation discipline:** this project previously caught 7 hallucinated/
misattributed citations by fetching source URLs. Apply the same gate to
every entry above before it enters the .bib — especially author and year.

---

## What the paper claims (revised)

A unified active-inference account of paradigm lock-in in scientific
communities, organized around precision collapse, with four contributions:

1. **Unified precision account (headline).** Paradigm lock-in is the
   collapse of effective precision on disconfirming evidence. Three
   structurally distinct mechanisms — epistemic conviction, social trust,
   material resources — each drive that collapse, and because they share
   one currency they interact. This subsumes the IWAI structural mechanism
   as the epistemic route and adds the social and material routes.

2. **The trust × resource interaction (key empirical result).** Trust
   learning is a near-no-op for lock-in in isolation (E13) but is
   *super-additive* with the resource channel (E14, interaction +0.31):
   combined, they produce active regression away from a truth the
   community had already reached. The single-currency account predicts
   this; separate-mechanism accounts do not.

3. **Material lock-in as path dependence, with hysteresis.** The resource
   cost barrier is an increasing-returns switching cost (Arthur; David).
   It produces textbook hysteresis (E16) — adapt to a new paradigm, spend
   the budget, then fail to revert when truth reverses — occupying a real
   region of parameter space, not a tuned point (E17 phase diagram: three
   regimes — adapt+revert / hysteresis / blocked).

4. **Principled substrates (the vehicle).** Hierarchical-context
   theory-ladenness (`simple_step`), the multi-agent lift of Hyland &
   Albarracin's variational cost of mind-change (`cont_step`,
   `motivated_step`), and the per-agent dependency network with in-loop
   Bayesian model reduction (`structural_step`). All closed-form, 105
   tests. The structural substrate reproduces the IWAI gamma-crossover
   (E10b), demonstrating the bridge rather than asserting it.

## Relationship to the IWAI 2026 paper: connected, not competing

The earlier draft framed this as a *separate* paper on a *different*
substrate. The revised framing is stronger: the two papers share one
theoretical spine.

- The IWAI 2026 paper (Hallgren/Hyland/Albarracin) develops the
  **epistemic/structural route** in depth: a per-agent Bayes-net of
  commitments, the propagation operator T = (I − A)⁻¹, conviction-driven
  endogenous gamma, and structure learning via Bayesian model
  expansion/reduction over a hidden dependency graph.

- This paper develops the **unifying precision account** and the two
  routes the IWAI paper does not model — social (trust) and material
  (resources) — and shows all three modulate the same quantity and
  interact. The structural route is reproduced here (`structural_step.py`,
  E10b) as the bridge between the two papers.

- Concretely: the IWAI paper is the deep dive on column 1 of the
  three-channel table; this paper is the unifying frame plus columns 2-3.
  A reader of either is pointed to the other. The appendix-credit David
  offered becomes a genuine theoretical cross-reference, not a courtesy.

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

Organized around the three-channel / precision-collapse thesis. The
substrate-validation experiments (E1-E9) become support; the channel
experiments (E10b-E17) carry the headline.

Substrate validation (background, condense):
- **E1**: stationary, no social — individual theory-ladenness baseline
- **E2/E7**: discrete shift, (q_reliability × eps_resolve) phase diagram
- **E3**: slow drift
- **E4/E5**: continuous-lambda (tempered update)
- **E6/E8/E9**: resource + motivated-update — multi-agent realization of
  Hyland & Albarracin 2025

The three channels (headline):
- **E10b**: structural substrate reproduces the IWAI gamma-crossover —
  the epistemic route, and the bridge to the IWAI paper
- **E13**: trust alone is a convergence accelerator, NOT a lock-in
  mechanism (asymptotic state unchanged; ~3x faster convergence)
- **E14**: trust × resource is SUPER-ADDITIVE (+0.31). **Headline figure** —
  the four-curve trajectory plot where trust+resource peaks then regresses
  away from truth (`experiments/results/E14/trajectories.png`)
- **E12 / E15**: resource channel blocks adaptation; sharp phase transition
  in cost scale c0
- **E16**: textbook hysteresis arc — adapt then fail to revert
  (`experiments/results/E16/hysteresis_trajectories.png`)
- **E17**: phase diagram in (R_in, c0) — three regimes, hysteresis is a
  band not a point (`experiments/results/E17/phase_diagram.png`).
  **Second headline figure.**

Headline figures: E14 trajectories (super-additive regression) and E17
phase diagram (three regimes). Together they make the "one currency, three
routes, and they interact" argument visually.

Total: ~250 seeded runs across all experiments (E11-E17 alone = 105+).

Caveats to address before submission (from the empirical notes):
- Add the r_init axis to E17 so the hysteresis band's genericity is shown
  in 2D, not just along c0.
- Run the gamma × resource and gamma × trust cross-channel sweeps — the
  three-channel interaction story currently rests on trust × resource only.
- K > 2 paradigms untested (requires K-column likelihood generalization in
  gen_model).
- Scale ceiling: the Python step-loop makes large sweeps slow and the GPU
  counterproductive (benchmark: GPU 2.7x slower). A lax.scan + vmap
  refactor is the prerequisite for referee-scale robustness sweeps.

### §5 The bridge to the structural / BMR account (Hallgren/Hyland/Albarracin)

Reframed from "comparison" to "connection". This section:

- Shows the IWAI endogenous-γ mechanism IS the epistemic column of the
  three-channel table — same precision-collapse currency
- Presents E10b: the structural substrate (per-agent DAG + T propagation
  + in-loop BMR) reproduces the γ-crossover, demonstrating the bridge
- Locates the IWAI structure-learning machinery (BMR, expansion, Schur
  residue) as the deep treatment of the epistemic route, and this paper's
  trust/resource channels as the other two routes
- Frames the two papers as one research programme with a shared spine,
  each citing the other

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
| Motivated-update extension | `src/pomdp/motivated_step.py` | ✓ Done, 10 tests |
| Structural + BMR-in-loop | `src/pomdp/structural_step.py` | ✓ Done, 30+ tests |
| Sweep runner | `experiments/run_experiment.py` | ✓ all 4 substrates |
| E1-E10b results | `experiments/results/` | ✓ Done |
| E11-E17 results (channels) | `experiments/results/E11..E17` | ✓ Done, ~105 runs |
| Headline figs (E14, E16, E17) | `experiments/results/E1{4,6,7}/*.png` | ✓ Plotted |
| Empirical writeups | `notes/empirical_findings_consolidated_2026_06_09.md`, `experiment_results_E16_*.md` | ✓ Done |
| GPU benchmark / scale note | `notes/gpu_backend_benchmark_2026_06_09.md` | ✓ Done |
| Background research / lit review | `notes/framing_research_2026_06_05.md` | ✓ Substantial |
| §§1-2 prose | `notes/sections_1_2_draft_2026_06_05.tex` | ✓ Adaptable (reframe) |
| Connecting-lit citations | this outline, "Unifying framework" | ⚠ verify before .bib |
| §2 unifying-framework prose | needs writing | ✗ ~1.5 days |
| §3 model spec (4 substrates) | needs writing | ✗ ~2 days |
| §4 experiments writeup | strong (consolidated notes) | ~70% |
| §5 connection to IWAI | reframe from "positioning" to "bridge" | ✗ ~1 day |
| §6 discussion (path dependence) | needs writing | ✗ ~1 day |
| §7 related work | adapt from §§1-2 + connecting lit | ✗ ~1 day |

Realistic timeline: **3-4 weeks of focused part-time work** to a
submission-ready draft. AAAI Jul deadline reachable. The empirical core
is essentially complete; the main writing lift is the unifying-framework
§2 and folding the four substrates into one §3.

---

## Status of the two "still missing" items (now DONE)

The previous version of this outline listed two gaps. Both are now built:

1. **Per-agent dependency network.** ✓ DONE — `structural_step.py`:
   per-agent DAG, T = (I−A)⁻¹ propagation, conservatism κ = T·1,
   conviction U = T·u, endogenous γ. Validated against the IWAI mechanism
   in E10b.

2. **BMR (full, not just lite).** ✓ DONE — Savage-Dickey edge pruning,
   wired into the multi-agent loop with soft-gated Gaussian conjugate
   edge inference (the variational E/M step), plus structure expansion and
   Schur-complement residue. 30+ tests.

So the standalone paper now has a *superset* of the IWAI mechanisms, not a
simpler flatter substrate. This is what makes the unifying framing
available: we can demonstrate the epistemic route on our own substrate
(E10b) and connect it to the social and material routes.

---

## Open work before submission (priority order)

1. **r_init axis on E17** — make the hysteresis-genericity claim 2D. ~2h.
2. **Cross-channel sweeps** (γ × resource, γ × trust) — the three-channel
   interaction story currently rests on trust × resource (E14) only. ~half
   a day once configs are written.
3. **lax.scan + vmap refactor** — prerequisite for any large robustness
   sweep; also unlocks the GPU. ~half a day. Optional unless a referee
   wants finer phase boundaries.
4. **Verify all connecting-lit citations** before they enter the .bib
   (author + year + venue via source URL). Non-negotiable given the prior
   7-citation error rate.
5. **Write §2 (unifying framework) and §3 (four substrates)** — the main
   prose lift.

The empirical core is essentially complete. The remaining work is
sharpening (1-2), de-risking scale (3), and writing (4-5).
