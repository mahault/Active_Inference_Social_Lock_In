# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

Multi-agent active-inference simulations of Kuhnian paradigm lock-in. Three outputs share this repo:

- IWAI 2026 paper with Jonas Hallgren and David Hyland (accepted, oral; camera-ready retitled *Can Network Models of Science Do Without Pluralist Agents?*). Its minimal code lives in David's repo `dkhyland/epistemic-pluralism`.
- `paper-precision-collapse/`: Mahault's standalone paper (two-channel precision collapse) on the categorical substrates.
- `paper-pluralist-lockin/`: the joint follow-up (agreed 1 Oct 2026, Mahault leads, Royal Society journal), built on David's pluralist model ported to `src/pluralist/`.

Results and decisions are tracked in `notes/results_2026-10-05_reruns_and_pluralist.md`.

## Commands

The repo `.venv` is currently empty; the global Python 3.12 has jax, scipy and pytest. Run everything from the repo root.

```bash
# Tests (163 pass as of 2026-10-06)
python -m pytest tests -q
python -m pytest tests/test_pluralist_mechanisms.py -v   # pluralist port + extensions

# Config-driven experiments (E1-E12, E20)
python -m experiments.run_experiment experiments/configs/E8_principled_ablation.yaml

# Standalone experiment scripts need module mode so `src` resolves
python -m scripts.run_e18_cross_channel
N_SEEDS=20 python -m scripts.run_e19_regime_flip      # E13-E15, E18, E19 default to 5 seeds
python -m scripts.run_p1c_mixed_communities           # P-series (pluralist model)
python -m scripts.p1_stats                            # paired stats for P1b-P1e

# Paper (from the paper folder)
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Architecture

Five model substrates:

### 1. Continuous Gaussian substrate (`src/*.py`)
Agents hold Gaussian beliefs N(mu, 1/tau) over a continuous parameter theta. Per-step cycle: policy (softmax over EIG-minus-cost) -> world observation -> private Bayes -> lambda*U social blend -> precision-pool -> Gamma-conjugate trust update -> resource flow. Orchestrated by `population.py` as a JIT-compiled equinox Module. **Proven monostable** via Jacobian analysis (NB14); retained as a regression baseline.

### 2. Categorical POMDP scaffold (`src/pomdp/step.py`, `agent_pop.py`, `gen_model.py`)
Fixed-state epistemic POMDP: theta in {A, B} paradigms, actions are experiments (measurements, not control), B=identity (truth is fixed). Per-paradigm likelihoods A_theta make theory-ladenness structural. Belief-utility U(theta) enters the EFE policy evaluation only, never the state update — avoids the precision-additive blow-up. Social coupling via softmax-precision reliability channel; validated pitchfork bifurcation at K_c=2 (NB15). **Martingale wall**: belief-utility in the policy cannot flip the inference basin; capture requires a different mechanism.

### 3. Simplified theory-laden model (`src/pomdp/simple_step.py`)
Drops the POMDP action loop per David Hyland's suggestion. Adds a latent context variable c in {normal_science, crisis} to the joint state (theta, c). Under c=0, paradigm likelihood columns are averaged (observations uninformative about paradigms); under c=1, original discriminating columns. The joint posterior is carried forward through a proper transition model B = B_theta x B_c, where B_c encodes paradigm inertia via `eps_crisis` (anomaly detection rate) and `eps_resolve` (crisis resolution rate). Lock-in emerges from the dynamics: committed agents see uninformative data -> reinforce c=0 -> self-insulating loop. This is the active experimental substrate for E1-E3.

### 4. Continuous-lambda model (`src/pomdp/cont_step.py`) and structural model (`src/pomdp/structural_step.py`)
`cont_step` tempers the likelihood by a per-agent lambda (Hyland & Albarracin 2025, Eq. 13); `structural_step` adds the per-agent dependency network, conviction-driven endogenous gamma and in-loop BMR. Both carry the trust and resource channels used in E11-E19.

### 5. Pluralist model (`src/pluralist/`)
Port of David Hyland's `epistemic-pluralism` (linear-Gaussian, K theories in information form, support weights; `model.py` and `experiments.py` copied verbatim with credit). `extended.py` adds conviction and resource gates on the discriminating channel, learned trust and support forgetting; `mixed.py` allows communities mixing pluralists, committed monists and neutral monists; `verified.py` adds trust learned from outcomes revealed after a delay. Every extension switched off reproduces the original model exactly (tested). `tests/pluralist_x64.py` keeps float64 scoped to these tests, because importing David's module enables it globally.

### Observation sampling (2026-10-05)
With resources on, each agent observes at the experiment it chose, and trust scores neighbours at the experiment the observer ran (`simple_step`, `structural_step`, `cont_step`; tests in `tests/test_chosen_experiment_observation.py`). E8, E11 and E14-E19 were rerun with this; earlier versions of their results are at `85e7ad0`.

### Key relationships
- `gen_model.build_generative_model()` constructs A/B/C/D/U from a `PomdpConfig`; discretizes `world.py`'s regression family into categorical likelihood bins
- `agent_pop.infer_state()` is pure categorical Bayes on any state dimension S; both the POMDP scaffold (S=K) and simplified model (S=K*C) reuse it unchanged via vmap
- `step.build_trust()` and `step.readout_trust_mixture()` are shared across substrates
- `network.build_adjacency()` provides Watts-Strogatz / Barabasi-Albert / planted-SBM graphs

## Config hierarchy

`SimpleConfig` wraps `PomdpConfig` which contains `WorldConfig`. All frozen dataclasses. The experiment runner (`experiments/run_experiment.py`) loads YAML configs that map to `SimpleConfig` fields, with a `sweep` dict for Cartesian grid expansion over parameter values.

## Test tiers

- **Tier 1** (`test_tier1_invariants.py`): Algebraic contracts — conservation laws, row-stochasticity, reduction to baseline at lambda=0
- **Tier 2** (`test_tier2_sanity.py`): Behavioral — trust concentrates on aligned neighbors, resource enables exploration, lambda damps revision
- **POMDP scaffold** (`test_pomdp_scaffold.py`): Generative-model algebra, EFE decomposition, martingale wall proof, social-fold basin-locking
- **Simplified model** (`test_simple_step.py`): Joint-model algebra, transition dynamics (stationary convergence, normalization), context learning from evidence, capture/escape pair
- **Pluralist** (`test_pluralist_*.py`): original-model reproduction against David's reference, each extension reduces exactly to the original when off, mixed-community and delayed-trust contracts
- **Observation sampling** (`test_chosen_experiment_observation.py`): observations follow the chosen experiment; trust is scored at the observer's experiment

## Constraints

- **Anonymization**: Paper compiles with `\author{Anonymous}`. Real authors in gitignored `paper/authors-deanon.tex`. Do not link this repo from the paper.
- **All POMDP code is pure JAX**: no side effects, explicit PRNG splitting, frozen dataclass state. Functions are vmapped over the agent batch dimension.
- **No modifications to existing modules** when adding new model variants — the POMDP scaffold was built by a co-author and has its own test gate. Add new files alongside.
