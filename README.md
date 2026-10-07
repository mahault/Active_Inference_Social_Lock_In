# Paradigm_Shift_Act_Inf

Multi-agent active-inference simulations of paradigm lock-in in scientific communities. The repo backs three papers:

- the IWAI 2026 paper with Jonas Hallgren and David Hyland, *Can Network Models of Science Do Without Pluralist Agents?* (accepted, oral);
- `paper-precision-collapse/`, a standalone paper on two-channel precision collapse;
- `paper-pluralist-lockin/`, the joint follow-up on when pluralist communities lock in (started Oct 2026).

Current results and decisions: `notes/results_2026-10-05_reruns_and_pluralist.md`.

## What

Extension of Hyland & Albarracin (2025), *On the Variational Costs of Changing Our Minds* (arXiv:2509.17957), from single-agent belief revision to a coupled multi-agent setting on a small-world network. Demonstrates Kuhnian paradigm-shift dynamics: communities lag behind environmental drift, with hysteretic phase transitions in their adaptation.

## Model substrates

The codebase contains three model substrates:

1. **Continuous Gaussian substrate** (`src/*.py`) — the original v2 implementation. Agents hold Gaussian beliefs over a continuous parameter; trust is Gamma-conjugate edge learning. Proven monostable via Jacobian analysis (NB14). Retained as a regression baseline.

2. **Categorical POMDP scaffold** (`src/pomdp/step.py`, `agent_pop.py`, `gen_model.py`) — fixed-state epistemic POMDP with per-paradigm likelihoods, belief-utility in the EFE policy, and softmax-precision social coupling. Phase-1 complete (28 tests pass). Discovered the martingale wall: belief-utility in the policy cannot flip the inference basin.

3. **Simplified theory-laden model** (`src/pomdp/simple_step.py`) — drops the POMDP action loop. Adds a latent context variable c in {normal_science, crisis} to the joint state (theta, c). Context carried forward through a proper transition model B_c. Used for E1-E3.

4. **Continuous-lambda model** (`src/pomdp/cont_step.py`) — direct multi-agent lift of Hyland & Albarracin (2025) Eq. 13: `q*(theta) ∝ prior(theta) · p(o|theta)^(1/lambda)`. Per-agent lambda tempers the likelihood; lambda evolves from KL cost of previous update (large changes → more conservative). Optional resource coupling gates exploration via trust-derived patronage. This is the active experimental substrate for E4-E6.

5. **Structural model** (`src/pomdp/structural_step.py`) — per-agent dependency network, conviction-driven endogenous gamma, in-loop Bayesian model reduction. Used for E10-E11 and E18-E19.

6. **Pluralist model** (`src/pluralist/`) — port of David Hyland's [epistemic-pluralism](https://github.com/dkhyland/epistemic-pluralism) (linear-Gaussian, rival theories held side by side), credited in each copied file. Extensions: conviction and resource gates on the discriminating channel, learned trust, support forgetting (`extended.py`); communities mixing pluralists, committed monists and neutral monists (`mixed.py`); trust learned from outcomes revealed after a delay (`verified.py`). Used for P1-P1e.

**Observation sampling (5 Oct 2026):** with resources on, each agent observes at the experiment it chose, and trust is scored at that experiment. E8, E11 and E14-E19 were rerun with this; earlier versions of their results are at `85e7ad0`.

## Layout

```
src/              Core simulation modules
  pomdp/          Categorical POMDP scaffold, simplified, cont and structural models
  pluralist/      Port of David Hyland's pluralist model + extensions (P-series)
experiments/      YAML configs + sweep runner; results/ per experiment
scripts/          Standalone experiment scripts (E13-E19, P1-P1e), stats, rerun scripts
tests/            Tiered test suite (163 tests)
notebooks/        Exploratory + figure generation (NB15-17 are active)
  _v1_archive/    Old continuous-substrate notebooks
notes/            Design docs, theory brief, architecture review
paper/            Original IWAI LNCS draft (anonymized; do NOT commit authors-deanon.tex)
paper-precision-collapse/  Standalone two-channel paper (June numbers in main.tex)
paper-pluralist-lockin/    Joint follow-up paper skeleton
lit-review/       Topical lit reviews (markdown)
archive/          Frozen v02 norms-as-shared-precision-priors draft
```

## Running

The repo `.venv` is currently empty; the global Python 3.12 (jax, scipy, pytest) is what was used in Oct 2026. Run from the repo root.

```bash
pip install -r requirements.txt

python -m pytest tests -q                        # full suite (163 tests)

# Config-driven experiments
python -m experiments.run_experiment experiments/configs/E8_principled_ablation.yaml

# Standalone scripts: module mode so `src` resolves; E13-E19 take N_SEEDS (default 5)
N_SEEDS=20 python -m scripts.run_e19_regime_flip
python -m scripts.run_p1c_mixed_communities
python -m scripts.p1_stats                       # paired stats -> experiments/results/P1_stats.md
```

## Experiments

| ID | Model | Environment | Sweep | What it tests |
|----|-------|-------------|-------|---------------|
| E1 | simple | Stationary, no social | eps_resolve x eps_crisis | Individual-level theory-ladenness (baseline) |
| E2 | simple | Discrete shift at t=100 | q_reliability x eps_resolve | Binary-context phase diagram |
| E3 | simple | Slow ramp over 200 steps | q_reliability x eps_resolve | Hysteresis under drift |
| E4 | cont | Discrete shift at t=100 | q_reliability x lambda_init | **Continuous-lambda phase diagram (Fig 3)** |
| E5 | cont | Discrete shift at t=100 | q_reliability x lambda_init | Continuous-lambda + resources |
| E6 | cont | Discrete shift at t=100 | lambda_init x resource ON/OFF | Resource ablation (heuristic) |
| E7 | simple (C=5) | Discrete shift at t=100 | q_reliability x eps_resolve | **Principled phase diagram (Fig 3)** |
| E8 | simple (C=5) | Discrete shift at t=100 | eps_resolve x resource ON/OFF | **Principled resource ablation (Fig 4)** |

| E9-E20 | cont / structural / motivated | various | see `notes/` | Channel experiments (trust, resources, conviction, motivated gate); E8, E11, E14-E19 rerun 5-6 Oct |
| P1-P1e | pluralist | Mass flips at round 80 | gamma x c0, composition, trust rule, delay | Do pluralist communities resist lock-in better than monist ones? |

Configs in `experiments/configs/`. Results in `experiments/results/`. Current results and stats: `notes/results_2026-10-05_reruns_and_pluralist.md` and `experiments/results/P1_stats.md`.
Detailed results: `notes/experiment_results_2026_05_27.md` (E1-E3),
`notes/experiment_results_2026_05_28.md` (E4, E6),
`notes/experiment_results_2026_05_28_principled.md` (E7, E8).

## Building the paper

Uses Springer LNCS class. On MiKTeX (Windows), `llncs` is auto-installed if missing; otherwise: `mpm --install=llncs`. On TeXLive: `tlmgr install llncs`.

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Strip identifying metadata before submission:

```bash
exiftool -all= main.pdf
```

## Anonymization

Submission is **double-blind**. The paper compiles with `\author{Anonymous}` by default. Real authors live in `paper/authors-deanon.tex` (gitignored). Do not link this repo from the paper; do not commit anything that identifies the authors.

## Status (Oct 2026)

- IWAI 2026 paper accepted as an oral; Jonas presents in Madrid (12-17 Oct).
- Joint follow-up started; Royal Society journal, no fixed deadline.
