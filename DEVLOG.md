# Devlog

## 2026-06-26
- Added an AGENTS.md project guide (uncommitted, work in progress)

## 2026-06-29
- Added standalone paper on precision collapse and two-channel lock-in (11 files, ~1043 insertions).

## 2026-06-30
- Added an AGENTS.md agent-instructions file. (uncommitted, work in progress)

## 2026-07-21
- Added AGENTS.md agent-instructions scaffolding. (uncommitted, work in progress)

## 2026-10-05
- Observation sampling: each agent now observes at the experiment it actually ran, and trust scores neighbours on that same experiment (`src/pomdp/*_step.py`, tests in `tests/test_chosen_experiment_observation.py`). E8 to E19 rerun (5-seed pass, then 20-seed reruns of E13 to E15, E18, E19).
- New pluralist model package `src/pluralist/` with tests against the original model, topology and a reference JSON; experiments P1 to P1e (Hyland Fig. 4 replication with 24 agents, lock-in boundary over gamma x c0, community composition, trust delay, sharp trust), stats in `scripts/p1_stats.py`.
- Lit review of formal epistemic-trust models (`notes/trust_models_lit_review_2026-10-05.md`): why the current surprisal-based trust channel does almost nothing, and candidate replacements (Bovens-Hartmann, Olsson, Shafto, Harris-Hahn, HGF, Sobel).
- Started `paper-pluralist-lockin/` (main.tex, refs, figures). All uncommitted, work in progress.

## 2026-10-06
- 20-seed reruns of E13-E15, E18, E19 (`scripts/rerun_e14_e18_e19_20seeds.sh`; seed count now set by `N_SEEDS`, E14 summary stores per-seed values). E19 sign flip holds (interaction +0.30 / +0.62 / +0.29 at 50 / 65 / 80% misled, p < 0.003); E14 shows no interaction (-0.03, n.s.).
- P1e (full-precision and sparse reveals, trust sharpness and memory sweeps) and paired stats for P1b-P1e (`experiments/results/P1_stats.md`, 49 contrasts, Holm).
- Docs updated for the reruns and the pluralist work: README, CLAUDE.md / AGENTS.md, `paper-precision-collapse/README.md` (updated numbers; main.tex flagged as older), new `paper-pluralist-lockin/README.md`, results note `notes/results_2026-10-05_reruns_and_pluralist.md`.
- Local result snapshots are gitignored; earlier E-series results are the versions at commit 85e7ad0.
- P1f artefact checks and P1g population size / bandwidth runs: new `src/pluralist/worlds.py`, checks in `verified.py` and `mixed.py`, run and stats scripts, two new test files, results in `experiments/results/P1f/` and `P1g/`, and section D of the results note. The pluralist advantage over committed monists holds under outcome-based scoring (share of agents reaching the truth 0.46 vs 0.01) but disappears when the rival theory is wrong. "Neutral monist does better" depends on the scoring rule. Larger populations (N=100 vs 8) shrink the across-seed lottery but are slower to reach the truth when only four agents observe.
