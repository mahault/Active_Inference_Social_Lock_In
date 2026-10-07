# Results: E-series reruns and the pluralist experiments (5-6 Oct 2026)

Living version with discussion: the Claude Doc "Social lock-in follow-up" (claude.ai/code/artifact/e27aff5b-8b92-4faf-9478-b703e780e8dd).

Each effect is a difference in final belief in the truth (0 to 1), paired by seed, 20 seeds, 95% bootstrap interval in brackets. "Holds" = survives Holm correction across the 49 headline contrasts in `experiments/results/P1_stats.md`. Intervals are narrow partly because each run averages 24 agents, so read effect sizes, not only significance. E-series stats were computed with the same method from the 20-seed reruns.

## Observation sampling

With resources on, each agent observes at the experiment it chose, and trust scores neighbours at the experiment the observer ran (5 Oct; `src/pomdp/simple_step.py`, `structural_step.py`, `cont_step.py`; tests in `tests/test_chosen_experiment_observation.py`). Earlier versions of the E-series results are at `85e7ad0`.

## A. E-series, rerun

| # | Result | Numbers | Status |
|---|---|---|---|
| 1 | Sign flip (E19): trust and resource limits compound once half the agents are misled | interaction +0.30 [0.15, 0.45] at 50%, +0.62 [0.47, 0.75] at 65%, +0.29 [0.14, 0.45] at 80%, all p < 0.003; -0.02 at 20% (p = 0.06) | solid; stronger than the June 5-seed runs |
| 2 | E14: no super-additivity | -0.03 [-0.07, 0.00], p = 0.12 | does not replicate with 20 seeds |
| 3 | E18: trust only partly rescues under resource limits | trust effect under resources +0.05 [0.01, 0.12]; trust x resource interaction +0.02 [-0.07, 0.11] | trust helps only a little under resource limits |
| 4 | Hysteresis (E16, E17) | changed by < 0.02 | stands; E17 has 3 seeds, not re-tested |
| 5 | E15 cost dose-response with trust on | misled bloc ends 0.69-0.81 | much milder |

Interaction = additive prediction minus observed; positive means the combination is worse than the parts.

## B. Pluralists vs monists (David's model, `src/pluralist/`)

| # | Result | Numbers | Status |
|---|---|---|---|
| 6 | Pluralists beat monists committed to the old theory | +0.34 [0.33, 0.34] | holds |
| 7 | A neutral monist beats pluralists | -0.11 [-0.11, -0.10] | holds; the advantage is about not being committed |
| 8 | Pluralist lock-in comes from support that never fades | with fading, pure pluralist communities never lock in but rarely fully recover | descriptive; raise with David |
| 9 | A committed minority locks a community in | 0% to 50% committed monists: -0.28; 25% committed already locks 14 of 25 conditions | holds |
| 10 | Pluralists help in proportion to their numbers | +0.11 (0% to 25%), +0.13 (25% to 75%); 75% pluralist locks 9 of 25 vs 12 for 100% | holds (last part untested) |
| 11 | Pluralists among the agents who see the decisive measurement help most | +0.08 | holds |
| 12 | Influence runs both ways | committed monists drag pluralists -0.18; a pluralist majority pulls committed monists +0.33 | holds; pull vanishes under strong conviction and costly experiments |

## C. How agents judge each other

| # | Result | Numbers | Status |
|---|---|---|---|
| 13 | Trusting those who agree with you makes lock-in worse | -0.14 (all pluralist), -0.07 (half pluralist) | holds; worse than fixed trust everywhere |
| 14 | Trusting whoever forecast revealed outcomes well helps, modestly | +0.08 / +0.03; sharp trust with outcomes used for trust only +0.10 / +0.24 | holds; sharp-trust numbers are best-of-4, optimistic |
| 15 | No trust rule rescues an all-committed community | identical across rules | everyone predicts the same |
| 16 | Delay before outcomes are revealed barely matters | differences <= 0.004, not significant even with sparse replications | robust null |
| 17 | Clearly revealed outcomes swamp everything | +0.43 to +0.65 over attention-weighted reveals | holds |
| 18 | Sharp trust slightly hurts mixed communities under clear reveals | -0.03 | holds |

## Caveats

Conviction strength, experiment cost and trust sharpness were set by hand or picked from a short sweep. "Correct" means being on the right side of 50%. Lit review behind the trust designs: `notes/trust_models_lit_review_2026-10-05.md`.

## Decisions (5 Oct)

- Monist type (committed, neutral) is a community-composition parameter, swept in proportions.
- Support fading is a secondary setup alongside the main one.
- Agents judge each other on outcomes revealed after a delay (relative log score); agree-with-me and the old accuracy rule are comparison arms.
- Ability/integrity trust, selective sharing vs lying, and motivated ignorance move to the follow-up paper.

## D. Artefact checks (P1f) and population size / bandwidth (P1g), 6 Oct

Stats: `experiments/results/P1f/stats.md` (Holm across 117), `experiments/results/P1g/stats.md` (Holm across 64).

| # | Check | Answer | Key numbers |
|---|---|---|---|
| 19 | Scoring by reaching the truth | Pluralists reach the truth far more often; the neutral monist's higher average is hedging near 0.5. Proper scores (log, Brier) favour the neutral monist, because pluralists are confidently wrong where they lock in | agents reaching truth: pluralist 0.46, neutral 0.12, committed 0.006; log score -1.05 / -0.52 / -1.84 |
| 20 | Changing world | Delay matters once the truth keeps flipping | D=1 vs D=50, Brier 0.197 vs 0.289 (pluralists, reveals every 10th round) |
| 21 | Wrong rival | Pluralist advantage over committed monists disappears when the rival is wrong; survives when the menu contains the truth | rival M=+3: identical log scores; K=3 menu {-1,+1,+3}: +0.79 |
| 22 | Anchoring (omega) | End state mostly tracks the support-weighted anchor; at omega=1 every type locks in | |end - anchor| 0.03-0.04 for omega <= 0.9 |
| 23 | Sparse/noisy reveals, more observers | Trust helps more when reveals are noisy, not when there are more observers | +0.01 (1x noise) vs +0.10-0.15 (3x noise) log score |
| 24 | Population size | Lock-in tracks the fraction of observers, not N; outcomes are not a lottery | 4 fixed observers, N 8 -> 100: pluralists reaching truth 0.46 -> 0.33 |
| 25 | Message bandwidth | Sampled messages have no significant effect; what matters is sharing theory by theory. Pluralists sharing only observable beliefs behave like neutral monists | reaching truth 0.46 -> 0.11 |
| 26 | Neutral-drift control | Sampled messages reproduce symmetry breaking (seed spread ~ (mN)^-1/2) but anchoring prevents fixation; only a partial match to Tanaka's QSG | SD 0.24 / 0.15 / 0.11 at N = 8 / 24 / 50, m = 1 |

What changes: "delay doesn't matter" holds only in a static world; the pluralist advantage depends on the rival containing the truth and on sharing beliefs per theory; "neutral monist does better" depends on the scoring rule; anchoring drives much of the end state.
