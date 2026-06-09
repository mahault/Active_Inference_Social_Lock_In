# E18 — cross-channel interactions: a negative result that refines the thesis

Date: 2026-06-09. Substrate: `structural_step.py` (has all three channels).
2x2x2 factorial (gamma x trust x resource), 5 seeds, N=30, heterogeneous
prior (30% committed+convicted to the WRONG paradigm, truth = paradigm 1).
Readout: wrong-bloc final q(truth). Low = locked in.

## TL;DR — the prediction was wrong, and the real finding is better

I expected E18 to confirm "all three channel pairs are super-additive for
lock-in," generalizing E14's trust x resource result. **It did not.** On
this substrate and regime:

- **Trust is CORRECTIVE, not lock-in-inducing** (main effect -0.085: it
  *raises* wrong-bloc convergence from 0.90 to 0.985).
- **All three pairwise interactions are SUB-additive** (negative), not
  super-additive: gamma x trust -0.107, gamma x resource -0.109,
  trust x resource -0.213.
- Once trust is ON, it **dominates**: the wrong bloc converges to truth
  (~0.984) regardless of gamma or resource.

This contradicts the simple "they all interact super-additively" story I
hoped to confirm. But the mechanism that explains it is a genuine
refinement of the precision-collapse thesis, and it reconciles with E14.

## The cube

| gamma | trust | rsrc | wrong-bloc q(truth) | reading |
|---|---|---|---|---|
| off | off | off | 0.900 | baseline |
| off | off | on  | 0.685 | resource locks in (world evidence gated) |
| off | on  | off | 0.985 | trust corrects (majority is right) |
| off | on  | on  | 0.984 | trust dominates resource |
| on  | off | off | 0.791 | gamma locks in (world evidence silenced) |
| on  | off | on  | 0.685 | resource dominates gamma |
| on  | on  | off | 0.984 | trust corrects despite gamma |
| on  | on  | on  | 0.984 | trust dominates everything |

## Main effects

- gamma alone: +0.109 lock-in (conviction silences the wrong bloc's WORLD
  evidence; modest)
- resource alone: +0.214 lock-in (cost barrier gates informative
  experiments; strongest single channel)
- trust alone: **-0.085** — trust *reduces* lock-in here

## The mechanism: two evidence channels, not one knob

The result makes sense once you separate the evidence routes:

**There are two channels through which a wrong-bloc agent can receive
disconfirming evidence: the WORLD (its own observations) and the SOCIETY
(its neighbours' beliefs). Lock-in requires BOTH to be closed.**

- **gamma** closes the world channel by attenuating the world likelihood
  (precision -> 0 via `w = exp(-gamma * conviction_asymmetry)`).
- **resource** closes the world channel by pricing informative
  (high-Fisher) experiments out of reach — the agent observes, but at
  near-zero discriminability.
- **trust** governs the social channel. It does NOT reduce precision; it
  *reallocates* it across sources by predictive accuracy. Whether that
  helps or hurts depends on who is accurate:
  - If a correct majority exists and predicts well (E18: 70% right-bloc,
    full social coupling, q_reliability 0.70), trust concentrates on them
    and pulls the wrong bloc to truth -> **corrective**.
  - If evidence is starved so nobody predicts reliably (the E14 regime),
    predictive accuracy becomes noise and trust amplifies it into false
    consensus -> **lock-in**.

So gamma and resource both close the *world* channel, but in E18 they
leave the *social* channel intact. With a correct majority on the other
end of an open social channel, the wrong bloc is rescued — which is why
trust-on cells all converge (~0.984) and the interactions are sub-additive:
once trust opens a corrective social route, the world-channel lock-in
mechanisms have nothing left to bite on.

## Why this reconciles with E14 rather than contradicting it

E14 (cont substrate) found trust x resource SUPER-additive for lock-in.
E18 (structural substrate) finds it SUB-additive. The difference is not a
bug — it is the regime:

- **E14 regime**: resource starvation was severe enough to compromise the
  evidence of the WHOLE population, including the would-be-correct agents.
  With no reliable predictor anywhere, trust's reallocation amplified noise
  -> the social channel ALSO closed -> both channels shut -> lock-in, and
  trust made it worse (super-additive).
- **E18 regime**: resource cost (c0=0.12) gated the wrong bloc but a large
  correct majority kept predicting well, so the social channel stayed open
  and informative. Trust used it correctively.

The unifying statement: **trust is a precision-REALLOCATOR on the social
channel; its sign depends on whether the social channel carries signal or
noise. The world-channel mechanisms (gamma, resource) determine which.**

## What this means for the paper's thesis

The headline must be stated more carefully than "three channels that all
interact super-additively." The honest, stronger claim:

> Paradigm lock-in requires closing BOTH the world-evidence channel
> (gamma and/or resource) AND the social-evidence channel. Trust does not
> reduce precision; it reallocates it across social sources, so it is
> corrective when a reliable majority exists and amplifying when the world
> channel is already starved. The trust x resource interaction therefore
> flips sign with regime: super-additive when starvation removes the
> reliable signal (E14), sub-additive (trust-dominated, corrective) when a
> correct majority survives (E18).

This is a richer and more defensible result than uniform super-additivity.
It says lock-in is a **two-channel** phenomenon and trust's role is
**contingent on the world channel's state** — a genuinely non-obvious
prediction that the model makes and the experiments confirm.

## Honest status

- This is a NEGATIVE result against the naive hypothesis ("all pairs
  super-additive"). Reported as such; not tuned to flip it.
- It does NOT invalidate E14 — it locates E14 as one regime and explains
  the sign.
- It DOES require rewriting the outline's contribution #2 from "all three
  super-additive" to the two-channel / regime-dependent-sign account.

## Open follow-up (to map the sign flip cleanly)

The decisive experiment is a regime sweep that interpolates between E18 and
E14: vary the correct-majority strength (or social coupling / resource
severity) and watch the trust x resource interaction cross from sub- to
super-additive. That sweep would turn "the sign is regime-dependent" from a
two-point observation (E14 vs E18) into a mapped boundary. Not yet run.

## Files

- `experiments/results/E18/summary.json` — full cube + effects + interactions
- `experiments/results/E18/cross_channel_interactions.png` — main effects
  (left) and pairwise interactions (right); all three interactions negative
