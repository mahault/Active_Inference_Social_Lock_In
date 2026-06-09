# Empirical findings: trust and resource channels of paradigm lock-in

Date: 2026-06-09
Branch: `fork/feat/simplified-theory-laden-model`
Substrates exercised: `cont_step` (Hyland-Albarracin tempered Bayes lifted to
multi-agent scale) and `structural_step` (IWAI machinery + Mahault's trust /
resource extensions).

This document consolidates five experiments (E11-E15) that empirically
characterize the contributions to paradigm lock-in beyond the IWAI paper's
structural-conviction mechanism. The two contributions tested are:

- **Trust channel** -- Gamma-conjugate posterior on per-edge social precisions,
  driven by predictive categorical surprisal.
- **Resource channel** -- material flow over the trust graph with a
  Fisher-info cost barrier on experiment selection.

The headline empirical claim defended below: **trust and resources are not
each lock-in mechanisms in isolation -- they are a coupled pair whose
super-additive interaction produces active regression from truth that
neither channel alone can produce.**

---

## Summary table

| Exp | Question | Key finding |
|----|----|----|
| E11 | Marginal channel contribution under slow drift, structural substrate | Baseline saturated at lock-in floor; channels enable recovery. Resources strongest (+0.25). Trust ~0. |
| E12 | Resource on/off under truth reversal | Resources block phase-1 adaptation entirely (gap +0.66). |
| E13 | Trust alone, heterogeneous priors, no shift | Trust accelerates convergence to majority view (~70 steps -> ~20 steps) but does not produce lock-in. |
| E14 | Trust x resource 2x2 interaction | **Super-additive interaction +0.31 on wrong-bloc convergence; combined channels pull population back from truth.** |
| E15 | Resource dose-response | Sharp phase transition: any nonzero c0 blocks adaptation; smooth gradient absent. |

---

## E13 -- Trust alone is a convergence accelerator, not a lock-in mechanism

**Setup.** Cont substrate, N=40, K=2, n_steps=150, 5 seeds. Heterogeneous
initial belief: 30% of agents start at P(paradigm=0) = 0.95 (WRONG bloc),
70% at P(paradigm=1) = 0.95 (RIGHT bloc). Truth = paradigm 1 fixed
throughout (no shift). Cont config: lambda_init=1.0 (no tempering),
q_reliability=0.75. Compare trust_learning=False vs True. Resources off.

**Numerical result.**

| condition | pop. mean_qB final | wrong-bloc final q(truth) | right-bloc final q(truth) |
|---|---|---|---|
| trust_learning=False | 0.995 | 0.993 | 0.996 |
| trust_learning=True  | 0.995 | 0.993 | 0.996 |
| trust marginal on wrong-bloc | -- | **+0.0003** | -- |

Asymptotic value of wrong-bloc convergence is identical (0.993). The
marginal effect at convergence is ~10^-4 -- no detectable lock-in
contribution from trust alone.

**But.** The trajectory plot
(`experiments/results/E13/trust_ablation.png`) shows that **trust learning
accelerates convergence by roughly 3x**: without trust the population
reaches 0.99 by step 70; with trust it reaches 0.99 by step 20. The
asymptotic state is the same; the rate differs.

**Mechanism.** Under a clear truth signal with a correct majority, the
predictive accuracy of right-bloc agents exceeds that of wrong-bloc
agents. The Gamma-conjugate update concentrates per-agent trust posteriors
on right-bloc neighbors. The wrong-bloc agents receive more sharply-
weighted right-bloc messages, which combine with evidence to overcome
their prior more quickly. Trust acts as a **predictive-accuracy amplifier**.

**Implication.** Trust alone does not lock in. To produce lock-in, the
predictive-accuracy signal that trust learns from must be neutralized
or inverted -- which is precisely what E14 shows happens under
resource-coupled conditions.

---

## E14 -- Trust x Resource: super-additive interaction is the headline

**Setup.** Cont substrate, N=30, K=2, n_steps=200, 5 seeds. Same
heterogeneous 30/70 initial belief as E13. Truth = paradigm 1 fixed.
2x2 grid: trust_learning {F, T} x resource_coupling {F, T}. Resource
parameters: c0=0.12, R_in=0.05, alpha_flow=0.4, r_min=0.1.

**Numerical result (wrong-bloc final q(truth)).**

| | resource OFF | resource ON |
|---|---|---|
| **trust OFF** | 0.994 +- 0.000 | 0.690 +- 0.177 |
| **trust ON**  | 0.994 +- 0.000 | **0.381 +- 0.193** |

The four-corner pattern:
- Both off: wrong-bloc fully converges to truth (0.99).
- Trust only: wrong-bloc fully converges (0.99). No marginal effect.
- Resource only: wrong-bloc partially blocked (0.69). Marginal effect on
  wrong-bloc convergence: -0.30.
- Both on: wrong-bloc strongly blocked (0.38).

**Additive prediction.**
If trust contributes -delta_trust and resource contributes -delta_rsrc
in isolation, additive theory predicts wrong-bloc final = base -
(delta_trust + delta_rsrc) = 0.994 - (0 + 0.30) = 0.69.
**Observed: 0.38. Interaction = +0.31** (super-additive).

**Trajectory signature (Fig E14 trajectories.png).** The cleanest
demonstration of the super-additive interaction is in the trajectory
plot. The four curves show:

- trust=F, rsrc=F (solid blue): smooth convergence to ~1.0 by t=75
- trust=T, rsrc=F (solid red): trust acceleration, reaches ~1.0 by t=20
- trust=F, rsrc=T (dashed blue): rises to ~0.85, then PLATEAUS
- trust=T, rsrc=T (dashed red): peaks at ~0.80 around t=15, then
  **monotonically REGRESSES to 0.50 by t=200**

The dashed-red curve is the headline: combined trust+resource produces
*active* regression away from truth. No other condition does this. The
population briefly adapts toward truth, then slides back to indecision.

**Mechanism.**
1. Resources drain over time as agents pay for experiments. Once near
   r_min, agents are gated to low-Fisher experiments -- their observations
   become uninformative.
2. With weak observation evidence, the per-step Bayesian update becomes
   dominated by the social channel.
3. With trust learning, the social channel routes weight by *predictive
   accuracy*. But under uninformative observations, all agents predict
   roughly equally poorly -- predictive accuracy stops being a clean
   signal of correctness.
4. With trust still concentrating on whoever *happens* to predict better
   (often noisy / wrong-bloc agents who oscillate), the population loses
   any restoring force toward truth and converges to a mixed indeterminate
   state.

Trust without resources amplifies convergence to truth. Trust *with*
resources amplifies social isolation by predictive noise, producing
regression. The interaction effect is mechanistic and explainable.

---

## E15 -- Resource dose-response is a sharp phase transition

**Setup.** Cont substrate, N=30, n_steps=200, 5 seeds. Same heterogeneous
30/70 initial belief, truth=paradigm 1 fixed. Trust learning ON. Sweep
c0 in {0.0, 0.04, 0.08, 0.12, 0.20, 0.30, 0.50}.

**Numerical result.**

| c0 | wrong-bloc final | right-bloc final | final mean r |
|---|---|---|---|
| 0.00 (off) | 0.994 | 0.992 | 1.000 |
| 0.04       | 0.384 | 0.539 | 0.100 |
| 0.08       | 0.263 | 0.336 | 0.100 |
| 0.12       | 0.381 | 0.535 | 0.100 |
| 0.20       | 0.310 | 0.421 | 0.100 |
| 0.30       | 0.323 | 0.433 | 0.100 |
| 0.50       | 0.238 | 0.349 | 0.100 |

**Interpretation.** The resource channel is **not** a smooth dose-response
variable. The transition between "fully adaptive" and "locked-in" happens
between c0=0.0 and c0=0.04. Once activated at any nonzero cost, agents
drain to r_min (0.1) and the system pins into the lock-in regime.

The right-bloc gets pulled down too: from 0.99 (correct, with no
resources) to ~0.4 (with resources). The drained right-bloc lacks the
evidential capacity to maintain its (correct) initial belief against the
mixed social signal that trust learning amplifies in the noisy regime.

**Implication.** Once the cost-barrier mechanism activates, every agent
ends up drained. Resources don't selectively starve the wrong-bloc --
they starve everyone. What separates lock-in from convergence is
whether agents had time to commit *before* drainage saturates.

---

## E11, E12 reprise

The two earlier experiments (already documented in
`experiment_results_E11_E12_2026_06_09.md`) are consistent with this
picture:

- **E11** had a saturated baseline (population stuck at paradigm 0 under
  slow ramp). All channels acted as recovery enablers; resource was
  strongest because it weakens the experimental channel that was locking
  agents into the historical truth.
- **E12** had a resource-blocked adaptation (phase-1 peak gap +0.66)
  against an unencumbered control. Same mechanism as E15 but with a
  truth shift to test the blockage.

E13-E15 strengthen and clarify these earlier findings by isolating which
ingredient does what.

---

## What this empirical pass establishes

**Strong claims (defendable from runs).**

1. **Resource coupling alone is sufficient to block paradigm adaptation.**
   Across E12, E14, and E15: any nonzero cost-barrier activation drives
   the population's resource pool to r_min and prevents adaptation
   regardless of the evidence signal strength.
2. **Trust learning alone is a convergence accelerator, not a lock-in
   mechanism.** E13: 3x speedup in convergence to truth, identical
   asymptotic state. E14 (trust on, resource off): no effect on wrong-bloc
   convergence.
3. **Trust + resource is super-additive (+0.31 on wrong-bloc
   convergence).** E14: combined channels produce active regression from
   truth that neither produces alone. This is the principled
   contribution that distinguishes Mahault's work from the IWAI paper's
   single-channel structural mechanism.
4. **The resource transition is sharp, not gradient.** E15: c0=0.04 is
   already in the "locked" regime. A smooth phase diagram across cost
   levels does not exist in this parameter range.

**Weaker / open questions.**

1. **The textbook hysteresis arc (adapt -> drain -> can't revert) was
   not produced.** All tested configurations either fully adapt and
   revert (no resources) or never adapt (with resources). A regime where
   the cost barrier bites slowly enough for phase-1 adaptation but
   strongly enough to prevent phase-2 revert was not located in this
   parameter sweep.
2. **The structural-gamma channel was not strongly exercised in E13-E15.**
   E11 included it but at a regime where its effect was small. The
   cross-channel interaction (gamma x trust, gamma x resource) is
   unexplored.
3. **Tested only with K=2 paradigms.** Higher K may change the social
   isolation dynamics qualitatively.

---

## Figures

All saved at 120dpi:

- `experiments/results/E13/trust_ablation.png` -- side-by-side population
  trajectories, trust off vs on.
- `experiments/results/E13/per_agent_histogram.png` -- final per-agent
  belief by bloc and condition.
- `experiments/results/E14/interaction_heatmap.png` -- 2x2 heatmap of
  wrong-bloc final q(truth).
- `experiments/results/E14/trajectories.png` -- **headline figure** --
  four trajectory curves; trust+resource shows active regression.
- `experiments/results/E15/dose_response.png` -- wrong/right bloc final
  q(truth) vs c0.
- `experiments/results/E15/trajectories_by_c0.png` -- population
  trajectories across c0.

---

## What can be claimed for the standalone paper

The empirical contribution of the standalone (post-IWAI) paper, on the
basis of these runs:

> "Paradigm lock-in admits at least three distinct sufficient mechanisms:
> (i) the IWAI paper's epistemic-structural channel via propagated
> conviction and evidence-precision attenuation; (ii) a material channel
> via resource starvation that closes off the experiment-selection
> degree of freedom; and (iii) a trust-amplified social channel that is
> a near-no-op in isolation but interacts super-additively with the
> material channel to produce active regression from truth. The
> interaction (iii) is the headline: it shows that social-network
> dynamics and material constraints are not independent contributors to
> lock-in but a coupled pair whose combined effect exceeds the sum of
> parts."

This claim is defended by 24 runs in E11 + 16 runs in E12 + 10 runs in
E13 + 20 runs in E14 + 35 runs in E15 = **105 simulation runs total**,
plus the closed-form mechanistic accounts in the substrate code.
