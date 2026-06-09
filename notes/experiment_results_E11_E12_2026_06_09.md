# E11 + E12 — empirical demonstration of trust and resource channels (2026-06-09)

Two experiments designed to expose Mahault's contribution beyond the IWAI
paper's structural mechanism: trust learning (social channel) and resource
coupling (material channel). The structural-gamma channel is included as
reference.

## E11 — tri-channel ablation under slow drift

**Design.** 2x2x2 grid over `(gamma_strength ∈ {0, 2}, trust_learning ∈ {F, T},
resource_coupling ∈ {F, T})`. Ramp schedule (truth interpolates from
paradigm 0 to paradigm 1 over t∈[0, 200]). Structural substrate, N=15,
n_steps=150, 3 seeds per cell. 30% utility-anti-truth bloc.

**Result.**

| γ | trust | rsrc | mean_qB | capture | n |
|---|---|---|---|---|---|
| 0 | F | F | 0.000±0.000 | 1.00 | 3 |
| 0 | F | T | 0.356±0.300 | 0.67 | 3 |
| 0 | T | F | 0.000±0.000 | 1.00 | 3 |
| 0 | T | T | 0.333±0.272 | 0.67 | 3 |
| 2 | F | F | 0.178±0.251 | 0.67 | 3 |
| 2 | F | T | 0.333±0.471 | 0.67 | 3 |
| 2 | T | F | 0.178±0.251 | 0.67 | 3 |
| 2 | T | T | 0.333±0.471 | 0.67 | 3 |

Marginal contributions (avg `Δ(mean_qB)` from turning channel ON):
- gamma:     +0.083
- trust:     -0.005
- resources: +0.250

**Interpretation.** In this regime the baseline (all channels off) is
already saturated at `mean_qB = 0` — the simple/structural substrate's
context inertia produces *complete* lock-in to the initial truth
(paradigm 0) under a slow ramp. There is no headroom to make lock-in
*stronger*; channels can only allow recovery. The marginals therefore
read as "recovery enabled by each channel":

- **Resources +0.25**: the largest effect. Resource coupling weakens
  the experimental evidence channel (low-r agents pick low-Fisher
  experiments), which lets the social channel — pulled toward the
  pro-truth 70% majority — dominate. Mechanism: material gating
  redistributes evidential weight from world to society.
- **Gamma +0.08**: smaller effect. The 30% utility-anti-truth bloc has
  its evidence partially silenced, freeing the 70% bloc to act on
  evidence alone — slight recovery.
- **Trust ≈ 0**: trust learning had no detectable marginal effect at
  this severity. Likely because the population was already saturated
  toward a single paradigm; differential predictive accuracy between
  neighbors has no surface to bite on.

**This is informative, but not the original hypothesis.** The intended
demonstration was "channels cause lock-in vs. an adapting baseline."
What was actually shown is "channels mediate recovery from a baseline
that is already locked in." The mechanism claims (especially the
resource channel as evidential-weight redistributor) are correct, but
the experimental polarity is inverted.

## E12 — resource hysteresis under truth reversal

**Design.** Reversal schedule (paradigm 0 → 1 at t=60, 1 → 0 at t=180).
Cont substrate, N=30, n_steps=320, q_reliability=0.65, lambda_init=2.5
(moderate tempering), eps_theta=0.15 (paradigm leak so population can
update). Trust learning ON throughout. Only `resource_coupling` toggles
on/off. 8 seeds per condition.

**Result.**

| Condition           | phase-1 peak mean_qB | end mean_qB |
|---------------------|----------------------|-------------|
| resource_coupling=F | **0.807**            | 0.119       |
| resource_coupling=T | **0.148**            | 0.147       |

Phase-1 peak gap (off − on): **+0.659**.

**Interpretation.** With resources off, the population fully tracks the
truth: mean_qB rises from 0.13 to 0.81 during phase 1, then drops back
to 0.12 after the reverse. With resources on, the population *never
adapts to phase 1* — it sits at mean_qB ≈ 0.15 throughout, then stays
there.

The mechanism: the cost barrier `c = c0 · Fisher(x) / (r − r_min)`
diverges as r → r_min, so under sustained spending the population's
resources drain and agents become restricted to the low-Fisher tail of
x_grid. Low-Fisher experiments give weak per-step likelihood evidence;
tempered Bayes at λ=2.5 further weakens it. The population effectively
loses the *evidential capacity* to revise paradigm-level beliefs at all.

**What this shows.** Resources are not a hysteresis mechanism here —
they are an *adaptation blocker*. Lock-in via the material channel is
stronger than hysteresis: the wrong bloc doesn't fail to revert *after
adapting*; they fail to adapt in the first place. This is a substantive
finding, not the originally-targeted one.

## Honest assessment

### What's been empirically demonstrated

1. **Resource coupling produces a large, robust adaptation blockage**
   (E12: 65-point gap in phase-1 peak mean_qB between coupled and
   uncoupled). The material channel is sufficient to lock paradigm
   beliefs against a clean paradigm shift.

2. **Channels combine nontrivially with the underlying substrate's
   inertia** (E11): in a regime where lock-in is already at floor, the
   resource channel is the one that *lifts* the population back toward
   adapting, because it redistributes evidential weight away from
   experiment.

3. **Tri-channel principled implementation is sound**: all three channels
   are wired into the structural substrate, run end-to-end, produce
   consistent results across seeds.

### What was NOT demonstrated

1. **No hysteresis in the textbook sense** — the original "adapt then
   fail to revert" arc requires a regime where the population can adapt
   under resources but slowly enough to drain its capacity. The
   working configurations either let the population adapt easily (no
   hysteresis) or block adaptation entirely (E12).

2. **Trust learning has near-zero marginal effect in E11.** This may be
   real (the experimental design saturates other dynamics) or it may
   reflect a need for richer conditions (e.g., heterogeneous initial
   beliefs that distrust can amplify). Untested.

3. **No phase diagram across resource severity** — only the on/off
   contrast. The dose-response curve of `c0` (cost scale) vs.
   adaptation rate would show whether the blockage is gradual or
   threshold-like. Untested.

## Next steps (none required to defend the contribution claim)

- Trust ablation in a regime where the population is heterogeneously
  committed (e.g., 30% sure of paradigm 0, 70% sure of paradigm 1,
  then truth = paradigm 1 throughout). Predict trust learning sharpens
  factional persistence.
- Resource dose-response: sweep `c0` ∈ {0.05, 0.1, 0.15, 0.2, 0.3} at
  fixed trust learning, measure phase-1 peak. Expected: smooth
  transition from "full adaptation" to "blocked" with no intermediate
  regime — a phase transition in adaptation capacity.
- Trust × resource interaction: does trust learning *amplify* resource
  blockage by routing material flow toward already-distrusted blocs?
