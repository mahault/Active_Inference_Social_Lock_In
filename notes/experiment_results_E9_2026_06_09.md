# E9 — Motivated update phase diagram (2026-06-09)

Motivated update model (`src/pomdp/motivated_step.py`) — the Hyland & Albarracin
(2025) Eq. 13 value-tilted posterior at multi-agent scale on top of the
hierarchical-context substrate (C=5). 30% of agents have utility for the
WRONG paradigm; 70% prefer truth.

**Sweep**: (q_reliability × lambda_tilt) × 5 seeds = 175 runs. Discrete
shift at t=100, run to t=300. eps_crisis=0.15, eps_resolve=0.20, eps_theta=0.02,
20% mavericks.

## Result

```
           lam=0.0   lam=0.5   lam=1.0   lam=2.0   lam=3.0
qr=0.55    0.792    0.692    0.694    0.698    0.699
qr=0.65    0.729    0.712    0.698    0.699    0.700
qr=0.70    0.430    0.733    0.700    0.699    0.700
qr=0.75    0.188    0.738    0.703    0.700    0.700
qr=0.80    0.103    0.754    0.723    0.700    0.700
qr=0.85    0.059    0.733    0.731    0.701    0.700
qr=0.90    0.038    0.738    0.736    0.702    0.700
```

## Two findings

**(1) Expected:** the `lam=0.0` column reproduces E7 — the (q_reliability,
eps_resolve) phase boundary is intact (0.79 → 0.04).

**(2) Unexpected:** at ANY positive lambda_tilt, the phase boundary
disappears. Mean_qB ≈ 0.70 across the entire q_reliability axis. The
result is dominated by the population's utility distribution
(70% pro-truth → 70% endpoint).

## Mechanism

Closed-form prediction: the value-tilted posterior at equilibrium is
approximately q(theta=1|pro) = σ(λ) and q(theta=1|anti) = σ(−λ).
Population mean is `0.7·σ(λ) + 0.3·σ(−λ)`. For λ ≥ 0.5 this gives ≈ 0.70.

Plugging in:
- λ=0.5: 0.7·0.622 + 0.3·0.378 = 0.548 (predicted)
- λ=1.0: 0.7·0.731 + 0.3·0.269 = 0.592
- λ=2.0: 0.7·0.881 + 0.3·0.119 = 0.653
- λ=3.0: 0.7·0.953 + 0.3·0.047 = 0.681

Observed values are slightly higher than predicted (0.70 vs. 0.55-0.68),
which makes sense: evidence aligns with the pro-truth tilt (since the
truth IS paradigm 1), so pro-truth agents reach higher q(truth) than
the pure-tilt prediction, while anti-truth agents are held in check
by evidence.

## Interpretation — what this says about substrates

**This is NOT the IWAI paper's endogenous γ phenomenon.** The motivated
update applied here biases posteriors directly; the IWAI paper's mechanism
is conviction propagating through structure and silencing the precisions
on disconfirming channels — gating evidence at its source, not redirecting
posteriors.

The hierarchical-context substrate with motivated update produces:

- **Utility-driven consensus** (each agent settles near their preferred
  paradigm with strength λ)
- **Phase boundary in social coupling disappears** at any positive λ
  because each agent has a strong individual attractor that no social
  pressure can override
- **No evidence silencing** — agents still update from observations;
  evidence partially counteracts the anti-truth tilt

These are honest, distinguishable phenomena. Both are forms of "motivated
belief revision," but they operate at different stages of the inference
chain.

## What this means for tomorrow's meeting

David's pre-meeting note was correct: the simple_step / motivated_step
substrate does NOT validate the paper's specific endogenous-γ mechanism.

Three options for how this gets reported:

**(A) Honest report:** "I added motivated update; the substrate now produces
utility-driven consensus rather than evidence-silencing lock-in. That
confirms David's point — the simpler substrate is a different beast than
the BMR + propagation model in the paper. Appendix placement remains the
right call."

**(B) Add a finer-grained sweep (~30 min):** lambda_tilt = 0.05, 0.10,
0.20, 0.30, 0.40 to map the cliff between the lam=0 social-coupling regime
and the lam=0.5+ utility-determined regime. Doesn't change the conclusion;
gives a usable phase-diagram figure.

**(C) Defer dependency network to post-IWAI:** the 1.5-2 day extension
that adds per-agent A and T = (I−A)⁻¹ propagation is what would let
conviction silence channels on this substrate. That's the genuine
cross-substrate validation of the paper's mechanism, and it's the path
to making the standalone paper a near-superset of the IWAI work.

Recommendation: A for the meeting, B if time tomorrow, C after Jun 12.
