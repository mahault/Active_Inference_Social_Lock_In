# E19 — the trust × resource interaction sign-flip, mapped

Date: 2026-06-09. The decisive follow-up to E18. Structural substrate,
gamma OFF (isolates trust × resource exactly as E14 did). Sweep the
wrong-bloc fraction; at each, run the 2×2 (trust × resource), 5 seeds,
and compute the interaction. Readout: population q(truth) at end.

## Result: the sign flip is real and goes in the predicted direction

| wrong_frac | base (neither) | both observed | interaction | sign |
|---|---|---|---|---|
| 0.20 | 0.985 | 0.984 | -0.051 | sub |
| 0.35 | 0.898 | 0.984 | **-0.137** | sub (most corrective) |
| 0.50 | 0.571 | 0.968 | -0.035 | sub |
| 0.65 | 0.296 | 0.694 | -0.023 | sub |
| 0.80 | 0.045 | 0.197 | **+0.070** | **super** |

The trust × resource interaction is sub-additive (negative) while a correct
majority exists, and flips super-additive once the wrong bloc becomes a
dominant majority (~0.8). Zero crossing is between wrong_frac 0.65 and 0.80.

This validates the E18 -> E14 reconciliation: the interaction sign is
regime-dependent, governed by whether the social channel carries signal
(reliable correct majority) or not.

## The cleaner finding underneath: trust follows accuracy, not the majority

The four-corner panel (`regime_flip.png`, right) shows something sharper
than the interaction sign alone:

- **trust-only (blue) is the highest curve at EVERY fraction** — trust is
  corrective even when 80% of agents are committed to the wrong paradigm
  (it lifts q(truth) from 0.045 to 0.296).
- **resource-only (orange) is the lowest** — the strongest single lock-in
  channel at every fraction.
- **both (red dashed)** tracks trust-only at low wrong_frac (trust
  dominates), but falls below it at high wrong_frac.

Why trust is corrective even against a wrong majority: trust learning
routes precision by *predictive accuracy*, not by headcount. The world
generates the truth, so the agents who predict it best are the
truth-trackers — and trust concentrates on them regardless of how few they
are. Trust is "follow the accurate," and accuracy tracks truth. This is the
opposite of naive "follow the crowd" social influence.

## So when does trust fail, and why is that the super-additive corner?

Trust only fails to correct when the evidence that makes the truth-trackers
accurate is removed. That is exactly what the resource channel does:
starvation forces low-Fisher experiments, so even the would-be-accurate
agents can no longer demonstrate accuracy, and trust loses its anchor.

- At low wrong_frac, the correct majority is large enough that even under
  resource starvation some agents stay accurate; trust corrects; resource
  adds little on top of trust -> sub-additive.
- At high wrong_frac (0.8), the truth-trackers are a small minority AND
  resource starvation removes their accuracy advantage; trust can no longer
  find a reliable anchor; resource starvation now *undercuts* trust's
  correction rather than running parallel to it -> super-additive
  (both-observed 0.197 < additive prediction 0.267).

This is the same mechanism as E14 ("resource starvation removes the
reliable signal trust needs"), now mapped as a function of majority size.
The magnitude here (+0.07) is smaller than E14's cont-substrate result
(+0.31) — the structural substrate's stickier priors and the moderate
c0=0.12 make a gentler flip — but the direction and mechanism match.

## The honest two-channel account (final form for the paper)

Combining E13, E14, E18, E19:

> Lock-in requires closing BOTH evidence channels. The world channel is
> closed by conviction (gamma) or material starvation (resource). The
> social channel is governed by trust, which reallocates precision by
> predictive accuracy: it is **corrective** whenever some agents can still
> track the truth-generating world, and **fails** only when the world
> channel is starved badly enough that no one can. Hence the trust ×
> resource interaction is sub-additive while accurate agents survive
> (trust corrects, resource adds little) and super-additive once
> starvation removes them (resource undercuts trust). The sign flip is a
> function of how many truth-trackers survive — here, of the correct-
> majority fraction.

This is more defensible and more interesting than the original "all three
super-additive" hypothesis, which E18 falsified. The model makes a
non-obvious, mechanistically-explained prediction (the sign flip and its
locus) and the sweep confirms it.

## Status / limits

- Confirms the regime-dependence; direction and mechanism match E14.
- Magnitude is substrate- and severity-dependent (+0.07 structural vs
  +0.31 cont). A 2D map (wrong_frac × c0) would show the full sign-flip
  surface; only the 1D majority cut is run here.
- gamma held off throughout, so this isolates trust × resource. The
  gamma × (trust, resource) interactions from E18 were sub-additive in the
  reliable-majority regime; their regime dependence is untested.

## Files

- `experiments/results/E19/summary.json`
- `experiments/results/E19/regime_flip.png` — left: interaction vs
  wrong_frac (green sub / red super); right: the four 2×2 corners.
