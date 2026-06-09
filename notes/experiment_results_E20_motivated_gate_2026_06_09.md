# E20 — fixing the "meh" motivated-update result (2026-06-09)

E9 found that adding the motivated update WASHED OUT the social-coupling
phase boundary: at any lambda_tilt > 0, mean_qB collapsed to ~0.70 across
the whole q_reliability axis. E20 re-runs the identical sweep with a
corrected update rule and the boundary SURVIVES.

## Root cause of the E9 flatness

The original `apply_value_tilt` implemented:

    q_lambda(theta, c) proportional to q_post(theta, c) * exp(lambda * U(theta))

This is a REWARD ON THE BELIEF STATE — a Boltzmann tilt toward high-utility
paradigms. Applied to the carried-forward posterior every step, it acts as a
constant external field and the belief equilibrates to its fixed point
sigma(lambda * U), independent of evidence history or social coupling. That
fixed attractor is why every lambda > 0 cell collapsed to the utility-
distribution value (~0.70 for a 70/30 pro/anti split). It saturates.

This is NOT what the theory says. Hyland & Albarracin (2025), *On the
Variational Costs of Changing Our Minds*, and this project's own
`theory_brief.md` (line 19) describe motivated reasoning as a COST OF
MIND-CHANGE: "resistance to revising belief proportional to KL-divergence
between the new posterior and its current prior." A cost on CHANGE, not a
reward on STATE.

## The fix: evidence-gate update mode

New `update_mode = "evidence_gate"` in `motivated_step.py`
(`apply_motivated_gate`). On the theta-marginal, per agent, per paradigm:

    delta(theta) = log m_post(theta) - log m_prior(theta)   # this step's move
    w(theta)     = exp(-lambda * relu(U(theta)))            # in (0, 1]
    delta_gated  = delta * w   if delta < 0 else delta       # gate DECREASES only
    m_gated proportional to m_prior * exp(delta_gated)

Confirming movement (toward a valued paradigm) flows at full strength;
disconfirming movement (away from a valued paradigm) is attenuated by
w = exp(-lambda * utility). The honest context conditional q(c | theta) is
preserved, so epistemic inference over context is untouched.

Three properties that distinguish it from the state tilt:
- **Asymmetric**: only resists losing a valued belief; does not reward
  gaining one. So it cannot manufacture a fixed sigma(lambda) attractor.
- **History-dependent**: the gate acts on the actual per-step move, so the
  trajectory and the prior matter — no saturation to a fixed point.
- **A precision mechanism**: gating disconfirming evidence by exp(-lambda*U)
  is exactly the precision-collapse currency of the structural substrate's
  endogenous gamma, but driven by raw utility on the FLAT substrate. This
  slots motivated_step into the three-channel / precision thesis as the
  flat-substrate epistemic channel.

## Result: the phase boundary survives

E20 = E9 config with `update_mode: evidence_gate`. 175 runs
(q_reliability x lambda_tilt x 5 seeds), discrete shift at t=100.

```
E9 (state_tilt) mean_qB             E20 (evidence_gate) mean_qB
qr\lam  0.0   0.5   1.0   2.0  3.0    0.0   0.5   1.0   2.0   3.0
0.55   0.79  0.69  0.69  0.70 0.70   0.79  0.79  0.78  0.77  0.77
0.65   0.73  0.71  0.70  0.70 0.70   0.73  0.71  0.70  0.78  0.78
0.70   0.43  0.73  0.70  0.70 0.70   0.43  0.39  0.29  0.34  0.37
0.75   0.19  0.74  0.70  0.70 0.70   0.19  0.17  0.15  0.28  0.31
0.80   0.10  0.75  0.72  0.70 0.70   0.10  0.10  0.09  0.22  0.30
0.85   0.06  0.73  0.73  0.70 0.70   0.06  0.06  0.06  0.24  0.26
0.90   0.04  0.74  0.74  0.70 0.70   0.04  0.04  0.07  0.25  0.26
```

Social-coupling boundary strength (max-min over q_reliability, per lambda):

```
lambda    E9 state_tilt   E20 evidence_gate
0.0           0.754             0.754   (identical -- both reduce to simple_step)
0.5           0.063             0.750   (E9 collapsed; E20 intact)
1.0           0.042             0.724
2.0           0.003             0.568
3.0           0.001             0.519
```

E9's boundary strength craters to ~0 the moment lambda > 0. E20 keeps a
strong boundary (0.75 -> 0.52) all the way to lambda = 3. The diagonal
lock-in structure — high social coupling traps the population in the
pre-shift paradigm — is preserved under motivated reasoning, which is the
phenomenon we wanted.

## Reading the E20 panel

- **Low q_reliability (0.55-0.65)**: weak social coupling. The population
  tracks truth (~0.78) regardless of lambda — motivated resistance alone
  is not enough to lock in without social reinforcement.
- **High q_reliability (0.75-0.90)**: strong social coupling locks the
  population into the pre-shift paradigm (mean_qB 0.04-0.31). Motivated
  resistance now COMPOUNDS the social lock-in rather than overriding it.
- **lambda 2-3 at high qr**: a modest UPTICK (0.06 -> 0.25) — strong
  motivated resistance lets some pro-truth agents cling to their (correct)
  paradigm against the wrong-paradigm social consensus. Motivated reasoning
  cuts both ways: it entrenches whoever you already are, so the convicted
  pro-truth minority resists the social pull too. This is a genuine,
  interpretable interaction, not a flat artifact.

## Status

- Principled fix faithful to the source theory; NOT a parameter tune. The
  state-tilt mode is retained (`update_mode="state_tilt"`, still the default)
  so the contrast is reproducible and the old E9 result stands.
- 15/15 motivated tests pass (10 legacy + 5 new gate tests: identity at
  lambda=0, confirming-flows-freely, disconfirming-resisted-monotonically,
  context-conditional-preserved, end-to-end run).
- Connects motivated_step to the precision-collapse thesis: it is the
  flat-substrate epistemic channel (utility-driven precision gating),
  parallel to structural_step's structure-driven endogenous gamma.

## Files

- `src/pomdp/motivated_step.py` — `apply_motivated_gate`, `update_mode`
- `experiments/configs/E20_motivated_gate.yaml`
- `experiments/results/E20/{results.json, e9_vs_e20_comparison.png}`
- `scripts/analyze_e9_vs_e20.py`
