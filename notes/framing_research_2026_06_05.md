# Framing notes for `changing_networked_mind.tex` — 2026-06-05

Working notes for the two things Jonas asked Mahault to tighten:

1. **Framing / background** — the "why" and general cohesion (his words)
2. **Replace the dark-energy/GR motivating example** (David's note)

Reference: full paper draft at `paper/changing_networked_mind.tex` (1324 lines).
Deadline pushed to **2026-06-12**.

---

## Part 1: Replacing the dark-energy example

### The problem

The dark-energy / GR / Λ-CDM net is used in §6 (Results) as the main quantitative
demonstration of the two-field decoupling and the Schur banana. David's email:

> "I would lean towards not using dark energy/GR as the main motivating example —
> most people will not be able to follow."

He is right. The IWAI audience is cognitive science / active inference / philosophy
of mind. Almost none of them work with cosmology nets. The supernova distance modulus,
the Friedmann equations, light-bending, BBN — all unfamiliar primitives. The reader
spends cognitive budget decoding the example instead of seeing what the example shows
about the model.

### Candidate replacements ranked

The replacement needs to satisfy:
- Familiar to AIF / cog-sci reviewers (no domain decoding required)
- Has a clear **core / belt** structure (so κ = T·1 can be computed)
- Has a concrete **wanted-true** vs **wanted-false** distinction (so U = T·u is meaningful)
- Has a documented **paradigm shift** with motivated resistance (so lock-in is interpretable)

**Phlogiston → oxygen (Chemical Revolution).** Already implemented in the paper
(§7.6, endogenous γ). Universal high-school-level recognition. Has the exact
structure the paper needs:
- Core: phlogiston as a substance released in combustion; bodies are phlogiston-rich
- Belt: specific combustion / calcination measurements
- Wanted-true: phlogiston exists (Priestley held this until his death in 1804)
- Wanted-false: oxygen is the active reactant
- Disconfirming evidence: metals *gain* weight when burned — a mass-balance violation
  Priestley repeatedly dismissed
- Lock-in well-documented in history of chemistry (Kuhn-loss thesis)

**Behaviorism → cognitivism (Cognitive Revolution).** Resonant with the AIF audience
because active inference itself emerges from this lineage. Has the structure:
- Core: anti-mentalism, S-R laws, observability requirement
- Belt: specific operant-conditioning protocols
- Wanted-true: language acquisition explained by reinforcement (Skinner 1957)
- Wanted-false: innate mental representations
- Disconfirming evidence: Chomsky (1959) on the poverty of the stimulus
- Lock-in: methodological behaviorists retained the core well into the 1980s

*Risk*: looks self-serving in an active inference paper.

**Germ theory (miasma → microbes, Semmelweis 1847, Pasteur).** The strongest
*social-rejection* example, with documented motivated resistance:
- Core: bad-air explanations, social hierarchy of contagion
- Belt: specific outbreak patterns, fevers
- Wanted-true: physicians' clean hands (their professional identity)
- Wanted-false: physicians transmit disease
- Disconfirming evidence: hand-washing reduces puerperal-fever mortality 10x
- Lock-in: Semmelweis ostracized, dies in an asylum 1865 — the textbook case of
  motivated rejection

### Recommendation

**Lead with phlogiston throughout.** Three reasons:

1. **It's already implemented.** The endogenous-γ result (§7.6) runs on phlogiston.
   No new code, no new figures to commission. The paper currently runs *both*
   dark-energy and phlogiston — collapsing to phlogiston cuts ~40 lines from §6
   alone.
2. **It's Kuhn's own canonical example.** Kuhn (1962) uses phlogiston as the central
   illustration of the Chemical Revolution. Anchoring there places the paper in the
   tradition it is extending, not contesting.
3. **The mass-balance violation is concrete.** A reader who never opens the paper
   knows "metal gets heavier when burned" is the smoking gun. The supernova distance
   modulus has no such grip.

**Where dark energy could survive (optional):** as a *brief* appendix worked example,
to show the formalism scales to a 12+ node real scientific net. Half a page. Cite the
banana plot as supporting figure only.

**For the cognitive-revolution example:** mention in §2 (Hidden structure) as a
*second* concrete vignette so the framing isn't tied to chemistry. One paragraph.

---

## Part 2: Tightening the framing — what needs to change in §§1–2

### What works in the current intro

- The two-debts framing (exploration posited, hypothesis space fixed) is sharp
- The "structure of reality is hidden" pivot is the genuine insight
- The six-desiderata list at the end of §2 is good architecture

### What doesn't work

1. **The headline insight is buried.** The single non-obvious move in this paper is
   that **one operator T = (I-A)⁻¹ acts on two sources** to generate both
   conservatism (T·1) and conviction (T·u). This appears in §4 (line 500). By then
   the reader has been told a lot of formalism without knowing why two fields will
   matter. *Fix:* tease this in the abstract (one sentence) and again in §2 (one
   sentence). The reader should know the unifying claim by line 200.

2. **The lit survey in §1 is too dense.** Six literatures named in one paragraph
   (Zollman, O'Connor & Weatherall, Kitcher, Strevens, Jonard, Albarracin, Hyland &
   Albarracin, decentralized science). The reader cannot remember which is which by
   the next page. *Fix:* group into two bins — (i) network epistemology
   (Zollman/O'Connor/Jonard), (ii) division of labour and motivated revision
   (Kitcher/Strevens, Albarracin, Hyland & Albarracin). Three sentences per bin
   maximum. Save the granular citations for the related-work section (§8).

3. **The "why structure learning" leap is unmotivated for the AIF reader.** §2
   argues: hidden structure → structure learning. But the AIF reader's intuition
   is state inference, not structure inference. The reader needs a sentence that
   says: *just as the brain has to learn what its hidden states are, a scientific
   community has to learn what its hypotheses are.* This frames structure learning
   as a natural extension of the AIF apparatus, not a foreign import.

4. **"Carry-over cost" is introduced before the reader needs it.** §3 launches into
   Schur complements at line 264. The intuitive picture (revising one belief forces
   re-reading of dependents) is buried in the math. *Fix:* one paragraph of plain
   prose before the equations. The phlogiston example does this work for free —
   "if you revise phlogiston, you must re-read every combustion experiment that
   used phlogiston as the active substance" is one sentence.

5. **Six desiderata read like a checklist.** They're correct but they don't say why
   this particular shape. *Fix:* fold them into a single closing paragraph of §2 in
   running prose: "We require six things, and the rest of the paper checks them off:
   (1) … (2) … …"

### Proposed reorganization of §§1–2

**§1 — What scalar models miss (renamed from "Models of collective belief revision")**

- *Lead vignette (NEW, ~3–5 sentences):* phlogiston defenders saw metals gain
  weight but did not change their minds. State the puzzle the paper will solve:
  why does evidence land and not move belief.
- *Network epistemology* (one paragraph, two refs):
  Zollman + O'Connor & Weatherall. The transient-diversity story. Acknowledge the
  achievement (slow networks help) and identify the limit (exploration is exogenous,
  menu is fixed).
- *Motivated revision* (one paragraph, two refs):
  Hyland & Albarracin (cost of mind-change), Albarracin (epistemic communities).
  Acknowledge the achievement (revision is asymmetric) and identify the limit (the
  cost is scalar, not structural).
- *Two gaps* (as currently): exploration posited, menu fixed. Hand off to §2.

**§2 — Why paradigms are networks and need structure learning**

- *The Kuhn line* (as currently, tightened): paradigm shapes what is seen,
  investigated, entertained.
- *The hidden-structure pivot* (as currently): the structure of the world is
  itself unknown.
- *The AIF bridge (NEW, one sentence):* AIF readers already accept that hidden
  states must be inferred; the move here is to accept that the hidden *structure
  among* those states must be inferred too.
- *Why two fields, one operator (NEW, one paragraph, ~5 sentences):*
  Tease the §4 unification. Conservatism = T·1 (cost of revising = mass of
  descendants). Conviction = T·u (utility propagated through the same
  dependencies). The same operator on two sources gives two fields that are
  linearly independent. This is the single non-obvious move; advertise it.
- *Desiderata, in prose:* fold the bullet list into one closing paragraph.

This reorg costs ~half a page in net length but front-loads the two things the
reader actually has to remember: (a) a paradigm is a network, and (b) the same
propagation operator generates cost and value.

---

## Part 3: References to add (for unfamiliar concepts)

David's specific concern: "a lot of the concepts will not be familiar to reviewers
in active inference, so we'd have to make sure they are clearly explained, ideally
in the main text or else in the appendix."

The concepts most likely to trip an AIF reviewer:

| Concept | Where it's used | Fix |
|---------|----------------|-----|
| **Bayesian model reduction (BMR)** | §3.2 | Already cite Friston 2011, 2018. Add one-line gloss: "BMR scores the evidence change of any prior-only edit against an already-inverted posterior in closed form (the Savage-Dickey ratio)." |
| **Schur complement** | §3.1 (Eq. 5) | Add one sentence: "The Schur complement is the residual coupling induced among a node's neighbours when the node is marginalized out — incommensurability written as block matrix algebra." |
| **ARD (automatic relevance determination)** | §3.2 | Currently cited MacKay 1992, Neal 1996, Tipping 2001. Good. Add a single-line gloss: "An edge with prior precision α_e ∈ [0,∞] is on at α=0 and pruned at α=∞; the data choose where on the cube to land." |
| **Propagation operator T = (I-A)⁻¹** | §3.1 (Eq. 6) | This is the paper's most important object. Currently introduced via the Neumann series with no plain-English gloss. Add: "T_iw is the precision-weighted total influence of i on its descendant w — every path summed." |
| **Causal Attitude Network (CAN)** | §4 | Cite Dalege 2016 + Dalege 2017 (Network Analysis on Attitudes) + Dalege 2019 (Network Perspective on Attitude Strength). Already partly in. Add one-line: "CAN treats attitudes as networks of evaluative reactions where connectivity *is* attitude strength — the empirical posture this paper formalizes." |
| **Free-energy functional with utility tilt** | §4 (Eq. 7-8) | This is the Hyland & Albarracin (2025) move. Already cited. Good. |
| **Susceptibility / cumulants** | §4 (Eq. 9-10) | These will be unfamiliar to most AIF readers. Add: "The first cumulant Z′(λ) is the mean — where the tilted ledger lands; the second Z″(λ) is the variance — how strongly evidence can still move it. A gated disconfirming channel contributes zero variance and so cannot move the agent regardless of evidence strength." |
| **Indian Buffet Process** | bibliography only | Cited but never used in text. Either use it in §3.2 as the proper prior on expansion, or remove from refs. |
| **Partial information decomposition (PID)** | bibliography only | Same. Either use or remove. |

The fixes are mostly one-sentence glosses that fit in line. Total addition: ~20 lines.

---

## Part 4: Suggested writing pass order

Given the deadline (2026-06-12, ~7 days) and Jonas's request to focus on framing/
background:

1. **Day 1 (today): scope agreement.** Reply to Jonas. Confirm phlogiston as primary
   example, propose the §§1–2 reorganization, ask if he wants me to rewrite or just
   redline. **Critical: get Jonas's signoff before rewriting.**
2. **Day 2–3:** Rewrite §§1–2 per the structure above. Length-neutral or shorter.
3. **Day 3:** Scrub §§6 (Results) for dark-energy uses, replace with phlogiston
   references throughout. Move §6.1 (two fields) and §6.2 (Schur banana) to use
   phlogiston, or relegate dark-energy to a half-page appendix.
4. **Day 4–5:** Add the one-line glosses for unfamiliar concepts. Spot-check for
   readability with a non-AIF reader if possible.
5. **Day 6:** Page-count + cuts. The current draft is over 12 pages. Identify
   sections that can be cut without losing the argumentative arc.
6. **Day 7 (2026-06-12):** Final pass, exiftool the metadata, submit.

---

---

## Part 5: Slotting in Mahault's E1–E8 experiments

The structure-learning paper currently uses Jonas's `src/structural/` results
(dark-energy net, phlogiston endogenous γ, 2×2 lock-in demonstration, forgetting
sweep). Mahault's experiments live on a different substrate: `src/pomdp/simple_step.py`
and `src/pomdp/cont_step.py` (multi-level context model, K×C joint state, no BMR).

The question is which experiments earn a place in 12 pages, and how to frame the
two substrates side-by-side.

### Cross-walk: Mahault's experiments vs. Jonas's sections

| Mahault | Substrate | What it shows | Closest Jonas section | Verdict |
|---------|-----------|---------------|----------------------|---------|
| E1 | simple_step (no social) | Individual-level theory-ladenness gradient | — | **Drop** — baseline only, not load-bearing |
| E2 | simple_step (binary c) | (qr, er) phase diagram, sharp boundary | §6.2 lock-in 2×2 | **Drop** — superseded by E7 |
| E3 | simple_step (binary c) | Slow drift, similar boundary | §6.3 forgetting | **Drop** — different mechanism, redundant |
| E4 | cont_step (continuous λ) | (qr, λ) phase diagram | — | **Maybe** — Hyland-style λ at multi-agent scale |
| E6 | cont_step + resources | Resource ablation (cont-λ) | §6.3 forgetting | **Drop** — superseded by E8 |
| E7 | simple_step (C=5) | (qr, er) **principled** phase diagram | §6.2 lock-in 2×2 | **KEEP** — strongest add |
| E8 | simple_step (C=5) + resources | Resource ablation, principled | §6.3 forgetting | **KEEP** — orthogonal mechanism |

### The pitch: E7 + E8 as a "cross-substrate" §6.X

Add a single subsection to §6 (Results) titled something like:

> **§6.X — The lock-in regime is robust to substrate**

with two figures (E7 phase diagram, E8 ablation curve) and ~half a page of text
making three points:

1. **E7 is a quantitative regime map.** Jonas's §6.2 lock-in 2×2 demonstrates that
   lock-in can occur at one operating point with hand-chosen weights. E7 sweeps
   the boundary across 56 points × 10 seeds and shows the regime structure
   (diagonal in (qr, eps_resolve)). This is the seed-averaged statistical evidence
   the 2×2 cannot provide.

2. **E8 confirms the resource mechanism on a simpler model.** Jonas's forgetting
   result shows that resource accumulation produces ratchet-like lock-in. E8
   shows that resource *gating* (per-agent affordability of informative
   experiments) produces a complementary form of irreversibility: a lock-in
   *floor* at ~0.08 regardless of paradigm inertia. Same qualitative
   phenomenon — different operationalization.

3. **The substrate independence is the point.** The hierarchical-context model
   (multi-level c, joint Bayes on (θ,c), no BMR) reproduces the lock-in regime
   that the structure-learning model produces with BMR. Lock-in is not an
   artifact of either machinery: it's a consequence of social coupling + bounded
   revision cost + finite evidence per step. This *strengthens* the paper's
   claim by showing the result isn't substrate-specific.

### Why this framing works

David's email said:

> "I think the central idea and framing is interesting — I didn't have time to check
> the math, though the core seems plausible. The text + figures would have to be
> heavily polished (and cut down to 12 pages)..."

He's saying: tight argumentative arc, polish prose, watch the page budget. Adding
two figures + half a page costs ~1 page. To stay at 12 pages, cut from:
- §6.5 (coarse net resolves slow) — confirms the obvious, can move to appendix
- §6.7 (staircase not recovered) — honest negative, but ~1 page; can be tightened to ~3 paragraphs

Net: +1 page (cross-substrate) − 0.5 page (coarse net) − 0.7 page (staircase) = under budget.

### Why this framing doesn't undermine Jonas

Jonas's pivot to structure learning is the paper's main contribution. Mahault's
E7/E8 don't compete — they reinforce. The pitch is *not* "here's a better model"
but "the same phenomenology shows up in a simpler model, which means the
structure-learning story is the *explanation*, not the artifact."

This is consistent with the ablation-build-up framing David and Jonas agreed on
in the May 28 call: simplest model first, then the richer model adds something
specific. Mahault's E7/E8 plays the role of the simpler model.

### What to ask Jonas for

In the reply to Jonas's "tighten framing and background" ask:

> "Two related questions: (1) Would you want E7 (the (qr, eps_resolve) phase
> diagram from `simple_step.py` with C=5) and E8 (resource ablation) added to
> §6 as a cross-substrate robustness result? They show the same lock-in regime
> as your 2×2 but as a quantitative regime map across 560 seeded runs.
> (2) If yes, do you want me to draft that subsection, or should we keep the
> two substrates separate (yours as the main result, mine as a brief footnote
> or appendix)?"

If Jonas says no — drop the experiments, focus on framing only. If yes — ~1 day
of writing.

### Honest fallback if E7/E8 don't fit

If page budget or framing reasons rule out a subsection, the minimum mention is
one footnote in §6.2 (lock-in) and one in §6.3 (forgetting):

> "The lock-in regime here is also recovered, with the same diagonal structure
> in (social coupling, paradigm inertia), by a simpler hierarchical-context
> model that does not use BMR (see `src/pomdp/simple_step.py` in the
> supplementary code, sweep E7)."

Two-line cost, preserves attribution, doesn't force a figure.

---

---

## Part 6: Deep research findings (2026-06-05)

A multi-agent fan-out research workflow (106 agents, 24 sources, 25 adversarially
verified claims) returned a sharper diagnosis than the initial pass.

### The sharpest framing (verified)

Canonical network epistemology (Zollman, Bala–Goyal, PolyGraphs, O'Connor &
Weatherall) **models beliefs as scalar credences in [0,1] over a fixed
proposition menu**, and **treats persistent disagreement as a normative
failure** ("ignorance of the community") rather than as endogenous paradigm
dynamics. PolyGraphs (Devlin et al., Nature HSSC 2024) makes this verbatim:

> "modeled as having a degree of belief, or credence, between 0 and 1 in the
> proposition that B is better"

This is the cleanest contrast class for §1. The paper's contribution is to
replace **scalar-credence-over-fixed-menu** with **Bayesian-network-over-
learned-structure** — a single move from which the dual-field formalism,
endogenous lock-in, motivated persistence, and exploration-as-expansion all
fall out.

### Three load-bearing pillars for the first two pages

| Citation | Function | Verified claim |
|----------|----------|----------------|
| **Albarracin et al. 2022 (Entropy)** | Precision-based lock-in on flat representation | "Agents tend to sample information in order to justify their own view... once they reach a certain level of certainty, it becomes very difficult to get them to change their views" |
| **Friston et al. 2023 (Neurosci Biobehav Rev)** + **Tschantz et al. 2025 (Entropy "As One and Many")** | Individual → group-level generative model bridge | "non-trivial relationship between the generative models of individual agents and the group-level agent they constitute" — license + warning |
| **Ullman & Tenenbaum 2020 (Annual Review Dev Psych)** | Structured representations + stochastic search legitimize exploration-as-expansion | "Symbolic representations are better suited to capturing children's intuitive theories but give rise to a harder learning problem, which can only be solved by exploratory search" |

Plus one mechanism reference:

- **Smith et al. 2022 (PLOS ONE)** — BMR operationalized in agents, explicitly
  "operates in the absence of further sensory experience... at the slowest
  timescale, structure learning proceeds by redistributing the products of
  learning to minimise model complexity." This directly grounds the forgetting/
  earned-lock-in mechanism in §3.3.

### Draft first paragraph (210 words, designed for LNCS page 1 column 1)

> "When the chemical community released phlogiston, it did not revise a single
> credence — it rewrote a dependency network in which combustion, calcination,
> respiration, and the theory of acids were all load-bearing on a hub that had
> to go. Standard network-epistemology models of scientific communities cannot
> represent this: agents are equipped with a scalar credence in [0,1] over a
> fixed proposition [PolyGraphs 2024; O'Connor & Weatherall 2018], so persistent
> disagreement appears only as collective failure rather than as the cost of
> revising a connected web of commitments. Active-inference accounts of
> epistemic communities supply an endogenous precision-based mechanism for
> lock-in [Albarracin et al. 2022] and a formal route from individual to
> group-level generative models [Friston et al. 2023; Tschantz et al. 2025],
> but they too leave belief structure fixed. We close this gap by treating a
> paradigm as a Bayesian network of interdependent commitments and identifying
> its dynamics with structure learning — Bayesian model expansion and reduction
> [Friston 2018; Smith et al. 2022; Ullman & Tenenbaum 2020] — over a hidden
> world. A single propagation operator T = (I−A)⁻¹ acts on two source vectors
> to generate two distinct fields on the same network: conservatism κ = T·1,
> the carry-over cost of revising a commitment, and conviction U = T·u,
> intrinsic utility propagated through the same dependencies. Lock-in,
> motivated persistence, and Kuhnian transitions all fall out of how these two
> fields interact."

### Citation discipline

**First two pages (§§1–2):**
Albarracin 2022 · Friston/Heins 2023 · Tschantz 2025 · Smith 2022 · Ullman &
Tenenbaum 2020 · PolyGraphs (Devlin) 2024 · O'Connor & Weatherall 2018

**Related work (§7):**
Michelini 2023 · Albarracin et al. 2024 (Shared Protentions) · Ullman–Goodman–
Tenenbaum 2010/2012 · Kemp & Tenenbaum 2008 · Hyland & Albarracin 2025 ·
Dalege CAN 2016/2019 · Lake et al. 2018 · Friston 2011 (BMR primary) · Friston
2018 (BMR review) · classic Zollman / Bala–Goyal

### Three framings that DON'T survive verification — avoid these

The adversarial verification killed 7 plausible-sounding claims. Three of them
matter for what NOT to write:

1. **DO NOT claim "BMR changes priors rather than posteriors."** Verification
   3-0 refuted. Friston/Heins 2023's framing is more subtle. Write the BMR ratio
   as the Savage–Dickey density ratio without making a priors-vs-posteriors
   slogan of it.

2. **DO NOT claim "a collective with a group-level Markov blanket itself
   constitutes a larger active inference agent."** Refuted 3-0 against Tschantz
   et al. 2025. This is a tempting move but the source does not support the
   scale-free reading. Use "individual-to-collective bridge" framing instead.

3. **DO NOT claim "Ullman & Tenenbaum unify scientific theory change and
   individual conceptual development under one model-building framework."**
   Refuted 1-2. They use the same Bayesian apparatus across both settings but
   do not explicitly claim unification — cite them for the structured-search
   licensing only.

### Adversarial caveat: which framings will land, which will not

**Will land for the IWAI audience:**
- "Scalar-credence-over-fixed-menu cannot represent X" — they think in
  generative models, this is a natural complaint
- "Lock-in is derived rather than imposed" — Albarracin 2022 is in their
  library
- "Exploration = model expansion" — provided each technical term gets one
  sentence of motivation

**Won't land without help:**
- **Schur complement** — needs one sentence: "marginalizing out latent
  dependencies leaves a connected covariance on the observed nodes — the rest
  of the formalism is bookkeeping over that marginal"
- **Indian Buffet Process / ARD / graphical lasso** — name the
  cognitive-science vocabulary first (Ullman–Tenenbaum stochastic search over
  symbolic structures), then introduce IBP/ARD as the technical machinery
- **Dalege CAN** — unfamiliar to AIF; mention only in related work as the
  empirical posture, not in §1–2 framing

### Recommended §1–2 ordering (5 moves)

1. **Open with phlogiston as a dependency-network puzzle.** One paragraph,
   ~5 sentences. Use the draft above.
2. **Name the gap in canonical network epistemology.** Cite PolyGraphs 2024
   + O'Connor & Weatherall 2018 + Michelini 2023 (the "more data → more
   polarization" result that the paper subsumes via T·u flattening Z″(λ)).
3. **Credit the active-inference precedent and name what it doesn't do.**
   Cite Albarracin 2022 for precision-based lock-in; Shared Protentions 2024
   for group intentionality. State that both leave belief structure fixed.
4. **Introduce structure learning as the missing level.** Cite Ullman &
   Tenenbaum 2020 + Smith 2022 (BMR). One sentence motivating each technical
   term that will appear in §3.
5. **State the dual-field contribution as the smallest formal addition that
   recovers all four phenomena.** Lock-in, motivated persistence, Kuhnian
   transitions, exploration. Hand off to §3.

### Open questions the research could not close

1. **No 2024–2026 primary source explicitly applies BMR to collective
   cognition or paradigm shifts.** This is either a real gap (the paper would
   be first) or an artifact of search coverage. **Recommended action:** a
   targeted Google Scholar pass on "Bayesian model reduction collective" /
   "structure learning paradigm shift" for 2024–2026 before submission.
2. **No primary historian-of-science source explicitly characterizes
   phlogiston as a hub node in a dependency network.** The framing is the
   authors' analytical move on a well-known case — should be phrased as such
   ("we read the phlogiston case as ...") rather than as established history.
3. **PolyGraphs (Devlin et al. 2024) is on Nature HSSC, unfamiliar to IWAI.**
   Pair with the Stanford Encyclopedia of Philosophy entry on social/formal
   epistemology to soften the unfamiliarity for the audience.

---

## Sources consulted

- [Cognitive Revolution — Wikipedia](https://en.wikipedia.org/wiki/Cognitive_revolution)
- [Bolhuis 2023 — Language and learning: the cognitive revolution at 60-odd](https://onlinelibrary.wiley.com/doi/10.1111/brv.12936)
- [Kuhn-loss thesis and the case of phlogiston theory](https://www.academia.edu/1573641/The_Kuhn_loss_thesis_and_the_case_of_phlogiston_theory)
- [Germ theory of disease — from miasmas to germs](https://www.researchgate.net/publication/223957556_From_miasmas_to_germs_A_historical_approach_to_theories_of_infectious_disease_transmission)
- [Dalege 2016 — Causal Attitude Network model (in ref list)](https://psycnet.apa.org/doiLanding?doi=10.1037%2Fa0039802)
- [Dalege 2019 — A Network Perspective on Attitude Strength](https://journals.sagepub.com/doi/full/10.1177/1948550618781062)
- [Kemp & Tenenbaum 2008 — The discovery of structural form (already in ref list)](https://www.pnas.org/doi/10.1073/pnas.0802631105)
- [Albarracin et al. 2022 — Epistemic Communities under Active Inference](https://www.mdpi.com/1099-4300/24/4/476)
- [Hyland & Albarracin 2025 — On the Variational Costs of Changing Our Minds](https://arxiv.org/abs/2509.17957)

### Added from deep research (2026-06-05)

- [Devlin et al. 2024 — PolyGraphs reflection (Nature HSSC)](https://www.nature.com/articles/s41599-024-02619-z)
- [Michelini et al. 2023 — JASSS, polarization from disagreement on Bayes factors](https://www.jasss.org/26/4/5.html)
- [Friston, Parr, Heins et al. 2023 — Federated inference and belief sharing (Neurosci Biobehav Rev)](https://www.sciencedirect.com/science/article/pii/S0149763423004694)
- [Tschantz, Heins, Da Costa et al. 2025 — "As One and Many" (Entropy)](https://www.mdpi.com/1099-4300/27/2/143)
- [Smith et al. 2022 — BMR operationalized in agents (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277199)
- [Ullman & Tenenbaum 2020 — Bayesian models of conceptual development (Annual Review Dev Psych)](https://klab.tch.harvard.edu/academia/classes/BAI/pdfs/UllmanEtAl_AnnRevPsych2020.pdf)
- [Ullman, Goodman & Tenenbaum 2010/2012 — Theory learning as stochastic search](https://cocolab.stanford.edu/papers/UllmanEtAl2010-Cogsci.pdf)
- [Albarracin et al. 2024 — Shared Protentions (Entropy)](https://www.mdpi.com/1099-4300/26/4/303)
