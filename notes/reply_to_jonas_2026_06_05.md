# Reply to Jonas — draft (2026-06-05)

Subject: Re: Meeting today — framing rewrite ready for your eyes

---

Hey Jonas,

I've got a rewrite of §§1-2 ready for you to look at, plus the bib entries
it needs. Both are on my fork branch
(`feat/simplified-theory-laden-model`) under `notes/`:

- `notes/sections_1_2_draft_2026_06_05.tex` — drop-in replacement
- `notes/bib_entries_to_add.bib` — 16 new entries with verified DOIs

The rewrite is ~4 pages, matching the current §§1-2 footprint. It opens with
phlogiston (your endogenous-γ example, lifted to the front), gives an
AIF-reader bridge into structure learning, names the BMR-for-collective-
cognition gap as the genuine novelty, and teases the T·1 / T·u unification
in §2 so the reader knows where the paper is going before the math hits in
§3.

**Three things I need to flag from the background work,** because they
affect what we can claim and how:

1. **Holmes (2000, Isis 91:4)** argues the Chemical Revolution was rival
   research programs, not paradigm replacement. Using phlogiston as a
   "clean Kuhnian shift" without engaging him reads as naive to historians
   of science. I engaged him in the opening — turns out it actually makes
   the structural framing stronger (rival programs IS what the dependency
   network formalizes).

2. **Thagard's ECHO (1989, 1990) is the closest structural precedent for
   treating phlogiston formally**, but ECHO uses connectionist
   excitatory/inhibitory links — NOT Bayesian DAGs with Schur complements.
   I cited Thagard as the conceptual-change precedent and explicitly
   distinguished the formalism. Worth a sanity-check from you.

3. **The BMR-for-collective-cognition gap is real** — Smith 2022, Friston
   2016 PEB, Friston 2024 collective intelligence, and Kaufmann/Gupta/Taylor
   2021 either work on independent agents, between-subject neuroimaging, or
   propose shared generative models without operationalizing BMR. The paper
   is genuinely first. I worded this as "does not operationalize BMR" rather
   than "does not invoke BMR" because Friston 2024 does mention model
   selection.

(I ran a hostile two-pass research workflow to ground everything; full
verification log is in `notes/framing_research_2026_06_05.md` if you want
to see what survived adversarial verification and what didn't.)

**Three scope questions before I keep going:**

(a) **Does the §§1-2 framing land for you?** If yes I'll proceed. If you'd
like surgery, redline freely — happy to iterate.

(b) **Should I touch the rest of the paper for cohesion?** Specifically:
   - **Abstract** — currently leads with dark energy / GR / SN banana,
     needs to match the new opening
   - **§7 results dark-energy uses** (§7.1 two fields, §7.2 Schur banana)
     — David asked us to cut dark-energy as the main example. I can swap
     to phlogiston (you already have it implemented) or move dark-energy
     to a half-page appendix
   - **One-line glosses for unfamiliar concepts** (BMR, Schur complement,
     ARD, propagation operator T, CAN) — David flagged that AIF reviewers
     won't have all this vocabulary
   - **§8 related work** — currently overlaps §1; would retarget to
     engage the new citations without duplicating
   - **§9 discussion** — quick pass so the closing echoes the opening

(c) **My E7/E8 experiments** — I have a (q_reliability, eps_resolve) phase
diagram on the simpler hierarchical-context substrate (multi-level c, no
BMR), with 560 seeded runs, plus a resource ablation. The boundary
qualitatively matches your §6.2 lock-in 2×2 but as a quantitative regime
map. Would you want this added as a "robustness to substrate" subsection
in §6, or kept separate as a footnote/appendix pointer? I'm fine either
way — your call.

I can have everything else done by Saturday evening if you give me the
green light. David's new deadline is Jun 12 so we've got runway, but I'd
like to converge on framing before I spend cycles on the rest.

Hope the retreat is going well!

Mahault
