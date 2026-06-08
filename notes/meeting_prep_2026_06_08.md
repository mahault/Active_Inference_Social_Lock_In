# Meeting prep — 2026-06-08

Tonight, 21:30 UK / 16:30 EDT (David's slot; David flagged he can't do
Jonas's original time). Jonas, David, Mahault.

Jonas's agenda: "tighten up the story and frame, maybe something like
asking what the most exciting thing within the paper is for you and how we
can bring that forth."

David's deadline now Jun 12 (4 days).

---

## State of play

- **§§1-2 rewrite** — Jonas green-lit Saturday morning ("the reframe works
  for me and most of the moves make the thing better, not just different")
- **Abstract** — still leads with dark-energy/SN banana, needs to match
- **§7 results** — still uses dark-energy net in §7.1 and §7.2
- **Concept glosses** — not added yet
- **§8 related work, §9 discussion** — not retargeted yet
- **Page-count** — currently > 12, no cuts made
- **E7/E8 experiments** — Jonas hasn't answered the inclusion question

So 4 of the 5 cohesion items from my Friday reply are open. The meeting
is the right place to decide which ones stay, which ones cut, in what
order.

---

## "What's most exciting in the paper" — options to pitch

Jonas's question is structurally a prioritization question. Whatever I
name as most exciting is what we'll bring forward and protect in the cuts.
Three honest candidates, in order of how distinctive each is for the
paper:

### Option A — Conviction as a second field that silences its own disconfirming channels (§7.6 endogenous γ)

This is the sharpest *prediction* in the paper. It says: a community that
values the incumbent doesn't need a curator to suppress dissent — the
conviction field, propagated through the same dependencies that carry the
revision cost, drives the disconfirming sensory precisions toward zero by
itself. The deaf-but-honest control then proves the stall is the gating
and not absent evidence.

**Why it's exciting for me specifically:** it's the falsifiable face of
motivated reasoning — the formal counterpart of "what gets through filters
into a researcher's working model." Connects directly to my prior work on
epistemic communities (Albarracin 2022) and to the Hyland & Albarracin
variational-cost-of-mind-change framework. It is the piece that
distinguishes this paper from pure structure-learning accounts (Ullman-
Tenenbaum) which don't carry a value field at all.

**What it asks the audience to swallow:** that "lock-in" is not noise, not
echo-chamber dynamics, not exogenous censorship — it is endogenous
self-silencing. That's the Kuhnian insight given a precise mathematical
face.

### Option B — One operator on two sources (§4 — T·1 and T·u)

This is the sharpest *theoretical move* in the paper. The same propagation
operator T = (I − A)⁻¹ generates conservatism (T·1, the cost-of-revision
field) and conviction (T·u, the value field), and they are linearly
independent unless u ∝ 1. The whole phenomenology — central-but-neutral,
cheap-but-cherished, wanted-false-but-load-bearing — falls out of one
operator acting on two source vectors.

**Why it's exciting:** it's parsimonious. Most accounts of cost-of-mind-
change and motivated reasoning posit two separate mechanisms. Here they
are two readings of one network through one operator. Elegant in a way
reviewers reward.

**What it asks the audience to swallow:** that the Schur-complement / 
propagation-operator algebra is worth their time. The math is heavier than
the average IWAI reader expects.

### Option C — Lock-in is derived, not imposed (the forgetting threshold)

This is the *honesty* of the paper. Without forgetting, every positive
conviction eventually dominates evidence — lock-in is automatic, a
degenerate ratchet. With forgetting, lock-in becomes a conviction
threshold one must clear: motivated persistence is *earned*, not posited.
This converts a stylistic "bias" into a measurable parameter.

**Why it's exciting:** it answers David's old skepticism about whether
information dynamics alone are sufficient. The answer turns out to be:
they're sufficient for re-tracking, but lock-in needs the value field
above a threshold, and the threshold itself is set by how fast the
community forgets.

**What it asks the audience to swallow:** that forgetting is a load-
bearing parameter, not a regularization detail.

---

## My recommendation: lead with Option A, motivate with B, defend with C

Option A is the headline that should be in the abstract, the discussion,
and the first sentence of related work. Option B is the *machinery* that
makes A non-trivial (one operator, not two mechanisms). Option C is the
adversarial defense (it's not magic; it requires forgetting to be
non-trivial).

If we're cutting to 12 pages, the cuts should fall on whatever doesn't
serve A:

- **§7.3 coarse net** (slow resolution) — interesting but doesn't earn its
  keep for A. Can move to appendix or cut.
- **§7.5 conservatism-gated revision rate** (the staircase negative) —
  honest negative but ~1 page. Can tighten to ~3 paragraphs.
- **§7.4 exploration-as-Poisson-rate** — serves A indirectly (gating
  exploration matters for what gets challenged), but could compress.
- **Dark-energy net** — serves A only indirectly. Phlogiston serves A
  directly (Priestley silencing mass-balance evidence IS the endogenous γ
  story). David's already asked us to cut dark-energy as the main example;
  Option A gives us positive reason to do it, not just David's preference.

---

## Questions to bring to the meeting

1. **Does Option A land for both of you as the headline?** Or does Jonas
   want to lead with B (the operator unification) instead?

2. **§7 dark-energy cut — appendix or full removal?** Phlogiston already
   carries the endogenous-γ result; dark-energy is illustrative but
   redundant if we lead with A.

3. **E7/E8 experiments — fit or not?** If A is the headline, E7 is
   evidence that the lock-in regime is real across substrates (helpful);
   E8 is evidence that resources create irreversibility on top
   (orthogonal to A, less useful for the headline framing). I'd offer E7
   as a §6.X subsection and drop E8 unless we're explicitly making the
   resource argument.

4. **Page-cut priority:** are we cutting from §7 (results) or §3-5 (math)?
   I'd argue keep all the math, cut from §7 by collapsing §7.3 and §7.5.

5. **Division of labour for the remaining 4 days:**
   - Abstract + opening discussion paragraph — me
   - §7 dark-energy → phlogiston swap — Jonas (he has the code)
   - Concept glosses (BMR, Schur, ARD, T) — Jonas (math owner) with me
     reviewing for AIF-reader accessibility
   - §8 related work retarget — me
   - §9 discussion echo — me (one pass)
   - Page-cut decisions — all three on the call

---

## Reply to David's "21:30UK onwards" email

> Hey both — 21:30 UK works for me. See you tonight.
>
> Mahault
