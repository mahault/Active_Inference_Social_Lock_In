# One Currency, Two Channels

**Precision Collapse and the Conditions for Paradigm Lock-In in Multi-Agent Active Inference**

Mahault Albarracin · working draft, June 2026

This folder holds the standalone follow-up paper to the IWAI 2026 structural-BMR
work. The LaTeX source is [`main.tex`](main.tex); the bibliography is
[`refs.bib`](refs.bib). This README is a readable rendering so the argument and
figures can be read without compiling. To build the PDF, see
[Building](#building) at the bottom, or drop the folder into Overleaf.

> **Updated 6 Oct 2026.** The results below are 20-seed reruns with each agent
> observing at the experiment it chose (see
> `notes/results_2026-10-05_reruns_and_pluralist.md`). `main.tex` and the copies
> in `figures/` still show the June numbers and need updating.

---

## Abstract

Scientific communities sometimes hold an outdated paradigm long after the
evidence against it has arrived. I model this as the collapse of effective
precision on disconfirming evidence, and I ask which mechanisms can produce that
collapse in a population of active-inference agents who share a world. Three
candidate mechanisms are studied on a common substrate: conviction that
attenuates an agent's own observations, material resource limits that price
informative experiments out of reach, and learned trust that reweights
neighbours' reports. The mechanisms act on a single quantity, the precision that
gates belief updating, but they do not act on the same channel. Conviction and
resources both close the agent's *world* channel, its own observations, while
trust governs the *social* channel, what it hears from others. Lock-in needs
both channels closed at once. Trust on its own does not lock a community in: it
reallocates precision across sources by predictive accuracy, so it pulls a
misled minority toward the truth whenever some agents can still track the world,
and it amplifies error only once the world channel has already been starved.
This makes the interaction between trust and resources change sign with regime,
sub-additive while a reliable signal survives and super-additive once it is
gone. I map the boundary as a function of how many truth-tracking agents remain,
and I show the material channel reproduces hysteresis, the signature of path
dependence from the economics of increasing returns.

---

## The three claims

**Precision is the common currency, and lock-in is a two-channel condition.**
Conviction, resources, and trust each modulate the precision on disconfirming
evidence. Conviction and resources close the world channel; trust governs the
social channel. A community locks in only when both are shut.

**Trust corrects more often than it traps.** Trust follows predictive accuracy,
not headcount, so it follows whoever tracks the world. It rescues a misled
minority while accurate agents survive and amplifies error only after starvation
removes them. So the trust-by-resource interaction flips sign with regime.

**The material channel is path dependence.** The cost barrier on informative
experiments is an increasing-returns switching cost (Arthur; David). It produces
hysteresis: a population adapts to a new paradigm, spends its budget, and then
fails to revert when the truth reverses.

---

## Key results

### The sign flip, mapped (E19)

Wrong-bloc final belief in the truth, with gamma off, 20 seeds. Trust alone
rescues a misled bloc even when it is the majority (0.985 at 50% misled, 0.85 at
65%). With resources also on, that rescue disappears and the bloc ends near the
resources-only level.

| wrong fraction | 0.20 | 0.35 | 0.50 | 0.65 | 0.80 |
|----------------|:----:|:----:|:----:|:----:|:----:|
| neither        | 0.985 | 0.886 | 0.601 | 0.213 | 0.052 |
| trust only     | 0.985 | 0.985 | 0.985 | 0.852 | 0.346 |
| resources only | 0.960 | 0.799 | 0.503 | 0.152 | 0.032 |
| both           | 0.984 | 0.849 | 0.592 | 0.174 | 0.039 |
| interaction    | −0.02 | +0.05 | **+0.30** | **+0.62** | **+0.29** |

Interaction = additive prediction minus observed (positive = the pair is worse
than its parts). 95% bootstrap intervals: [0.15, 0.45], [0.47, 0.75],
[0.14, 0.45] for the last three; Wilcoxon p < 0.003 for each. Below 0.35 misled
the two channels do not compound. This is the cleanest evidence for the
two-channel reading: trust corrects only while the world channel is open.

### Trust + resources on the cont substrate (E14)

The interaction is −0.03 [−0.07, 0.00], not significant: on this substrate
the two effects simply add up.

|            | resources off | resources on |
|------------|:-------------:|:------------:|
| trust off  | 0.992         | 0.817        |
| trust on   | 0.992         | 0.851        |

E14 and E19 differ in substrate and in how many agents are misled (E14 has a
30% misled bloc, where E19 also finds no interaction).

### Cross-channel factorial (E18)

A 2×2×2 factorial over conviction (γ), trust, and resources, 20 seeds. Trust
raises wrong-bloc convergence from 0.91 to 0.985 when resources are off, but adds
only +0.05 [0.01, 0.12] when resources are on (0.83 to 0.88). The trust ×
resource interaction is +0.02 [−0.07, 0.11], not significant at a 30% misled
bloc, consistent with E19 at that fraction.

### Material lock-in is path dependence, with hysteresis (E16, E17)

The cost barrier is an increasing-returns switching cost. A population adapts in
phase 1, drains its budget, and cannot afford to revert in phase 2 even though
phase 2 returns it to the truth it started from.

![E16 hysteresis](figures/E16_hysteresis.png)

The phase diagram over inflow and cost scale shows three regimes
(adapt-and-revert, hysteresis, blocked); hysteresis is a band, not a point.

![E17 phase diagram](figures/E17_phase_diagram.png)

### The motivated update as a cost of mind-change (E20)

The original rule put a reward on the belief state and saturated, washing out
the social-coupling boundary. The corrected rule gates disconfirming movement
(a cost on change, per Hyland & Albarracin 2025) and the boundary survives.

![E20 motivated gate](figures/E20_motivated_gate.png)

---

## Status

Not submitted to AAAI 2027. The headline is now the E19 sign flip and the
hysteresis results. To do:
rewrite the abstract, results and figures in `main.tex` around the updated
numbers; add the 2D sign-flip surface (misled fraction × cost scale); re-test
E17 with more than 3 seeds. Venue to decide.

The affiliation line in `main.tex` is a placeholder; set it before submission.

## Citation verification log

Every reference in `refs.bib` was checked against an authoritative source before
inclusion. Two reported attributions were corrected during verification:

| Key | Verified against | Note |
|-----|------------------|------|
| `arthur1994IncreasingReturns` | press.umich.edu | Univ. of Michigan Press, 1994 ✓ |
| `david1985QWERTY` | JSTOR 1805621 / RePEc | AER 75(2):332–337, 1985 ✓ (no DOI; P&P issue) |
| `liederGriffiths2020ResourceRational` | Cambridge Core | BBS 43:e1 (corrected from e19) |
| `zollman2010TransientDiversity` | Springer | Erkenntnis 72(1):17–35 ✓ |
| `oconnorWeatherall2019Misinformation` | PhilPapers | Yale UP, 2019 ✓ |
| `friston2024Ecosystems` | SAGE | Collective Intelligence 3(1); Albarracin is co-author ✓ |
| `friston2023Federated` | Elsevier / PubMed | Neurosci. Biobehav. Rev. 156:105500 ✓ |
| `catal2024BeliefSharing` | arXiv:2407.02465 | **Catal** et al. (not Heins, as first reported) |
| `heins2024CollectiveBehavior` | PNAS | 121(17):e2320239121 ✓ |
| `waade2025AsOneAndMany` | MDPI | Entropy 27(2):143; **Waade** et al. (not Tschantz/Da Costa) |
| `albarracin2022EpistemicCommunities`, `hylandAlbarracin2025MindChange`, `kuhn1962Structure`, `friston2015...`, `feldmanFriston2010...`, `parrFriston2019...`, `millidge2021EFE`, `wattsStrogatz1998...` | reused from vetted IWAI `refs.bib` | DOIs/arXiv on file |

## Building

No LaTeX install is assumed locally. Options:

- **Overleaf:** upload this folder, set `main.tex` as the root document, compile.
- **Local:** `pdflatex main && bibtex main && pdflatex main && pdflatex main`
- **CI:** [`ci/build-paper.yml`](ci/build-paper.yml) is a ready GitHub Actions
  workflow that compiles the PDF and uploads it as an artifact. Move it to
  `.github/workflows/` and push with a token that has `workflow` scope to enable
  it (it was kept out of `.github/` here to avoid that scope requirement).
