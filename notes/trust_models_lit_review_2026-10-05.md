# How agents judge each other: formal models of epistemic trust (lit review, 2026-10-05)

Context: the current social channel sets trust ∝ a Gamma-conjugate estimate of 1/(mean surprisal of neighbour's predictions on my observations). It does almost nothing, for two reasons. First, Gaussian surprisals barely differ: the mean log-score gap is about Δμ²/2σ², which is tiny when the theories' means differ by less than σ. Second, most agents never see the discriminating channel. The literature names this problem directly. Merdes, von Sydow & Hahn (2021) separate **outcome-based** reliability estimation (predictions checked against eventual outcomes) from **expectation-based** estimation (reports checked against my own current beliefs). Our model has outcome-based form but runs on data that can't discriminate. For agents without the channel, the only thing left is expectation-based trust, and that is the self-reinforcing Laputa/O'Connor–Weatherall loop.

Read status: **Full** = full text, model section read; **Part** = full text, skimmed; **Abs** = abstract or secondary summary only.

## Table

| Verified citation | What trust is | Update (sketch) | Agent must observe | Options | Read |
|---|---|---|---|---|---|
| Bovens & Hartmann 2003, *Bayesian Epistemology*, OUP (via Merdes et al.) | Binary latent REL. A reliable source is perfectly truthful; an unreliable one randomises with P(rep)=a | Joint Bayes net H→REP←REL: P(REP\|H,REL)=1/0, P(REP\|·,¬REL)=a; posterior on REL and H | Reports, plus own belief (or a later outcome) | 1, 5 | Book not read; CPTs from Merdes (Full) |
| Merdes, von Sydow & Hahn 2021, *Synthese* 198:5773–5801 | Compares BH and Olsson models | Sequential trust updating creates **order effects**. BH cannot reach "liar" (likelihood ∈[.5,1]); Olsson can (∈[0,1]). Outcome-based vs expectation-based estimation | — | 1, 5 | Full (equations are images in the PDF) |
| Olsson 2011 *Episteme* 8:127–143; Olsson 2013 in Zenker (ed.) *Bayesian Argumentation* (Springer); Olsson & Vallinder 2013 *Synthese* (Norms of assertion); Vallinder & Olsson 2014 *Synthese* 191:1991–2007 | Laputa: a density τ(ρ) over source reliability ρ∈[0,1] | C(p\|says p)=C(p)⟨τ⟩/[C(p)⟨τ⟩+(1−C(p))(1−⟨τ⟩)]; τ'(ρ)∝τ(ρ)[ρC(p)+(1−ρ)(1−C(p))]. Expected messages raise trust and unexpected ones lower it | Message + own credence only | 5 (self-reinforcing), 1 | V&O 2014 Full; others Abs |
| O'Connor & Weatherall 2018, *Eur. J. Phil. Sci.* 8:855–875 | Credence in a neighbour's reported evidence, falling with belief distance d | Jeffrey: P_f(H)=P(H\|E)P_f(E)+P(H\|¬E)(1−P_f(E)); P_f(E)=max(1−d·m·(1−P_i(E)),0) (anti-updating) or 1−min(1,dm)(1−P_i(E)) | Neighbour's credence + data | 5 (produces polarisation) | Full |
| Weatherall & O'Connor 2021, *Synthese* (Endogenous epistemic factionalization) | Same rule, belief distance taken over several propositions | as above | as above | 5 | Part |
| Holman & Bruner 2015, *Phil. Sci.* 82:956–968 | Weights on neighbours, lowered when their results clash with one's own | Endogenous reweighting (exact rule not seen) | Neighbour's data | 1, 3 (biased agent) | Abs |
| Shafto, Eaves, Navarro & Perfors 2012, *Dev. Sci.* 15:436–447 | **Two latent variables**: knowledge k∈{0,1} and helpfulness h∈{1,−1} | P(s,k,h\|ℓ)∝Σ_b P(ℓ\|b,h)P(b\|k,s)P(s)P(k)P(h); P(ℓ\|b,h)∝P(b\|ℓ)^h. A knowledgeable agent holds the true belief; an unhelpful one picks labels that mislead | The label, and ideally the truth s (otherwise consensus) | 2, 3 | Full (OSF preprint) |
| Landrum, Eaves & Shafto 2015, *TICS* 19:109–111 | Knowledge + intent framework | — | — | 2 | Abs |
| Harris, Hahn, Madsen & Hsu 2016, *Cogn. Sci.* 40:1496–1533 | Separate **Expertise** and **Trustworthiness** nodes | Rep has parents H, Exp, T. An expert reports the truth if trustworthy and the opposite if not; a non-expert randomises. "Expertise = honest mistakes; trustworthiness = systematic deceit" | Report + prior on H | 2, 3, 4 | Full |
| Sperber et al. 2010, *Mind & Language* 25:359–393 | Vigilance toward the **source** (competence vs benevolence/honesty) and toward the **content** (coherence) | Not formal | — | 2 (conceptual) | Part |
| Diaconescu et al. 2014, *PLoS Comput. Biol.* 10:e1003810 | Advice accuracy as a latent variable, plus the **volatility of the adviser's intentions** (HGF) | μ₂←μ₂+σ₂δ₁ with σ₂ scaled by volatility μ₃. The outcome is revealed every trial | Advice + **later outcome** | 1, 4, delayed verification | Part |
| Siegel, Mathys, Rutledge & Crockett 2018, *Nat. Hum. Behav.* 2:750–756 | Impression of moral character + its uncertainty (HGF) | Beliefs about "bad" agents are more uncertain, so they update faster. Gives a Bayesian **asymmetry** | Observed choices | 4 | Part |
| Sobel 1985, *Rev. Econ. Stud.* 52:557–573 | Reputation = P(sender is a "friend"). Enemies tell the truth while it pays and lie when the stakes are high | Bayes on type after each verified report. One detected lie is near-conclusive | **Ex-post truth** | 2, 4, delayed verification | Abs |
| Sobel 2020, *J. Polit. Econ.* 128:907–947 | Formal lying ≠ deception: "deception does not require lying" | Definitions | — | 3 | Part |
| Milgrom 1981, *Bell J. Econ.* 12:380–391 | Skeptical receiver: withheld information is read as unfavourable (unravelling) | Equilibrium inference about the undisclosed | **What the sender could have disclosed** | 3 | Abs |
| Weatherall, O'Connor & Bruner 2020, *BJPS* (How to beat science…) | No trust learning. A propagandist **shares only real, favourable results** | Policy-makers update on all received data and still converge on the false theory | — | 3 (cherry-picking works without lying) | Full (§3) |
| Gweon, Pelton, Konopka & Schulz 2014, *Cognition* 132:335–341 | Children rate **under-informative** (omitting) teachers lower | Empirical | What else was true | 3 | Abs |
| Oey, Schachner & Vul 2023, *JEP:Gen* 152:346–362 | Recursive ToM model of lie design and detection | Receiver: P(lie\|msg)∝P(msg\|lie,sender model)… | Message vs prior | 3 | Part |
| Kirfel, Gerstenberg & Zultan 2025, *Curr. Opin. Psychol.* 65:102090 | Willful ignorance leads observers to make epistemic, counterfactual and **character** inferences | Review; not formal | That information was **avoided** | 6 | Full (it's about **blame**, not trust) |
| Slovic 1993, *Risk Analysis* 13:675–682 | Asymmetry principle: trust is easier to destroy than to build | Empirical | — | 4 | Abs |
| Vasil et al. 2020, *Front. Psychol.* 11:417 | Trust = **precision** given to another's signals. Repeated surprise means low precision | Conceptual AIF (this is our current model) | Surprise | 1, 5 | Part |
| Albarracin, Demekas, Ramstead & Heins 2022, *Entropy* 24:476 | Fixed "epistemic confirmation bias" precision on agreeing neighbours | Precision raised when the neighbour's message matches own belief | Message | 5 | Part |
| Shah & Pashea 2026 arXiv:2609.24876; Mahadewa & Shock 2026 arXiv:2608.02440 | Shah & Pashea: per-partner precision. Mahadewa & Shock: **intent vs execution noise** as separate latents (AIF POMDP) | S&P track **predictability, not truth**; the confidence lags change | Partner actions | 2, 5 | Abs (preprints, not peer-reviewed) |

Not found: I found no peer-reviewed 2020–2026 model that turns Mayer, Davis & Schoorman's ability/benevolence/integrity split into a Bayesian latent-variable model for *epistemic* sources. The closest are Shafto 2012 (k, h) and Harris 2016 (Exp, T). What exists for ABI is engineering or HRI frameworks (abstracts only). Treat ABI as the conceptual label and Shafto/Harris as the formal model.

## Recommendation: three designs for the social channel

The common fix is to replace "mean surprisal on my data" with an **accumulated log-likelihood ratio on a channel that discriminates**, and to give each signature of untrustworthiness its own latent variable.

### D1. Delayed public verification (outcome-based track record)
*Grounded in:* Diaconescu 2014, Sobel 1985, Merdes et al. (outcome-based estimation).
*Observables:* every L steps there is a **public replication event**: a draw y* of the discriminating channel that all agents see, including the ones who can't observe it otherwise. Each neighbour j commits a forecast N(μ_j, s_j²) beforehand.
*Update:* use a relative proper score, s_j = log N(y*; μ_j, s_j²) − log N(y*; μ_ref, s_ref²), where ref is the agent's own forecast or the population mixture. Then
ℓ_j ← λℓ_j + s_j  (λ<1 = forgetting), trust w_j = σ(ℓ_j).
Summing, rather than averaging, lets small per-event gaps add up (a Bovens–Hartmann-style Bayes factor).
*Tests:* how lock-in decay depends on L and on verification frequency. Do pluralists, with hedged forecasts, gain trust under the log score but lose it under a 0/1 "hit" rule? That would make the choice of scoring rule a lever on lock-in.

### D2. Ability × integrity source model (Shafto/Harris, Gaussian)
*Observables:* verified outcomes (D1), plus each neighbour's **stake** d_j: the sign of their favoured theory's prediction relative to the ensemble.
*Generative model of a report:* r_j = y + β_j d_j + ε, ε~N(0, 1/π_j). Ability = π_j. Integrity is a spike-and-slab on β_j: P(β_j=0)=ι_j.
*Update:* on each verified residual e=r_j−y,
- slab: Normal-Gamma regression of e on d_j, giving (m,κ,a,b)
- ι_j' = ι_j·N(e;0,1/π_j) / [ι_j·N(e;0,1/π_j) + (1−ι_j)·∫N(e;βd_j,1/π_j)p(β)dβ]
- E[π_j]=a/b

Effective social precision = ι_j·E[π_j], used as the precision of j's message in the A-matrix (a mixture likelihood).
Random errors lower ability only. Errors that line up with j's stake lower integrity. With a prior ι≈0.9 and Sobel-type liars, one large lie aligned with j's stake is near-diagnostic, so **asymmetric updating emerges from the model rather than being imposed**. Optionally, add a Siegel-style higher volatility for low-ι agents.
*Tests:* hypothesis 3. Each datum a cherry-picker shares is true, so β≈0 and the cherry-picker should keep both ι and π, while liars lose ι. Check that a lie costs more trust than an honest mistake of equal size.

### D3. Disclosure- and avoidance-aware integrity (Milgrom / Gweon / Kirfel)
*Observables:* **the number of experiments n_j that j ran and which channel j sampled** (preregistration, or visible actions). This is the extra observable that cherry-picking detection needs.
*Update:* j reports m_j of n_j results. Under selection rate ρ_j (the probability an unfavourable result is withheld), the unreported count is u=n_j−m_j ~ Bin(n_j(1−q_j(θ)), ρ_j), where q_j(θ)=P(result favours j \| θ). Keep a Beta posterior on ρ_j. On the belief side, treat the u withheld results as unfavourable draws (unravelling).
*Avoidance:* compare j's channel choice under an epistemic policy, softmax(γ·infogain), with a confirmation policy, softmax(γ·expected support for j's theory). Integrity odds are multiplied by that likelihood ratio.
*Tests:* cherry-pickers are caught **only** when n_j is visible. Without it, detection is only possible through statistical over-representation against the ensemble, the heuristic O'Connor & Weatherall suggest. Do monists avoid the discriminating channel more often, and does the resulting integrity penalty break their lock-in?

Keep **expectation-based trust (Laputa, or O'Connor–Weatherall's d·m rule)** as the control condition for option 5. It should reproduce self-reinforcing lock-in.

## Flags and uncertainties
- Merdes and Vallinder–Olsson print their equations as images. The Laputa formulas above are the standard published forms, reconstructed and checked against the qualitative rules in the text (Table 1 of V&O).
- Not verified: the page ranges for Olsson 2013, Olsson & Vallinder 2013, Weatherall & O'Connor 2021, and Weatherall, O'Connor & Bruner 2020.
- I read Bovens & Hartmann only through Merdes et al.
- I didn't see Holman & Bruner's exact reweighting rule.
- Kirfel et al. 2025 is about blame, so using it for trust is our extrapolation.
- I didn't fetch Kamenica & Gentzkow (Bayesian persuasion), so it is not cited.
- Hahn, Merdes & von Sydow 2018 (*TopiCS* 10:660–678) is the source of the outcome/expectation distinction. I saw it only as cited inside Merdes et al.
