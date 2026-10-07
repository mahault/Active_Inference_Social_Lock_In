# When Do Pluralist Communities Lock In?

Joint follow-up to the IWAI 2026 paper (Hallgren, Hyland, Albarracin). Scope agreed on the 1 Oct 2026 call: David's pluralist agents as the base; the question is whether pluralist communities resist two-channel precision collapse better than monist ones; belief utility and the motivated cost of mind-change come back in. Trust and deception, and motivated ignorance, are planned as follow-up papers.

Target: a Royal Society journal (rolling submission). Author order to settle.

- `main.tex`: skeleton with section plans; results sections are placeholders.
- `refs.bib`: inherits the verified bibliography from `paper-precision-collapse/` plus the references David sent (Williams 2021, Freedman 2024, Hu & Orquin 2025, Kirfel et al. 2025, Hagenbach & Saucet 2025, McCraw 2015, Mayer et al. 1995). IWAI proceedings details for `hallgren2026Pluralist` are still to fill in.
- Code: `src/pluralist/`; experiments: `scripts/run_p1*.py`; results: `experiments/results/P1*` and `P1_stats.md`.
- Results so far: `notes/results_2026-10-05_reruns_and_pluralist.md`, sections B and C.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
