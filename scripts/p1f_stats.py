"""Paired statistics for P1f (artefact checks). Same method as scripts/p1_stats.py:
per-seed cell-averaged paired differences, 95% bootstrap CI (10k), Wilcoxon
signed-rank, Holm across the whole P1f family. Writes experiments/results/P1f/stats.{json,md}.

Usage: python -m scripts.p1f_stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.p1_stats import contrast, holm  # noqa: E402

OUT = ROOT / "experiments" / "results" / "P1f"
LOWER_BETTER = {"brier", "t_reach", "track_brier", "abs_err_M", "dev_from_anchor"}


def load(n):
    p = OUT / f"check{n}.npz"
    return dict(np.load(p)) if p.exists() else {}


def main():
    rng = np.random.default_rng(12345)
    d = {n: load(n) for n in range(1, 6)}
    rows = []

    def add(check, name, A, B, metric, src):
        a, b = src.get(f"{A}::{metric}"), src.get(f"{B}::{metric}")
        if a is None or b is None:
            return
        r = contrast(a, b, rng)
        r.update(check=check, contrast=name, metric=metric,
                 better=("lower" if metric in LOWER_BETTER else "higher"))
        rows.append(r)

    def add_arrays(check, name, a, b, metric):
        if a is None or b is None:
            return
        r = contrast(a, b, rng)
        r.update(check=check, contrast=name, metric=metric,
                 better=("lower" if metric in LOWER_BETTER else "higher"))
        rows.append(r)

    # ---- check 1: outcome-based re-scoring
    c4 = d[4]
    for m in ("frac_agents_reach", "community_reach", "log_score", "brier", "t_reach"):
        add(1, "pluralist - neutral monist", "omega0.9|pluralist|noforget", "omega0.9|neutral|noforget", m, c4)
        add(1, "pluralist - committed monist", "omega0.9|pluralist|noforget", "omega0.9|committed|noforget", m, c4)
        add(1, "neutral - committed monist", "omega0.9|neutral|noforget", "omega0.9|committed|noforget", m, c4)
    c1 = d[1]
    for m in ("frac_agents_reach", "log_score"):
        add(1, "P1c 75% - 25% pluralists (monists 50:50)", "P1c|plural0.75_half", "P1c|plural0.25_half", m, c1)
        add(1, "P1c pluralists at observers - none", "P1c|plural0.25_half_obsP", "P1c|plural0.25_half_obsM", m, c1)
    for comp in ("all_pluralist", "half_pluralist_committed", "all_committed"):
        for m in ("frac_agents_reach", "log_score"):
            add(1, f"P1d relative(k1) - fixed, mode b attention [{comp}]",
                f"P1de|{comp}|trust+belief|attention|relative|10|1.0",
                f"P1de|{comp}|trust+belief|attention|fixed|10|1.0", m, c1)
            add(1, f"P1e relative(k30) - fixed, mode a [{comp}]", f"P1de|{comp}|trust|-|relative|10|30.0",
                f"P1de|{comp}|trust|-|fixed|None|1.0", m, c1)
        add(1, f"P1d agree - fixed, mode b attention [{comp}]", f"P1de|{comp}|trust+belief|attention|agree|10|1.0",
            f"P1de|{comp}|trust+belief|attention|fixed|10|1.0", "log_score", c1)

    # ---- check 2: changing worlds
    c2 = d[2]
    for w in ("multiflip", "drift"):
        for comp in ("all_pluralist", "half_pluralist_committed"):
            for rule in ("relative", "fixed"):
                add(2, f"D=1 - D=50 [{w}, {comp}, {rule}, every 10th]", f"{w}|{comp}|every10|{rule}|1",
                    f"{w}|{comp}|every10|{rule}|50", "track_brier", c2)
            add(2, f"relative(k30) - fixed, D=10 [{w}, {comp}, every 10th]", f"{w}|{comp}|every10|relative|10",
                f"{w}|{comp}|every10|fixed|10", "track_brier", c2)
            add(2, f"D=1 - D=50 [{w}, {comp}, relative, every round]", f"{w}|{comp}|every1|relative|1",
                f"{w}|{comp}|every1|relative|50", "track_brier", c2)

    # ---- check 3: wrong rival
    c3 = d[3]
    for v in ("original", "rival_M+3", "rival_M-3", "K3_-1_+1_+3"):
        add(3, f"pluralist - committed [{v}]", f"{v}|pluralist", f"{v}|committed", "log_score", c3)
        add(3, f"pluralist - committed [{v}]", f"{v}|pluralist", f"{v}|committed", "abs_err_M", c3)
        add(3, f"pluralist - neutral [{v}]", f"{v}|pluralist", f"{v}|neutral", "log_score", c3)

    # ---- check 4: anchoring
    for om in (0.5, 0.8, 0.9, 0.97, 1.0):
        add(4, f"pluralist - committed [omega {om}]", f"omega{om}|pluralist|noforget", f"omega{om}|committed|noforget",
            "log_score", c4)
        add(4, f"pluralist - neutral [omega {om}]", f"omega{om}|pluralist|noforget", f"omega{om}|neutral|noforget",
            "log_score", c4)
        add(4, f"pluralist(forget) - committed [omega {om}]", f"omega{om}|pluralist|forget",
            f"omega{om}|committed|noforget", "log_score", c4)

    # ---- check 5: sparse/noisy reveals x observers
    c5 = d[5]
    settings = [("trust+belief", 10, 1.0), ("trust+belief", 10, 3.0), ("trust+belief", 30, 1.0),
                ("trust+belief", 30, 3.0), ("trust", 10, 1.0)]
    for comp in ("all_pluralist", "half_pluralist_committed"):
        for mode, ev, nz in settings:
            for nf in (4, 8, 12):
                pre = f"nfull{nf}|{comp}|{mode}|every{ev}|noise{nz}"
                add(5, f"relative(k30) - fixed [{comp}, {mode}, every {ev}, noise {nz}x, {nf} observers]",
                    f"{pre}|relative", f"{pre}|fixed", "log_score", c5)
            k = lambda nf, r: c5.get(f"nfull{nf}|{comp}|{mode}|every{ev}|noise{nz}|{r}::log_score")  # noqa: E731
            if all(k(nf, r) is not None for nf in (4, 12) for r in ("relative", "fixed")):
                add_arrays(5, f"trust gain at 12 - at 4 observers [{comp}, {mode}, every {ev}, noise {nz}x]",
                           k(12, "relative") - k(12, "fixed"), k(4, "relative") - k(4, "fixed"), "log_score")

    adj = holm([r["p_wilcoxon"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
        r["survives_holm"] = bool(a < 0.05)
        r["negligible"] = bool(abs(r["mean_diff"]) < (0.005 if r["metric"] not in ("t_reach",) else 1.0))

    # descriptive tables
    desc = {"check1_means": {}, "check4_anchor": {}}
    for t in ("pluralist|noforget", "pluralist|forget", "neutral|noforget", "committed|noforget"):
        k = f"omega0.9|{t}"
        desc["check1_means"][t] = {m: float(np.mean(c4[f"{k}::{m}"])) for m in
                                   ("end_pM", "frac_agents_reach", "community_reach", "log_score", "brier",
                                    "t_reach", "t_reach_censored") if f"{k}::{m}" in c4}
    for om in (0.5, 0.8, 0.9, 0.97, 1.0):
        for t in ("pluralist|noforget", "pluralist|forget", "neutral|noforget", "committed|noforget"):
            k = f"omega{om}|{t}"
            if f"{k}::end_pM" in c4:
                end, anc = c4[f"{k}::end_pM"].ravel(), c4[f"{k}::anchor_pM"].ravel()
                desc["check4_anchor"][k] = {
                    "end_pM": float(end.mean()), "anchor_pM": float(anc.mean()),
                    "mean_abs_dev": float(np.mean(c4[f"{k}::dev_from_anchor"])),
                    "corr_end_anchor": float(np.corrcoef(end, anc)[0, 1]) if anc.std() > 0 else None,
                    "log_score": float(np.mean(c4[f"{k}::log_score"]))}
    (OUT / "stats.json").write_text(json.dumps({"contrasts": rows, "descriptive": desc}, indent=1))

    L = ["# P1f statistics (artefact checks)", "",
         "Paired by seed (seeds 0-19, identical observation noise across arms); each seed's value is averaged "
         "over the 3x3 gamma x c0 grid (and seer set). diff = A - B with 95% bootstrap CI (10k), d_z, Wilcoxon "
         "signed-rank p, Holm across all " + str(len(rows)) + " rows. 'better' says which sign is good for A. "
         "Metrics: log_score = mean log P(true sign of M) of agents' final beliefs (0 is perfect); brier = mean "
         "(1-P(true))^2; frac_agents_reach = share of agents with P(true) >= .75; community_reach = community "
         "mean >= .75; t_reach = rounds after the shift until the community mean first reaches .75 (censored "
         "at the horizon); track_brier = time-averaged Brier of the community against the current sign of M; "
         "abs_err_M = |posterior mean of M - 1| at the end.", ""]
    for chk, title in ((1, "Check 1: outcome-based scoring"), (2, "Check 2: changing worlds"),
                       (3, "Check 3: wrong rival"), (4, "Check 4: anchoring strength omega"),
                       (5, "Check 5: sparse/noisy reveals x number of full observers")):
        L += [f"## {title}", "", "| contrast (A - B) | metric (better) | mean A | mean B | diff [95% CI] | d_z | p | p_Holm | Holm |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["check"] != chk:
                continue
            sig = ("yes (negligible)" if r["negligible"] else "yes") if r["survives_holm"] else "no"
            L.append(f"| {r['contrast']} | {r['metric']} ({r['better']}) | {r['mean_A']:.3f} | {r['mean_B']:.3f} | "
                     f"{r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:.2f} | "
                     f"{r['p_wilcoxon']:.2g} | {r['p_holm']:.2g} | {sig} |")
        L.append("")
        if chk == 1 and desc["check1_means"]:
            L += ["Pure-type means at omega 0.9 (3x3 grid, 20 seeds):", "",
                  "| type | end P(M>0) | agents reach | community reach | log score | Brier | t_reach (censored share) |",
                  "|---|---|---|---|---|---|---|"]
            for t, v in desc["check1_means"].items():
                L.append(f"| {t} | {v['end_pM']:.3f} | {v['frac_agents_reach']:.3f} | {v['community_reach']:.3f} | "
                         f"{v['log_score']:.3f} | {v['brier']:.3f} | {v['t_reach']:.0f} ({v['t_reach_censored']:.2f}) |")
            L.append("")
        if chk == 4 and desc["check4_anchor"]:
            L += ["Anchor tracking: anchor_pM = P(M>0) implied by the agent's own reference Gaussians weighted by "
                  "its final support; mean |dev| = mean over agents of |end P(M>0) - anchor_pM|.", "",
                  "| omega | type | end P(M>0) | anchor P(M>0) | mean abs dev | corr(end, anchor) over cells x seeds | log score |",
                  "|---|---|---|---|---|---|---|"]
            for k, v in desc["check4_anchor"].items():
                om, t = k.split("|", 1)
                cr = "n/a" if v["corr_end_anchor"] is None else f"{v['corr_end_anchor']:.2f}"
                L.append(f"| {om[5:]} | {t} | {v['end_pM']:.3f} | {v['anchor_pM']:.3f} | {v['mean_abs_dev']:.3f} | "
                         f"{cr} | {v['log_score']:.3f} |")
            L.append("")
    (OUT / "stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
