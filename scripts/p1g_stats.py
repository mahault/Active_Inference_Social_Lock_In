"""Statistics for P1g (population size, bandwidth, neutral drift).

Part 2 (bandwidth) contrasts are paired by seed (same observation stream; message
sampling uses a separately folded key). Part 1 compares different population
sizes, which share seed indices but not random draws, so those contrasts are
UNPAIRED: difference of means with an independent-resampling bootstrap CI and a
two-sided Mann-Whitney U test. The "lottery" indicator is the across-seed SD of
the community-mean end P(M>0) within a gamma x c0 cell, averaged over cells;
its N-contrast CI comes from resampling seeds independently in each arm.
Holm across all tests in this file. Part 3 is descriptive.

Usage: python -m scripts.p1g_stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from scripts.p1_stats import contrast, holm  # noqa: E402

OUT = ROOT / "experiments" / "results" / "P1g"
N_BOOT = 10_000


def load(k):
    p = OUT / f"part{k}.npz"
    return dict(np.load(p)) if p.exists() else {}


def unpaired(a, b, rng):
    """a, b: (cells, S). Per-seed cell means compared between independent arms."""
    da, db = a.mean(0), b.mean(0)
    ia = rng.integers(0, da.size, (N_BOOT, da.size))
    ib = rng.integers(0, db.size, (N_BOOT, db.size))
    diff = da[ia].mean(1) - db[ib].mean(1)
    p = 1.0 if np.allclose(da, db) else float(mannwhitneyu(da, db, alternative="two-sided").pvalue)
    return {"mean_A": float(da.mean()), "mean_B": float(db.mean()), "mean_diff": float(da.mean() - db.mean()),
            "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "p": p,
            "test": "Mann-Whitney U (unpaired)"}


def sd_contrast(a, b, rng):
    """Across-seed SD within cells (averaged over cells), A - B, independent bootstrap."""
    sa, sb = a.std(1, ddof=1).mean(), b.std(1, ddof=1).mean()
    ia = rng.integers(0, a.shape[1], (2000, a.shape[1]))
    ib = rng.integers(0, b.shape[1], (2000, b.shape[1]))
    d = np.array([a[:, i].std(1, ddof=1).mean() - b[:, j].std(1, ddof=1).mean() for i, j in zip(ia, ib)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / len(d)))
    return {"mean_A": float(sa), "mean_B": float(sb), "mean_diff": float(sa - sb), "ci95": [float(lo), float(hi)],
            "p": p, "test": "bootstrap (2000) sign test on SD difference (unpaired)"}


def main():
    rng = np.random.default_rng(12345)
    p1, p2, p3 = load(1), load(2), load(3)
    rows = []

    def row(part, name, metric, r):
        r.update(part=part, contrast=name, metric=metric)
        rows.append(r)

    # ---- part 1: population size (unpaired)
    for scheme in ("sixth", "four"):
        for kind in ("pluralist", "committed", "neutral", "mix"):
            a, b = f"N100|{scheme}|{kind}", f"N8|{scheme}|{kind}"
            if f"{a}::end_pM" not in p1 or f"{b}::end_pM" not in p1:
                continue
            for m in ("end_pM", "frac_agents_reach", "log_score", "t_reach"):
                row(1, f"N=100 - N=8 [{kind}, observers {scheme}]", m, unpaired(p1[f"{a}::{m}"], p1[f"{b}::{m}"], rng))
            row(1, f"N=100 - N=8 [{kind}, observers {scheme}]", "across-seed SD of end P(M>0) (lottery)",
                sd_contrast(p1[f"{a}::end_pM"], p1[f"{b}::end_pM"], rng))

    # ---- part 2: bandwidth (paired)
    def pc(name, A, B, m):
        a, b = p2.get(f"{A}::{m}"), p2.get(f"{B}::{m}")
        if a is None or b is None:
            return
        r = contrast(a, b, rng)
        r["p"] = r.pop("p_wilcoxon")
        r["test"] = "Wilcoxon signed-rank (paired)"
        row(2, name, m, r)

    for kind in ("pluralist", "committed", "neutral", "mix"):
        for m in ("end_pM", "log_score", "frac_agents_reach"):
            pc(f"m=1 - exact moments [{kind}]", f"{kind}|observables|msg1|sup0", f"{kind}|observables|msg0|sup0", m)
        pc(f"exact observable moments - full natural parameters [{kind}]", f"{kind}|observables|msg0|sup0",
           f"{kind}|rule|msg0|sup0", "end_pM")
        if kind in ("pluralist", "mix"):
            for m in ("end_pM", "log_score"):
                pc(f"sampled support m=1 - exact support (message m=1) [{kind}]", f"{kind}|observables|msg1|sup1",
                   f"{kind}|observables|msg1|sup0", m)
    # interaction: does low bandwidth hurt pluralists more than committed monists?
    for m in ("end_pM", "log_score"):
        k = lambda kind, mm: p2.get(f"{kind}|observables|msg{mm}|sup0::{m}")  # noqa: E731
        if all(k(t, mm) is not None for t in ("pluralist", "committed", "neutral") for mm in (0, 1)):
            for other in ("committed", "neutral"):
                r = contrast(k("pluralist", 1) - k("pluralist", 0), k(other, 1) - k(other, 0), rng)
                r["p"] = r.pop("p_wilcoxon")
                r["test"] = "Wilcoxon signed-rank (paired)"
                row(2, f"bandwidth loss: pluralist - {other} [(m=1) - (exact)]", m, r)

    adj = holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
        r["survives_holm"] = bool(a < 0.05)

    # ---- part 3: descriptive
    drift = {}
    for key in sorted({k.split("::")[0] for k in p3}):
        g = lambda m: p3[f"{key}::{m}"]  # noqa: E731
        drift[key] = {"end_abs_dev": float(g("end_abs_dev").mean()),
                      "frac_seeds_pM_gt_half": float((g("end_pM") > 0.5).mean()),
                      "median_t_consensus": float(np.median(g("t_consensus_pM_dev_0.25"))),
                      "frac_seeds_reaching_dev_0.25": float((g("t_consensus_pM_dev_0.25") < 600).mean()),
                      "within_seed_agent_sd_end": float(g("within_seed_agent_sd_end").mean()),
                      "across_seed_sd_end_pM": float(g("end_pM").std(ddof=1))}
    lottery = {}
    for key in sorted({k.split("::")[0] for k in p1}):
        e = p1[f"{key}::end_pM"]
        lottery[key] = {"end_pM": float(e.mean()), "across_seed_sd": float(e.std(1, ddof=1).mean()),
                        "frac_seeds_locked": float((e < 0.5).mean()),
                        "bimodality_p(1-p)": float(((e < 0.5).mean(1) * (e >= 0.5).mean(1)).mean()),
                        "frac_agents_reach": float(p1[f"{key}::frac_agents_reach"].mean()),
                        "t_reach": float(p1[f"{key}::t_reach"].mean())}
    bw = {k.split("::")[0]: {m: float(p2[f"{k.split('::')[0]}::{m}"].mean()) for m in
                            ("end_pM", "log_score", "frac_agents_reach")}
          for k in p2 if k.endswith("::end_pM")}
    (OUT / "stats.json").write_text(json.dumps({"contrasts": rows, "lottery": lottery, "bandwidth_means": bw,
                                                "drift": drift}, indent=1))

    L = ["# P1g statistics: population size, message bandwidth, neutral drift", "",
         "Part 1 contrasts are unpaired (different N = different random draws): difference of seed-level means "
         "(cell-averaged), independent bootstrap CI (10k), Mann-Whitney U. The lottery indicator is the across-seed "
         "SD of the community-mean end P(M>0) within a cell, averaged over the 3x3 grid. Part 2 is paired by seed "
         "(Wilcoxon). Holm across all " + str(len(rows)) + " tests.", "",
         "| part | contrast (A - B) | metric | mean A | mean B | diff [95% CI] | p | p_Holm | Holm |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['part']} | {r['contrast']} | {r['metric']} | {r['mean_A']:.3f} | {r['mean_B']:.3f} | "
                 f"{r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.2g} | "
                 f"{r['p_holm']:.2g} | {'yes' if r['survives_holm'] else 'no'} |")
    L += ["", "## Part 1: population size (means over 3x3 grid x 20 seeds)", "",
          "| arm | end P(M>0) | across-seed SD | share of seeds locked | p(1-p) bimodality | agents reach | t_reach |",
          "|---|---|---|---|---|---|---|"]
    for k, v in lottery.items():
        L.append(f"| {k} | {v['end_pM']:.3f} | {v['across_seed_sd']:.3f} | {v['frac_seeds_locked']:.2f} | "
                 f"{v['bimodality_p(1-p)']:.3f} | {v['frac_agents_reach']:.2f} | {v['t_reach']:.0f} |")
    L += ["", "## Part 2: bandwidth (N=24, means over 3x3 grid x 20 seeds)", "",
          "| arm (type, fusion, message m, support m) | end P(M>0) | log score | agents reach |", "|---|---|---|---|"]
    for k, v in bw.items():
        L.append(f"| {k} | {v['end_pM']:.3f} | {v['log_score']:.3f} | {v['frac_agents_reach']:.2f} |")
    L += ["", "## Part 3: neutral-drift control (world closed for everyone, symmetric start, 40 seeds)", "",
          "| arm | mean abs(P(M>0)-.5) at end | share of seeds ending > .5 | median t to abs dev .25 | share of seeds reaching .25 | within-seed agent SD | across-seed SD |",
          "|---|---|---|---|---|---|---|"]
    for k, v in drift.items():
        L.append(f"| {k} | {v['end_abs_dev']:.3f} | {v['frac_seeds_pM_gt_half']:.2f} | {v['median_t_consensus']:.0f} | "
                 f"{v['frac_seeds_reaching_dev_0.25']:.2f} | {v['within_seed_agent_sd_end']:.3f} | "
                 f"{v['across_seed_sd_end_pM']:.3f} |")
    (OUT / "stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
