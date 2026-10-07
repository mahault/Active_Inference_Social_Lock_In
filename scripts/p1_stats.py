"""Paired-by-seed statistics for P1b, P1c, P1d and P1e headline contrasts.

For every contrast A - B (same seeds = same observation noise in both arms):
  * per seed, the difference of community-mean P(M>0) at round 599 (or a
    per-type mean where stated), averaged over the arm's gate cells (and seer
    sets for P1d/P1e) -> one paired value per seed ("pooled");
  * effect = mean paired difference, 95% percentile bootstrap CI (10,000
    resamples over seeds, fixed RNG seed 12345), paired Cohen's d_z;
  * Wilcoxon signed-rank test (two-sided, scipy; zero differences dropped by
    the "wilcox" zero method; all-zero -> p = 1).
Holm correction across the whole family of pooled headline contrasts reported
here. Per-cell versions are also computed, with Holm correction within each
contrast's own family of cells; their survivor counts are reported.

Per-seed data: P1b/P1c are regenerated with src/pluralist/mixed.py (pure
communities reproduce the P1b pure-type runs exactly, see
tests/test_pluralist_mixed.py) and cached in
experiments/results/P1_stats_per_seed.npz; P1d/P1e come from
experiments/results/P1e/per_seed.npz (the P1d settings - kappa 1, lam .98,
attention-weighted reveals - are re-run there).

Usage: python scripts/p1_stats.py [--seeds 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

RES = ROOT / "experiments" / "results"
N_BOOT = 10_000


def boot_ci(d, rng):
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_test(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0.0, atol=1e-15):
        return 1.0
    try:
        return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def contrast(a, b, rng):
    """a, b: (..., S) per-seed arrays with cells on leading axes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = (a - b).reshape(-1, a.shape[-1])            # (cells, S)
    d = diff.mean(0)
    lo, hi = boot_ci(d, rng)
    sd = d.std(ddof=1)
    cell_p = np.array([paired_test(x) for x in diff])
    cell_adj = holm(cell_p) if cell_p.size > 1 else cell_p
    return {"mean_diff": float(d.mean()), "ci95": [lo, hi],
            "dz": float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else float("inf")),
            "p_wilcoxon": paired_test(d), "n_seeds": int(d.size),
            "mean_A": float(a.mean()), "mean_B": float(b.mean()),
            "cells": int(diff.shape[0]),
            "cells_sig_holm_within": int((cell_adj < 0.05).sum()),
            "cells_positive_sig": int(((cell_adj < 0.05) & (diff.mean(1) > 0)).sum()),
            "cells_negative_sig": int(((cell_adj < 0.05) & (diff.mean(1) < 0)).sum())}


# --------------------------------------------------------------------------
# P1b / P1c per-seed data (regenerated, cached)
# --------------------------------------------------------------------------

def p1bc_per_seed(seeds):
    cache = RES / "P1_stats_per_seed.npz"
    if cache.exists():
        return dict(np.load(cache))
    from scripts.run_p1c_mixed_communities import CELLS, N, N_FULL, SHIFT, STEPS, compositions
    from src.pluralist import extended as X
    from src.pluralist import mixed as M
    from src.pluralist import model

    base = model.Config(*model.candidate_priors())
    _, pre = X.run_prehistory(X.ExtConfig(base), seeds, STEPS, SHIFT, N, N_FULL)
    q_ox = float(pre["oxygen"][:, SHIFT - 1].mean())
    comps = compositions()
    wanted = [("monist_committed0.0", 1.0), ("monist_committed0.5", 1.0), ("monist_committed1.0", 1.0),
              ("plural1.0", 1.0), ("plural1.0", 0.95), ("plural0.25_half", 1.0), ("plural0.75_half", 1.0),
              ("plural0.25_half_obsP", 1.0), ("plural0.25_half_obsM", 1.0), ("plural0.25_allC", 1.0),
              ("plural0.75_allC", 1.0)]
    out = {}
    for name, rho in wanted:
        types = np.asarray(comps[name]["types"])
        mc = M.MixedConfig(base, tuple(types), q_ox, X.ExtConfig(base, rho_L=rho))
        hist = M.mixed_shift_protocol(mc, seeds, [g for g, _ in CELLS], [c for _, c in CELLS],
                                      steps=STEPS, shift=SHIFT, n_full=N_FULL)
        end = hist["p_mass_pos"][:, :, -1]                    # (G,S,N)
        tag = f"{name}|{'noforget' if rho == 1.0 else 'forget'}"
        out[f"{tag}::end"] = end.mean(-1)
        for t, tn in M.TYPE_NAMES.items():
            if (types == t).any():
                out[f"{tag}::end_{tn}"] = end[..., types == t].mean(-1)
        print("P1b/c per-seed:", tag, flush=True)
    np.savez_compressed(cache, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    rng = np.random.default_rng(12345)
    bc = p1bc_per_seed(seeds)
    e_path = RES / "P1e" / "per_seed.npz"
    e = dict(np.load(e_path)) if e_path.exists() else {}
    e_meta = json.loads((RES / "P1e" / "summary.json").read_text()) if e_path.exists() else {}
    kbest = e_meta.get("meta", {}).get("kappa_best")

    rows = []

    def add(study, name, A, B, desc, family="headline"):
        if A is None or B is None:
            return
        r = contrast(A, B, rng)
        r.update(study=study, contrast=name, description=desc, family=family)
        rows.append(r)

    # ---------------- P1b: pure communities, 5x5 grid ----------------
    P = "plural1.0|noforget::end"
    add("P1b", "pluralist - committed monist", bc[P], bc["monist_committed1.0|noforget::end"],
        "pure communities, support never forgets, 25 gate cells")
    add("P1b", "pluralist - neutral monist", bc[P], bc["monist_committed0.0|noforget::end"],
        "pure communities, 25 gate cells")
    add("P1b", "pluralist(forget .95) - committed monist", bc["plural1.0|forget::end"],
        bc["monist_committed1.0|noforget::end"], "pure communities, 25 gate cells")
    # ---------------- P1c: composition effects ----------------
    add("P1c", "monists 50% committed - 0% committed", bc["monist_committed0.5|noforget::end"],
        bc["monist_committed0.0|noforget::end"], "monist-only communities")
    add("P1c", "25% pluralists - 0% (monists 50:50)", bc["plural0.25_half|noforget::end"],
        bc["monist_committed0.5|noforget::end"], "community end P(M>0)")
    add("P1c", "75% pluralists - 25% (monists 50:50)", bc["plural0.75_half|noforget::end"],
        bc["plural0.25_half|noforget::end"], "community end P(M>0)")
    add("P1c", "75% pluralists (50:50) - 100% pluralists", bc["plural0.75_half|noforget::end"],
        bc[P], "community end P(M>0)")
    add("P1c", "25% pluralists at observers - none at observers", bc["plural0.25_half_obsP|noforget::end"],
        bc["plural0.25_half_obsM|noforget::end"], "placement, monists 50:50")
    add("P1c", "drag: pluralists in 25%-pluralist/committed community - pure pluralists",
        bc["plural0.25_allC|noforget::end_pluralist"], bc[P], "per-type end P(M>0) of pluralists")
    add("P1c", "pull: committed in 75%-pluralist community - pure committed",
        bc["plural0.75_allC|noforget::end_committed"], bc["monist_committed1.0|noforget::end"],
        "per-type end P(M>0) of committed monists")

    # ---------------- P1d / P1e ----------------
    def g(comp, forget, mode, prec, rule, D, kappa=1.0, lam=0.98, what="end", every=1):
        suffix = f"|every{every}" if every != 1 else ""
        return e.get(f"{comp}|{forget}|{mode}|{prec}|{rule}|{D}|{kappa}|{lam}{suffix}::{what}")

    comps = ["all_pluralist", "half_pluralist_committed", "all_committed"]
    for c in comps:
        nb = "noforget"
        add("P1d", f"relative - fixed W [{c}, mode b, D=10]", g(c, nb, "trust+belief", "attention", "relative", 10),
            g(c, nb, "trust+belief", "attention", "fixed", 10), "P1d settings (kappa 1, attention-weighted reveals)")
        add("P1d", f"agree - fixed W [{c}, mode b, D=10]", g(c, nb, "trust+belief", "attention", "agree", 10),
            g(c, nb, "trust+belief", "attention", "fixed", 10), "P1d settings")
        add("P1d", f"relative - fixed W [{c}, mode a, D=10]", g(c, nb, "trust", "-", "relative", 10),
            g(c, nb, "trust", "-", "fixed", None), "trust only")
        add("P1d", f"D=1 - D=50 [{c}, relative, mode b]", g(c, nb, "trust+belief", "attention", "relative", 1),
            g(c, nb, "trust+belief", "attention", "relative", 50), "delay effect at P1d settings")
        add("P1e", f"full - attention-weighted reveals [{c}, fixed W, mode b, D=10]",
            g(c, nb, "trust+belief", "full", "fixed", 10), g(c, nb, "trust+belief", "attention", "fixed", 10),
            "reveal precision")
        if kbest is not None:
            add("P1e", f"relative(k={kbest:g}) - fixed W [{c}, full, D=10]",
                g(c, nb, "trust+belief", "full", "relative", 10, kbest), g(c, nb, "trust+belief", "full", "fixed", 10),
                "sharp trust vs fixed W, full reveals")
            add("P1e", f"D=1 - D=50 [{c}, relative k={kbest:g}, full]",
                g(c, nb, "trust+belief", "full", "relative", 1, kbest),
                g(c, nb, "trust+belief", "full", "relative", 50, kbest), "does delay matter with sharp trust?")
            add("P1e", f"D=1 - D=50 [{c}, fixed W, full]", g(c, nb, "trust+belief", "full", "fixed", 1),
                g(c, nb, "trust+belief", "full", "fixed", 50), "delay effect without trust learning")
            add("P1e", f"relative(k={kbest:g}) - fixed W [{c}, mode a, D=10]",
                g(c, nb, "trust", "-", "relative", 10, kbest), g(c, nb, "trust", "-", "fixed", None), "trust only")
        add("P1e", f"sparse reveals: relative(k=30) - fixed W [{c}, every 10th, full, D=1]",
            g(c, nb, "trust+belief", "full", "relative", 1, 30.0, every=10),
            g(c, nb, "trust+belief", "full", "fixed", 1, every=10), "sparse public replications")
        add("P1e", f"sparse reveals: D=1 - D=50 [{c}, relative k=30, every 10th]",
            g(c, nb, "trust+belief", "full", "relative", 1, 30.0, every=10),
            g(c, nb, "trust+belief", "full", "relative", 50, 30.0, every=10), "delay with sparse reveals")
        add("P1e", f"sparse reveals: D=1 - D=50 [{c}, fixed W, every 10th]",
            g(c, nb, "trust+belief", "full", "fixed", 1, every=10),
            g(c, nb, "trust+belief", "full", "fixed", 50, every=10), "delay with sparse reveals, no trust learning")
        add("P1e", f"agree - fixed W [{c}, full, D=10]", g(c, nb, "trust+belief", "full", "agree", 10),
            g(c, nb, "trust+belief", "full", "fixed", 10), "full reveals")

    adj = holm([r["p_wilcoxon"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
        r["survives_holm"] = bool(a < 0.05)
        r["negligible"] = bool(abs(r["mean_diff"]) < 0.005)
    meta = {"method": __doc__.strip().splitlines()[0], "n_boot": N_BOOT, "bootstrap_seed": 12345,
            "test": "Wilcoxon signed-rank, two-sided, on per-seed pooled differences",
            "holm_family": f"all {len(rows)} pooled contrasts in this file",
            "kappa_best_P1e": kbest, "seeds": seeds}
    (RES / "P1_stats.json").write_text(json.dumps({"meta": meta, "contrasts": rows}, indent=1))

    lines = ["# P1 statistics: paired-by-seed contrasts (P1b-P1e)", "",
             f"Seeds 0-{len(seeds) - 1} (paired: identical observation noise across arms). Outcome: community-mean "
             "P(M>0) at round 599, averaged per seed over the arm's gate cells (and seer sets in P1d/P1e). "
             "Effect = mean paired difference A - B with 95% bootstrap CI (10k resamples over seeds); d_z = "
             "mean/sd of paired differences; p = Wilcoxon signed-rank; Holm across all rows below. "
             "'cells +/-' = gate cells (x seer sets) individually significant after Holm within that contrast.", "",
             "| study | contrast (A - B) | mean A | mean B | diff [95% CI] | d_z | p | p_Holm | Holm | cells +/- |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    NEG = " (negligible)"
    for r in rows:
        lines.append(f"| {r['study']} | {r['contrast']} | {r['mean_A']:.3f} | {r['mean_B']:.3f} | "
                     f"{r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:.2f} | "
                     f"{r['p_wilcoxon']:.2g} | {r['p_holm']:.2g} | {'yes' if r['survives_holm'] else 'no'}{NEG if r['survives_holm'] and r['negligible'] else ''} | "
                     f"{r['cells_positive_sig']}/{r['cells_negative_sig']} of {r['cells']} |")
    lines += ["", "Notes: with 20 seeds the smallest attainable two-sided Wilcoxon p is ~1.9e-6, so the Holm "
              "threshold for the most significant of the contrasts above is 0.05/" + str(len(rows)) + ". Contrasts with all-zero "
              "paired differences get p = 1. 'yes (negligible)' marks Holm-significant contrasts whose mean "
              "paired difference is below 0.005 in P(M>0) - numerically consistent but practically null "
              "(e.g. trust rules in an all-committed community, where nobody forecasts correctly, differ "
              "from fixed W only at the 1e-4 level). Differences are tiny d_z-inflated because paired seeds "
              "share observation noise; read the CI width, not p alone."]
    (RES / "P1_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
