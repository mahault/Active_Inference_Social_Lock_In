"""P1d: social channel = trust learned from delayed public verification (design D1).

Compositions: all pluralist, all committed monist, 50/50 pluralist/committed
(stratified: 2 of the 4 observers are pluralists). Fig-4 shift protocol, gates
from round 80, coarse grid gamma {0,3,10} x c0 {0,.3,1}, 20 paired seeds.
Seer sets (vmapped): 25% / 50% of agents = restricted agents 4-9 / 4-15
(the agents who otherwise cannot judge M), and 100% = everyone.
Rules: fixed W | relative log score (D1; kappa=1, lam=.98) | Gamma accuracy on
the same revealed outcomes | agree-with-me (O'Connor-Weatherall, m=2).
Modes: (a) "trust" - outcomes update trust only; (b) "trust+belief".
Delays D in {1,10,50}; "never" = the fixed-W, no-reveal run (identical for every
rule except agree, see the tests).

Support forgetting off (main) for everything; on (rho_L=.95, secondary) for
mode (b) fixed and relative only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.pluralist import extended as X  # noqa: E402
from src.pluralist import mixed as M  # noqa: E402
from src.pluralist import model  # noqa: E402
from src.pluralist import verified as V  # noqa: E402

N, N_FULL, SHIFT, STEPS = 24, 4, 80, 600
GAMMAS, C0S = [0.0, 3.0, 10.0], [0.0, 0.3, 1.0]
CELLS = list(itertools.product(GAMMAS, C0S))
DELAYS = [1, 10, 50]
SEER_FRACS = [0.25, 0.5, 1.0]


def seer_masks():
    m = np.zeros((3, N))
    m[0, 4:10] = 1
    m[1, 4:16] = 1
    m[2, :] = 1
    return m


def compositions():
    P, Cm = M.PLURALIST, M.COMMITTED
    return {"all_pluralist": (P,) * N,
            "all_committed": (Cm,) * N,
            "half_pluralist_committed": (P, P, Cm, Cm) + (P,) * 10 + (Cm,) * 10}


def runs():
    out = []
    for rule in ("fixed", "agree"):
        out.append(("noforget", "trust", rule, None if rule == "fixed" else 10))
    for rule, D in itertools.product(("relative", "gamma"), DELAYS):
        out.append(("noforget", "trust", rule, D))
    for rule, D in itertools.product(("fixed", "relative", "gamma", "agree"), DELAYS):
        out.append(("noforget", "trust+belief", rule, D))
    for rule, D in itertools.product(("fixed", "relative"), DELAYS):
        out.append(("forget", "trust+belief", rule, D))
    return out


def summarise(hist, p_final, types):
    types = np.asarray(types)
    res = []
    for f in range(hist["p_comm"].shape[0]):
        for g, (gam, c0) in enumerate(CELLS):
            e = hist["p_comm"][f, g, :, -1]
            post = slice(SHIFT, STEPS)
            tc = hist["trust_correct_seers"][f, g, :, post]
            fc = hist["frac_correct"][f, g, :, post]
            cell = {"seer_frac": SEER_FRACS[f], "gamma": gam, "c0": c0,
                    "pM_end": float(e.mean()), "locked_frac": float((e < 0.5).mean()),
                    "recovered_frac": float((e >= 0.75).mean()),
                    "trust_on_correct_seers_post": float(tc.mean()),
                    "frac_correct_post": float(fc.mean()),
                    "trust_excess_post": float((tc - fc).mean())}
            for t, key in M.TYPE_NAMES.items():
                if (types == t).any():
                    cell[f"pM_end_{key}"] = float(p_final[f, g][:, types == t].mean())
            res.append(cell)
    return res


def phase_counts(cells, f):
    cs = [c for c in cells if c["seer_frac"] == f]
    locked = sum(c["locked_frac"] >= 0.5 for c in cs)
    rec = sum(c["recovered_frac"] >= 0.5 for c in cs)
    return {"locked": locked, "recovered": rec, "unresolved": len(cs) - locked - rec,
            "mean_pM_end": float(np.mean([c["pM_end"] for c in cs]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "P1d"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    base = model.Config(*model.candidate_priors())
    t0 = time.time()
    _, pre = X.run_prehistory(X.ExtConfig(base), seeds, STEPS, SHIFT, N, N_FULL)
    q_ox = float(pre["oxygen"][:, SHIFT - 1].mean())
    masks = seer_masks()
    results, traces = {}, {}
    gam, c0s = [g for g, _ in CELLS], [c for _, c in CELLS]
    for (cname, types), (fname, mode, rule, D) in itertools.product(compositions().items(), runs()):
        mc = M.MixedConfig(base, types, q_ox, X.ExtConfig(base, rho_L=1.0 if fname == "noforget" else 0.95))
        vcfg = V.VerifyConfig(mc, rule=rule, mode=mode, delay=D)
        _, hist, p_final = V.verified_shift_protocol(vcfg, seeds, gam, c0s, masks, steps=STEPS, shift=SHIFT)
        key = f"{cname}|{fname}|{mode}|{rule}|{D}"
        cells = summarise(hist, p_final, types)
        results[key] = {"composition": cname, "forget": fname, "mode": mode, "rule": rule, "delay": D,
                        "cells": cells, "phases": {str(f): phase_counts(cells, f) for f in SEER_FRACS}}
        traces[key.replace("|", "__")] = np.stack([hist["trust_correct_seers"].mean(2),
                                                   hist["frac_correct"].mean(2)])  # (2,F,G,T)
        ph = results[key]["phases"]
        print(f"[{time.time() - t0:6.0f}s] {key:58s} " + "  ".join(
            f"f{f}: L{ph[str(f)]['locked']}/R{ph[str(f)]['recovered']} p{ph[str(f)]['mean_pM_end']:.2f}"
            for f in SEER_FRACS), flush=True)
    meta = {"seeds": seeds, "cells": CELLS, "delays": DELAYS, "seer_fracs": SEER_FRACS,
            "seer_sets": {"0.25": "agents 4-9 (restricted)", "0.5": "agents 4-15 (restricted)", "1.0": "all"},
            "committed_q_ox": q_ox, "kappa": 1.0, "lam": 0.98, "gamma_rho": 0.95, "agree_m": 2.0,
            "never": "delay never == fixed W, no reveals (identical for relative/gamma; tested)",
            "phases": "community-mean P(M>0) at round 599: locked <.5, recovered >=.75; cell phase by >=50% of seeds",
            "elapsed_seconds": time.time() - t0}
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    np.savez_compressed(out_dir / "trust_traces.npz", **traces)
    plot(results, traces, out_dir)
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")


def plot(results, traces, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rules = {"fixed": "#888888", "relative": "#2a78d6", "gamma": "#1baf7a", "agree": "#eb6834"}
    comps = list(compositions())
    xs = DELAYS + ["never"]
    fig, axes = plt.subplots(len(comps), 2, figsize=(13, 10), sharey=True)
    for r, cname in enumerate(comps):
        never = results[f"{cname}|noforget|trust|fixed|None"]["phases"]
        for c, mode in enumerate(("trust", "trust+belief")):
            ax = axes[r, c]
            for rule, color in rules.items():
                for f, ls in ((0.25, ":"), (1.0, "-")):
                    ys = []
                    for D in DELAYS:
                        k = f"{cname}|noforget|{mode}|{rule}|{D}"
                        if mode == "trust" and rule in ("fixed", "agree"):
                            k = f"{cname}|noforget|trust|{rule}|{None if rule == 'fixed' else 10}"
                        ys.append(results[k]["phases"][str(f)]["mean_pM_end"])
                    ys.append(never[str(f)]["mean_pM_end"] if rule != "agree" else np.nan)
                    ax.plot(range(len(xs)), ys, "o", ls=ls, color=color, lw=2, ms=5,
                            label=f"{rule}, seers {int(f * 100)}%")
            ax.axhline(0.5, color="#bbb", lw=0.8)
            ax.set_xticks(range(len(xs)), [str(x) for x in xs])
            ax.set_xlabel("reveal delay D (rounds)")
            ax.set_title(f"{cname}, mode: {mode}", fontsize=10)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[r, 0].set_ylabel("end P(M>0), mean over 9 gate cells")
    axes[0, 1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("P1d: community end belief vs reveal delay, by trust rule (support never forgets; 20 seeds; "
                 "seers 25% = restricted agents 4-9)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "p1d_end_belief_vs_delay.png", dpi=150)
    plt.close(fig)

    # Trust concentration over time: excess trust share on correct agents (seers), D=10, mode trust
    fig, axes = plt.subplots(1, len(comps), figsize=(15, 4.3), sharey=True)
    t = np.arange(1, STEPS + 1)
    for ax, cname in zip(axes, comps):
        for rule in ("relative", "gamma", "agree"):
            tr = traces[f"{cname}__noforget__trust__{rule}__10"]
            for gi, ls in ((0, "-"), (3, "--")):     # cells (0,0) and (3,0)
                ex = tr[0, 1, gi] - tr[1, 1, gi]       # seers 50%
                ax.plot(t, ex, color=rules[rule], ls=ls, lw=1.6,
                        label=f"{rule}, gamma={CELLS[gi][0]:g} c0={CELLS[gi][1]:g}")
        ax.axvline(SHIFT, color="#888", lw=0.8)
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.set_title(cname, fontsize=10)
        ax.set_xlabel("round")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("trust share on correct agents\nminus fraction correct (seers)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("P1d: does trust concentrate on truth-tracking agents? (D=10, mode trust, seers = 50% restricted)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "p1d_trust_concentration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
