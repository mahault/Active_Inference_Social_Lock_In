"""P1: do pluralist agents resist two-channel precision collapse better than monists?

Factorial on the Hyland Fig. 4 shift protocol (24 agents, 4 full observers,
20 restricted to attention .01 on G and M after the mass sign flips at round 80;
600 rounds; fused beliefs + support sharing alpha = .02, i.e. the N11 arm):

  agent   {pluralist (K=2), monist (K=1, gaussian_reference)}
  world   {open, conviction (gamma=3), resources (c0=.1)}
  social  {fixed W, trust (Gamma, full surprisal), trust (excess surprisal)}
  forget  {off (rho_L=1), on (rho_L=.95)}
  + supplementary: pluralist, conviction measured on support (max-min softmax L).

Seeds 0..49 (phase-2 RNG seed+1000), paired across all arms.

Usage: python scripts/run_p1_pluralist_two_channel.py [--seeds 50] [--out experiments/results/P1]
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

from src.pluralist import model  # noqa: E402
from src.pluralist import extended as X  # noqa: E402

SHIFT, STEPS, N, N_FULL = 80, 600, 24, 4
GAMMA, C0, RHO_L = 3.0, 0.1, 0.95
END = slice(STEPS - 50, STEPS)  # last 50 rounds

WORLD = {"open": {}, "conviction": {"gamma": GAMMA}, "resources": {"resources": True, "c0": C0}}
SOCIAL = {"fixedW": {}, "trust": {"trust": True},
          "trust_excess": {"trust": True, "trust_surprisal": "excess"}}
FORGET = {"noforget": {}, "forget": {"rho_L": RHO_L}}


def sustained_recovery_time(series):
    """First post-shift round t with series > .5 at every round >= t; nan if never."""
    post = series[SHIFT:] > 0.5
    if not post[-1]:
        return np.nan
    bad = np.nonzero(~post)[0]
    return float(bad[-1] + 1 if bad.size else 0)


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "se": None, "n": 0}
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0,
            "n": int(x.size)}


def summarise(out, pluralist):
    pm = out["p_mass_pos"]                     # (S, T, N)
    groups = {"all": slice(0, N), "observers": slice(0, N_FULL), "restricted": slice(N_FULL, N)}
    per_seed = {}
    for g, ids in groups.items():
        per_seed[f"pM_end_{g}"] = pm[:, END, ids].mean((1, 2))
        per_seed[f"frac_pM_gt_half_final_{g}"] = (pm[:, -1, ids] > 0.5).mean(1)
        per_seed[f"t_recover_{g}"] = np.array([sustained_recovery_time(s) for s in pm[:, :, ids].mean(2)])
        if pluralist:
            q = out["oxygen"]
            per_seed[f"qox_end_{g}"] = q[:, END, ids].mean((1, 2))
            per_seed[f"frac_qox_gt_half_final_{g}"] = (q[:, -1, ids] > 0.5).mean(1)
    per_seed["pM_preshift_all"] = pm[:, SHIFT - 1].mean(1)
    per_seed["attnM_post_observers"] = out["attention_M"][:, SHIFT:, :N_FULL].mean((1, 2))
    per_seed["attnM_pre_all"] = out["attention_M"][:, :SHIFT].mean((1, 2))
    per_seed["r_post_mean"] = out["r"][:, SHIFT:].mean((1, 2))
    per_seed["trust_on_obs_post_restricted_rows"] = out["trust_on_observers"][:, SHIFT:, N_FULL:].mean((1, 2))
    per_seed["trust_on_obs_post_observer_rows"] = out["trust_on_observers"][:, SHIFT:, :N_FULL].mean((1, 2))
    summary = {k: mean_ci(v) for k, v in per_seed.items()}
    for g in groups:
        summary[f"n_seeds_never_recovered_{g}"] = int(np.isnan(per_seed[f"t_recover_{g}"]).sum())
    return summary, per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "P1"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    base = model.Config(*model.candidate_priors())
    bases = {"pluralist": base, "monist": X.monist_config(base)}

    arms = [(a, w, s, f) for a, w, s, f in itertools.product(bases, WORLD, SOCIAL, FORGET)]
    arms += [("pluralist", "conviction_support", "fixedW", f) for f in FORGET]
    results, traces, per_seed_all = {}, {}, {}
    t0 = time.time()
    for agent, world, social, forget in arms:
        name = f"{agent}|{world}|{social}|{forget}"
        wkw = ({"gamma": GAMMA, "conviction_mode": "support"} if world == "conviction_support"
               else WORLD[world])
        ecfg = X.ExtConfig(bases[agent], **wkw, **SOCIAL[social], **FORGET[forget])
        out = X.shift_protocol(ecfg, seeds, steps=STEPS, shift=SHIFT, n=N, n_full=N_FULL)
        summ, per_seed = summarise(out, agent == "pluralist")
        results[name] = {"agent": agent, "world": world, "social": social, "forget": forget, **summ}
        per_seed_all[name] = per_seed
        traces[name] = {"pM_all": out["p_mass_pos"].mean(2), "pM_restricted": out["p_mass_pos"][:, :, N_FULL:].mean(2)}
        print(f"[{time.time() - t0:6.0f}s] {name:48s} pM_end_all={summ['pM_end_all']['mean']:.3f} "
              f"frac_final={summ['frac_pM_gt_half_final_all']['mean']:.3f} "
              f"never={summ['n_seeds_never_recovered_all']}", flush=True)

    # Paired (same seeds, same observation noise) pluralist - monist contrasts
    contrasts = {}
    for world, social, forget in itertools.product(WORLD, SOCIAL, FORGET):
        p = per_seed_all[f"pluralist|{world}|{social}|{forget}"]
        m = per_seed_all[f"monist|{world}|{social}|{forget}"]
        d = p["pM_end_all"] - m["pM_end_all"]
        dt = p["t_recover_all"] - m["t_recover_all"]
        contrasts[f"{world}|{social}|{forget}"] = {
            "d_pM_end_all": mean_ci(d), "frac_seeds_pluralist_better": float((d > 0).mean()),
            "d_t_recover_all_bothrecovered": mean_ci(dt),
        }

    meta = {
        "protocol": "Hyland Fig.4 (N11): 24 agents, observers 0-3, restricted 4-23 at attention .01 on G,M after shift",
        "shift": SHIFT, "steps": STEPS, "seeds": seeds, "phase2_seed_offset": 1000,
        "params": {"gamma": GAMMA, "c0": C0, "r_min": 0.1, "r_init": 1.0, "R_in": 0.1, "delta": 0.05,
                   "budget_fraction": 0.5, "trust_rho": 0.95, "trust_alpha0": 1.0, "trust_beta0": 1.0,
                   "rho_L_forget": RHO_L, "sigma": base.sigma, "omega": base.omega, "alpha": base.alpha,
                   "conviction_mode": "predictive |2P(M>0)-1| unless world=conviction_support"},
        "readouts": {"pM": "P(M>0) under agent's mixture belief (theory-free, common to both agent types)",
                     "qox": "softmax(L)[oxygen], pluralist only",
                     "END": "rounds 550-599", "t_recover": "rounds after shift until community-mean pM stays >.5"},
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "arms": results, "pluralist_minus_monist": contrasts}, indent=1))
    np.savez_compressed(out_dir / "traces.npz", **{k.replace("|", "__") + "__" + kk: v
                                                  for k, d in traces.items() for kk, v in d.items()})
    plot(traces, results, out_dir)
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")


def plot(traces, results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"pluralist": "#2a78d6", "monist": "#eb6834"}
    styles = {"fixedW": "-", "trust": "--", "trust_excess": ":"}
    worlds = list(WORLD)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    t = np.arange(1, STEPS + 1)
    for r, forget in enumerate(FORGET):
        for c, world in enumerate(worlds):
            ax = axes[r, c]
            for agent in ("pluralist", "monist"):
                for social in SOCIAL:
                    y = traces[f"{agent}|{world}|{social}|{forget}"]["pM_all"]
                    ax.plot(t, y.mean(0), color=colors[agent], ls=styles[social], lw=1.6,
                            label=f"{agent}, {social}")
            ax.axvline(SHIFT, color="#888", lw=0.8)
            ax.axhline(0.5, color="#bbb", lw=0.6)
            ax.set_title(f"world: {world} | support: {forget}", fontsize=10)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel("community mean P(M>0)")
            if r == 1:
                ax.set_xlabel("round (mass flips at 80)")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="center right")
    fig.suptitle("P1: post-shift recovery, pluralist (blue) vs monist (orange); "
                 f"mean over {traces[next(iter(traces))]['pM_all'].shape[0]} paired seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "p1_recovery_timecourses.png", dpi=150)
    plt.close(fig)

    # End-state dot plot with +-1.96 SE
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, forget in zip(axes, FORGET):
        rows = [(w, s) for w in worlds for s in SOCIAL]
        for i, (w, s) in enumerate(rows):
            for k, agent in enumerate(("pluralist", "monist")):
                m = results[f"{agent}|{w}|{s}|{forget}"]["pM_end_all"]
                ax.errorbar(m["mean"], i + (k - 0.5) * 0.3, xerr=1.96 * m["se"], fmt="o", ms=6,
                            color=colors[agent], label=agent if i == 0 else None)
        ax.set_yticks(range(len(rows)), [f"{w} / {s}" for w, s in rows])
        ax.axvline(0.5, color="#bbb", lw=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("community mean P(M>0), rounds 550-599")
        ax.set_title(f"support {forget}", fontsize=10)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "p1_end_state.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
