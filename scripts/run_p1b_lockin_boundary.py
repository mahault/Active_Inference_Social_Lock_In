"""P1b: where does post-shift lock-in happen, and does the boundary move for pluralists?

Changes vs P1 (scripts/run_p1_pluralist_two_channel.py):
  * world-channel gates switch on only at the shift (round 80), so every arm of
    one agent type enters the shift from the identical belief state;
  * fair monist: reference = moment-matched Gaussian of the pluralist's prior
    mixture (``monist_moment_matched``); the original log-linear pooled monist
    (``gaussian_reference``) is kept as a secondary arm;
  * gamma x c0 grid, forgetting on/off; lock-in := community-mean P(M>0) < .5
    at the final round (599);
  * social arms where trust is learnable.

Grid A (fixed W, Fig-4 attention .01 for restricted agents):
  gamma {0,1,3,6,10} x c0 {0,.05,.1,.3,1}  (c0 = 0 -> resource gate off)
  agents: pluralist (rho_L 1 / .95), monist_mm, monist_pool (rho_L irrelevant, K=1),
          monist_incumbent = moment-matched with the pluralist community's pre-shift
          support weights (q at round 79), i.e. a single-theory agent anchored to
          what the community believed when the world changed.
Phases at round 599 (community-mean P(M>0)): locked < .5 <= unresolved < .75 <= recovered.
Arms B (social channel), at (gamma, c0) in {(0,0), (6,0), (0,.3), (6,.3)}:
  fig4/fixedW, fig4/trust_own, fig4/trust_CG, reduced/fixedW, reduced/trust_own
  where reduced = restricted agents see G and M at base attention .1 (not .01);
  trust_own = judge neighbours on own post-gate attention; trust_CG = restricted
  agents judge neighbours only on C and G (weights 1,1,0), observers on all.
  Trust uses the "excess" surprisal (P1 showed full vs excess makes no
  difference to outcomes; excess gives the larger trust contrast).

Seeds 0..19 (phase-2 RNG seed+1000), paired across all arms.
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
GAMMAS = [0.0, 1.0, 3.0, 6.0, 10.0]
C0S = [0.0, 0.05, 0.1, 0.3, 1.0]
RHO_L = 0.95
B_POINTS = [(0.0, 0.0), (6.0, 0.0), (0.0, 0.3), (6.0, 0.3)]


def gate_kw(gamma, c0):
    kw = {"gamma": gamma}
    if c0 > 0:
        kw.update(resources=True, c0=c0)
    return kw


def recovery_time(series):
    post = series[SHIFT:] > 0.5
    if not post[-1]:
        return np.nan
    bad = np.nonzero(~post)[0]
    return float(bad[-1] + 1 if bad.size else 0)


def cell_summary(pre, post, pluralist):
    pm = np.concatenate((pre["p_mass_pos"], post["p_mass_pos"]), axis=1)
    comm = pm.mean(2)                                   # (S, T)
    end = comm[:, -1]
    t = np.array([recovery_time(s) for s in comm])
    out = {
        "pM_preshift_mean": float(comm[:, SHIFT - 1].mean()),
        "pM_end_mean": float(end.mean()), "pM_end_se": float(end.std(ddof=1) / np.sqrt(len(end))),
        "pM_end_restricted_mean": float(pm[:, -1, N_FULL:].mean()),
        "lockin_frac": float((end < 0.5).mean()),
        "recovered_frac": float((end >= 0.75).mean()),
        "t_recover_median": None if np.isnan(t).all() else float(np.nanmedian(t)),
        "attnM_post_observers": float(post["attention_M"][:, :, :N_FULL].mean()),
        "trust_on_obs_restricted_rows_end": float(post["trust_on_observers"][:, -50:, N_FULL:].mean()),
        "trust_on_obs_observer_rows_end": float(post["trust_on_observers"][:, -50:, :N_FULL].mean()),
    }
    if pluralist:
        out["qox_end_mean"] = float(post["oxygen"][:, -1].mean())
    return out, comm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "P1b"))
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    base = model.Config(*model.candidate_priors())
    agents = {
        "pluralist_noforget": (base, 1.0, True),
        "pluralist_forget": (base, RHO_L, True),
        "monist_mm": (X.monist_moment_matched(base), 1.0, False),
        "monist_pool": (X.monist_config(base), 1.0, False),
    }
    t0 = time.time()
    # Incumbent monist: weights = pluralist community support at the shift.
    _, plural_pre = X.run_prehistory(X.ExtConfig(base), seeds, STEPS, SHIFT, N, N_FULL)
    q_ox = float(plural_pre["oxygen"][:, SHIFT - 1].mean())
    agents["monist_incumbent"] = (X.monist_moment_matched(base, (1 - q_ox, q_ox)), 1.0, False)
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)  # noqa: E731

    # ---------------- Grid A ----------------
    grid, comm_traces = {}, {}
    for name, (cfg, rho, plural) in agents.items():
        root_cfg = X.ExtConfig(cfg, rho_L=rho)
        state, pre = X.run_prehistory(root_cfg, seeds, STEPS, SHIFT, N, N_FULL)
        grid[name] = {}
        for gamma, c0 in itertools.product(GAMMAS, C0S):
            ecfg = X.ExtConfig(cfg, rho_L=rho, **gate_kw(gamma, c0))
            _, post = X.run_post_shift(ecfg, state, seeds, STEPS, SHIFT, N, N_FULL)
            summ, comm = cell_summary(pre, post, plural)
            grid[name][f"{gamma}|{c0}"] = summ
            comm_traces[f"{name}__g{gamma}__c{c0}"] = comm.mean(0)
            log(f"A {name:20s} gamma={gamma:4} c0={c0:4}  pre={summ['pM_preshift_mean']:.3f} "
                f"end={summ['pM_end_mean']:.3f} lock={summ['lockin_frac']:.2f}")

    # ---------------- Arms B ----------------
    social = {}
    if not args.skip_b:
        cg = np.ones((N, 3))
        cg[N_FULL:, 2] = 0.0
        variants = {  # name: (restricted attention, trust?, trust_attention)
            "fig4|fixedW": (0.01, False, None),
            "fig4|trust_own": (0.01, True, None),
            "fig4|trust_CG": (0.01, True, cg),
            "reduced|fixedW": (0.1, False, None),
            "reduced|trust_own": (0.1, True, None),
        }
        for name in ("pluralist_noforget", "pluralist_forget", "monist_mm"):
            cfg, rho, plural = agents[name]
            for vname, (ra, trust, tatt) in variants.items():
                tkw = {"trust": True, "trust_surprisal": "excess"} if trust else {}
                state, pre = X.run_prehistory(X.ExtConfig(cfg, rho_L=rho, **tkw), seeds,
                                              STEPS, SHIFT, N, N_FULL, trust_attention=tatt)
                for gamma, c0 in B_POINTS:
                    ecfg = X.ExtConfig(cfg, rho_L=rho, **tkw, **gate_kw(gamma, c0))
                    _, post = X.run_post_shift(ecfg, state, seeds, STEPS, SHIFT, N, N_FULL,
                                               restricted_attention=ra, trust_attention=tatt)
                    summ, _ = cell_summary(pre, post, plural)
                    social[f"{name}|{vname}|{gamma}|{c0}"] = summ
                    log(f"B {name:20s} {vname:18s} g={gamma} c0={c0} end={summ['pM_end_mean']:.3f} "
                        f"lock={summ['lockin_frac']:.2f} trR={summ['trust_on_obs_restricted_rows_end']:.3f} "
                        f"trO={summ['trust_on_obs_observer_rows_end']:.3f}")

    meta = {
        "seeds": seeds, "steps": STEPS, "shift": SHIFT, "gammas": GAMMAS, "c0s": C0S, "rho_L_forget": RHO_L,
        "gates": "active from round 80 only", "lockin": "community-mean P(M>0) at round 599 < .5",
        "monist_mm": "K=1 reference = moment-matched Gaussian of the pluralist prior mixture (weights .5/.5)",
        "monist_pool": "K=1 reference = model.gaussian_reference (log-linear pool), as in P1",
        "B_points": B_POINTS, "elapsed_seconds": time.time() - t0,
    }
    meta["monist_incumbent"] = f"moment-matched with weights (1-q, q), q = pluralist pre-shift q_ox = {q_ox:.5f}"
    meta["phases"] = "locked: end pM < .5; unresolved: .5-.75; recovered: >= .75"
    prev = out_dir / "summary.json"
    if args.skip_b and prev.exists():
        social = json.loads(prev.read_text()).get("social", {})
        meta["social_from_previous_run"] = True
    prev.write_text(json.dumps({"meta": meta, "grid": grid, "social": social}, indent=1))
    np.savez_compressed(out_dir / "community_traces.npz", **comm_traces)
    plot(grid, out_dir)
    log(f"done -> {out_dir}")


def plot(grid, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    titles = {"pluralist_noforget": "pluralist, no forgetting", "pluralist_forget": "pluralist, forgetting .95",
              "monist_mm": "monist, moment-matched prior", "monist_incumbent": "monist, incumbent anchor",
              "monist_pool": "monist, pooled (P1)"}
    colors = {"pluralist_noforget": "#2a78d6", "pluralist_forget": "#1baf7a", "monist_mm": "#eb6834",
              "monist_incumbent": "#eda100", "monist_pool": "#e87ba4"}
    styles = {"pluralist_noforget": "-", "pluralist_forget": "--", "monist_mm": "-",
              "monist_incumbent": "-.", "monist_pool": ":"}
    # Diverging: orange (locked, < .5) - neutral grey at .5 - blue (recovered)
    cmap = LinearSegmentedColormap.from_list("div", ["#b8461b", "#eb6834", "#e8e7e3", "#2a78d6", "#104281"])
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    names = [n for n in titles if n in grid]
    G, C = len(GAMMAS), len(C0S)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4.1 * (len(names) + 1), 4.8))
    for ax, name in zip(axes, names):
        end = np.array([[grid[name][f"{g}|{c}"]["pM_end_mean"] for c in C0S] for g in GAMMAS])
        lock = np.array([[grid[name][f"{g}|{c}"]["lockin_frac"] for c in C0S] for g in GAMMAS])
        im = ax.imshow(end, origin="lower", cmap=cmap, norm=norm, aspect="auto")
        for i, j in itertools.product(range(G), range(C)):
            ax.text(j, i, f"{end[i, j]:.2f}" + ("\nL" if lock[i, j] >= 0.5 else ""),
                    ha="center", va="center", fontsize=7.5,
                    color="white" if abs(end[i, j] - 0.5) > 0.3 else "#222")
        ax.set_xticks(range(C), [str(c) for c in C0S])
        ax.set_yticks(range(G), [str(g) for g in GAMMAS])
        ax.set_xlabel("c0 (resource cost)")
        ax.set_title(titles[name], fontsize=10)
    axes[0].set_ylabel("gamma (conviction gate)")
    cb = fig.colorbar(im, ax=axes[len(names) - 1], fraction=0.046)
    cb.set_label("community mean P(M>0) at round 599")
    ax = axes[-1]
    for name in names:
        lock = np.array([[grid[name][f"{g}|{c}"]["lockin_frac"] for c in C0S] for g in GAMMAS])
        if lock.min() < 0.5 < lock.max():
            ax.contour(np.arange(C), np.arange(G), lock, levels=[0.5], colors=colors[name],
                       linestyles=styles[name], linewidths=2)
            ax.plot([], [], color=colors[name], ls=styles[name], lw=2, label=titles[name])
        else:
            state = "always locked" if lock.min() >= 0.5 else "never locked"
            ax.plot([], [], color=colors[name], ls=styles[name], lw=2, label=f"{titles[name]} ({state})")
    ax.set_xticks(range(C), [str(c) for c in C0S])
    ax.set_yticks(range(G), [str(g) for g in GAMMAS])
    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(-0.5, G - 0.5)
    ax.set_xlabel("c0 (resource cost)")
    ax.set_title("lock-in boundary (>=50% seeds end < .5)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18))
    ax.grid(alpha=0.2)
    fig.suptitle("P1b: end state on the gamma x c0 plane, gates on from round 80, 20 paired seeds "
                 "(L = locked in >= 50% of seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "p1b_lockin_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
