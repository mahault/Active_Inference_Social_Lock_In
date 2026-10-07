"""P1c: mixed communities of pluralists, committed monists and neutral monists.

Agent type is a per-agent parameter (src/pluralist/mixed.py). Fig-4 shift
protocol (24 agents, observers 0-3 at full attention, restricted 4-23 at .01 on
G and M after the mass sign flips at round 80; fixed all-to-all W; fused beliefs,
support sharing alpha=.02 among pluralists). Gates on from round 80 only.

Compositions
  (a) monist-only, fraction committed in {0, .25, .5, .75, 1}
  (b) pluralist fraction in {.25, .5, .75, 1}, monist remainder split
      {all committed, 50/50, all neutral}  (fraction 0 = row (a))
  (c) placement at 25% pluralists: pluralists = all 4 observers + 2 restricted
      ("obsP") vs no pluralist observers ("obsM"); default rows are stratified
      (1 pluralist observer).
Grid gamma {0,1,3,6,10} x c0 {0,.05,.1,.3,1}; support forgetting off (main) /
rho_L = .95 (secondary). Seeds 0..19 paired across everything.

Phases on community-mean P(M>0) at round 599: locked < .5 <= unresolved < .75 <= recovered.
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

N, N_FULL, SHIFT, STEPS = 24, 4, 80, 600
GAMMAS = [0.0, 1.0, 3.0, 6.0, 10.0]
C0S = [0.0, 0.05, 0.1, 0.3, 1.0]
CELLS = list(itertools.product(GAMMAS, C0S))
FORGET = {"noforget": 1.0, "forget": 0.95}
SPLITS = {"allC": 1.0, "half": 0.5, "allN": 0.0}  # committed share of the monist remainder
TYPE_KEYS = {M.PLURALIST: "pluralist", M.COMMITTED: "committed", M.NEUTRAL: "neutral"}


def allocate(counts, slots):
    """Largest-remainder share of ``slots`` observer seats per type."""
    total = sum(counts.values())
    raw = {t: c * slots / total for t, c in counts.items()}
    seats = {t: int(np.floor(v)) for t, v in raw.items()}
    for t in sorted(raw, key=lambda t: raw[t] - seats[t], reverse=True)[: slots - sum(seats.values())]:
        seats[t] += 1
    return seats


def build_types(counts, observer_seats=None):
    obs = observer_seats or allocate(counts, N_FULL)
    rest = {t: counts[t] - obs.get(t, 0) for t in counts}
    order = [M.PLURALIST, M.COMMITTED, M.NEUTRAL]
    types = [t for t in order for _ in range(obs.get(t, 0))] + [t for t in order for _ in range(rest[t])]
    assert len(types) == N and all(v >= 0 for v in rest.values())
    return tuple(types)


def compositions():
    comps = {}
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        nc = int(round(f * N))
        comps[f"monist_committed{f}"] = dict(group="a", plural=0.0, committed_share=f,
                                             types=build_types({M.PLURALIST: 0, M.COMMITTED: nc, M.NEUTRAL: N - nc}))
    for f in (0.25, 0.5, 0.75):
        npl = int(round(f * N))
        for split, share in SPLITS.items():
            nc = int(round((N - npl) * share))
            counts = {M.PLURALIST: npl, M.COMMITTED: nc, M.NEUTRAL: N - npl - nc}
            comps[f"plural{f}_{split}"] = dict(group="b", plural=f, committed_share=share, types=build_types(counts))
            if f == 0.25:
                mono = {M.COMMITTED: nc, M.NEUTRAL: N - npl - nc}
                comps[f"plural{f}_{split}_obsP"] = dict(
                    group="c", plural=f, committed_share=share, placement="pluralists at all observers",
                    types=build_types(counts, {M.PLURALIST: 4, M.COMMITTED: 0, M.NEUTRAL: 0}))
                comps[f"plural{f}_{split}_obsM"] = dict(
                    group="c", plural=f, committed_share=share, placement="no pluralist observers",
                    types=build_types(counts, {M.PLURALIST: 0, **allocate(mono, N_FULL)}))
    comps["plural1.0"] = dict(group="b", plural=1.0, committed_share=None,
                              types=build_types({M.PLURALIST: N, M.COMMITTED: 0, M.NEUTRAL: 0}))
    return comps


def summarise(hist, types):
    pm = hist["p_mass_pos"]                 # (G, S, T, N)
    types = np.asarray(types)
    comm_end = pm[:, :, -1].mean(-1)        # (G, S)
    out = []
    for g in range(pm.shape[0]):
        e = comm_end[g]
        cell = {"pM_end": float(e.mean()), "pM_end_se": float(e.std(ddof=1) / np.sqrt(e.size)),
                "locked_frac": float((e < 0.5).mean()),
                "recovered_frac": float((e >= 0.75).mean()),
                "unresolved_frac": float(((e >= 0.5) & (e < 0.75)).mean())}
        for t, key in TYPE_KEYS.items():
            if (types == t).any():
                cell[f"pM_end_{key}"] = float(pm[g, :, -1][:, types == t].mean())
        cell["pM_end_observers"] = float(pm[g, :, -1, :N_FULL].mean())
        cell["pM_end_restricted"] = float(pm[g, :, -1, N_FULL:].mean())
        out.append(cell)
    return out


def phase(cell):
    if cell["locked_frac"] >= 0.5:
        return "locked"
    if cell["recovered_frac"] >= 0.5:
        return "recovered"
    return "unresolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "P1c"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    base = model.Config(*model.candidate_priors())
    t0 = time.time()

    # committed anchor = pluralist community's pre-shift oxygen support (as P1b)
    _, pre = X.run_prehistory(X.ExtConfig(base), seeds, STEPS, SHIFT, N, N_FULL)
    q_ox = float(pre["oxygen"][:, SHIFT - 1].mean())

    comps = compositions()
    gam = [g for g, _ in CELLS]
    c0s = [c for _, c in CELLS]
    results = {}
    for (cname, comp), (fname, rho) in itertools.product(comps.items(), FORGET.items()):
        mc = M.MixedConfig(base, comp["types"], q_ox, X.ExtConfig(base, rho_L=rho))
        hist = M.mixed_shift_protocol(mc, seeds, gam, c0s, steps=STEPS, shift=SHIFT, n_full=N_FULL)
        cells = summarise(hist, comp["types"])
        key = f"{cname}|{fname}"
        results[key] = {k: v for k, v in comp.items() if k != "types"}
        results[key].update(
            forget=fname, types=list(comp["types"]),
            cells={f"{g}|{c}": cell for (g, c), cell in zip(CELLS, cells)},
            n_locked=sum(phase(c) == "locked" for c in cells),
            n_recovered=sum(phase(c) == "recovered" for c in cells),
            n_unresolved=sum(phase(c) == "unresolved" for c in cells))
        print(f"[{time.time() - t0:6.0f}s] {key:34s} locked={results[key]['n_locked']:2d} "
              f"unres={results[key]['n_unresolved']:2d} rec={results[key]['n_recovered']:2d}", flush=True)

    meta = {"seeds": seeds, "steps": STEPS, "shift": SHIFT, "gammas": GAMMAS, "c0s": C0S,
            "committed_q_ox": q_ox, "forget": FORGET, "W": "fixed all-to-all (Fig 4)",
            "phases": "community-mean P(M>0) at round 599: locked <.5, unresolved .5-.75, recovered >=.75; "
                      "a cell's phase = the phase of >=50% of seeds (else unresolved)",
            "elapsed_seconds": time.time() - t0}
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    plot(results, out_dir)
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")


def plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncell = len(CELLS)
    split_style = {"allC": ("#eb6834", "-", "monists all committed"),
                   "half": ("#7a5cc2", "--", "monists 50/50"),
                   "allN": ("#1baf7a", ":", "monists all neutral")}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for r, fname in enumerate(FORGET):
        # (a) monist-only
        ax = axes[r, 0]
        xs = [0.0, 0.25, 0.5, 0.75, 1.0]
        for metric, color, label in (("n_locked", "#b8461b", "locked"),
                                     ("n_recovered", "#2a78d6", "recovered")):
            ys = [results[f"monist_committed{x}|{fname}"][metric] / ncell for x in xs]
            ax.plot(xs, ys, "o-", color=color, lw=2, ms=6, label=label)
        ax.set_xlabel("fraction of monists that are committed")
        ax.set_title(f"(a) monist-only communities, support {fname}", fontsize=10)
        ax.set_ylabel("fraction of gamma x c0 cells")
        ax.legend(frameon=False, fontsize=8)
        # (b) pluralist fraction, locked and recovered
        for col, (metric, title) in enumerate((("n_locked", "locked"), ("n_recovered", "recovered")), start=1):
            ax = axes[r, col]
            for split, (color, ls, label) in split_style.items():
                xs = [0.0, 0.25, 0.5, 0.75, 1.0]
                ys = []
                for x in xs:
                    if x == 0.0:
                        k = f"monist_committed{SPLITS[split]}|{fname}"
                    elif x == 1.0:
                        k = f"plural1.0|{fname}"
                    else:
                        k = f"plural{x}_{split}|{fname}"
                    ys.append(results[k][metric] / ncell)
                ax.plot(xs, ys, "o", color=color, ls=ls, lw=2, ms=6, label=label)
                for place, marker in (("obsP", "^"), ("obsM", "v")):
                    k = f"plural0.25_{split}_{place}|{fname}"
                    ax.plot([0.25], [results[k][metric] / ncell], marker, color=color, ms=8,
                            mfc="white", mew=1.5)
            ax.plot([], [], "^", color="#555", mfc="white", label="25%: pluralists = observers")
            ax.plot([], [], "v", color="#555", mfc="white", label="25%: no pluralist observers")
            ax.set_xlabel("fraction of agents that are pluralists")
            ax.set_title(f"(b) cells {title}, support {fname}", fontsize=10)
            if col == 1 and r == 0:
                ax.legend(frameon=False, fontsize=7.5)
        for ax in axes[r]:
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("P1c: share of the 5x5 gamma x c0 grid that ends locked (<.5) or recovered (>=.75), "
                 "by community composition (20 paired seeds per cell)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "p1c_lockin_vs_composition.png", dpi=150)
    plt.close(fig)

    # Per-type end beliefs at four representative cells, 50/50 and all-committed splits
    show = [(0.0, 0.0), (3.0, 0.0), (3.0, 0.3), (10.0, 1.0)]
    tcol = {"pluralist": "#2a78d6", "committed": "#eb6834", "neutral": "#1baf7a"}
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    fname = "noforget"
    for r, split in enumerate(("allC", "half")):
        for c, (g, c0) in enumerate(show):
            ax = axes[r, c]
            xs = [0.0, 0.25, 0.5, 0.75, 1.0]
            keys = [f"monist_committed{SPLITS[split]}|{fname}" if x == 0 else
                    f"plural1.0|{fname}" if x == 1 else f"plural{x}_{split}|{fname}" for x in xs]
            for t in ("pluralist", "committed", "neutral"):
                pts = [(x, results[k]["cells"][f"{g}|{c0}"].get(f"pM_end_{t}")) for x, k in zip(xs, keys)]
                pts = [(x, y) for x, y in pts if y is not None]
                if pts:
                    ax.plot(*zip(*pts), "o-", color=tcol[t], lw=2, ms=6, label=t)
            ys = [results[k]["cells"][f"{g}|{c0}"]["pM_end"] for k in keys]
            ax.plot(xs, ys, "s--", color="#333", lw=1.2, ms=4, label="community")
            ax.axhline(0.5, color="#bbb", lw=0.8)
            ax.axhline(0.75, color="#bbb", lw=0.8, ls=":")
            ax.set_title(f"gamma={g}, c0={c0}; monists {split}", fontsize=10)
            ax.set_xlabel("fraction pluralists")
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[r, 0].set_ylabel("end P(M>0) by agent type")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("P1c: who drags whom? end-of-run P(M>0) per agent type (support never forgets; "
                 "20 seeds; grey lines at .5 lock-in and .75 recovered)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "p1c_per_type_beliefs.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
