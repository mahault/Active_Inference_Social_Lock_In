"""Figure for P1e: trust sharpness (mode a) and delay under full / sparse reveals (mode b)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = ROOT / "experiments" / "results" / "P1e"
R = json.loads((OUT / "summary.json").read_text())["runs"]
COMPS = {"all_pluralist": "#2a78d6", "half_pluralist_committed": "#7a5cc2", "all_committed": "#eb6834"}


def m(key):
    v = R.get(key)
    return None if v is None else np.mean([v["phases"][f]["mean_pM_end"] for f in ("0.25", "0.5", "1.0")])


fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
ax = axes[0]
for c, col in COMPS.items():
    ks = [1.0, 3.0, 10.0, 30.0]
    ys = [m(f"{c}|noforget|trust|-|relative|10|{k}|0.98") for k in ks]
    pts = [(k, y) for k, y in zip(ks, ys) if y is not None]
    ax.plot(*zip(*pts), "o-", color=col, lw=2, label=c)
    ax.axhline(m(f"{c}|noforget|trust|-|fixed|None|1.0|0.98"), color=col, ls=":", lw=1.2)
ax.set_xscale("log")
ax.set_xlabel("trust sharpness kappa")
ax.set_title("(a) trust only, D=10: relative rule (dotted = fixed W)", fontsize=10)
ax.set_ylabel("end P(M>0), mean over 9 cells x 3 seer sets")
ax.legend(frameon=False, fontsize=8)
for ax, (every, title) in zip(axes[1:], ((1, "(b) every round revealed, full precision"),
                                          (10, "(b) every 10th round revealed, full precision"))):
    suf = "" if every == 1 else f"|every{every}"
    for c, col in COMPS.items():
        Ds = [1, 10, 50] if every == 1 else [1, 50]
        for rule, k, ls in (("fixed", 1.0, ":"), ("relative", 30.0, "-")):
            ys = [m(f"{c}|noforget|trust+belief|full|{rule}|{D}|{k}|0.98{suf}") for D in Ds]
            pts = [(i, y) for i, y in enumerate(ys) if y is not None]
            if pts:
                ax.plot(*zip(*pts), "o", ls=ls, color=col, lw=2,
                        label=f"{c}, {'fixed W' if rule == 'fixed' else 'relative k=30'}")
        ax.set_xticks(range(len(Ds)), [str(D) for D in Ds])
    ax.set_xlabel("reveal delay D")
    ax.set_title(title, fontsize=10)
axes[2].legend(frameon=False, fontsize=7)
for ax in axes:
    ax.axhline(0.5, color="#bbb", lw=0.8)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("P1e: sharper trust helps when outcomes only update trust; with full-precision reveals the "
             "outcomes themselves dominate and delay does not matter (20 paired seeds, support never forgets)",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "p1e_summary.png", dpi=150)
