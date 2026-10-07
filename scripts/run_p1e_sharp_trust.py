"""P1e: P1d with full-precision public reveals and a trust-sharpness/memory sweep.

Setup as P1d (scripts/run_p1d_verified_trust.py): compositions all pluralist /
all committed / 50:50 pluralist:committed; Fig-4 shift protocol; gates from round
80; grid gamma {0,3,10} x c0 {0,.3,1}; seers = restricted 4-9 (25%), 4-15 (50%),
all (100%); 20 paired seeds.

Stage "main" (support never forgets):
  mode (b) trust+belief, FULL-precision reveals (y* at precision 1/sigma^2):
      relative x kappa {1,3,10,30} x D {1,10,50}, lam = .98
      fixed W x D {1,10,50}; agree (m=2) x D {1,10,50}
  mode (a) trust only (reveal precision irrelevant):
      relative x kappa {1,3,10,30} at D = 10, plus kappa {1,30} at D {1,50}; lam = .98
      fixed W (= D never); agree at D = 10
  secondary, mode (b) with ATTENTION-weighted reveals (= P1d settings):
      relative kappa=1, fixed W, each x D {1,10,50}; agree D = 10
  "never" for every rule = fixed W without reveals (exact, tested).
  all_committed: relative only at kappa {1,30} (trust has no correct agent to find).
Stage "second" (needs --kappa = best kappa from main):
  lam {.9, 1.0} at best kappa, D = 10, modes (a) and (b, full)
  mode (b) attention-weighted relative at best kappa x D {1,10,50}
  support forgetting on (rho_L = .95): mode (b) full, relative(best kappa) and fixed x D {1,10,50}
Stage "sparse" (extra): only every 10th round's outcome is revealed (sparse
public replications), full precision, mode (b): fixed W and relative kappa {1,30}
x D {1,50}.
Per-seed end values are saved to per_seed.npz for the statistics script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.run_p1d_verified_trust import CELLS, N, N_FULL, SEER_FRACS, SHIFT, STEPS, compositions, seer_masks  # noqa: E402
from src.pluralist import extended as X  # noqa: E402
from src.pluralist import mixed as M  # noqa: E402
from src.pluralist import model  # noqa: E402
from src.pluralist import verified as V  # noqa: E402

KAPPAS = [1.0, 3.0, 10.0, 30.0]
DELAYS = [1, 10, 50]


def run_key(r):
    key = "|".join(str(x) for x in (r["comp"], r["forget"], r["mode"], r["prec"], r["rule"], r["D"], r["kappa"], r["lam"]))
    return key + (f"|every{r['every']}" if r.get("every", 1) != 1 else "")


def spec(comp, forget, mode, prec, rule, D, kappa=1.0, lam=0.98, every=1):
    return dict(comp=comp, forget=forget, mode=mode, prec=prec, rule=rule, D=D, kappa=kappa, lam=lam, every=every)


def sparse_runs():
    """Extra: sparse public replications (only every 10th round's outcome is
    revealed), full precision, mode (b)."""
    runs = []
    for comp in compositions():
        for D in (1, 50):
            runs.append(spec(comp, "noforget", "trust+belief", "full", "fixed", D, every=10))
            for k in ((30.0,) if comp == "all_committed" else (1.0, 30.0)):
                runs.append(spec(comp, "noforget", "trust+belief", "full", "relative", D, k, every=10))
    return runs


def main_runs():
    runs = []
    for comp in compositions():
        ks = [1.0, 30.0] if comp == "all_committed" else KAPPAS
        for k in ks:
            for D in DELAYS:
                runs.append(spec(comp, "noforget", "trust+belief", "full", "relative", D, k))
        for D in DELAYS:
            runs.append(spec(comp, "noforget", "trust+belief", "full", "fixed", D))
            runs.append(spec(comp, "noforget", "trust+belief", "full", "agree", D))
        for k in ks:
            runs.append(spec(comp, "noforget", "trust", "-", "relative", 10, k))
        for k in (1.0, 30.0):
            for D in (1, 50):
                runs.append(spec(comp, "noforget", "trust", "-", "relative", D, k))
        runs.append(spec(comp, "noforget", "trust", "-", "fixed", None))
        runs.append(spec(comp, "noforget", "trust", "-", "agree", 10))
        for D in DELAYS:
            runs.append(spec(comp, "noforget", "trust+belief", "attention", "relative", D, 1.0))
            runs.append(spec(comp, "noforget", "trust+belief", "attention", "fixed", D))
        runs.append(spec(comp, "noforget", "trust+belief", "attention", "agree", 10))
    return runs


def second_runs(kbest):
    runs = []
    for comp in compositions():
        for lam in (0.9, 1.0):
            runs.append(spec(comp, "noforget", "trust", "-", "relative", 10, kbest, lam))
            runs.append(spec(comp, "noforget", "trust+belief", "full", "relative", 10, kbest, lam))
        for D in DELAYS:
            runs.append(spec(comp, "noforget", "trust+belief", "attention", "relative", D, kbest))
            runs.append(spec(comp, "forget", "trust+belief", "full", "relative", D, kbest))
            runs.append(spec(comp, "forget", "trust+belief", "full", "fixed", D))
    return runs


def per_seed(hist, p_final, types):
    types = np.asarray(types)
    out = {"end": hist["p_comm"][..., -1],                                   # (F,G,S)
           "trust_excess": (hist["trust_correct_seers"] - hist["frac_correct"])[..., SHIFT:].mean(-1)}
    for t, name in M.TYPE_NAMES.items():
        if (types == t).any():
            out[f"end_{name}"] = p_final[..., types == t].mean(-1)
    return out


def cell_table(ps):
    cells = []
    for f, sf in enumerate(SEER_FRACS):
        for g, (gam, c0) in enumerate(CELLS):
            e = ps["end"][f, g]
            c = {"seer_frac": sf, "gamma": gam, "c0": c0, "pM_end": float(e.mean()),
                 "locked_frac": float((e < 0.5).mean()), "recovered_frac": float((e >= 0.75).mean()),
                 "trust_excess_post": float(ps["trust_excess"][f, g].mean())}
            for k, v in ps.items():
                if k.startswith("end_"):
                    c["pM_" + k] = float(v[f, g].mean())
            cells.append(c)
    return cells


def phases(cells):
    out = {}
    for sf in SEER_FRACS:
        cs = [c for c in cells if c["seer_frac"] == sf]
        lk = sum(c["locked_frac"] >= 0.5 for c in cs)
        rc = sum(c["recovered_frac"] >= 0.5 for c in cs)
        out[str(sf)] = {"locked": lk, "recovered": rc, "unresolved": len(cs) - lk - rc,
                        "mean_pM_end": float(np.mean([c["pM_end"] for c in cs]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["main", "second", "sparse"], default="main")
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "P1e"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    base = model.Config(*model.candidate_priors())
    t0 = time.time()
    _, pre = X.run_prehistory(X.ExtConfig(base), seeds, STEPS, SHIFT, N, N_FULL)
    q_ox = float(pre["oxygen"][:, SHIFT - 1].mean())
    masks = seer_masks()
    comps = compositions()
    runs = {"main": main_runs, "second": lambda: second_runs(args.kappa), "sparse": sparse_runs}[args.stage]()

    summ_path, npz_path = out_dir / "summary.json", out_dir / "per_seed.npz"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {"meta": {}, "runs": {}}
    arrays = dict(np.load(npz_path)) if npz_path.exists() else {}
    gam, c0s = [g for g, _ in CELLS], [c for _, c in CELLS]
    for r in runs:
        key = run_key(r)
        if key in summary["runs"]:
            continue
        types = comps[r["comp"]]
        mc = M.MixedConfig(base, types, q_ox, X.ExtConfig(base, rho_L=1.0 if r["forget"] == "noforget" else 0.95))
        vcfg = V.VerifyConfig(mc, rule=r["rule"], mode=r["mode"], delay=r["D"], kappa=r["kappa"], lam=r["lam"],
                              reveal_precision="full" if r["prec"] == "full" else "attention",
                              reveal_every=r.get("every", 1))
        _, hist, p_final = V.verified_shift_protocol(vcfg, seeds, gam, c0s, masks, steps=STEPS, shift=SHIFT)
        ps = per_seed(hist, p_final, types)
        cells = cell_table(ps)
        summary["runs"][key] = {**r, "cells": cells, "phases": phases(cells)}
        for k, v in ps.items():
            arrays[f"{key}::{k}"] = v
        ph = summary["runs"][key]["phases"]
        print(f"[{time.time() - t0:6.0f}s] {key:72s} " + "  ".join(
            f"f{f}: L{ph[str(f)]['locked']}/R{ph[str(f)]['recovered']} p{ph[str(f)]['mean_pM_end']:.3f}"
            for f in SEER_FRACS), flush=True)
        if args.kappa is not None:
            summary["meta"]["kappa_best"] = args.kappa
        summary["meta"] = {**summary["meta"], "seeds": seeds, "cells": CELLS, "seer_fracs": SEER_FRACS, "kappas": KAPPAS,
                           "delays": DELAYS, "committed_q_ox": q_ox, "agree_m": 2.0, "gamma_rho": 0.95,
                           "key": "comp|forget|mode|reveal_precision|rule|D|kappa|lam",
                           "never": "D never == fixed W without reveals (exact)"}
        summary["meta"].setdefault("kappa_best", None)
        summ_path.write_text(json.dumps(summary, indent=1))
        np.savez_compressed(npz_path, **arrays)
    print(f"stage {args.stage} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
