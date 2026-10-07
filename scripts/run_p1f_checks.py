"""P1f: artefact checks for the weaker P1b-P1e findings.

  --check 1  outcome-based scoring (reach truth >= .75, log score, Brier, time to
             truth) of P1b pure types, P1c compositions, P1d/P1e trust arms
  --check 2  changing worlds (M flips every 150 rounds; slow drift); delay
             D {1,10,50} for relative (kappa 30) vs fixed W; sparse full-precision
             reveals every 10th round (primary), every round (secondary)
  --check 3  wrong rival: rival M=+3, rival M=-3, K=3 {-1,+1,+3}, vs original
  --check 4  anchoring: omega {.5,.8,.9,.97,1} for pure types; anchor tracking
  --check 5  sparse/noisy reveals (every 10/30; noise 1x/3x sensor) x full observers {4,8,12}

All: Fig-4 protocol (restricted agents at .01 on G, M from round 80), gates from
round 80, coarse grid gamma {0,3,10} x c0 {0,.3,1}, seeds 0-19 paired, support
never forgets unless stated. Per-seed metric arrays go to
experiments/results/P1f/check{N}.npz (keys "arm::metric", arrays (..., cells, seeds)).
Resumable: arms already saved are skipped.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.pluralist import extended as X  # noqa: E402
from src.pluralist import mixed as M  # noqa: E402
from src.pluralist import model  # noqa: E402
from src.pluralist import verified as V  # noqa: E402
from src.pluralist import worlds as Wd  # noqa: E402

OUT = ROOT / "experiments" / "results" / "P1f"
N, SHIFT, STEPS = 24, 80, 600
GAMMAS, C0S = [0.0, 3.0, 10.0], [0.0, 0.3, 1.0]
CELLS = list(itertools.product(GAMMAS, C0S))
G_ARR, C_ARR = [g for g, _ in CELLS], [c for _, c in CELLS]
SEEDS = list(range(20))
P, C, Nn = M.PLURALIST, M.COMMITTED, M.NEUTRAL
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


class Store:
    def __init__(self, check):
        self.path = OUT / f"check{check}.npz"
        self.meta_path = OUT / f"check{check}.json"
        self.arrays = dict(np.load(self.path)) if self.path.exists() else {}
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {"arms": {}}

    def has(self, arm):
        return arm in self.meta["arms"]

    def put(self, arm, metrics, info):
        for k, v in metrics.items():
            self.arrays[f"{arm}::{k}"] = np.asarray(v)
        self.meta["arms"][arm] = {**info, **{f"mean_{k}": float(np.mean(v)) for k, v in metrics.items()}}
        np.savez_compressed(self.path, **self.arrays)
        self.meta_path.write_text(json.dumps(self.meta, indent=1))


def base_cfg():
    return model.Config(*model.candidate_priors())


def incumbent_weights(base):
    """Pluralist community's pre-shift support (seed/agent mean of softmax L at round 79)."""
    state, _ = X.run_prehistory(X.ExtConfig(base), SEEDS, STEPS, SHIFT, N, 4)
    L = np.asarray(state.L)
    q = np.exp(L - L.max(-1, keepdims=True))
    q /= q.sum(-1, keepdims=True)
    return tuple(float(x) for x in q.mean((0, 1)))


def mixed_metrics(hist, types, signals=None, truth_positive=True):
    """hist from mixed_shift_protocol: (G,S,T,N). Returns metric arrays (G,S)."""
    pm = hist["p_mass_pos"]
    out = Wd.outcome_metrics(pm[:, :, -1], pm.mean(-1), SHIFT, truth_positive)
    out["end_pM"] = pm[:, :, -1].mean(-1)
    types = np.asarray(types)
    for t, name in M.TYPE_NAMES.items():
        if (types == t).any():
            out[f"end_pM_{name}"] = pm[:, :, -1][..., types == t].mean(-1)
    if "mean_M" in hist:
        out["abs_err_M"] = np.abs(hist["mean_M"][:, :, -1] - 1.0).mean(-1)
    if signals is not None:
        out.update(Wd.tracking_metrics(pm.mean(-1), signals))
    return out


def verified_metrics(hist, p_final, signals=None):
    """verified histories (F,G,S,T), p_final (F,G,S,N) -> metric arrays (F,G,S)."""
    if signals is None:
        out = Wd.outcome_metrics(p_final, hist["p_comm"], SHIFT)
        out["end_pM"] = hist["p_comm"][..., -1]
    else:
        out = Wd.tracking_metrics(hist["p_comm"], signals)
        out["end_right_side"] = np.where(signals[-1, 3] > 0, p_final, 1 - p_final).mean(-1)
    out["trust_excess"] = (hist["trust_correct_seers"] - hist["frac_correct"])[..., SHIFT:].mean(-1)
    return out


def pure_types():
    return {"pluralist": (P,) * N, "neutral": (Nn,) * N, "committed": (C,) * N}


# ---------------------------------------------------------------- check 4 (+1A)
def check4():
    st = Store(4)
    base = base_cfg()
    q = incumbent_weights(base)
    for omega in (0.5, 0.8, 0.9, 0.97, 1.0):
        b = replace(base, omega=omega)
        for tname, types in pure_types().items():
            for forget, rho in (("noforget", 1.0), ("forget", 0.95)):
                if forget == "forget" and tname != "pluralist":
                    continue
                arm = f"omega{omega}|{tname}|{forget}"
                if st.has(arm):
                    continue
                mc = M.MixedConfig(b, types, q, X.ExtConfig(b, rho_L=rho))
                hist = M.mixed_shift_protocol(mc, SEEDS, G_ARR, C_ARR, steps=STEPS, shift=SHIFT)
                met = mixed_metrics(hist, types)
                # anchor-implied P(M>0): the agent's own references weighted by its final support
                P0, h0 = M.references(mc)
                mean0 = np.linalg.solve(np.asarray(P0), np.asarray(h0)[..., None])[..., 0][..., 3]
                sd0 = np.sqrt(np.linalg.inv(np.asarray(P0))[..., 3, 3])
                from scipy.stats import norm
                phi0 = norm.cdf(mean0 / sd0)                                    # (N,K)
                qox = hist["oxygen"][:, :, -1]                                  # (G,S,N)
                anchor = (1 - qox) * phi0[None, None, :, 0] + qox * phi0[None, None, :, 1]
                met["anchor_pM"] = anchor.mean(-1)
                met["dev_from_anchor"] = np.abs(hist["p_mass_pos"][:, :, -1] - anchor).mean(-1)
                st.put(arm, met, {"omega": omega, "type": tname, "forget": forget, "committed_w": q})
                log(f"check4 {arm:34s} end={met['end_pM'].mean():.3f} anchor={met['anchor_pM'].mean():.3f} "
                    f"reach={met['frac_agents_reach'].mean():.2f} logS={met['log_score'].mean():.2f}")


# ---------------------------------------------------------------- check 1 (B, C)
def check1():
    st = Store(1)
    base = base_cfg()
    q = incumbent_weights(base)
    from scripts.run_p1c_mixed_communities import compositions as p1c_comps
    comps = p1c_comps()
    for name in ("monist_committed0.5", "plural0.25_half", "plural0.75_half", "plural0.25_half_obsP",
                 "plural0.25_half_obsM", "plural0.25_allC", "plural0.75_allC"):
        arm = f"P1c|{name}"
        if st.has(arm):
            continue
        types = comps[name]["types"]
        mc = M.MixedConfig(base, types, q, X.ExtConfig(base))
        hist = M.mixed_shift_protocol(mc, SEEDS, G_ARR, C_ARR, steps=STEPS, shift=SHIFT)
        met = mixed_metrics(hist, types)
        st.put(arm, met, {"composition": name})
        log(f"check1 {arm:34s} reach={met['frac_agents_reach'].mean():.2f} logS={met['log_score'].mean():.2f}")
    from scripts.run_p1d_verified_trust import compositions as p1d_comps, seer_masks
    masks = seer_masks()
    arms = [("trust+belief", "attention", "fixed", 10, 1.0), ("trust+belief", "attention", "relative", 10, 1.0),
            ("trust+belief", "attention", "agree", 10, 1.0), ("trust+belief", "full", "fixed", 10, 1.0),
            ("trust+belief", "full", "relative", 10, 30.0), ("trust", "-", "fixed", None, 1.0),
            ("trust", "-", "relative", 10, 30.0)]
    for cname, types in p1d_comps().items():
        for mode, prec, rule, D, k in arms:
            arm = f"P1de|{cname}|{mode}|{prec}|{rule}|{D}|{k}"
            if st.has(arm):
                continue
            mc = M.MixedConfig(base, types, q[1], X.ExtConfig(base))
            vc = V.VerifyConfig(mc, rule=rule, mode=mode, delay=D, kappa=k,
                                reveal_precision="full" if prec == "full" else "attention")
            _, hist, pf = V.verified_shift_protocol(vc, SEEDS, G_ARR, C_ARR, masks, steps=STEPS, shift=SHIFT)
            met = verified_metrics(hist, pf)
            st.put(arm, met, {"composition": cname, "mode": mode, "prec": prec, "rule": rule, "D": D, "kappa": k})
            log(f"check1 {arm:60s} reach={met['frac_agents_reach'].mean():.2f} logS={met['log_score'].mean():.2f}")


# ---------------------------------------------------------------- check 2
def check2():
    st = Store(2)
    base = base_cfg()
    q = incumbent_weights(base)
    from scripts.run_p1d_verified_trust import compositions as p1d_comps
    comps = {k: v for k, v in p1d_comps().items() if k != "all_committed"}
    worlds = {"multiflip": Wd.multi_flip_signals(), "drift": Wd.drift_signals()}
    seer = np.zeros((1, N))
    seer[0, 4:16] = 1                                        # 50% of agents, restricted
    plan = [(every, rule, D) for every in (10,) for rule in ("fixed", "relative") for D in (1, 10, 50)]
    plan += [(1, rule, D) for rule in ("fixed", "relative") for D in (1, 50)]
    for (wname, sig), (cname, types), (every, rule, D) in itertools.product(worlds.items(), comps.items(), plan):
        arm = f"{wname}|{cname}|every{every}|{rule}|{D}"
        if st.has(arm):
            continue
        mc = M.MixedConfig(base, types, q[1], X.ExtConfig(base))
        vc = V.VerifyConfig(mc, rule=rule, mode="trust+belief", delay=D, kappa=30.0,
                            reveal_precision="full", reveal_every=every)
        _, hist, pf = V.verified_shift_protocol(vc, SEEDS, G_ARR, C_ARR, seer, steps=len(sig), shift=SHIFT,
                                                signals=sig)
        met = verified_metrics(hist, pf, signals=sig)
        st.put(arm, met, {"world": wname, "composition": cname, "every": every, "rule": rule, "D": D})
        log(f"check2 {arm:50s} brier={met['track_brier'].mean():.3f} right={met['track_right_side'].mean():.3f}")


# ---------------------------------------------------------------- check 3
def check3():
    st = Store(3)
    for variant in Wd.RIVALS:
        b = Wd.rival_config(base_cfg(), variant)
        q = incumbent_weights(b)
        for tname, types in pure_types().items():
            arm = f"{variant}|{tname}"
            if st.has(arm):
                continue
            mc = M.MixedConfig(b, types, q, X.ExtConfig(b))
            hist = M.mixed_shift_protocol(mc, SEEDS, G_ARR, C_ARR, steps=STEPS, shift=SHIFT)
            met = mixed_metrics(hist, types)
            st.put(arm, met, {"variant": variant, "type": tname, "committed_w": q})
            log(f"check3 {arm:30s} end={met['end_pM'].mean():.3f} reach={met['frac_agents_reach'].mean():.2f} "
                f"|E[M]-1|={met['abs_err_M'].mean():.2f}")


# ---------------------------------------------------------------- check 5
def half_types(n_full):
    obs = tuple(P if i % 2 == 0 else C for i in range(n_full))
    rest = N - n_full
    return obs + (P,) * (rest // 2) + (C,) * (rest - rest // 2)


def check5():
    st = Store(5)
    base = base_cfg()
    q = incumbent_weights(base)
    plan = [("trust+belief", every, noise) for every in (10, 30) for noise in (1.0, 3.0)]
    plan += [("trust", 10, 1.0)]
    for n_full in (4, 8, 12):
        comps = {"all_pluralist": (P,) * N, "half_pluralist_committed": half_types(n_full)}
        seer = np.zeros((1, N))
        seer[0, n_full:] = 1                                  # all restricted agents
        for (cname, types), (mode, every, noise), rule in itertools.product(comps.items(), plan, ("fixed", "relative")):
            arm = f"nfull{n_full}|{cname}|{mode}|every{every}|noise{noise}|{rule}"
            if st.has(arm):
                continue
            mc = M.MixedConfig(base, types, q[1], X.ExtConfig(base))
            vc = V.VerifyConfig(mc, rule=rule, mode=mode, delay=10, kappa=30.0, reveal_precision="full",
                                reveal_every=every, reveal_noise=noise)
            _, hist, pf = V.verified_shift_protocol(vc, SEEDS, G_ARR, C_ARR, seer, steps=STEPS, shift=SHIFT,
                                                    n_full=n_full)
            met = verified_metrics(hist, pf)
            st.put(arm, met, {"n_full": n_full, "composition": cname, "mode": mode, "every": every,
                              "noise": noise, "rule": rule})
            log(f"check5 {arm:62s} end={met['end_pM'].mean():.3f} logS={met['log_score'].mean():.2f} "
                f"tx={met['trust_excess'].mean():+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, nargs="+", default=[4, 1, 3, 5, 2])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for c in args.check:
        {1: check1, 2: check2, 3: check3, 4: check4, 5: check5}[c]()
    log("done")


if __name__ == "__main__":
    main()
