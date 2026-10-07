"""P1g: population size, message bandwidth and a neutral-drift control.

Motivated by Tanaka (2026), arXiv:2603.24676, "When Is Collective Intelligence a
Lottery?" (Quantized Simplex Gossip: sampled messages inject drift ~ alpha^2/(m N^2);
consensus time ~ m N^2 / alpha^2; drift -> selection crossover at
Gamma = (m N / alpha) h).

  --part 1  population size N {8, 24, 50, 100}; observers = round(N/6) ("sixth")
            or 4 ("four"); types pluralist / committed / neutral / 50:50
            pluralist-committed mix; Fig-4 protocol, gates from round 80, 3x3 grid.
  --part 2  bandwidth at N = 24: fusion "rule" (P1c, full natural parameters),
            "observables" with exact moments (m = inf), and sampled message means
            m {10, 3, 1}; pluralists/mix also with sampled support (same m).
  --part 3  neutral-drift control: world channel closed for everyone from the
            start (attention 0 on every channel), symmetric priors, no prehistory;
            sampled messages (and support) m {1, 3, 10} vs exact (m = 0);
            N {8, 24, 50}; 40 seeds.

Seeds 0-19 (parts 1-2), 0-39 (part 3). Outputs: experiments/results/P1g/part{k}.npz/.json.
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

from src.pluralist import experiments as E  # noqa: E402
from src.pluralist import extended as X  # noqa: E402
from src.pluralist import mixed as M  # noqa: E402
from src.pluralist import model  # noqa: E402
from src.pluralist import worlds as Wd  # noqa: E402

OUT = ROOT / "experiments" / "results" / "P1g"
SHIFT, STEPS = 80, 600
GAMMAS, C0S = [0.0, 3.0, 10.0], [0.0, 0.3, 1.0]
CELLS = list(itertools.product(GAMMAS, C0S))
G_ARR, C_ARR = [g for g, _ in CELLS], [c for _, c in CELLS]
SEEDS = list(range(20))
P, C, Nn = M.PLURALIST, M.COMMITTED, M.NEUTRAL
COMMITTED_Q = 0.0030558345262316758   # pluralist pre-shift oxygen support at N = 24 (P1c-P1e)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


class Store:
    def __init__(self, part):
        self.path, self.meta_path = OUT / f"part{part}.npz", OUT / f"part{part}.json"
        self.arrays = dict(np.load(self.path)) if self.path.exists() else {}
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {"arms": {}}

    def has(self, arm):
        return arm in self.meta["arms"]

    def put(self, arm, metrics, info):
        for k, v in metrics.items():
            self.arrays[f"{arm}::{k}"] = np.asarray(v)
        self.meta["arms"][arm] = {**info, **{f"mean_{k}": float(np.nanmean(v)) for k, v in metrics.items()
                                             if np.asarray(v).dtype.kind == "f"}}
        np.savez_compressed(self.path, **self.arrays)
        self.meta_path.write_text(json.dumps(self.meta, indent=1))


def types_for(kind, n, n_full):
    if kind in ("pluralist", "committed", "neutral"):
        return ({"pluralist": P, "committed": C, "neutral": Nn}[kind],) * n
    obs = tuple(P if i % 2 == 0 else C for i in range(n_full))
    rest = n - n_full
    return obs + (P,) * (rest - rest // 2) + (C,) * (rest // 2)


def metrics(hist, types):
    pm = hist["p_mass_pos"]                                  # (G,S,T,N)
    out = Wd.outcome_metrics(pm[:, :, -1], pm.mean(-1), SHIFT)
    out["end_pM"] = pm[:, :, -1].mean(-1)
    types = np.asarray(types)
    for t, name in M.TYPE_NAMES.items():
        if (types == t).any() and len(set(types.tolist())) > 1:
            out[f"end_pM_{name}"] = pm[:, :, -1][..., types == t].mean(-1)
    return out


def part1():
    st = Store(1)
    base = model.Config(*model.candidate_priors())
    for n, scheme, kind in itertools.product((8, 24, 50, 100), ("sixth", "four"),
                                             ("pluralist", "committed", "neutral", "mix")):
        n_full = int(round(n / 6)) if scheme == "sixth" else 4
        if n == 24 and scheme == "four":
            continue                                          # identical to "sixth"
        arm = f"N{n}|{scheme}|{kind}"
        if st.has(arm):
            continue
        types = types_for(kind, n, n_full)
        mc = M.MixedConfig(base, types, COMMITTED_Q, X.ExtConfig(base))
        hist = M.mixed_shift_protocol(mc, SEEDS, G_ARR, C_ARR, steps=STEPS, shift=SHIFT, n_full=n_full)
        met = metrics(hist, types)
        st.put(arm, met, {"N": n, "n_full": n_full, "scheme": scheme, "type": kind})
        log(f"part1 {arm:26s} end={met['end_pM'].mean():.3f} sd_seeds={met['end_pM'].std(1, ddof=1).mean():.3f} "
            f"reach={met['frac_agents_reach'].mean():.2f}")


def part2():
    st = Store(2)
    base = model.Config(*model.candidate_priors())
    arms = [("rule", 0, 0), ("observables", 0, 0)] + [("observables", m, 0) for m in (10, 3, 1)]
    for kind in ("pluralist", "committed", "neutral", "mix"):
        extra = [("observables", m, m) for m in (10, 3, 1)] if kind in ("pluralist", "mix") else []
        for fusion, mm, sm in arms + extra:
            arm = f"{kind}|{fusion}|msg{mm}|sup{sm}"
            if st.has(arm):
                continue
            types = types_for(kind, 24, 4)
            mc = M.MixedConfig(base, types, COMMITTED_Q, X.ExtConfig(base), fusion=fusion,
                               message_m=mm, support_m=sm)
            hist = M.mixed_shift_protocol(mc, SEEDS, G_ARR, C_ARR, steps=STEPS, shift=SHIFT)
            met = metrics(hist, types)
            st.put(arm, met, {"type": kind, "fusion": fusion, "message_m": mm, "support_m": sm})
            log(f"part2 {arm:36s} end={met['end_pM'].mean():.3f} logS={met['log_score'].mean():.2f} "
                f"reach={met['frac_agents_reach'].mean():.2f}")


def part3():
    st = Store(3)
    base = model.Config(*model.candidate_priors())
    seeds = list(range(40))
    steps = 600
    for n, kind, mm in itertools.product((8, 24, 50), ("pluralist", "neutral"), (0, 10, 3, 1)):
        arm = f"N{n}|{kind}|m{mm}"
        if st.has(arm):
            continue
        types = types_for(kind, n, 0)
        sm = mm if kind == "pluralist" else 0
        mc = M.MixedConfig(base, types, COMMITTED_Q, X.ExtConfig(base), fusion="observables",
                           message_m=mm, support_m=sm)
        W = E.social_matrix(n)
        sig = np.broadcast_to([0.0, 1.0, 1.0, 1.0], (steps, 4))
        _, hist = M.mixed_rollout(mc, W, np.zeros((n, 3)), sig, seeds, [0.0], [0.0])
        pm = hist["p_mass_pos"][0].mean(-1)                  # (S,T) community P(M>0)
        qx = hist["oxygen"][0].mean(-1)                      # (S,T) community oxygen support
        dev = np.abs(pm - 0.5)
        thr = 0.25
        hit = dev >= thr
        t_cons = np.where(hit.any(-1), hit.argmax(-1), steps).astype(float)
        met = {"end_pM": pm[:, -1], "end_qox": qx[:, -1], "end_abs_dev": dev[:, -1],
               "mean_abs_dev": dev.mean(-1), "t_consensus_pM_dev_0.25": t_cons,
               "within_seed_agent_sd_end": hist["p_mass_pos"][0][:, -1].std(-1)}
        st.put(arm, met, {"N": n, "type": kind, "m": mm, "seeds": len(seeds)})
        log(f"part3 {arm:22s} |pM-.5| end={dev[:, -1].mean():.3f} frac(pM>.5)={(pm[:, -1] > .5).mean():.2f} "
            f"t_cons={np.median(t_cons):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, nargs="+", default=[3, 2, 1])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for p in args.part:
        {1: part1, 2: part2, 3: part3}[p]()
    log("done")


if __name__ == "__main__":
    main()
