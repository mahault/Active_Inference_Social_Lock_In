"""Analyze E11 tri-channel ablation: marginal contribution of structural
(gamma), social (trust learning), and material (resource coupling) channels
to paradigm lock-in.

Run after `python -m experiments.run_experiment experiments/configs/E11_tri_channel_ablation.yaml`.

Output: a printed table per channel + a saved summary JSON to
experiments/results/E11/summary.json.
"""

from __future__ import annotations

import json
import numpy as np


def load(path="experiments/results/E11/results.json"):
    with open(path) as f:
        return json.load(f)


def key(r):
    s = r["sweep"]
    return (bool(s["gamma_strength"]),
            bool(s["trust_learning"]),
            bool(s["resource_coupling"]))


def summarize(records):
    """Group records by 8-cell key. For each cell, mean and std of
    final_mean_qB across seeds."""
    groups: dict[tuple, list[float]] = {}
    for r in records:
        k = key(r)
        groups.setdefault(k, []).append(r["final_mean_qB"])
    out = {}
    for k, vals in groups.items():
        arr = np.asarray(vals)
        out[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                  "n": int(arr.size),
                  "capture_rate": float((arr < 0.5).mean())}
    return out


def channel_marginal(summary, channel_idx):
    """For each of the 4 settings of the OTHER two channels, compute the
    delta in mean_qB from turning channel `channel_idx` ON.
    channel_idx: 0=gamma, 1=trust, 2=resource."""
    deltas = []
    other_settings = []
    for a in (False, True):
        for b in (False, True):
            key_off = list((False, False, False))
            key_on = list((False, False, False))
            other_axes = [i for i in range(3) if i != channel_idx]
            key_off[other_axes[0]] = a
            key_off[other_axes[1]] = b
            key_off[channel_idx] = False
            key_on[other_axes[0]] = a
            key_on[other_axes[1]] = b
            key_on[channel_idx] = True
            off_mean = summary[tuple(key_off)]["mean"]
            on_mean = summary[tuple(key_on)]["mean"]
            deltas.append(on_mean - off_mean)
            other_settings.append((a, b))
    return deltas, other_settings


def main():
    records = load()
    summary = summarize(records)
    print("\nE11 -- Tri-channel ablation (final_mean_qB, mean +/- std over seeds)")
    print("=" * 78)
    print(f"{'gamma':>6}  {'trust':>6}  {'rsrc':>6}  {'mean_qB':>12}  {'capture':>10}  {'n':>3}")
    print("-" * 78)
    for k in sorted(summary.keys()):
        gm, tr, rs = k
        s = summary[k]
        print(f"{str(gm):>6}  {str(tr):>6}  {str(rs):>6}  "
              f"{s['mean']:>6.3f}+-{s['std']:>4.3f}  "
              f"{s['capture_rate']:>10.2f}  {s['n']:>3}")

    print("\nMarginal channel contributions (delta_mean_qB from turning channel ON):")
    print("=" * 78)
    names = ["gamma (structural)", "trust learning (social)", "resources (material)"]
    rows = []
    for idx in range(3):
        deltas, others = channel_marginal(summary, idx)
        avg = np.mean(deltas)
        rows.append({"channel": names[idx], "mean_delta": float(avg),
                     "deltas_per_cell": [float(d) for d in deltas]})
        # mean_qB DROPS when a channel locks in the anti-truth bloc, so
        # a NEGATIVE delta is the lock-in signal.
        print(f"  {names[idx]:<30} avg delta_qB = {avg:+.4f}  "
              f"(per-cell: {[f'{d:+.3f}' for d in deltas]})")

    print("\nInterpretation:")
    print("  More negative avg delta = stronger lock-in contribution from that channel.")
    print("  If a channel has avg delta near zero, it does not affect capture under")
    print("  this experimental design (anti-truth bloc fraction = 30 percent).")

    out_path = "experiments/results/E11/summary.json"
    with open(out_path, "w") as f:
        json.dump({"cells": {str(k): v for k, v in summary.items()},
                   "channel_marginals": rows}, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
