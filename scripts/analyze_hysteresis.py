"""Analyze E12 under truth reversal (theta: 0 -> 1 at t_shift, 1 -> 0 at
t_reverse). Compares resource_coupling on vs off.

What we measure:
  * Phase-1 peak: max(mean_qB) during the t_shift..t_reverse window. Is
    the population able to adapt to the new paradigm at all?
  * Phase-2 recovery: mean_qB at the end. Does it track back to truth=0?
  * Cross-condition gap: how big is (resource_on - resource_off) at end?

The original hypothesis was "resources create hysteresis after adaptation."
The actual finding (under the working severity) is stronger: resources
completely block phase-1 adaptation, while the no-resource control adapts
and then reverts cleanly. The "material lock-in" is at the adaptation
step, not the revert step.
"""

from __future__ import annotations

import json
import numpy as np


def load(path="experiments/results/E12/results.json"):
    with open(path) as f:
        return json.load(f)


def main():
    records = load()
    # Group by resource_coupling
    groups: dict[bool, list[dict]] = {False: [], True: []}
    for r in records:
        groups[bool(r["sweep"]["resource_coupling"])].append(r)

    print("\nE12 -- Resource hysteresis under truth reversal (theta 0->1 at t=80, 1->0 at t=200)")
    print("=" * 80)

    summary = {}
    for rc in (False, True):
        traj = np.asarray([r["mean_qB_trajectory"] for r in groups[rc]])
        theta = np.asarray(groups[rc][0]["theta_star_trace"])
        n_steps = traj.shape[1]

        # Identify phase-1 window: theta = post (paradigm 1)
        in_phase1 = theta > 0.5
        if in_phase1.any():
            t_shift = int(np.argmax(in_phase1))
            # First step where theta drops back to pre AFTER phase 1
            after_shift_in_post = in_phase1.copy()
            after_shift_in_post[:t_shift] = False
            if (~after_shift_in_post).any() and after_shift_in_post.any():
                t_reverse = int(np.argmax((~in_phase1) & (np.arange(n_steps) > t_shift)))
            else:
                t_reverse = n_steps - 1
        else:
            t_shift, t_reverse = 0, n_steps - 1

        phase1_peak = float(traj[:, t_shift:t_reverse].mean(0).max())
        mean_just_before_reverse = float(traj[:, max(0, t_reverse - 5):t_reverse].mean())
        mean_final = float(traj[:, -10:].mean())

        print(f"\nresource_coupling = {rc}:")
        print(f"  phase-1 window: [t={t_shift}, t={t_reverse})  truth=paradigm 1")
        print(f"  phase-1 peak mean_qB:           {phase1_peak:.3f}  (1.0 = full adaptation)")
        print(f"  mean_qB just before reverse:    {mean_just_before_reverse:.3f}")
        print(f"  mean_qB at end (truth=paradigm 0): {mean_final:.3f}  (0.0 = full recovery)")
        if phase1_peak > 0.5:
            print(f"  -> Population ADAPTED to paradigm 1 during phase 1")
        else:
            print(f"  -> Population FAILED to adapt to paradigm 1 (blocked)")

        summary[str(rc)] = {
            "phase1_peak": phase1_peak,
            "mean_just_before_reverse": mean_just_before_reverse,
            "mean_final": mean_final,
        }

    # Compare conditions
    if "True" in summary and "False" in summary:
        peak_gap = summary["False"]["phase1_peak"] - summary["True"]["phase1_peak"]
        final_gap = summary["True"]["mean_final"] - summary["False"]["mean_final"]
        print("\n" + "=" * 80)
        print(f"ADAPTATION GAP (peak_off - peak_on during phase 1): {peak_gap:+.3f}")
        if peak_gap > 0.3:
            print(f"  Resources BLOCK adaptation: no-resource control reaches {summary['False']['phase1_peak']:.2f}")
            print(f"  while resource-coupled population stays near {summary['True']['phase1_peak']:.2f}.")
            print(f"  This is the material channel of lock-in: the agent can't afford")
            print(f"  the informative experiments needed to update under shift.")
        print(f"END GAP (resource_on - resource_off, after reverse): {final_gap:+.3f}")

    out_path = "experiments/results/E12/summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
