#!/usr/bin/env bash
# Reruns of E13-E15, E18, E19 with 20 seeds (originals used 5).
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" N_SEEDS=20
LOG=experiments/results/_rerun_logs; mkdir -p "$LOG"
for m in run_e18_cross_channel run_e19_regime_flip run_e13_e14_e15; do
  echo "[$(date '+%F %T')] START $m"
  if python -m scripts.$m > "$LOG/${m}_20seeds.log" 2>&1; then echo "[$(date '+%F %T')] DONE  $m"; else echo "[$(date '+%F %T')] FAIL  $m"; fi
done
echo "[$(date '+%F %T')] ALL DONE"
