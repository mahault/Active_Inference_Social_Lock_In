#!/usr/bin/env bash
# Rerun the E-series experiments that use resources or learned trust, with
# observations drawn at each agent's chosen experiment (2026-10-05).
# Usage: bash scripts/rerun_e_series.sh   (from the repo root)

cd "$(dirname "$0")/.."
LOG=experiments/results/_rerun_logs
mkdir -p "$LOG"
PY=${PY:-python}

run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] START $name"
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $name"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $LOG/$name.log)"
  fi
}

# structural substrate, resources on
run E18 $PY scripts/run_e18_cross_channel.py
run E19 $PY scripts/run_e19_regime_flip.py
run E11 $PY -m experiments.run_experiment experiments/configs/E11_tri_channel_ablation.yaml
# simple substrate, resources on
run E8  $PY -m experiments.run_experiment experiments/configs/E8_principled_ablation.yaml
# cont substrate, trust + resources
run E13_E14_E15 $PY scripts/run_e13_e14_e15.py
run E16 $PY scripts/run_e16_hysteresis.py
run E17 $PY scripts/run_e17_phase_diagram.py
run E12 $PY -m experiments.run_experiment experiments/configs/E12_resource_hysteresis.yaml

echo "[$(date '+%F %T')] ALL DONE"
