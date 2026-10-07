#!/usr/bin/env bash
# Second pass of the E-series reruns: the standalone scripts need the repo
# root on PYTHONPATH (first pass failed on "No module named 'src'").
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
LOG=experiments/results/_rerun_logs
mkdir -p "$LOG"
PY=${PY:-python}
run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] START $name"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "[$(date '+%F %T')] DONE  $name"
  else echo "[$(date '+%F %T')] FAIL  $name (see $LOG/$name.log)"; fi
}
run E18 $PY -m scripts.run_e18_cross_channel
run E19 $PY -m scripts.run_e19_regime_flip
run E13_E14_E15 $PY -m scripts.run_e13_e14_e15
run E16 $PY -m scripts.run_e16_hysteresis
run E17 $PY -m scripts.run_e17_phase_diagram
echo "[$(date '+%F %T')] ALL DONE"
