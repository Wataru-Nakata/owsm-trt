#!/bin/bash
# Submit scripts/gpu_eval.pbs from the current directory so that it starts only after every job submitted
# through this script before it has ended: one GPU job at a time, however many people/agents submit.
#   scripts/qsub_serial.sh "MODEL=...,STAGES=a:b,REF=..." [walltime, default 01:00:00]
# Prints the job id. Wait for it with: until ! qstat <id> >/dev/null 2>&1; do sleep 30; done
set -euo pipefail
CHAIN_DIR="${OWSM_GPU_CHAIN:-/path/to/owsm_trt/.cache/gpu_chain}"
mkdir -p "$CHAIN_DIR"
exec 9>"$CHAIN_DIR/lock"
flock 9
dep=()
last=$(cat "$CHAIN_DIR/last" 2>/dev/null || true)
if [ -n "$last" ] && qstat "$last" >/dev/null 2>&1; then
  dep=(-W "depend=afterany:$last")
fi
id=$(qsub -P <group> -q <queue> "${dep[@]}" -v "RTYPE=rt_HG,$1" -l select=1 -l "walltime=${2:-01:00:00}" \
  scripts/gpu_eval.pbs)
echo "$id" > "$CHAIN_DIR/last"
echo "$(date '+%F %T') $id $PWD $1" >> "$CHAIN_DIR/log"
echo "$id"
