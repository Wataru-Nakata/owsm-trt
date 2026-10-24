#!/bin/bash
# Submit scripts/gpu_eval.pbs from the current directory on one of N single-GPU lanes (default 8,
# OWSM_GPU_LANES): an idle lane starts the job at once, otherwise it queues behind the job of the lane
# picked in turn. At most N jobs run at the same time, however many people/agents submit.
#   scripts/qsub_pool.sh "MODEL=...,STAGES=a:b,REF=..." [walltime, default 01:00:00]
# Prints the job id. Wait for it with: until ! qstat <id> >/dev/null 2>&1; do sleep 30; done
# Two jobs from one checkout that build the same engines must not run at once: let the first one build.
# The cluster account comes from the environment, or from .cache/qsub.env (not in git):
#   OWSM_QSUB_GROUP (qsub -P), OWSM_QSUB_QUEUE (qsub -q), OWSM_QSUB_RTYPE (the job's RTYPE resource type)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "${OWSM_QSUB_ENV:-$root/.cache/qsub.env}" ] && . "${OWSM_QSUB_ENV:-$root/.cache/qsub.env}"
: "${OWSM_QSUB_GROUP:?set OWSM_QSUB_GROUP (qsub -P), e.g. in .cache/qsub.env}"
: "${OWSM_QSUB_QUEUE:?set OWSM_QSUB_QUEUE (qsub -q)}"
: "${OWSM_QSUB_RTYPE:?set OWSM_QSUB_RTYPE (resource type)}"
CHAIN_DIR="${OWSM_GPU_CHAIN:-$root/.cache/gpu_chain}"
LANES="${OWSM_GPU_LANES:-8}"
mkdir -p "$CHAIN_DIR"
exec 9>"$CHAIN_DIR/lock"
flock 9
lane="" dep=()
for ((i = 0; i < LANES; i++)); do
  last=$(cat "$CHAIN_DIR/lane$i" 2>/dev/null || true)
  if [ -z "$last" ] || ! qstat "$last" >/dev/null 2>&1; then
    lane=$i
    break
  fi
done
if [ -z "$lane" ]; then  # every lane busy: queue behind the lanes in turn
  lane=$(($(cat "$CHAIN_DIR/turn" 2>/dev/null || echo 0) % LANES))
  echo $((lane + 1)) > "$CHAIN_DIR/turn"
  dep=(-W "depend=afterany:$(cat "$CHAIN_DIR/lane$lane")")
fi
id=$(qsub -P "$OWSM_QSUB_GROUP" -q "$OWSM_QSUB_QUEUE" "${dep[@]}" -v "RTYPE=$OWSM_QSUB_RTYPE,$1" -l select=1 \
  -l "walltime=${2:-01:00:00}" \
  scripts/gpu_eval.pbs)
echo "$id" > "$CHAIN_DIR/lane$lane"
echo "$(date '+%F %T') lane$lane $id $PWD $1" >> "$CHAIN_DIR/log"
echo "$id"
