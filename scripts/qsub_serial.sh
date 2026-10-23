#!/bin/bash
# Kept for older callers: submits through scripts/qsub_pool.sh (up to OWSM_GPU_LANES parallel GPU jobs;
# OWSM_GPU_LANES=1 makes it one job at a time again).
exec "$(dirname "$0")/qsub_pool.sh" "$@"
