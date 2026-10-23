#!/bin/bash
# Freeze src/ and scripts/ under outputs/jobs/<tag>/ and submit scripts/long_job.py from there (one GPU job,
# through scripts/qsub_serial.sh), so that later edits do not leak into a queued job.
#   scripts/long_submit.sh <tag> "LONG_PLAN=seq:dump:b16[,LONG_LIMIT=..,LONG_REF=..]" [walltime]
set -euo pipefail
tag=$1
dir="outputs/jobs/$tag"
rm -rf "$dir" && mkdir -p "$dir"
cp -r src scripts "$dir/"
scripts/qsub_serial.sh "MODEL=.cache/models/owsm_v4_medium_1B,STAGES=pyfile,PYFILE=$dir/scripts/long_job.py,$2" "${3:-01:00:00}"
