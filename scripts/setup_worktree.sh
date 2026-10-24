#!/bin/bash
# In a git worktree of this repo: share the main checkout's environments, models and test data, and keep
# TensorRT engines (whose cache key does not see uncommitted export changes) separate.
#   scripts/setup_worktree.sh /path/to/the/main/checkout
set -euo pipefail
main="${1:?path of the main checkout}"
mkdir -p .cache/engines
for d in envs models testdata hf; do
  [ -e ".cache/$d" ] || ln -s "$main/.cache/$d" ".cache/$d"
done
ls -la .cache
