#!/bin/bash
# Build a self-contained Python environment on node-local disk and pack it into one tar.zst on /groups.
# Lustre is slow with the ~100k small files of torch/TensorRT, so jobs unpack this archive into
# $PBS_LOCALDIR instead of running from a .venv on /groups (see scripts/gpu_eval.pbs).
#
#   scripts/build_env_tar.sh            # main env  -> .cache/envs/owsm-env.tar.zst
#   scripts/build_env_tar.sh ref        # ESPnet reference env (CPU torch) -> .cache/envs/espnet-ref.tar.zst
#
# The env is a relocatable python-build-standalone interpreter with packages installed directly into it,
# so it runs from wherever it is extracted: <dir>/bin/python3.
set -euo pipefail
: "${PBS_JOBID:?environment builds require an allocated compute job}"
: "${PBS_LOCALDIR:?environment builds require node-local scratch}"
cd "$(dirname "$0")/.."
REPO=$PWD
KIND="${1:-main}"
BUILD_ROOT="${BUILD_ROOT:-$PBS_LOCALDIR/env-build}"
OUT_DIR="$REPO/.cache/envs"
ZSTD="$(command -v zstd || echo "$HOME/miniconda3/bin/zstd")"

case $KIND in
  main) NAME=owsm-env; PROJECT=$REPO ;;
  ref) NAME=espnet-ref; PROJECT=$REPO/tools/espnet_ref ;;
  *) echo "usage: $0 [main|ref]" >&2; exit 2 ;;
esac

mkdir -p "$BUILD_ROOT" "$OUT_DIR"
# Only this invocation owns WORK_DIR; concurrent builds never share caches or delete each other's files.
WORK_DIR=$(mktemp -d "$BUILD_ROOT/$NAME.XXXXXXXX")
# PIDs can coincide on different compute nodes; reuse the private build's random identity on Lustre.
BUILD_ID="${WORK_DIR##*/}"
ARCHIVE_TMP="$OUT_DIR/$NAME.tar.zst.tmp.$BUILD_ID"
cleanup() {
    local status=$?
    trap - EXIT
    if [ -f "$WORK_DIR/$NAME.txt" ]; then
        # Keep failed-build inputs without replacing metadata belonging to a previous archive.
        cp "$WORK_DIR/$NAME.txt" "$OUT_DIR/$BUILD_ID.requirements.txt" || status=1
    fi
    # A failed metadata publication must not prevent releasing the much larger build/cache tree.
    rm -f -- "$ARCHIVE_TMP" || status=1
    rm -rf -- "$WORK_DIR" || status=1
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export UV_CACHE_DIR="$WORK_DIR/uv-cache" UV_PYTHON_INSTALL_DIR="$WORK_DIR/pythons"
export PIP_CACHE_DIR="$WORK_DIR/pip-cache" TMPDIR="$WORK_DIR/tmp"
mkdir -p "$TMPDIR"
ENV="$WORK_DIR/$NAME"

uv python install 3.12
# resolve symlinks: uv also creates a minor-version alias dir that is a symlink to the real install
cp -a "$(dirname "$(dirname "$(readlink -f "$(uv python find 3.12 --managed-python)")")")" "$ENV"
test ! -L "$ENV"
rm -f "$ENV"/lib/python3.12/EXTERNALLY-MANAGED
PY="$ENV/bin/python3"

(cd "$PROJECT" && uv lock && uv export --frozen --no-hashes --no-emit-project --all-groups -o "$WORK_DIR/$NAME.txt")
cp "$WORK_DIR/$NAME.txt" "$ENV/requirements.txt"  # authoritative dependency list travels with the archive
uv pip install --python "$PY" -r "$WORK_DIR/$NAME.txt"
if [ "$KIND" = main ]; then
  # editable: the .pth points at $REPO/src, so code edits need no rebuild
  uv pip install --python "$PY" --no-deps -e "$REPO"
fi
"$PY" -c "import torch, sys; print(sys.version.split()[0], 'torch', torch.__version__, torch.version.cuda)"

tar -C "$WORK_DIR" -cf - "$NAME" | "$ZSTD" -T16 -3 -q -o "$ARCHIVE_TMP" -f
mv "$ARCHIVE_TMP" "$OUT_DIR/$NAME.tar.zst"
ls -lh "$OUT_DIR/$NAME.tar.zst"
echo "environment archived; temporary interpreter and uv/pip caches are removed on exit"
