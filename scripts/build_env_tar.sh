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
cd "$(dirname "$0")/.."
REPO=$PWD
KIND="${1:-main}"
BUILD_ROOT="${BUILD_ROOT:-${TMPDIR:-/tmp}/owsm-env-build-$USER}"
OUT_DIR="$REPO/.cache/envs"
ZSTD="$(command -v zstd || echo "$HOME/miniconda3/bin/zstd")"

case $KIND in
  main) NAME=owsm-env; PROJECT=$REPO ;;
  ref) NAME=espnet-ref; PROJECT=$REPO/tools/espnet_ref ;;
  *) echo "usage: $0 [main|ref]" >&2; exit 2 ;;
esac

export UV_CACHE_DIR="$BUILD_ROOT/uv-cache"  # keep the cache off Lustre too
export UV_PYTHON_INSTALL_DIR="$BUILD_ROOT/pythons"
ENV="$BUILD_ROOT/$NAME"
rm -rf "$ENV"
mkdir -p "$BUILD_ROOT" "$OUT_DIR"

uv python install 3.12
# resolve symlinks: uv also creates a minor-version alias dir that is a symlink to the real install
cp -a "$(dirname "$(dirname "$(readlink -f "$(uv python find 3.12 --managed-python)")")")" "$ENV"
test ! -L "$ENV"
rm -f "$ENV"/lib/python3.12/EXTERNALLY-MANAGED
PY="$ENV/bin/python3"

(cd "$PROJECT" && uv lock && uv export --frozen --no-hashes --no-emit-project --all-groups -o "$BUILD_ROOT/$NAME.txt")
uv pip install --python "$PY" -r "$BUILD_ROOT/$NAME.txt"
if [ "$KIND" = main ]; then
  # editable: the .pth points at $REPO/src, so code edits need no rebuild
  uv pip install --python "$PY" --no-deps -e "$REPO"
fi
"$PY" -c "import torch, sys; print(sys.version.split()[0], 'torch', torch.__version__, torch.version.cuda)"

tar -C "$BUILD_ROOT" -cf - "$NAME" | "$ZSTD" -T16 -3 -q -o "$OUT_DIR/$NAME.tar.zst.tmp" -f
mv "$OUT_DIR/$NAME.tar.zst.tmp" "$OUT_DIR/$NAME.tar.zst"
ls -lh "$OUT_DIR/$NAME.tar.zst"
echo "local copy (usable on this node): $PY"
