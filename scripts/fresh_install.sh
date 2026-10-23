#!/bin/bash
# Clean-install QA (scripts/gpu_eval.pbs stage `fresh`): a new venv on node-local disk with the wheel and the
# dependencies it pulls from PyPI, run under an empty home (no engine, Triton, CUDA JIT or Hugging Face caches):
# the README quick start, scripts/engaged.py, eval_wer at batch 1 and 16; then the same evals in the dev env.
#   scripts/fresh_install.sh <out dir>   with WHEEL=dist/owsm_trt-*.whl, FRESH_MODELS=espnet/a:espnet/b,
#   FRESH_PKGS="torch==2.6.0 tensorrt==10.9.0.34" (pins added to the install), FRESH_NAME=tag, DEV_SRC=src dir,
#   DEV_MODELS=<local dirs for the dev runs, ':'-separated, '-' to skip>, SEED_ENGINES=<engine dir>+<model cache
#   name> (copied into the empty cache first)
set -uo pipefail
OUT=$1
NAME="${FRESH_NAME:-fresh}"
L="$PBS_LOCALDIR/$NAME"
T=.cache/testdata/librispeech_dummy
WAV=$T/1272-128104-0000.flac
mkdir -p "$OUT" "$L/bin" "$L/home"
cp "$(command -v uv || echo "$HOME/.local/bin/uv")" "$L/bin/uv"
FRESH=(env -i PATH="$L/venv/bin:$L/bin:/usr/bin:/bin" HOME="$L/home" LANG=C.UTF-8 PYTHONUNBUFFERED=1
       TMPDIR="$L/tmp" UV_CACHE_DIR="$L/uv-cache" UV_PYTHON_INSTALL_DIR="$L/pythons"
       ${CUDA_VISIBLE_DEVICES+CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES})
mkdir -p "$L/tmp"

step() {  # step <name> <command...>: timed, logged
  local name=$1 t0=$SECONDS; shift
  echo "=== [$NAME] $name: $*"
  "$@" > "$OUT/$NAME-$name.log" 2>&1
  local rc=$?
  echo "--- [$NAME] $name: exit $rc in $((SECONDS - t0))s"
  tail -n "${TAIL:-25}" "$OUT/$NAME-$name.log"
}

for url in https://pypi.org/simple/torch/ https://huggingface.co/api/models/espnet/owsm_v4_medium_1B \
           https://github.com; do
  echo "net $url: $(curl -sS -m 15 -o /dev/null -w '%{http_code}' "$url" 2>&1)"
done
step venv "${FRESH[@]}" uv venv --python "${FRESH_PY:-3.12}" --managed-python "$L/venv"
[ -x "$L/venv/bin/python" ] ||  # no download of a Python: the dev env's interpreter (its packages not included)
  step venv2 "${FRESH[@]}" uv venv --python "$PBS_LOCALDIR/owsm-env/bin/python3" "$L/venv"
# shellcheck disable=SC2086
step install "${FRESH[@]}" uv pip install --python "$L/venv/bin/python" "$WHEEL" ${FRESH_PKGS:-}
"${FRESH[@]}" uv pip freeze --python "$L/venv/bin/python" > "$OUT/$NAME-freeze.txt"
du -sh "$L/venv" | sed "s/^/venv size: /"

if [ -n "${SEED_ENGINES:-}" ]; then  # <engine dir>+<model cache name>: prebuilt engines (no build here)
  mkdir -p "$L/home/.cache/owsm_trt/${SEED_ENGINES#*+}"
  cp -r "${SEED_ENGINES%%+*}" "$L/home/.cache/owsm_trt/${SEED_ENGINES#*+}/"
fi
IFS=: read -ra MODELS <<< "${FRESH_MODELS:-espnet/owsm_v4_medium_1B}"
IFS=: read -ra DEVS <<< "${DEV_MODELS:--}"
for i in "${!MODELS[@]}"; do
  m=${MODELS[$i]} tag=$(basename "${MODELS[$i]}")
  step "cli1-$tag" "${FRESH[@]}" owsm-trt transcribe --model "$m" --lang eng "$WAV"  # cold: download, build
  step "cli2-$tag" "${FRESH[@]}" owsm-trt transcribe --model "$m" --lang eng "$WAV"  # warm caches
  # (warm engines, an empty Triton cache: the megakernels' compile time)
  TAIL=40 step "engaged-$tag" "${FRESH[@]}" TRITON_CACHE_DIR="$L/triton-$tag" python \
    "$PBS_LOCALDIR/snap/scripts/engaged.py" "$m" "$WAV"
  for b in 1 16; do
    step "eval-b$b-$tag" "${FRESH[@]}" python "$PBS_LOCALDIR/snap/scripts/eval_wer.py" --model "$m" \
      --batch-size $b --save "$OUT/$NAME-b$b-$tag.json"
  done
  d=${DEVS[$i]:--}
  [ "$d" = - ] && continue
  for b in 1 16; do
    step "dev-b$b-$tag" env -u CONDA_PREFIX PYTHONPATH="${DEV_SRC:-$PBS_LOCALDIR/snap/src}" \
      "$PBS_LOCALDIR/owsm-env/bin/python3" "$PBS_LOCALDIR/snap/scripts/eval_wer.py" --model "$d" --batch-size $b \
      --save "$OUT/dev-b$b-$tag.json" --compare "$OUT/$NAME-b$b-$tag.json"
  done
done
ls -R "$L/home/.cache/owsm_trt" 2>/dev/null | head -20
du -sh "$L/home"/.cache/* "$L/home"/.triton 2>/dev/null
