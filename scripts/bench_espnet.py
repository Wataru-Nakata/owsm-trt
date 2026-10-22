"""Baseline: official ESPnet ``Speech2Text`` on GPU (run in the tools/espnet_ref env, PYTHONPATH=src:scripts).

python scripts/bench_espnet.py --model .cache/models/owsm_v4_medium_1B --beam 5 --dtype float32
"""

import argparse
import json
import os
import time

import torch
from eval_wer import edit_distance, normalize

from owsm_trt.audio import load_audio
from owsm_trt.config import load_espnet_config
from owsm_trt.hub import resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--save")
    args = ap.parse_args()

    from espnet2.bin.s2t_inference import Speech2Text

    files = resolve(args.model)
    raw = load_espnet_config(files.config)
    cwd = os.getcwd()
    os.chdir(files.root)  # ESPnet resolves the config's relative paths (stats file) from the cwd
    s2t = Speech2Text(
        s2t_train_config=str(files.config),
        s2t_model_file=str(files.checkpoint),
        bpemodel=str(files.root / raw["bpemodel"]),
        device="cuda",
        dtype=args.dtype,
        beam_size=args.beam,
        ctc_weight=0.0,
        lang_sym="<eng>",
        task_sym="<asr>",
        predict_time=False,
    )
    os.chdir(cwd)
    items = json.load(open(args.data))
    waves = [load_audio(it["path"]) for it in items]
    audio_s = sum(len(w) for w in waves) / 16000
    for w in waves[:2]:
        s2t(w)  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    hyps = [s2t(w)[0][3] for w in waves]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    errs = sum(edit_distance(normalize(it["text"]), normalize(h)) for it, h in zip(items, hyps))
    words = sum(len(normalize(it["text"])) for it in items)
    print(f"backend=espnet dtype={args.dtype} beam={args.beam}")
    print(f"WER {100 * errs / words:.2f}% over {len(items)} utts ({words} words)")
    print(f"decode {dt:.2f}s for {audio_s:.0f}s audio: RTFx {audio_s / dt:.0f}")
    if args.save:
        json.dump(
            {it["id"]: h for it, h in zip(items, hyps)}, open(args.save, "w"), indent=1, ensure_ascii=False
        )


if __name__ == "__main__":
    main()
