"""TensorRT fp16 vs PyTorch fp32 for several models: transcripts, WER and speed.

python scripts/check_models.py espnet/owls_025B_180K espnet/powsm@pr espnet/owsm_v1@asr@en

A model may carry a task and a language: id[@task[@lang]] (default asr, eng). fp32 parity with ESPnet is scripts/parity_espnet.py;
this checks that the fp16 engines agree with the fp32 reimplementation and are fast.
"""

import argparse
import gc
import json
import time

import torch
from eval_wer import edit_distance, normalize

from owsm_trt import OWSM
from owsm_trt.audio import load_audio


def run(model: OWSM, waves, task: str, lang: str, batch_size: int) -> tuple[list[str], float]:
    kw = {"task": task, "lang": lang, "batch_size": batch_size}
    model.transcribe(waves[:batch_size], **kw)  # warm-up / graph capture
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = model.transcribe(waves, **kw)
    torch.cuda.synchronize()
    return [r.text for r in res], time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    args = ap.parse_args()
    items = json.load(open(args.data))
    waves = [load_audio(it["path"]) for it in items]
    audio_s = sum(len(w) for w in waves) / 16000
    words = sum(len(normalize(it["text"])) for it in items)

    for spec in args.models:
        name, *rest = spec.split("@")
        task, lang = (rest + ["asr", "eng"][len(rest) :])[:2]
        print(f"=== {name} task={task} lang={lang}", flush=True)
        ref = OWSM.from_pretrained(name, backend="torch", dtype="float32")
        ref_text, _ = run(ref, waves, task, lang, 8)
        del ref
        gc.collect()
        torch.cuda.empty_cache()

        t0 = time.perf_counter()
        trt = OWSM.from_pretrained(name)
        load = time.perf_counter() - t0
        text1, dt1 = run(trt, waves, task, lang, 1)
        text8, dt8 = run(trt, waves, task, lang, 8)
        same = sum(a == b for a, b in zip(text1, ref_text))
        line = f"engines+load {load:.0f}s | b1 RTFx {audio_s / dt1:.0f}, b8 RTFx {audio_s / dt8:.0f}"
        line += f" | identical to torch fp32: b1 {same}/{len(items)}"
        line += f", b8 {sum(a == b for a, b in zip(text8, ref_text))}/{len(items)}"
        if task == "asr":
            wer = [
                sum(edit_distance(normalize(it["text"]), normalize(h)) for it, h in zip(items, hyp))
                for hyp in (ref_text, text1)
            ]
            line += f" | WER fp32 {100 * wer[0] / words:.2f}% trt {100 * wer[1] / words:.2f}%"
        print(line)
        for a, b, it in list(zip(ref_text, text1, items))[:2]:
            print(f"  {it['id']}\n    fp32: {a}\n    trt : {b}")
        del trt
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
