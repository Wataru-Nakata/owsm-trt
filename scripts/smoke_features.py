"""Exercise the API beyond plain ASR: language ID, timestamps, translation, prompts, long-form.

python scripts/smoke_features.py --model .cache/models/owsm_v4_medium_1B --backend tensorrt
"""

import argparse
import json
import time

import torch
from eval_wer import edit_distance, normalize

from owsm_trt import OWSM
from owsm_trt.audio import load_audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--long-seconds", type=float, default=100.0)
    args = ap.parse_args()

    model = OWSM.from_pretrained(args.model, backend=args.backend)
    items = json.load(open(args.data))
    wav = load_audio(items[0]["path"])
    print("reference :", items[0]["text"])

    if model.cfg.ctc_only:  # no decoder: no language identification, no timestamps
        print("ctc       :", model.transcribe(wav).text)
    else:
        print("language  :", [(lang, round(p, 3)) for lang, p in model.detect_language(wav, top_k=3)])
        res = model.transcribe(wav, timestamps=True)
        print("timestamps:", res.text)
        for s in res.segments:
            print(f"    [{s.start:6.2f} - {s.end:6.2f}] {s.text}")
    for task in ("st_deu", "st_jpn"):
        if task in model.tokenizer.tasks:
            print(f"{task:10s}:", model.transcribe(wav, task=task).text)
    print("prompted  :", model.transcribe(wav, prompt="Quilter, the apostle of the middle classes.").text)

    # long-form: concatenate utterances until --long-seconds
    waves, refs, total = [], [], 0.0
    for it in items:
        w = load_audio(it["path"])
        waves.append(w)
        refs.append(it["text"])
        total += len(w) / 16000
        if total >= args.long_seconds:
            break
    long = torch.cat(waves)
    t0 = time.perf_counter()
    res = model.transcribe_long(long)
    dt = time.perf_counter() - t0
    ref = " ".join(refs)
    errs = edit_distance(normalize(ref), normalize(res.text))
    print(f"long-form : {total:.0f}s audio in {dt:.2f}s, {len(res.segments)} segments, "
          f"WER {100 * errs / len(normalize(ref)):.2f}% vs concatenated references")  # fmt: skip
    for s in res.segments[:6]:
        print(f"    [{s.start:6.2f} - {s.end:6.2f}] {s.text}")
    print(
        f"    ... last: [{res.segments[-1].start:6.2f} - {res.segments[-1].end:6.2f}] {res.segments[-1].text}"
    )


if __name__ == "__main__":
    main()
