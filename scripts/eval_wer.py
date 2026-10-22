"""WER / speed check for one backend configuration.

    python scripts/eval_wer.py --model .cache/models/owsm_v4_base_102M --backend tensorrt --dtype float16 \
        --save outputs/trt_fp16.json --compare outputs/torch_fp32.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from owsm_trt import OWSM
from owsm_trt.audio import load_audio


def normalize(text: str) -> list[str]:
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return text.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-batch-size", type=int, default=16)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--save")
    ap.add_argument("--compare")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = OWSM.from_pretrained(
        args.model,
        backend=args.backend,
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
        use_cuda_graph=not args.no_cuda_graph,
        optimization_level=args.opt_level,
    )
    load_s = time.perf_counter() - t0
    items = json.load(open(args.data))
    waves = [load_audio(it["path"]) for it in items]
    audio_s = sum(len(w) for w in waves) / 16000

    model.transcribe(waves[: args.batch_size], batch_size=args.batch_size)  # warm-up / graph capture
    times = []
    for _ in range(args.repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model.transcribe(waves, batch_size=args.batch_size)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    dt = sorted(times)[len(times) // 2]  # median

    errs = sum(edit_distance(normalize(it["text"]), normalize(r.text)) for it, r in zip(items, results))
    words = sum(len(normalize(it["text"])) for it in items)
    n_tok = sum(len(r.tokens) for r in results)
    print(f"backend={args.backend} dtype={args.dtype} batch={args.batch_size} load={load_s:.1f}s")
    print(f"WER {100 * errs / words:.2f}% over {len(items)} utts ({words} words)")
    print(
        f"decode {dt:.2f}s (median of {args.repeat}) for {audio_s:.0f}s audio: "
        f"RTFx {audio_s / dt:.0f}, {n_tok / dt:.0f} tok/s"
    )

    hyps = {it["id"]: r.text for it, r in zip(items, results)}
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        json.dump(hyps, open(args.save, "w"), indent=1, ensure_ascii=False)
    if args.compare:
        ref = json.load(open(args.compare))
        diff = [k for k in hyps if hyps[k] != ref.get(k)]
        print(f"identical to {args.compare}: {len(hyps) - len(diff)}/{len(hyps)}")
        for k in diff[:5]:
            print(f"  {k}\n    ref : {ref.get(k)}\n    this: {hyps[k]}")


if __name__ == "__main__":
    main()
