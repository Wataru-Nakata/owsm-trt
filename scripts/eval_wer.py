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
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--no-draft", action="store_true", help="plain greedy steps, no CTC-drafted windows")
    ap.add_argument("--timestamps", action="store_true", help="save generated token ids, not text")
    ap.add_argument("--max-batch-size", type=int, default=16)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument(
        "--opt-level", type=int, default=None, help="TensorRT builder level (default: 3 / 5 for decoder)"
    )
    ap.add_argument("--save")
    ap.add_argument("--compare", nargs="*", default=[], help="transcript JSON files to diff against")
    ap.add_argument(
        "--fp8-calib", type=int, default=0, help="FP8 encoder calibrated on the first N utterances"
    )
    args = ap.parse_args()

    items = json.load(open(args.data))
    waves = [load_audio(it["path"]) for it in items]
    audio_s = sum(len(w) for w in waves) / 16000
    t0 = time.perf_counter()
    model = OWSM.from_pretrained(
        args.model,
        backend=args.backend,
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
        use_cuda_graph=not args.no_cuda_graph,
        optimization_level=args.opt_level,
        ctc_draft=not args.no_draft,
        fp8_calibration=waves[: args.fp8_calib] if args.fp8_calib else None,
    )
    load_s = time.perf_counter() - t0

    kw = {"batch_size": args.batch_size, "beam_size": args.beam, "timestamps": args.timestamps}
    model.transcribe(waves[: args.batch_size], **kw)  # warm-up / graph capture
    model.draft_stats = [0, 0]
    times = []
    for _ in range(args.repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model.transcribe(waves, **kw)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    dt = sorted(times)[len(times) // 2]  # median

    errs = sum(edit_distance(normalize(it["text"]), normalize(r.text)) for it, r in zip(items, results))
    words = sum(len(normalize(it["text"])) for it in items)
    n_tok = sum(len(r.tokens) for r in results)
    print(
        f"backend={args.backend} dtype={args.dtype} batch={args.batch_size} beam={args.beam} "
        f"draft={not args.no_draft} load={load_s:.1f}s"
    )
    print(f"WER {100 * errs / words:.2f}% over {len(items)} utts ({words} words)")
    if args.fp8_calib:  # the calibration utterances are part of the test set: also report the others
        rest = list(zip(items, results))[args.fp8_calib :]
        e = sum(edit_distance(normalize(it["text"]), normalize(r.text)) for it, r in rest)
        n = sum(len(normalize(it["text"])) for it, _ in rest)
        print(f"WER {100 * e / n:.2f}% over the {len(rest)} utts not used for FP8 calibration")
    print(
        f"decode {dt:.2f}s (median of {args.repeat}) for {audio_s:.0f}s audio: "
        f"RTFx {audio_s / dt:.0f}, {n_tok / dt:.0f} tok/s"
    )
    if model.draft_stats[1]:
        tokens, calls = model.draft_stats
        print(f"CTC drafts: {tokens / calls:.2f} tokens per decoder call ({tokens} tokens, {calls} calls)")

    if args.timestamps:
        hyps = {it["id"]: " ".join(map(str, r.tokens)) for it, r in zip(items, results)}
    else:
        hyps = {it["id"]: r.text for it, r in zip(items, results)}
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        json.dump(hyps, open(args.save, "w"), indent=1, ensure_ascii=False)
    for path in args.compare:
        if not Path(path).exists():
            print(f"(no {path} to compare against)")
            continue
        ref = json.load(open(path))
        diff = [k for k in hyps if hyps[k] != ref.get(k)]
        print(f"identical to {path}: {len(hyps) - len(diff)}/{len(hyps)}")
        for k in diff[:5]:
            print(f"  {k}\n    ref : {ref.get(k)}\n    this: {hyps[k]}")


if __name__ == "__main__":
    main()
