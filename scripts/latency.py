"""Single-utterance latency: ``transcribe`` of one recording, median of many calls.

python scripts/latency.py --model espnet/owsm_ctc_v4_1B [--seconds 8] [--iters 100]
"""

import argparse
import json
import statistics
import time

import torch

from owsm_trt import OWSM
from owsm_trt.audio import load_audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--seconds", type=float, default=8.0, help="the utterance closest to this length")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--save", help="(ignored; for scripts/ab_eval.py)")
    args = ap.parse_args()
    model = OWSM.from_pretrained(args.model, backend=args.backend)
    items = json.load(open(args.data))
    waves = [load_audio(it["path"]) for it in items]
    i = min(range(len(waves)), key=lambda j: abs(len(waves[j]) / 16000 - args.seconds))
    for _ in range(10):
        text = model.transcribe(waves[i]).text
    ts = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.transcribe(waves[i])
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    print(f"{items[i]['id']} ({len(waves[i]) / 16000:.2f} s): {text}")
    print(f"latency median {statistics.median(ts):.3f} ms (min {min(ts):.3f}, over {len(ts)} calls)")


if __name__ == "__main__":
    main()
