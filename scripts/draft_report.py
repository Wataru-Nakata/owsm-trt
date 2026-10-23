"""Where do CTC drafts and the greedy decoder disagree? Per-utterance calls and the first differences.

python scripts/draft_report.py --model .cache/models/owsm_v4_medium_1B
"""

import argparse
import difflib
import json
from collections import Counter

import torch

from owsm_trt import OWSM
from owsm_trt.audio import load_audio


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--show", type=int, default=12, help="utterances to print in full")
    args = ap.parse_args()

    model = OWSM.from_pretrained(args.model, backend=args.backend)
    t = model.tokenizer
    items = json.load(open(args.data))
    ops, calls_hist, total = Counter(), Counter(), [0, 0, 0]  # tokens, calls, drafted tokens
    for n, it in enumerate(items):
        before = list(model.draft_stats)
        wave = load_audio(it["path"])
        res = model.transcribe(wave)
        calls = model.draft_stats[1] - before[1]
        draft = model._ctc_drafts(model._session(1).encode(model._features([wave])))[0]
        out = res.tokens
        total[0] += len(out)
        total[1] += calls
        total[2] += len(draft)
        calls_hist[calls] += 1
        sm = difflib.SequenceMatcher(a=draft, b=out, autojunk=False)
        diffs = [op for op in sm.get_opcodes() if op[0] != "equal"]
        for tag, a0, a1, b0, b1 in diffs:
            ops[tag] += 1
        if n < args.show:
            print(f"{it['id']}: {len(out)} tokens, {len(draft)} drafted, {calls} calls")
            for tag, a0, a1, b0, b1 in diffs[:6]:
                d = [t.tokens[i] for i in draft[a0:a1]]
                o = [t.tokens[i] for i in out[b0:b1]]
                print(f"    {tag:8s} draft {d} -> decoder {o}")
    tok, calls, drafted = total
    print(
        f"\n{len(items)} utterances: {tok} tokens, {drafted} drafted, {calls} decoder calls (incl. prefill)"
    )
    print(f"tokens per call {tok / calls:.2f}; draft/decoder edit ops: {dict(ops)}")
    print("calls per utterance:", dict(sorted(calls_hist.items())))


if __name__ == "__main__":
    main()
