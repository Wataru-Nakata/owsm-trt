"""Dump each utterance's CTC draft and greedy output (token ids), for tuning drafted decoding offline
with scripts/draft_sim.py.

python scripts/dump_drafts.py --model .cache/models/owsm_v4_medium_1B \
    --data .cache/testdata/librispeech_test_clean/manifest.json --out outputs/drafts.json
"""

import argparse
import json

import torch

from owsm_trt import OWSM
from owsm_trt.audio import load_audio


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    items = json.load(open(args.data))
    model = OWSM.from_pretrained(args.model, backend=args.backend, device=args.device)
    t = model.tokenizer
    primer = model._primer("eng", "asr", False, None)
    rows = []
    for i in range(0, len(items), args.batch):
        chunk = items[i : i + args.batch]
        sess = model._session(len(chunk))
        sess.encode(model._features([load_audio(it["path"]) for it in chunk]))
        drafts = [model._ctc_draft(r) for r in sess.ctc_ids.tolist()]
        outs, _ = sess.decode_drafted(
            torch.tensor([primer] * len(chunk), device=model.device), drafts, model._max_new(), t.not_drafted
        )
        for it, d, o in zip(chunk, drafts, outs):
            rows.append({"id": it["id"], "draft": d, "out": o})
    meta = {"primer": primer, "eos": t.eos, "insertions": sorted(t.not_drafted), "utterances": rows}
    meta["punctuation"] = sorted(t.punctuation)
    json.dump(meta, open(args.out, "w"))
    print(f"{len(rows)} utterances -> {args.out}")


if __name__ == "__main__":
    main()
