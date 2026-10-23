"""Does the 30 s input padding matter for accuracy? Greedy WER (PyTorch backend) with every input padded
to 30 s, as ESPnet does, against inputs padded only up to a multiple of a shorter length.

python scripts/pad_experiment.py --model .cache/models/owsm_v4_medium_1B \
    --data .cache/testdata/librispeech_test_clean/manifest.json --steps 30,10,5,2
"""

import argparse
import dataclasses
import json
import math
import time
from collections import defaultdict

import torch
from eval_wer import edit_distance, normalize

from owsm_trt import OWSM
from owsm_trt.audio import load_audio
from owsm_trt.pipeline import ctc_collapse
from owsm_trt.runtime import CTCSession, GreedySession


@torch.inference_mode()
def transcribe(model: OWSM, cfg, waves: list[torch.Tensor], batch: int) -> list[str]:
    """Greedy transcripts (CTC drafts, timestamp rules as in the pipeline) with inputs padded to ``cfg``."""
    t, out = model.tokenizer, []
    for i in range(0, len(waves), batch):
        chunk = waves[i : i + batch]
        speech = torch.zeros(len(chunk), cfg.num_samples, device=model.device)
        for row, w in zip(speech, chunk):
            row[: min(len(w), cfg.num_samples)] = w[: cfg.num_samples].to(model.device)
        feats = model.frontend(speech)
        if cfg.ctc_only:
            sess = CTCSession(model.backend, cfg, len(chunk), 1, model.device, use_cuda_graph=False)
            prefix = torch.tensor([[t.lang_id("eng"), t.task_id("asr")]] * len(chunk), device=model.device)
            ids = sess(feats, torch.tensor([[t.na]] * len(chunk), device=model.device), prefix)
            out += [t.decode(ctc_collapse(row, t.blank)) for row in ids.tolist()]
            continue
        sess = GreedySession(
            model.backend, cfg, len(chunk), model.opts.capacity, t.eos, model.dtype, model.device,
            use_cuda_graph=False, draft_windows=(1, 2, 4, 8, 16, 32), rules=model.rules,
        )  # fmt: skip
        paths = sess.encode(feats)
        drafts = [ctc_collapse(row, t.blank) for row in paths.tolist()]
        primer = torch.tensor([model._primer("eng", "asr", False, None)] * len(chunk), device=model.device)
        ids, _ = sess.decode_drafted(primer, drafts, cfg.enc_frames, t.not_drafted)
        out += [t.decode(x) for x in ids]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--steps", default="30,10,5,2", help="pad up to a multiple of this many seconds")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    items = json.load(open(args.data))
    items = items[: args.limit] if args.limit else items
    waves = [load_audio(it["path"]) for it in items]
    model = OWSM.from_pretrained(args.model, backend="torch", dtype=args.dtype, device=args.device)
    sr, full = model.cfg.sample_rate, model.cfg.speech_seconds
    refs = [normalize(it["text"]) for it in items]
    words = sum(map(len, refs))
    print(f"{len(items)} utterances, {sum(map(len, waves)) / sr / 3600:.2f} h, {words} words", flush=True)
    base = None
    for step in map(float, args.steps.split(",")):
        groups = defaultdict(list)  # padded length -> utterance indices
        for i, w in enumerate(waves):
            groups[min(full, step * math.ceil(len(w) / sr / step))].append(i)
        hyps: list[str | None] = [None] * len(items)
        t0 = time.time()
        for length, idx in sorted(groups.items()):
            cfg = dataclasses.replace(model.cfg, speech_seconds=length)
            for j, text in zip(idx, transcribe(model, cfg, [waves[i] for i in idx], args.batch)):
                hyps[j] = text
        errs = sum(edit_distance(r, normalize(h)) for r, h in zip(refs, hyps))
        frames = sum(min(full, step * math.ceil(len(w) / sr / step)) for w in waves) / (full * len(waves))
        line = f"pad to a multiple of {step:g} s: WER {100 * errs / words:.2f}% ({errs} errors)"
        line += f", encoder frames {100 * frames:.0f}% of 30 s padding, {time.time() - t0:.0f}s"
        if base is None:
            base = hyps
        else:
            line += f", {sum(a == b for a, b in zip(base, hyps))}/{len(items)} transcripts as with 30 s"
        print(line, flush=True)


if __name__ == "__main__":
    main()
