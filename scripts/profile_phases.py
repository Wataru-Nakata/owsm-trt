"""Per-phase GPU timing (CUDA events): frontend, encoder, prefill, per-token decode step.

python scripts/profile_phases.py --model .cache/models/owsm_v4_base_102M --backend tensorrt --batch 1
"""

import argparse
import json

import torch

from owsm_trt import OWSM
from owsm_trt.audio import load_audio


def timed(fn, iters: int = 20) -> float:
    """Mean milliseconds of ``fn`` on the current stream."""
    fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    args = ap.parse_args()

    model = OWSM.from_pretrained(
        args.model, backend=args.backend, dtype=args.dtype, optimization_level=args.opt_level
    )
    items = json.load(open(args.data))[: args.batch]
    waves = [load_audio(it["path"]) for it in items]
    t = model.tokenizer
    primer = [t.sos, t.lang_id("eng"), t.task_id("asr"), t.notime]
    sess = model._session(args.batch)
    primers = torch.tensor([primer] * args.batch, device=model.device)

    feats = model._features(waves)
    res = {
        "features (pad+H2D+frontend)": timed(lambda: model._features(waves)),
        "frontend": timed(
            lambda: model.frontend(torch.zeros(args.batch, model.cfg.num_samples, device="cuda"))
        ),
        "encode": timed(lambda: sess.encode(feats)),
    }
    sess.encode(feats)
    res["decode 1 token (encode excluded, prefill incl.)"] = timed(lambda: sess.decode(primers, 1), 10)
    n = 64
    sess.eos.fill_(-1)  # never stop early, so exactly n tokens are generated
    res[f"decode {n} tokens"] = timed(lambda: sess.decode(primers, n), 5)
    sess.eos.fill_(t.eos)
    res["=> per decode step"] = (
        res[f"decode {n} tokens"] - res["decode 1 token (encode excluded, prefill incl.)"]
    ) / (n - 1)
    for c, graph in sess.graphs.items():
        sess.pos.zero_()  # replays advance pos; stay inside the bucket
        res[f"graph replay, cache bucket {c}"] = timed(graph.replay, min(50, c - 2))
    res["full transcribe()"] = timed(lambda: model.transcribe(waves, batch_size=args.batch), 5)
    out = model.transcribe(waves, batch_size=args.batch)
    print(f"backend={args.backend} dtype={args.dtype} batch={args.batch} tokens/utt={len(out[0].tokens)}")
    for k, v in res.items():
        print(f"  {k:48s} {v:8.3f} ms")


if __name__ == "__main__":
    main()
