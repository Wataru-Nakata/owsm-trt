"""Per-layer timing of one TensorRT decode step (IProfiler) + engine inspector dump.

python scripts/trt_layers.py --model .cache/models/owsm_v4_medium_1B --batch 1 --out outputs/layers
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import tensorrt as trt
import torch

from owsm_trt import OWSM
from owsm_trt.audio import load_audio


class LayerTimes(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.ms: dict[str, float] = defaultdict(float)
        self.runs = 0

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.ms[layer_name] += ms


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--out", default="outputs/layers")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = OWSM.from_pretrained(args.model, backend="tensorrt")
    items = json.load(open(args.data))[: args.batch]
    t = model.tokenizer
    sess = model._session(args.batch)
    sess.encode(model._features([load_audio(it["path"]) for it in items]))
    primer = torch.tensor([[t.sos, t.lang_id("eng"), t.task_id("asr"), t.notime]] * args.batch, device="cuda")
    sess.decode(primer, 8)  # realistic cache state; pos is now past the prompt

    for name, engine, run_ctx in (
        ("decoder_step", model.backend.decoder, None),
        ("encoder", model.backend.encoder, model.backend.enc_ctx),
    ):
        if name == "decoder_step":
            b = sess
            run = model.backend.step_fn(
                b.tok, b.pos, b.self_k, b.self_v, b.cross_k, b.cross_v, b.logits, b.new_kv
            )
            ctx = run.ctx
        else:
            feats = model._features([load_audio(it["path"]) for it in items])

            def run():
                sess.encode(feats)

            ctx = run_ctx
        run()
        torch.cuda.synchronize()
        prof = LayerTimes()
        ctx.profiler = prof
        for _ in range(args.iters):
            run()
            torch.cuda.synchronize()
        total = sum(prof.ms.values()) / args.iters
        top = sorted(prof.ms.items(), key=lambda kv: -kv[1])
        print(f"== {name}: {len(prof.ms)} layers, {total:.3f} ms per run (sum of layer times)")
        for layer, ms in top[:25]:
            print(f"  {ms / args.iters:8.4f} ms  {layer[:150]}")
        kinds = defaultdict(lambda: [0, 0.0])
        for layer, ms in prof.ms.items():
            k = layer.split(" ")[0].split("_")[0][:40]
            kinds[k][0] += 1
            kinds[k][1] += ms / args.iters
        print("  by name prefix (count, ms):")
        for k, (n, ms) in sorted(kinds.items(), key=lambda kv: -kv[1][1])[:15]:
            print(f"    {n:4d} {ms:8.4f}  {k}")

        insp = engine.create_engine_inspector()
        insp.execution_context = ctx
        (out / f"{name}.json").write_text(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    print(f"inspector dumps in {out}")


if __name__ == "__main__":
    main()
