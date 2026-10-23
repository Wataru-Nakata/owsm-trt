"""Per-layer timing of one TensorRT decode step (IProfiler) + engine inspector dump.

python scripts/trt_layers.py --model .cache/models/owsm_v4_medium_1B --batch 1 --out outputs/layers

OWSM-CTC models: the CTC engine, on the first ``--batch`` utterances and on whole 30 s windows, with the
TensorRT layers' time summed per part of the model (see ``ctc_part``).
"""

import argparse
import ast
import json
import re
import tempfile
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
    ap.add_argument("--windows", default="", help="also profile batch-1 verify windows, e.g. 2,8,32")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = OWSM.from_pretrained(args.model, backend="tensorrt")
    items = json.load(open(args.data))[: args.batch]
    if model.cfg.ctc_only:
        profile_ctc(model, args, items, out)
        return
    t = model.tokenizer
    sess = model._session(args.batch)
    sess.encode(model._features([load_audio(it["path"]) for it in items]))
    primer = torch.tensor([[t.sos, t.lang_id("eng"), t.task_id("asr"), t.notime]] * args.batch, device="cuda")
    sess.decode(primer, 8)  # realistic cache state; pos is now past the prompt

    targets = [("decoder_step", model.backend.decoder, None)]
    targets += [
        (f"verify_{w}", model.backend.verifier, None) for w in map(int, filter(None, args.windows.split(",")))
    ]
    targets += [("encoder", model.backend.encoder, model.backend.enc_ctx(args.batch))]
    for name, engine, run_ctx in targets:
        if name == "decoder_step":
            b = sess
            run = model.backend.step_fn(
                b.tok, b.offs, b.self_k, b.self_v, b.cross_k, b.cross_v, b.logits, b.new_kv
            )
            ctx = run.ctx
        elif name.startswith("verify_"):
            w = int(name.split("_")[1])
            k, v = sess.caches[64]
            vt = torch.full((args.batch, w), t.eos, device="cuda")
            offs = torch.zeros(args.batch, dtype=torch.long, device="cuda")
            vl = torch.zeros(args.batch, w, model.cfg.vocab_size, device="cuda")
            vkv = torch.zeros(
                sess.new_kv.shape[:4] + (w, sess.new_kv.shape[-1]), dtype=sess.new_kv.dtype, device="cuda"
            )
            run = model.backend.step_fn(vt, offs, k, v, sess.cross_k, sess.cross_v, vl, vkv, kind="verify")
            ctx = run.ctx
        else:
            feats = model._features([load_audio(it["path"]) for it in items])

            def run():  # the backend's own context, not the session's CUDA graph
                model.backend.encode(feats, sess.cross_k, sess.cross_v, sess.ctc_ids)

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

        times = {layer: ms / args.iters for layer, ms in prof.ms.items()}
        (out / f"{name}_times.json").write_text(json.dumps(times, indent=0))
        insp = engine.create_engine_inspector()
        insp.execution_context = ctx
        (out / f"{name}.json").write_text(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    print(f"inspector dumps in {out}")


def ctc_part(scopes: list[str], op: str) -> str:
    """The part of an OWSM-CTC model an ONNX node belongs to, from its module scopes."""
    s = [x for x in scopes[:-1] if x]  # the last entry is the node itself
    deepest = s[-1] if s else ""
    if any(x.startswith(("model.prompt_encoder", "model.embed", "model.prompt_proj")) for x in s):
        return "prompt encoder"
    if any(x.startswith("model.encoder.embed") for x in s):
        return "subsampling"
    if "model.ctc" in s:
        return "final CTC (ctc_lo + argmax)"
    if any(x == "model.ctc.ctc_lo" for x in s):
        return "interCTC ctc_lo"
    if any(x.startswith("model.encoder.conditioning_layer") for x in s):
        return "interCTC conditioning linear"
    m = re.match(r"model\.encoder\.encoders\.\d+\.?(.*)", deepest)
    if m:
        sub = m.group(1)
        if not sub:
            return "layer: cross-attention core / residual" if op in ("Softmax", "MatMul") else "layer: misc"
        if sub.startswith("cgmlp.csgu.conv"):
            return "layer: cgmlp dwconv"
        if sub.startswith("cgmlp"):
            return "layer: cgmlp " + sub.split(".")[1].replace("csgu", "csgu norm")
        if sub.startswith(("norm_", "norm")):
            return "layer: " + sub.split(".")[0]
        return "layer: " + sub.split(".")[0]
    if deepest.startswith("model.encoder.after_norm"):
        return "after_norm"
    if deepest == "model.encoder":
        return "interCTC softmax/add" if op in ("Softmax", "Cast", "Add") else "encoder misc"
    return "other"


def onnx_parts(model, name: str) -> dict[str, str]:
    """ONNX node name -> part (ctc_part), from an export of the model as its engine was built."""
    import onnx

    from owsm_trt import hub
    from owsm_trt.build import export_ctc_onnx
    from owsm_trt.config import load_espnet_config, parse_config
    from owsm_trt.modeling import load_model

    files = hub.resolve(name)
    cfg = parse_config(load_espnet_config(files.config))
    net = load_model(cfg, str(files.checkpoint), "cuda", torch.float16)
    with tempfile.TemporaryDirectory() as tmp:
        path = export_ctc_onnx(net, Path(tmp), model.opts)
        graph = onnx.load(str(path), load_external_data=False).graph
    del net
    torch.cuda.empty_cache()
    parts = {}
    for n in graph.node:
        meta = {p.key: p.value for p in n.metadata_props}
        scopes = ast.literal_eval(meta.get("pkg.torch.onnx.name_scopes", "[]"))
        parts[n.name] = ctc_part(scopes, n.op_type)
    return parts


def profile_ctc(model, args, items, out: Path) -> None:
    from owsm_trt.runtime import CTCSession

    t = model.tokenizer
    b = args.batch
    waves = [load_audio(it["path"]) for it in items]
    feats = model._features(waves)
    padded = torch.zeros(b, model.cfg.num_samples, device="cuda")
    for row, w in zip(padded, waves):
        row[: len(w)] = w[: len(row)].cuda()
    full = model.frontend(padded)  # features of the whole window, as in long-form windows
    prompt = torch.tensor([[t.na]] * b, device="cuda")
    prefix = torch.tensor([[t.lang_id("eng"), t.task_id("asr")]] * b, device="cuda")
    try:
        parts = onnx_parts(model, args.model)
    except Exception as e:  # noqa: BLE001
        print(f"(no ONNX attribution: {e!r})")
        parts = {}
    engine = model.backend.engine
    insp = engine.create_engine_inspector()
    for name, x in (("data", feats), ("full30s", full)):
        sess = CTCSession(model.backend, model.cfg, b, 1, "cuda", use_cuda_graph=True)
        sess(x, prompt, prefix)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ms = []
        for _ in range(args.iters):
            start.record()
            sess(x, prompt, prefix)
            end.record()
            torch.cuda.synchronize()
            ms.append(start.elapsed_time(end))
        wall = sorted(ms)[len(ms) // 2]
        view, run, _ = sess._runs[tuple(x.shape)]
        ctx = run.ctx
        prof = LayerTimes()
        ctx.profiler = prof
        run()
        torch.cuda.synchronize()
        prof.ms.clear()
        for _ in range(args.iters):
            run()
            torch.cuda.synchronize()
        times = {layer: v / args.iters for layer, v in prof.ms.items()}
        total = sum(times.values())
        print(f"== ctc {name}: batch {b}, feats {tuple(x.shape)}: {wall:.3f} ms per call (CUDA graph), "
              f"{total:.3f} ms sum of {len(times)} layer times")  # fmt: skip
        insp.execution_context = ctx
        dump = insp.get_engine_information(trt.LayerInformationFormat.JSON)
        if not dump:  # (seen with the batch-1 profile's context) without the context's shapes
            insp = engine.create_engine_inspector()
            dump = insp.get_engine_information(trt.LayerInformationFormat.JSON)
        info = json.loads(dump)
        (out / f"ctc_{name}.json").write_text(json.dumps(info))
        (out / f"ctc_{name}_times.json").write_text(json.dumps(times, indent=0))
        meta = {}
        for layer in info["Layers"]:
            if isinstance(layer, dict):
                meta[layer["Name"]] = layer
        by_part = defaultdict(lambda: [0, 0.0])
        by_kind = defaultdict(lambda: [0, 0.0])
        rows = []
        for layer, v in times.items():
            info_l = meta.get(layer, {})
            nodes = re.findall(r"\[ONNX Layer: ([^\]]+)\]", info_l.get("Metadata", ""))
            names = [p for p in (parts.get(n) for n in nodes) if p]
            part = max(set(names), key=names.count) if names else "?"
            by_part[part][0] += 1
            by_part[part][1] += v
            kind = info_l.get("LayerType", "?")
            by_kind[kind][0] += 1
            by_kind[kind][1] += v
            rows.append((v, layer, part, kind, info_l.get("TacticName", "")))
        print("  by part (layers, ms, %):")
        for p, (n, v) in sorted(by_part.items(), key=lambda kv: -kv[1][1]):
            print(f"    {n:5d} {v:8.3f} {100 * v / total:5.1f}%  {p}")
        print("  by TensorRT layer type (layers, ms):")
        for k, (n, v) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
            print(f"    {n:5d} {v:8.3f}  {k}")
        print("  top layers:")
        for v, layer, part, kind, tactic in sorted(rows, reverse=True)[:40]:
            print(f"    {v:7.4f} ms  [{part}] {kind} {layer[:90]} | {tactic[:70]}")
    print(f"inspector dumps in {out}")


if __name__ == "__main__":
    main()
