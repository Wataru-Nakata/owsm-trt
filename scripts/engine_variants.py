"""TensorRT builder settings (tiling optimization level, auxiliary streams) and decoder export variants
(attention plugins, split self-attention), tried on the encoder and on decoder calls of the real network:
each variant is built and timed with CUDA graphs. Decoder variants are also compared with the fp32
PyTorch decoder on inputs with varied per-row cache offsets, and get a per-layer profile of a batch-1
8-token call.

python scripts/engine_variants.py --model .cache/models/owsm_v4_medium_1B --variants base,tile_fast,aux0
"""

import argparse
import json
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import tensorrt as trt
import torch

from owsm_trt import attn_plugin, hub
from owsm_trt.build import BuildOptions, decoder_profile_shapes, export_decoder_onnx, export_onnx
from owsm_trt.config import load_espnet_config, parse_config
from owsm_trt.modeling import load_model

TILING = trt.TilingOptimizationLevel
VARIANTS = {  # name: builder config settings
    "base": {},
    "tile_fast": {"tiling_optimization_level": TILING.FAST},
    "tile_moderate": {"tiling_optimization_level": TILING.MODERATE},
    "tile_full": {"tiling_optimization_level": TILING.FULL},
    "aux0": {"max_aux_streams": 0},
    "aux4": {"max_aux_streams": 4},
    "split": {},  # decoder exported with split self-attention (cache and new tokens apart)
}
# decoder exported with attention plugins (attn_plugin; "fixed": one kernel configuration, no tactics)
PLUGINS = {
    "attn": {"self", "cross"},
    "attn_self": {"self"},
    "attn_cross": {"cross"},
    "attn_fixed": {"self", "cross"},
}
VARIANTS |= dict.fromkeys(PLUGINS, {})
DTYPES = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
}


def build(
    onnx_path: Path, engine_path: Path, profiles: list[dict], settings: dict, level: int, aot: bool = False
) -> float:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    if aot:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    assert parser.parse_from_file(str(onnx_path)), parser.get_error(0)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.builder_optimization_level = level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # layer names; no runtime cost
    for key, value in settings.items():
        setattr(config, key, value)
    dynamic = [
        network.get_input(i).name for i in range(network.num_inputs) if -1 in network.get_input(i).shape
    ]
    for prof in profiles:
        p = builder.create_optimization_profile()
        for name in dynamic:
            p.set_shape(name, *prof[name])
        config.add_optimization_profile(p)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, f"failed to build {engine_path.name}"
    engine_path.write_bytes(serialized)
    return time.time() - t0


def bind(engine, profile: int, inputs: dict):
    """An execution context on ``profile`` with ``inputs`` bound, and its output tensors."""
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()
    ctx.set_optimization_profile_async(profile, stream.cuda_stream)
    stream.synchronize()
    outputs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(name, tuple(inputs[name].shape))
            ctx.set_tensor_address(name, inputs[name].data_ptr())
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            t = torch.empty(
                tuple(ctx.get_tensor_shape(name)), dtype=DTYPES[engine.get_tensor_dtype(name)], device="cuda"
            )
            outputs[name] = t
            ctx.set_tensor_address(name, t.data_ptr())
    return ctx, outputs


def run_once(engine, profile: int, inputs: dict) -> dict:
    ctx, outputs = bind(engine, profile, inputs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        assert ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    return outputs


def time_engine(engine, profile: int, inputs: dict, iters: int = 50) -> float:
    ctx, _ = bind(engine, profile, inputs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


class LayerTimes(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.ms: dict[str, float] = defaultdict(float)
        self.order: list[str] = []

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        if layer_name not in self.ms:
            self.order.append(layer_name)
        self.ms[layer_name] += ms


def profile_layers(engine, profile: int, inputs: dict, iters: int = 20) -> None:
    """Per-layer times (IProfiler, so each layer is a little slower than in a CUDA graph)."""
    ctx, _ = bind(engine, profile, inputs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    prof = LayerTimes()
    ctx.profiler = prof
    for _ in range(iters):
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
    total = sum(prof.ms.values()) / iters
    print(f"   layers: {len(prof.ms)}, {total:.3f} ms (sum of profiled layer times); the first 40:")
    for name in prof.order[:40]:
        print(f"   {prof.ms[name] / iters * 1000:7.1f} us  {name[:140]}")
    kinds = defaultdict(lambda: [0, 0.0])
    for name, ms in prof.ms.items():
        kind = "MatMul" if name.startswith("node_MatMul") else name.split("_myl")[0][:60]
        kinds[kind][0] += 1
        kinds[kind][1] += ms / iters
    for kind, (n, ms) in sorted(kinds.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"   {n:4d} {ms:8.4f} ms  {kind}")
    # what the inspector says about the plugin layers (chosen tactic, formats)
    try:
        info = json.loads(
            engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
        )
    except Exception as e:  # noqa: BLE001 - TensorRT 11.4 fails on some plugin engines
        print(f"   inspector failed: {e}")
        return
    shown = defaultdict(int)
    for layer in info.get("Layers", []):
        text = json.dumps(layer)
        for key in ("self_attn", "cross_attn"):
            if key in text and shown[key] < 2:
                shown[key] += 1
                print(f"   inspector: {text[:1200]}")


def decoder_inputs(cfg, b: int, s: int, cache: int = 64, offsets: list[int] | None = None) -> dict:
    H, dk, T = cfg.dec_heads, cfg.d_model // cfg.dec_heads, cfg.enc_frames
    f16 = {"device": "cuda", "dtype": torch.float16}
    inp = {
        "tokens": torch.randint(0, cfg.vocab_size, (b, s), device="cuda"),
        "offset": torch.full((b,), 20, dtype=torch.long, device="cuda"),
    }
    if offsets is not None:
        inp["offset"] = torch.tensor(offsets, dtype=torch.long, device="cuda")
    for i in range(cfg.dec_layers):
        inp[f"self_k_{i}"] = torch.randn(b, H, cache, dk, **f16)
        inp[f"self_v_{i}"] = torch.randn(b, H, cache, dk, **f16)
        inp[f"cross_k_{i}"] = torch.randn(b, H, T, dk, **f16)
        inp[f"cross_v_{i}"] = torch.randn(b, H, T, dk, **f16)
    return inp


def check_cases(cfg) -> list[tuple[str, int, dict]]:
    """(label, profile, inputs) with varied per-row cache offsets; slots past them hold finite garbage."""
    torch.manual_seed(0)
    return [
        ("b1 s8 c64 off37", 1, decoder_inputs(cfg, 1, 8, 64, [37])),
        ("b1 s32 c128 off90", 1, decoder_inputs(cfg, 1, 32, 128, [90])),
        ("b5 s8 c64", 2, decoder_inputs(cfg, 5, 8, 64, [0, 9, 23, 41, 56])),
        ("b16 s16 c128", 2, decoder_inputs(cfg, 16, 16, 128, torch.randint(0, 113, (16,)).tolist())),
        ("b3 s32 c512", 2, decoder_inputs(cfg, 3, 32, 512, [0, 200, 480])),
        ("b2 s1 c64", 2, decoder_inputs(cfg, 2, 1, 64, [0, 63])),
    ]


def reference_outputs(model, cfg, cases: list) -> dict[str, dict]:
    """The fp32 PyTorch decoder's outputs on the check cases."""
    out = {}
    for label, _, inp in cases:
        n = cfg.dec_layers
        caches = [
            [inp[f"{kind}_{i}"].float() for i in range(n)]
            for kind in ("self_k", "self_v", "cross_k", "cross_v")
        ]
        logits, new_kv = model.decoder.step(inp["tokens"], inp["offset"], *caches)
        out[label] = {"logits": logits, "new_kv": new_kv}
    return out


def compare(label: str, out: dict, ref: dict) -> None:
    diff = (out["logits"] - ref["logits"]).abs()
    agree = (out["logits"].argmax(-1) == ref["logits"].argmax(-1)).float().mean().item()
    dkv = (out["new_kv"].float() - ref["new_kv"].float()).abs()
    print(
        f"   {label:18s} vs fp32: logits max/mean diff {diff.max().item():.4f}/{diff.mean().item():.5f}, "
        f"argmax agree {agree:.4f}, new_kv max/mean diff {dkv.max().item():.4f}/{dkv.mean().item():.5f}"
    )


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--parts", default="decoder,encoder")
    ap.add_argument("--no-layers", action="store_true", help="no per-layer profile of a b1 w8 decoder call")
    args = ap.parse_args()
    names = args.variants.split(",")
    files = hub.resolve(args.model)
    cfg = parse_config(load_espnet_config(files.config))
    model = load_model(cfg, str(files.checkpoint), "cuda", torch.float16)
    opts = BuildOptions()
    tmp = Path(tempfile.mkdtemp())
    if "encoder" in args.parts:
        enc_onnx, dec_onnx = export_onnx(model, tmp / "onnx", opts)
    else:
        dec_onnx = export_decoder_onnx(model, tmp / "onnx" / "decoder.onnx", opts)
    onnx = dict.fromkeys(names, dec_onnx)
    if "split" in names:
        onnx["split"] = export_decoder_onnx(model, tmp / "onnx_split" / "decoder.onnx", opts, True)
    for name in names:
        if name in PLUGINS:
            path = tmp / f"onnx_{name}" / "decoder.onnx"
            onnx[name] = export_decoder_onnx(model, path, opts, plugins=frozenset(PLUGINS[name]))
    if any(name in PLUGINS for name in names):
        attn_plugin.register_plugins(cfg.dec_heads, cfg.d_model // cfg.dec_heads)
    del model
    torch.cuda.empty_cache()
    checks = check_cases(cfg) if "decoder" in args.parts else []
    if checks:
        model = load_model(cfg, str(files.checkpoint), "cuda", torch.float32)
        reference = reference_outputs(model, cfg, checks)
        del model
        torch.cuda.empty_cache()
    B = opts.max_batch
    dec_profiles = [
        {"kind": "decode", "batch": (1, 1, 1)},
        {"kind": "verify", "batch": (1, 1, 1)},
        {"kind": "verify", "batch": (2, B // 2, B)},
    ]
    dec_shapes = [decoder_profile_shapes(cfg, opts, p) for p in dec_profiles]
    frames = (64, cfg.num_frames // 3, cfg.num_frames)
    enc_shapes = [
        {"feats": tuple((n, t, cfg.n_mels) for n, t in zip(batch, frames))}
        for batch in ((1, 1, 1), (2, B, B))
    ]
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    for name in names:
        settings = VARIANTS[name]
        line = f"== {name:14s}"
        try:
            if "decoder" in args.parts:
                path = tmp / f"dec_{name}.engine"
                attn_plugin.TUNE = name != "attn_fixed"
                secs = build(onnx[name], path, dec_shapes, settings, 4, aot=name in PLUGINS)
                engine = runtime.deserialize_cuda_engine(path.read_bytes())
                res = [
                    ("b1 step", 0, decoder_inputs(cfg, 1, 1)),
                    ("b1 w8", 1, decoder_inputs(cfg, 1, 8)),
                    ("b8 w8", 2, decoder_inputs(cfg, 8, 8)),
                    ("b16 w16", 2, decoder_inputs(cfg, 16, 16)),
                ]
                line += f" decoder (built in {secs:.0f}s):" + "".join(
                    f" {label} {time_engine(engine, p, inp):.3f}" for label, p, inp in res
                )
                print(line + " (ms)", flush=True)
                line = " " * 17
                for label, p, inp in checks:
                    compare(label, run_once(engine, p, inp), reference[label])
                if not args.no_layers:
                    profile_layers(engine, 1, res[1][2])
                del engine
            if "encoder" in args.parts:
                path = tmp / f"enc_{name}.engine"
                secs = build(enc_onnx, path, enc_shapes, settings, 3)
                engine = runtime.deserialize_cuda_engine(path.read_bytes())
                res = [
                    ("b1 T1000", 0, {"feats": torch.randn(1, 1000, cfg.n_mels, device="cuda")}),
                    ("b1 T3001", 0, {"feats": torch.randn(1, cfg.num_frames, cfg.n_mels, device="cuda")}),
                    ("b16 T1000", 1, {"feats": torch.randn(16, 1000, cfg.n_mels, device="cuda")}),
                ]
                line += f" | encoder (built in {secs:.0f}s):" + "".join(
                    f" {label} {time_engine(engine, p, inp):.3f}" for label, p, inp in res
                )
                del engine
                print(line + " (ms)", flush=True)
        except Exception:  # noqa: BLE001 - report it and go on with the next variant
            print(f"{line}: FAILED", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
