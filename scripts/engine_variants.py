"""TensorRT builder settings (tiling optimization level, auxiliary streams), tried on the encoder and on
decoder calls of the real network: each variant is built and timed with CUDA graphs.

python scripts/engine_variants.py --model .cache/models/owsm_v4_medium_1B --variants base,tile_fast,aux0
"""

import argparse
import tempfile
import time
from pathlib import Path

import tensorrt as trt
import torch

from owsm_trt import hub
from owsm_trt.build import BuildOptions, decoder_profile_shapes, export_onnx
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
DTYPES = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
}


def build(onnx_path: Path, engine_path: Path, profiles: list[dict], settings: dict, level: int) -> float:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    assert parser.parse_from_file(str(onnx_path)), parser.get_error(0)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.builder_optimization_level = level
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
    engine_path.write_bytes(builder.build_serialized_network(network, config))
    return time.time() - t0


def time_engine(engine, profile: int, inputs: dict, iters: int = 50) -> float:
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()
    ctx.set_optimization_profile_async(profile, stream.cuda_stream)
    stream.synchronize()
    keep = []
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
            keep.append(t)
            ctx.set_tensor_address(name, t.data_ptr())
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


def decoder_inputs(cfg, b: int, s: int, cache: int = 64) -> dict:
    H, dk, T = cfg.dec_heads, cfg.d_model // cfg.dec_heads, cfg.enc_frames
    f16 = {"device": "cuda", "dtype": torch.float16}
    inp = {
        "tokens": torch.randint(0, cfg.vocab_size, (b, s), device="cuda"),
        "offset": torch.full((b,), 20, dtype=torch.long, device="cuda"),
    }
    for i in range(cfg.dec_layers):
        inp[f"self_k_{i}"] = torch.randn(b, H, cache, dk, **f16)
        inp[f"self_v_{i}"] = torch.randn(b, H, cache, dk, **f16)
        inp[f"cross_k_{i}"] = torch.randn(b, H, T, dk, **f16)
        inp[f"cross_v_{i}"] = torch.randn(b, H, T, dk, **f16)
    return inp


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--parts", default="decoder,encoder")
    args = ap.parse_args()
    files = hub.resolve(args.model)
    cfg = parse_config(load_espnet_config(files.config))
    model = load_model(cfg, str(files.checkpoint), "cuda", torch.float16)
    opts = BuildOptions()
    tmp = Path(tempfile.mkdtemp())
    enc_onnx, dec_onnx = export_onnx(model, tmp / "onnx", opts)
    if "split" in args.variants.split(","):
        _, split_onnx = export_onnx(model, tmp / "onnx_split", opts, split_self_attention=True)
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
    for name in args.variants.split(","):
        settings = VARIANTS[name]
        line = f"== {name:14s}"
        if "decoder" in args.parts:
            path = tmp / f"dec_{name}.engine"
            secs = build(split_onnx if name == "split" else dec_onnx, path, dec_shapes, settings, 4)
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


if __name__ == "__main__":
    main()
