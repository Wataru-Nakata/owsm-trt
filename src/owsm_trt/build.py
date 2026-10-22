"""Export OWSM to ONNX and build TensorRT engines (done once per GPU / TensorRT version)."""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .config import OWSMConfig
from .modeling import OWSMModel
from .runtime import cache_buckets

log = logging.getLogger(__name__)

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

# Bump whenever the exported graphs change, so cached engines from older exports are not reused.
EXPORT_VERSION = 4


@dataclass(frozen=True)
class BuildOptions:
    dtype: str = "float16"
    max_batch: int = 16
    capacity: int = 512  # decoder KV-cache slots = max prompt + generated tokens
    max_prefill: int = 256  # longest prompt (incl. previous-text condition) processed in one call
    workspace_gb: float = 8.0
    opt_level: int = 3  # TensorRT builder optimization level (0-5); higher = longer build, maybe faster


class _EncoderGraph(nn.Module):
    def __init__(self, model: OWSMModel):
        super().__init__()
        self.encoder, self.decoder = model.encoder, model.decoder

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.decoder.cross_kv(self.encoder(feats))  # [L, 2, B, H, T, d_k]
        return kv[:, 0], kv[:, 1]


class _DecoderGraph(nn.Module):
    def __init__(self, model: OWSMModel):
        super().__init__()
        self.decoder = model.decoder
        self.n = model.cfg.dec_layers

    def forward(self, tokens: torch.Tensor, offset: torch.Tensor, *caches: torch.Tensor):
        n = self.n
        sk, sv, ck, cv = (list(caches[i * n : (i + 1) * n]) for i in range(4))
        return self.decoder.step(tokens, offset, sk, sv, ck, cv)


def decoder_input_names(n_layers: int) -> list[str]:
    names = ["tokens", "offset"]
    for kind in ("self_k", "self_v", "cross_k", "cross_v"):
        names += [f"{kind}_{i}" for i in range(n_layers)]
    return names


def export_onnx(model: OWSMModel, out_dir: Path, opts: BuildOptions) -> tuple[Path, Path]:
    cfg = model.cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = next(model.decoder.parameters()).device
    dtype = DTYPES[opts.dtype]
    H, dk, T = cfg.dec_heads, cfg.d_model // cfg.dec_heads, cfg.enc_frames
    batch = torch.export.Dim("batch", min=1, max=opts.max_batch)
    seq = torch.export.Dim("seq", min=1, max=opts.max_prefill)
    cache = torch.export.Dim("cache", min=16, max=opts.capacity)  # KV-cache bucket length

    enc_path = out_dir / "encoder.onnx"
    feats = torch.randn(2, cfg.num_frames, cfg.n_mels, device=dev)
    torch.onnx.export(
        _EncoderGraph(model).eval(),
        (feats,),
        enc_path,
        input_names=["feats"],
        output_names=["cross_k", "cross_v"],
        dynamic_shapes={"feats": {0: batch}},
        dynamo=True,
        external_data=True,
    )

    dec_path = out_dir / "decoder.onnx"
    b, s = 2, 3  # sizes > 1 so the exporter does not specialise them
    tokens = torch.zeros(b, s, dtype=torch.long, device=dev)
    offset = torch.zeros(1, dtype=torch.long, device=dev)
    caches = [
        torch.zeros(b, H, opts.capacity, dk, device=dev, dtype=dtype) for _ in range(2 * cfg.dec_layers)
    ]
    caches += [torch.zeros(b, H, T, dk, device=dev, dtype=dtype) for _ in range(2 * cfg.dec_layers)]
    names = decoder_input_names(cfg.dec_layers)
    n_self = 2 * cfg.dec_layers
    cache_shapes = tuple({0: batch, 2: cache} if i < n_self else {0: batch} for i in range(len(caches)))
    shapes = {"tokens": {0: batch, 1: seq}, "offset": None, "caches": cache_shapes}
    split = model.decoder.split_attention
    model.decoder.split_attention = False  # TensorRT fuses concat + SDPA into a single MHA kernel
    try:
        torch.onnx.export(
            _DecoderGraph(model).eval(),
            (tokens, offset, *caches),
            dec_path,
            input_names=names,
            output_names=["logits", "new_kv"],
            dynamic_shapes=shapes,
            dynamo=True,
            external_data=True,
        )
    finally:
        model.decoder.split_attention = split
    return enc_path, dec_path


def _build(onnx_path: Path, engine_path: Path, profiles: list[dict], opts: BuildOptions) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(opts.workspace_gb * (1 << 30)))
    config.builder_optimization_level = opts.opt_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # layer names for profiling; no runtime cost
    dynamic = {
        network.get_input(i).name
        for i in range(network.num_inputs)
        if -1 in tuple(network.get_input(i).shape)
    }
    for prof in profiles:
        p = builder.create_optimization_profile()
        for name in dynamic:
            p.set_shape(name, *prof[name])
        config.add_optimization_profile(p)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {engine_path.name}")
    engine_path.write_bytes(serialized)
    log.info("built %s in %.0fs", engine_path.name, time.time() - t0)


def build_engines(model: OWSMModel, engine_dir: Path, opts: BuildOptions, keep_onnx: bool = False) -> None:
    """Export ``model`` (already on GPU in the target dtype) and build both engines into ``engine_dir``."""
    cfg = model.cfg
    engine_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = engine_dir / "onnx"
    log.info("exporting ONNX to %s", onnx_dir)
    enc_onnx, dec_onnx = export_onnx(model, onnx_dir, opts)

    B = opts.max_batch
    enc_profile = {"feats": tuple((n, cfg.num_frames, cfg.n_mels) for n in (1, B, B))}
    log.info("building encoder engine (this can take a few minutes)")
    _build(enc_onnx, engine_dir / "encoder.engine", [enc_profile], opts)

    log.info("building decoder engine (this can take a few minutes)")
    profiles = decoder_profiles(opts.max_batch)
    _build(
        dec_onnx,
        engine_dir / "decoder.engine",
        [decoder_profile_shapes(cfg, opts, p) for p in profiles],
        opts,
    )

    meta = {"options": asdict(opts), "config": asdict(cfg), "decoder_profiles": profiles}
    (engine_dir / "build.json").write_text(json.dumps(meta, indent=1))
    if not keep_onnx:
        shutil.rmtree(onnx_dir, ignore_errors=True)


def decoder_profile_shapes(cfg: OWSMConfig, opts: BuildOptions, profile: dict) -> dict:
    """(min, opt, max) shape of every decoder input for one entry of :func:`decoder_profiles`."""
    H, dk, T = cfg.dec_heads, cfg.d_model // cfg.dec_heads, cfg.enc_frames
    b = profile["batch"]
    s = (1, 1, 1) if profile["kind"] == "decode" else (1, 4, opts.max_prefill)  # new tokens per call
    smallest = cache_buckets(opts.capacity)[0]
    cache_len = (smallest, smallest, opts.capacity)
    shapes = {"tokens": tuple(zip(b, s)), "offset": ((1,), (1,), (1,))}
    for i in range(cfg.dec_layers):
        for kind in ("self_k", "self_v"):
            shapes[f"{kind}_{i}"] = tuple((n, H, c, dk) for n, c in zip(b, cache_len))
        for kind in ("cross_k", "cross_v"):
            shapes[f"{kind}_{i}"] = tuple((n, H, T, dk) for n in b)
    return shapes


def decoder_profiles(max_batch: int) -> list[dict]:
    """Decoder optimization profiles as (min, opt, max) batch sizes; the index is the TensorRT profile."""
    # Decode steps get a batch-1 profile of their own: kernels tuned for M=16 waste most of the GPU at M=1.
    profiles = [{"kind": "decode", "batch": (1, 1, 1)}]
    if max_batch > 1:
        profiles.append({"kind": "decode", "batch": (2, max(2, max_batch // 2), max_batch)})
    profiles.append({"kind": "prefill", "batch": (1, max_batch, max_batch)})
    return profiles


def engine_tag(opts: BuildOptions) -> str:
    """Directory name identifying an engine build: engines are GPU- and TensorRT-version-specific."""
    import tensorrt as trt

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu = props.name.replace(" ", "_").replace("/", "_")
    digest = hashlib.sha1(json.dumps(asdict(opts), sort_keys=True).encode()).hexdigest()[:8]
    return f"v{EXPORT_VERSION}-{gpu}-sm{props.major}{props.minor}-trt{trt.__version__}-{opts.dtype}-{digest}"


def check_config(cfg: OWSMConfig, engine_dir: Path) -> bool:
    meta = engine_dir / "build.json"
    return meta.exists() and json.loads(meta.read_text())["config"] == asdict(cfg)
