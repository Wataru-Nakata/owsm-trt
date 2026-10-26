"""Export OWSM to ONNX and build TensorRT engines (done once per GPU / TensorRT version)."""

import copy
import hashlib
import json
import logging
import shutil
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from . import attn_plugin, fp8, plugins
from .config import OWSMConfig
from .modeling import (
    CTC,
    Conv2dSubsampling,
    DepthwiseConv1d,
    FeedForward,
    FusableSiLU,
    MultiHeadedAttention,
    OWSMCTCModel,
    OWSMModel,
    padding_frame,
)
from .runtime import cache_buckets, trt_logger

log = logging.getLogger(__name__)

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

# Bump whenever the exported graphs change, so cached engines from older exports are not reused.
EXPORT_VERSION = 621
MIN_FEATURE_FRAMES = 64  # shortest feature input of a trimming encoder engine


@dataclass(frozen=True)
class BuildOptions:
    dtype: str = "float16"
    max_batch: int = 16
    capacity: int = 512  # decoder KV-cache slots = max prompt + generated tokens
    max_prefill: int = 256  # longest prompt (incl. previous-text condition) processed in one call
    max_verify: int = 32  # longest window of drafted tokens checked in one batch-1 decoder call
    workspace_gb: float = 8.0
    opt_level: int = 3  # TensorRT builder optimization level (0-5); higher = longer build, maybe faster
    # Multi-token decoder calls get kernels about twice as fast from level 4 on (1.3 vs 2.3 ms on the 1B
    # model), so they get an engine of their own (verify.engine) built at this level. (Level 5: same
    # kernels, 3x the build time; the encoder and single-token steps gain nothing above level 3.)
    verify_opt_level: int = 4
    fp8_encoder: bool = False  # experimental: FP8 encoder GEMMs/convolutions (needs calibration audio)
    fp8_calibration: str = ""  # digest of that audio: engines calibrated on other audio are other engines
    # Decoder attention as TensorRT plugins (attn_plugin): Triton kernels that read the KV cache and the
    # cross-attention keys/values where they are, instead of TensorRT's copies into its own layouts
    # every layer. In verify.engine (multi-token calls) and in decoder.engine (single-token steps and
    # prefill). Triton is needed to build them, not to run them; without it they are left out.
    attn_plugin: bool = True
    attn_plugin_steps: bool = True
    # E-Branchformer depthwise convolutions as a TensorRT plugin in [B, T, C] (plugins.py; fp16 only,
    # needs NVRTC at build time, else the usual grouped convolution between transposes)
    dwconv_plugin: bool = True
    # The subsampling's first convolution (one input channel) + ReLU as a TensorRT plugin (plugins.py; fp16,
    # NVRTC), writing the layout of the convolution after it (1.2 vs 2.5 ms for 16 x 30 s on an H200); and
    # its last ReLU with the flattening for the linear layer, reading that layout.
    conv1_plugin: bool = True
    # The CTC head's argmax (and OWSM-CTC's intermediate CTC softmax) over the vocabulary as TensorRT
    # plugins (plugins.py; fp16 only, needs NVRTC at build time). TensorRT's own kernels take 4x longer.
    ctc_plugins: bool = True


class _EncoderGraph(nn.Module):
    def __init__(self, model: OWSMModel):
        super().__init__()
        self.encoder, self.decoder, self.ctc = model.encoder, model.decoder, model.ctc
        self.strides = model.cfg.strides

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder(feats)
        kv = self.decoder.cross_kv(enc)  # [L, 2, B, H, T, d_k]
        # The CTC head only drafts tokens, so it reads just the subsampled frames computed from `feats`
        # (when the features stop after the audio, the frames filled in after them see padding only) and
        # the window's last frame, where OWSM's CTC puts the final punctuation. A blank stands for the
        # frames in between, which are blank (on the 73 LibriSpeech utterances of the tests, all of them).
        # ctc_ids [B, t + 2]: best path on frames 0 .. t-1, blank, frame T'-1 (blank if t = T').
        n, t = enc.shape[1], feats.shape[1]
        for s in self.strides:  # as OWSMConfig.enc_frames
            t = (t - 3) // s + 1
        pos = torch.arange(t + 2, device=enc.device)
        # (Not a slice enc[:, :t]: the exporter would specialise that view on whether it is contiguous.)
        frames = enc.index_select(1, torch.where(pos < t, pos, n - 1))
        return kv[:, 0], kv[:, 1], self.ctc(frames).masked_fill((pos == t) | (pos > n), 0)  # blank: 0


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


@contextmanager
def _encoder_export(model: OWSMModel | OWSMCTCModel, opts: BuildOptions, batch: torch.export.Dim):
    """Export settings of an encoder (of either model kind); yields the dynamic dimensions of ``feats``
    and the exporter's custom translation table."""
    cfg = model.cfg
    feats_dims = {0: batch}
    embed = model.encoder.embed
    convs = [m for m in model.encoder.modules() if isinstance(m, DepthwiseConv1d)]
    ffns = [
        m for m in model.encoder.modules() if isinstance(m, FeedForward) and isinstance(m.activation, nn.SiLU)
    ]
    try:
        if trims_features(cfg):  # feats may stop after the audio: the encoder fills in frames of padding
            feats_dims[1] = torch.export.Dim("time", min=MIN_FEATURE_FRAMES, max=cfg.num_frames)
            model.encoder.fill = padding_frame(model)
        MultiHeadedAttention.explicit = True  # 3-4% faster encoder at batch 8-16 than the exporter's SDPA
        for m in ffns:
            m.activation = FusableSiLU()
        DepthwiseConv1d.plugin = opts.dwconv_plugin and opts.dtype == "float16" and plugins.available()
        if DepthwiseConv1d.plugin:
            plugins.register()
            for m in convs:  # the plugins' weights as constants of the graph
                m.freeze_plugin_weights()
        Conv2dSubsampling.plugin = opts.conv1_plugin and opts.dtype == "float16" and plugins.available()
        if Conv2dSubsampling.plugin:
            plugins.register_conv1()
            plugins.register_flatten()
            embed.freeze_plugin_weights()
        translations = {
            **(fp8.ONNX_TRANSLATIONS if opts.fp8_encoder else {}),
            **(plugins.ONNX_TRANSLATIONS if DepthwiseConv1d.plugin else {}),
            **(plugins.CONV1_ONNX_TRANSLATIONS if Conv2dSubsampling.plugin else {}),
            **(plugins.FLATTEN_ONNX_TRANSLATIONS if Conv2dSubsampling.plugin else {}),
        }
        yield feats_dims, translations or None
    finally:
        model.encoder.fill = None
        MultiHeadedAttention.explicit = False
        Conv2dSubsampling.plugin = False
        embed.freeze_plugin_weights(False)
        DepthwiseConv1d.plugin = False
        for m in convs:
            m.freeze_plugin_weights(False)
        for m in ffns:
            m.activation = nn.SiLU()


def export_onnx(
    model: OWSMModel, out_dir: Path, opts: BuildOptions, split_self_attention: bool = False
) -> tuple[Path, Path]:
    """Encoder and decoder ONNX graphs. ``split_self_attention``: the decoder attends to the cache and
    to the new tokens separately (see modeling.cached_attention) instead of to their concatenation."""
    enc_path = export_encoder_onnx(model, out_dir, opts)
    dec_path = export_decoder_onnx(model, out_dir / "decoder.onnx", opts, split_self_attention)
    return enc_path, dec_path


def export_encoder_onnx(model: OWSMModel, out_dir: Path, opts: BuildOptions) -> Path:
    """Encoder/cross-KV/CTC graph, without exporting decoder engines that may be reused."""
    cfg = model.cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = next(model.decoder.parameters()).device
    batch = torch.export.Dim("batch", min=1, max=opts.max_batch)

    enc_path = out_dir / "encoder.onnx"
    feats = torch.randn(2, example_frames(cfg), cfg.n_mels, device=dev)
    # (the CTC head drafts tokens: its argmax as in the OWSM-CTC export, on a padded vocabulary)
    with (
        _encoder_export(model, opts, batch) as (feats_dims, translations),
        _ctc_export(model, opts) as extra,
        _quiet_exporter(),
    ):
        torch.onnx.export(
            _EncoderGraph(model).eval(),
            (feats,),
            enc_path,
            input_names=["feats"],
            output_names=["cross_k", "cross_v", "ctc_ids"],
            dynamic_shapes={"feats": feats_dims},
            dynamo=True,
            verbose=False,
            external_data=True,
            custom_translation_table={**(translations or {}), **extra} or None,
        )

    return enc_path


def export_decoder_onnx(
    model: OWSMModel,
    path: Path,
    opts: BuildOptions,
    split_self_attention: bool = False,
    plugins: frozenset[str] = frozenset(),
) -> Path:
    """The decoder call (``Decoder.step``) as ONNX. ``plugins``: attention as attn_plugin nodes (see
    ``Decoder.plugins``)."""
    cfg = model.cfg
    path.parent.mkdir(parents=True, exist_ok=True)
    dev = next(model.decoder.parameters()).device
    dtype = DTYPES[opts.dtype]
    H, dk, T = cfg.dec_heads, cfg.d_model // cfg.dec_heads, cfg.enc_frames
    batch = torch.export.Dim("batch", min=1, max=opts.max_batch)
    seq = torch.export.Dim("seq", min=1, max=max(opts.max_prefill, opts.max_verify))
    cache = torch.export.Dim("cache", min=16, max=opts.capacity)  # KV-cache bucket length
    b, s = 2, 3  # sizes > 1 so the exporter does not specialise them
    tokens = torch.zeros(b, s, dtype=torch.long, device=dev)
    offset = torch.zeros(b, dtype=torch.long, device=dev)
    caches = [
        torch.zeros(b, H, opts.capacity, dk, device=dev, dtype=dtype) for _ in range(2 * cfg.dec_layers)
    ]
    caches += [torch.zeros(b, H, T, dk, device=dev, dtype=dtype) for _ in range(2 * cfg.dec_layers)]
    names = decoder_input_names(cfg.dec_layers)
    n_self = 2 * cfg.dec_layers
    cache_shapes = tuple({0: batch, 2: cache} if i < n_self else {0: batch} for i in range(len(caches)))
    shapes = {"tokens": {0: batch, 1: seq}, "offset": {0: batch}, "caches": cache_shapes}
    split, plug = model.decoder.split_attention, model.decoder.plugins
    model.decoder.split_attention = split_self_attention  # concatenated: TensorRT fuses it for one token
    model.decoder.plugins = frozenset(plugins)
    MultiHeadedAttention.explicit = True  # cross-attention
    try:
        with _quiet_exporter():
            torch.onnx.export(
                _DecoderGraph(model).eval(),
                (tokens, offset, *caches),
                path,
                input_names=names,
                output_names=["logits", "new_kv"],
                dynamic_shapes=shapes,
                dynamo=True,
                verbose=False,
                external_data=True,
                custom_translation_table=attn_plugin.onnx_translations() if plugins else None,
            )
    finally:
        model.decoder.split_attention, model.decoder.plugins = split, plug
        MultiHeadedAttention.explicit = False
    return path


def _build(
    onnx_path: Path,
    engine_path: Path,
    profiles: list[dict],
    opts: BuildOptions,
    opt_level: int | None = None,
) -> None:
    import tensorrt as trt

    logger = trt_logger(trt)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    if hasattr(trt.NetworkDefinitionCreationFlag, "PREFER_AOT_PYTHON_PLUGINS"):  # plugins in the engine
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(opts.workspace_gb * (1 << 30)))
    config.builder_optimization_level = opts.opt_level if opt_level is None else opt_level
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
    log.warning("built %s in %.0fs", engine_path.name, time.time() - t0)


class _CTCGraph(nn.Module):
    def __init__(self, model: OWSMCTCModel):
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor, prompt: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        return self.model(feats, prompt, prefix)


def export_ctc_onnx(model: OWSMCTCModel, out_dir: Path, opts: BuildOptions) -> Path:
    cfg = model.cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = next(model.encoder.parameters()).device
    batch = torch.export.Dim("batch", min=1, max=opts.max_batch)
    plen = torch.export.Dim("prompt", min=1, max=opts.max_prefill)
    path = out_dir / "ctc.onnx"
    args = (
        torch.randn(2, example_frames(cfg), cfg.n_mels, device=dev),
        torch.zeros(2, 3, dtype=torch.long, device=dev),
        torch.zeros(2, 2, dtype=torch.long, device=dev),
    )
    with (
        _encoder_export(model, opts, batch) as (feats_dims, translations),
        _ctc_export(model, opts) as extra,
        _quiet_exporter(),
    ):
        torch.onnx.export(
            _CTCGraph(model).eval(),
            args,
            path,
            input_names=["feats", "prompt", "prefix"],
            output_names=["ctc_ids"],
            dynamic_shapes={"feats": feats_dims, "prompt": {0: batch, 1: plen}, "prefix": {0: batch}},
            dynamo=True,
            verbose=False,
            external_data=True,
            custom_translation_table={**(translations or {}), **extra} or None,
        )
    return path


@contextmanager
def _quiet_exporter():
    """The exporter without its expected warnings (no torchvision, unannotated custom-op arguments, dynamic axis
    names it merges, ...) and progress output; its errors still raise."""
    logger = logging.getLogger("torch.onnx")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logger.setLevel(level)


@contextmanager
def _ctc_export(model: OWSMModel | OWSMCTCModel, opts: BuildOptions):
    """Export settings of the CTC head (OWSM-CTC's output, the encoder-decoder models' drafts): the
    vocabulary padded (:func:`_padded_vocab`), softmax and argmax as plugins (``opts.ctc_plugins``).
    Yields the custom translations for the exporter."""
    CTC.plugin = opts.ctc_plugins and opts.dtype == "float16" and plugins.available()
    if CTC.plugin:
        plugins.register_ctc()
    try:
        with _padded_vocab(model, plugins.VOCAB_MULTIPLE):
            yield plugins.CTC_ONNX_TRANSLATIONS if CTC.plugin else {}
    finally:
        CTC.plugin = False


@contextmanager
def _padded_vocab(model: OWSMModel | OWSMCTCModel, multiple: int):
    """The CTC head and the intermediate CTC conditioning with the vocabulary padded to a multiple of
    ``multiple``: tokens that never win (logit -3e4: probability exactly 0 after the softmax in fp32) and
    add nothing (conditioning weights 0). The same results; TensorRT copies the logits and posteriors
    into aligned rows (a multiple of 8) otherwise. (Works on FP8 layers too: copies with padded weights.)"""
    lo = model.ctc.ctc_lo
    cond = getattr(model.encoder, "conditioning_layer", None)
    v = lo.weight.shape[0]
    pad = -v % multiple
    if not pad:
        yield
        return
    model.ctc.ctc_lo = _with_weight(
        lo, lambda w: F.pad(w, (0, 0, 0, pad)), lambda b: F.pad(b, (0, pad), value=-3e4)
    )
    if cond is not None:
        model.encoder.conditioning_layer = _with_weight(cond, lambda w: F.pad(w, (0, pad)))
    try:
        yield
    finally:
        model.ctc.ctc_lo = lo
        if cond is not None:
            model.encoder.conditioning_layer = cond


def _with_weight(layer: nn.Module, weight, bias=None) -> nn.Module:
    """A copy of ``layer`` (nn.Linear or fp8.QLinear) with ``weight(layer.weight)`` and ``bias(layer.bias)``."""
    new = copy.copy(layer)  # shares the buffers (e.g. FP8 scales); the parameters are replaced below
    new._parameters = dict(layer._parameters)
    with torch.no_grad():
        new.weight = nn.Parameter(weight(layer.weight), requires_grad=False)
        if bias is not None:
            new.bias = nn.Parameter(bias(layer.bias), requires_grad=False)
    return new


def build_engines(
    model: OWSMModel | OWSMCTCModel,
    engine_dir: Path,
    opts: BuildOptions,
    keep_onnx: bool = False,
    decoders_from: Path | None = None,
    skip_decoder_export: bool = True,
) -> None:
    """Export ``model`` (already on GPU in the target dtype) and build its engines into ``engine_dir``.
    ``decoders_from``: an engine directory of the same model and options but for the encoder's (an FP8
    encoder's fp16 counterpart), whose decoder engines are copied instead of built.
    ``skip_decoder_export`` omits their unused ONNX export; set False to retain the joint export."""
    cfg = model.cfg
    engine_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = engine_dir / "onnx"
    log.info("exporting ONNX to %s", onnx_dir)
    B = opts.max_batch
    # Batch 1 gets an encoder profile of its own: kernels picked for 16 recordings are ~20% slower on one.
    batches = [{"batch": (1, 1, 1)}] + ([{"batch": (2, B, B)}] if B > 1 else [])
    frames = (MIN_FEATURE_FRAMES, cfg.num_frames // 3, cfg.num_frames) if trims_features(cfg) else None
    frames = frames or (cfg.num_frames,) * 3
    if cfg.ctc_only:
        path = export_ctc_onnx(model, onnx_dir, opts)
        shapes = [
            {
                "feats": tuple((n, t, cfg.n_mels) for n, t in zip(p["batch"], frames)),
                "prompt": tuple((n, t) for n, t in zip(p["batch"], (1, 1, opts.max_prefill))),
                "prefix": tuple((n, 2) for n in p["batch"]),
            }
            for p in batches
        ]
        log.info("building the CTC engine (this can take a few minutes)")
        _build(path, engine_dir / "ctc.engine", shapes, opts)
        _finish(engine_dir, onnx_dir, opts, cfg, {"ctc": batches}, keep_onnx)
        return
    if decoders_from is not None and skip_decoder_export:
        enc_onnx, dec_onnx = export_encoder_onnx(model, onnx_dir, opts), None
    else:
        enc_onnx, dec_onnx = export_onnx(model, onnx_dir, opts)

    shapes = [{"feats": tuple((n, t, cfg.n_mels) for n, t in zip(p["batch"], frames))} for p in batches]
    log.info("building encoder engine (this can take a few minutes)")
    _build(enc_onnx, engine_dir / "encoder.engine", shapes, opts)

    profiles = decoder_profiles(opts.max_batch)
    steps = [p for p in profiles if p["kind"] != "verify"]
    verify = [p for p in profiles if p["kind"] == "verify"]
    if decoders_from is not None:
        meta = json.loads((decoders_from / "build.json").read_text())
        for name in ("decoder", "verify"):
            shutil.copyfile(decoders_from / f"{name}.engine", engine_dir / f"{name}.engine")
        log.warning("decoder engines copied from %s", decoders_from)
        extra = {"attn_plugin": meta.get("attn_plugin", {}), "decoders_from": decoders_from.name}
        _finish(engine_dir, onnx_dir, opts, cfg, {"encoder": batches, "decoder": steps, "verify": verify},
                keep_onnx, extra)  # fmt: skip
        return

    log.info("building decoder engines (this can take a few minutes)")
    plugin_onnx = None
    if opts.attn_plugin or opts.attn_plugin_steps:
        plugin_onnx = _export_plugin_decoder(model, onnx_dir / "plugins" / "decoder.onnx", opts)
    plugins = {}
    for name, profs, level, use in (
        ("decoder", steps, opts.opt_level, opts.attn_plugin_steps),
        ("verify", verify, opts.verify_opt_level, opts.attn_plugin),
    ):
        shapes = [decoder_profile_shapes(cfg, opts, p) for p in profs]
        plugins[name] = bool(use and plugin_onnx)
        if plugins[name]:
            try:
                _build(plugin_onnx, engine_dir / f"{name}.engine", shapes, opts, level)
            except Exception as e:  # noqa: BLE001 - e.g. kernels that do not compile for this GPU
                log.warning("building %s.engine without attention plugins: %s", name, e)
                plugins[name] = False
        if not plugins[name]:
            _build(dec_onnx, engine_dir / f"{name}.engine", shapes, opts, level)

    _finish(
        engine_dir,
        onnx_dir,
        opts,
        cfg,
        {"encoder": batches, "decoder": steps, "verify": verify},
        keep_onnx,
        {"attn_plugin": plugins},
    )


def _export_plugin_decoder(model: OWSMModel, path: Path, opts: BuildOptions) -> Path | None:
    """The decoder with attention as attn_plugin nodes, its plugins registered; None if they cannot be
    built here."""
    cfg = model.cfg
    head_dim = cfg.d_model // cfg.dec_heads
    if not attn_plugin.available(head_dim):
        reason = attn_plugin.unavailable() or f"no kernels for head size {head_dim}"
        log.warning("decoder engines without attention plugins (slower decoding): %s", reason)
        return None
    try:
        attn_plugin.register_plugins(cfg.dec_heads, head_dim)
        return export_decoder_onnx(model, path, opts, plugins=frozenset({"self", "cross"}))
    except Exception as e:  # noqa: BLE001
        log.warning("decoder engines without attention plugins: %s", e)
        return None


def example_frames(cfg: OWSMConfig) -> int:
    """Feature frames of the encoder input the export traces with. A trimming encoder's time dimension
    is dynamic: less than its maximum, so that the exporter does not specialise it. Otherwise the input
    always has ``num_frames``, and the example sets that (static) length."""
    return cfg.num_frames - 8 if trims_features(cfg) else cfg.num_frames


def trims_features(cfg: OWSMConfig) -> bool:
    """Whether the encoder engine takes features that stop after the audio (:func:`padding_frame`):
    when padding has the same features in every recording (global MVN)."""
    return cfg.normalize == "global_mvn"


def _finish(
    engine_dir: Path,
    onnx_dir: Path,
    opts: BuildOptions,
    cfg: OWSMConfig,
    profiles: dict,
    keep_onnx: bool,
    extra: dict | None = None,
) -> None:
    meta = {"options": asdict(opts), "config": asdict(cfg), "profiles": profiles, **(extra or {})}
    meta["trimmed_features"] = trims_features(cfg)
    (engine_dir / "build.json").write_text(json.dumps(meta, indent=1))
    if not keep_onnx:
        shutil.rmtree(onnx_dir, ignore_errors=True)


def decoder_profile_shapes(cfg: OWSMConfig, opts: BuildOptions, profile: dict) -> dict:
    """(min, opt, max) shape of every decoder input for one entry of :func:`decoder_profiles`."""
    H, dk, T = cfg.dec_heads, cfg.d_model // cfg.dec_heads, cfg.enc_frames
    b = profile["batch"]
    s = {  # new tokens per call
        "decode": (1, 1, 1),
        "verify": (1, max(1, opts.max_verify // 2), opts.max_verify),
        "prefill": (1, 4, opts.max_prefill),
    }[profile["kind"]]
    smallest = cache_buckets(opts.capacity)[0]
    cache_len = (smallest, smallest, opts.capacity)
    shapes = {"tokens": tuple(zip(b, s)), "offset": tuple((n,) for n in b)}
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
    # Several drafted tokens per row, with batch 1 again on its own profile. (A static profile per
    # window size was tried: no faster.)
    profiles.append({"kind": "verify", "batch": (1, 1, 1)})
    if max_batch > 1:
        profiles.append({"kind": "verify", "batch": (2, max(2, max_batch // 2), max_batch)})
    profiles.append({"kind": "prefill", "batch": (1, max_batch, max_batch)})
    return profiles


def buildable(opts: BuildOptions, cfg: OWSMConfig) -> BuildOptions:
    """``opts`` without the plugins this environment cannot build (each logs why). Engines built without them
    get a directory of their own, so they are rebuilt with the plugins once those can be built."""
    off = {}
    if (opts.dwconv_plugin or opts.conv1_plugin or opts.ctc_plugins) and not plugins.available():
        off.update(dwconv_plugin=False, conv1_plugin=False, ctc_plugins=False)
    if not cfg.ctc_only and (opts.attn_plugin or opts.attn_plugin_steps) and attn_plugin.unavailable():
        off.update(attn_plugin=False, attn_plugin_steps=False)
        log.warning(
            "decoder engines without attention plugins (slower decoding): %s", attn_plugin.unavailable()
        )
    return replace(opts, **off) if off else opts


def engine_tag(opts: BuildOptions) -> str:
    """Directory name identifying an engine build: engines are GPU- and TensorRT-version-specific."""
    import tensorrt as trt

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu = props.name.replace(" ", "_").replace("/", "_")
    digest = hashlib.sha1(json.dumps(asdict(opts), sort_keys=True).encode()).hexdigest()[:8]
    return f"v{EXPORT_VERSION}-{gpu}-sm{props.major}{props.minor}-trt{trt.__version__}-{opts.dtype}-{digest}"


def check_config(cfg: OWSMConfig, engine_dir: Path) -> bool:
    meta = engine_dir / "build.json"
    expected = json.loads(json.dumps(asdict(cfg)))  # tuples come back from JSON as lists
    return meta.exists() and json.loads(meta.read_text())["config"] == expected
