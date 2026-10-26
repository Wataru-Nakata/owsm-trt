"""Execution backends and the CUDA-graph decoding loops (greedy and beam search).

Both backends expose the same two operations on caller-owned buffers:

* ``encode(feats, cross_k, cross_v, ctc_ids)`` -- encoder, per-layer cross-attention K/V projection and
  the CTC head's best-path ids; returns the ids as a view [B, t] of ``ctc_ids`` [B, T' + 2] (the
  TensorRT encoder reads only some of the frames, see build._EncoderGraph)
* ``step_fn(...)`` -- returns a closure that runs one decoder call writing logits and new K/V into
  fixed buffers, so that the whole decode step can be captured in a CUDA graph.
"""

import importlib.metadata
import json
import logging
import os
import re
from array import array
from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from . import beam_kernels, kernels
from .config import OWSMConfig
from .modeling import OWSMCTCModel, OWSMModel

log = logging.getLogger(__name__)


def _check_cuda_versions() -> None:
    """Warn when torch and TensorRT ship different CUDA major versions (two runtimes in one process), or when
    TensorRT is installed for two (tensorrt-cu12 and tensorrt-cu13 write the same files)."""
    majors = set()
    for dist in importlib.metadata.distributions():
        m = re.fullmatch(r"tensorrt[-_]cu(\d+)[-_]bindings", (dist.metadata["Name"] or "").lower())
        if m:
            majors.add(int(m.group(1)))
    if len(majors) > 1:
        log.warning("TensorRT is installed for CUDA %s: their packages overwrite each other's files; keep one",
                    " and ".join(map(str, sorted(majors))))  # fmt: skip
    trt_major = max(majors) if len(majors) == 1 else None
    torch_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else None
    if trt_major and torch_major and trt_major != torch_major:
        log.warning(
            "torch is built for CUDA %d but TensorRT for CUDA %d: two CUDA runtimes in one process, which needs a"
            " driver for CUDA %d (tested: torch 2.10 with CUDA 12 + TensorRT 11.4 with CUDA 13)",
            torch_major, trt_major, max(torch_major, trt_major),
        )  # fmt: skip


class TorchBackend:
    name = "torch"
    trimmed = False  # the encoder wants features of the whole (padded) input

    def __init__(self, model: OWSMModel):
        self.model = model
        self.cfg = model.cfg

    def encode(
        self, feats: torch.Tensor, cross_k: torch.Tensor, cross_v: torch.Tensor, ctc_ids: torch.Tensor
    ) -> torch.Tensor:
        # feats: [B, T, n_mels] fp32; cross_k/v: [L, B, H, T', d_k]; ctc_ids: [B, T'] int32
        enc = self.model.encoder(feats)
        kv = self.model.decoder.cross_kv(enc)  # [L, 2, B, H, T', d_k]
        cross_k.copy_(kv[:, 0])
        cross_v.copy_(kv[:, 1])
        ids = self.model.ctc(enc)
        out = ctc_ids.view(-1)[: ids.numel()].view(ids.shape)
        out.copy_(ids)
        return out

    def encode_fn(
        self,
        feats: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        ctc_ids: torch.Tensor,
        lane: int = 0,
    ) -> Callable[[], None]:
        """``encode`` on fixed buffers; ``run.ctc_ids`` is what it returns."""

        def run() -> None:
            self.encode(feats, cross_k, cross_v, ctc_ids)

        b, t = feats.shape[0], self.cfg.enc_frames
        run.ctc_ids = ctc_ids.view(-1)[: b * t].view(b, t)
        return run

    def step_fn(
        self,
        tokens: torch.Tensor,
        offset: torch.Tensor,
        self_k: torch.Tensor,
        self_v: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        logits: torch.Tensor,
        new_kv: torch.Tensor,
        kind: str = "decode",
    ) -> Callable[[], None]:
        dec = self.model.decoder
        layers = range(self.cfg.dec_layers)
        sk, sv = [self_k[i] for i in layers], [self_v[i] for i in layers]
        ck, cv = [cross_k[i] for i in layers], [cross_v[i] for i in layers]

        def run() -> None:
            lg, kv = dec.step(tokens, offset, sk, sv, ck, cv)
            logits.copy_(lg)
            new_kv.copy_(kv)

        return run


# TensorRT warnings that are expected here: multi-profile engines' tensor queries, the int64 token inputs, the
# Triton plugin kernels' extra arguments
_EXPECTED = ("The engine contains", "has Int64 binding", "hidden args")


def trt_logger(trt):
    """A TensorRT logger for warnings and errors, without the expected ones (to owsm_trt's logging)."""

    class Logger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg):
            if severity == trt.ILogger.Severity.WARNING:
                if not any(e in msg for e in _EXPECTED):
                    log.warning("TensorRT: %s", msg)
            elif severity in (trt.ILogger.Severity.ERROR, trt.ILogger.Severity.INTERNAL_ERROR):
                log.error("TensorRT: %s", msg)

    return Logger()


class _TRTEngines:
    """Loads serialized engines and runs them on torch-owned device buffers."""

    name = "tensorrt"

    def __init__(self, cfg: OWSMConfig):
        import tensorrt as trt

        _check_cuda_versions()
        self.trt = trt
        self.cfg = cfg
        self.logger = trt_logger(trt)
        self.runtime = trt.Runtime(self.logger)
        self._scratch: dict[tuple, torch.Tensor] = {}

    def _load(self, path: Path):
        engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize {path}; rebuild it with `owsm-trt build`")
        return engine

    def _context(self, engine, profile: int, lane: int = 0):
        # Every context of an engine shares one scratch buffer: they all run on the same stream, one at a
        # time, and a session creates one context per (cache bucket, window size). Contexts of another
        # ``lane`` run at the same time as those of lane 0 (on a stream of their own): a scratch buffer
        # of their own, per optimization profile (the size it needs).
        ctx = engine.create_execution_context(self.trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        key = (id(engine),) if lane == 0 else (id(engine), lane, profile)
        if key not in self._scratch:
            size = engine.device_memory_size_v2
            if lane and hasattr(engine, "get_device_memory_size_for_profile_v2"):
                size = engine.get_device_memory_size_for_profile_v2(profile)
            self._scratch[key] = torch.empty(max(size, 1), dtype=torch.uint8, device="cuda")
        scratch = self._scratch[key]
        ctx.set_device_memory(scratch.data_ptr(), scratch.numel())
        stream = torch.cuda.current_stream()
        ctx.set_optimization_profile_async(profile, stream.cuda_stream)
        stream.synchronize()
        return ctx

    @staticmethod
    def _bind(ctx, name: str, t: torch.Tensor, is_input: bool) -> None:
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if is_input:
            ctx.set_input_shape(name, tuple(t.shape))
        ctx.set_tensor_address(name, t.data_ptr())

    @staticmethod
    def _execute(ctx, what: str) -> None:
        if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(f"{what} engine execution failed")


class TRTBackend(_TRTEngines):
    """The engines of an encoder-decoder model: encoder, decoder (single-token steps and prefill) and
    verify (multi-token decoder calls, the same network built at a higher optimization level)."""

    def __init__(self, cfg: OWSMConfig, engine_dir: Path):
        super().__init__(cfg)
        self.encoder = self._load(engine_dir / "encoder.engine")
        self.decoder = self._load(engine_dir / "decoder.engine")
        self.verifier = self._load(engine_dir / "verify.engine")
        meta = json.loads((engine_dir / "build.json").read_text())
        self.profiles = meta["profiles"]
        self._enc_ctxs: dict[int, object] = {}  # per encoder profile, for encode()
        # the encoder takes features that stop after the audio (see modeling.padding_frame)
        self.trimmed = meta.get("trimmed_features", False)
        self.prefill_ctx = self._context(*self._profile("prefill", 1))

    def _profile(self, kind: str, batch: int) -> tuple:
        """(engine, profile index) for ``kind`` calls of ``batch`` rows."""
        engine, name = (self.verifier, "verify") if kind == "verify" else (self.decoder, "decoder")
        for i, p in enumerate(self.profiles[name]):
            if p["kind"] == kind and p["batch"][0] <= batch <= p["batch"][2]:
                return engine, i
        raise ValueError(f"no {kind} profile in the {name} engine covers batch size {batch}")

    def enc_ctx(self, batch: int):
        """The execution context :meth:`encode` uses for ``batch`` recordings."""
        i = _batch_profile(self.profiles.get("encoder"), batch)
        if i not in self._enc_ctxs:
            self._enc_ctxs[i] = self._context(self.encoder, i)
        return self._enc_ctxs[i]

    def encode(
        self, feats: torch.Tensor, cross_k: torch.Tensor, cross_v: torch.Tensor, ctc_ids: torch.Tensor
    ) -> torch.Tensor:
        ctx = self.enc_ctx(feats.shape[0])
        ids = self._bind_encoder(ctx, feats.contiguous(), cross_k, cross_v, ctc_ids)
        self._execute(ctx, "encoder")
        return ids

    def encode_fn(
        self,
        feats: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        ctc_ids: torch.Tensor,
        lane: int = 0,
    ) -> Callable[[], None]:
        """``encode`` on fixed buffers with a context of its own, for CUDA-graph capture;
        ``run.ctc_ids`` is what it returns. ``lane``: see :meth:`_context`."""
        ctx = self._context(self.encoder, _batch_profile(self.profiles.get("encoder"), feats.shape[0]), lane)
        ids = self._bind_encoder(ctx, feats, cross_k, cross_v, ctc_ids)

        def run() -> None:
            self._execute(ctx, "encoder")

        run.ctx, run.ctc_ids = ctx, ids
        return run

    def _bind_encoder(self, ctx, feats, cross_k, cross_v, ctc_ids: torch.Tensor) -> torch.Tensor:
        """Bind the encoder's inputs and outputs; returns the view of ``ctc_ids`` [B, T'] it writes: the
        CTC ids of the frames computed from ``feats`` only [B, t], packed (t < T' on trimmed features)."""
        self._bind(ctx, "feats", feats, True)
        self._bind(ctx, "cross_k", cross_k, False)
        self._bind(ctx, "cross_v", cross_v, False)
        self._bind(ctx, "ctc_ids", ctc_ids, False)
        ctx.infer_shapes()
        b, t = ctx.get_tensor_shape("ctc_ids")
        if not 0 <= b * t <= ctc_ids.numel():
            raise ValueError(f"ctc_ids holds {ctc_ids.numel()} ids, the encoder writes {b} x {t}")
        return ctc_ids.view(-1)[: b * t].view(b, t)

    def step_fn(
        self,
        tokens: torch.Tensor,
        offset: torch.Tensor,
        self_k: torch.Tensor,
        self_v: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        logits: torch.Tensor,
        new_kv: torch.Tensor,
        kind: str = "decode",
    ) -> Callable[[], None]:
        # Prefill reuses one context. Decode / verify closures get their own, so that a captured CUDA
        # graph never sees its shapes change.
        if kind == "prefill":
            ctx = self.prefill_ctx
        else:
            ctx = self._context(*self._profile(kind, tokens.shape[0]))
        self._bind(ctx, "tokens", tokens, True)
        self._bind(ctx, "offset", offset, True)
        for i in range(self.cfg.dec_layers):
            self._bind(ctx, f"self_k_{i}", self_k[i], True)
            self._bind(ctx, f"self_v_{i}", self_v[i], True)
            self._bind(ctx, f"cross_k_{i}", cross_k[i], True)
            self._bind(ctx, f"cross_v_{i}", cross_v[i], True)
        self._bind(ctx, "logits", logits, False)
        self._bind(ctx, "new_kv", new_kv, False)

        def run() -> None:
            self._execute(ctx, "decoder")

        run.ctx = ctx  # keep the context alive as long as the closure
        return run


class TorchCTCBackend:
    name = "torch"

    def __init__(self, model: OWSMCTCModel):
        self.model = model
        self.cfg = model.cfg

    def run_fn(
        self,
        feats: torch.Tensor,
        prompt: torch.Tensor,
        prefix: torch.Tensor,
        ctc_ids: torch.Tensor,
        lane: int = 0,
    ) -> Callable[[], None]:
        # feats [B, T, n_mels] fp32, prompt [B, P], prefix [B, 2] -> ctc_ids [B, 2 + T'] int32
        return lambda: ctc_ids.copy_(self.model(feats, prompt, prefix))


class TRTCTCBackend(_TRTEngines):
    """The single engine of an OWSM-CTC model: prompt + prefix + features -> CTC best path."""

    def __init__(self, cfg: OWSMConfig, engine_dir: Path):
        super().__init__(cfg)
        self.engine = self._load(engine_dir / "ctc.engine")
        meta = json.loads((engine_dir / "build.json").read_text())
        self.profiles = meta["profiles"].get("ctc")
        self.trimmed = meta.get("trimmed_features", False)  # as TRTBackend.trimmed

    def run_fn(
        self,
        feats: torch.Tensor,
        prompt: torch.Tensor,
        prefix: torch.Tensor,
        ctc_ids: torch.Tensor,
        lane: int = 0,
    ) -> Callable[[], None]:
        ctx = self._context(self.engine, _batch_profile(self.profiles, feats.shape[0]), lane)
        self._bind(ctx, "feats", feats, True)
        self._bind(ctx, "prompt", prompt, True)
        self._bind(ctx, "prefix", prefix, True)
        self._bind(ctx, "ctc_ids", ctc_ids, False)

        def run() -> None:
            self._execute(ctx, "CTC")

        run.ctx = ctx
        return run


def _batch_profile(profiles: list[dict] | None, batch: int) -> int:
    """Index of the optimization profile for ``batch`` rows (engines built with one profile: 0)."""
    for i, p in enumerate(profiles or []):
        if p["batch"][0] <= batch <= p["batch"][2]:
            return i
    return 0


class CTCSession:
    """Static buffers and CUDA graphs for OWSM-CTC calls of one batch size and prompt length (a graph
    per feature length).

    ``shared``: (features, best paths) buffers of at least this session's size, to share with the
    sessions of other batch sizes (calls on one stream run one at a time, and a caller reads the best
    paths of a call before it makes the next one, or copies them in stream order). ``lane``: sessions of
    lanes > 0 run at the same time as those of lane 0, on other streams (TensorRT contexts with scratch
    memory of their own; buffers of their own)."""

    def __init__(
        self,
        backend,
        cfg: OWSMConfig,
        batch: int,
        prompt_len: int,
        device,
        use_cuda_graph: bool,
        shared: tuple[torch.Tensor, torch.Tensor] | None = None,
        lane: int = 0,
    ):
        self.backend, self.use_cuda_graph, self.lane = backend, use_cuda_graph, lane
        cols = cfg.n_prefix + cfg.enc_frames
        if shared is None:
            shared = ctc_buffers(cfg, batch, device)
        self.feats = shared[0]  # viewed per feature length
        self.ids = shared[1].view(-1)[: batch * cols].view(batch, cols)
        self.prompt = torch.zeros(batch, prompt_len, dtype=torch.long, device=device)
        self.prefix = torch.zeros(batch, 2, dtype=torch.long, device=device)
        self.conditioning = None  # what prompt and prefix hold (see __call__)
        self._runs: dict[tuple, tuple] = {}  # feats shape -> (input view, closure, graph)

    def __call__(
        self, feats: torch.Tensor, prompt: torch.Tensor | None = None, prefix: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Best paths [B, 2 + T'] (frame-wise argmax; the first two frames are the prefix). ``prompt`` and
        ``prefix`` [B, 2]: None keeps those of the previous call."""
        key = tuple(feats.shape)
        if key not in self._runs:
            view = self.feats[: feats.numel()].view(feats.shape)
            if self.lane:
                run = self.backend.run_fn(view, self.prompt, self.prefix, self.ids, lane=self.lane)
            else:
                run = self.backend.run_fn(view, self.prompt, self.prefix, self.ids)
            self._runs[key] = (view, run, capture_graph(run) if self.use_cuda_graph else None)
        view, run, graph = self._runs[key]
        view.copy_(feats)
        if prompt is not None:
            self.prompt.copy_(prompt)
        if prefix is not None:
            self.prefix.copy_(prefix)
        graph.replay() if graph is not None else run()
        return self.ids


def ctc_buffers(cfg: OWSMConfig, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Features and best-path buffers for CTC sessions of up to ``batch`` rows (CTCSession ``shared``)."""
    feats = torch.zeros(batch * cfg.num_frames * cfg.n_mels, device=device)
    ids = torch.zeros(batch, cfg.n_prefix + cfg.enc_frames, dtype=torch.int32, device=device)
    return feats, ids


_END = object()  # decode_stream: no batches left
# decode_stream with at most ENCODER_LANE_ROWS rows: its two staging slots are encoded on two streams, so
# that two encoder calls run at the same time (TensorRT contexts with scratch memory of their own; the same
# kernels on the same inputs: the same outputs). OWSM_TRT_ENCODER_LANES=0 turns it off.
ENCODER_LANES = os.environ.get("OWSM_TRT_ENCODER_LANES", "1") != "0"
ENCODER_LANE_ROWS = 1


def capture_graph(body: Callable[[], None], stream: torch.cuda.Stream | None = None) -> torch.cuda.CUDAGraph:
    """Warm ``body`` up on a side stream (TensorRT shape setup, allocator), then capture it (on
    ``stream``, by default PyTorch's capture stream)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        body()
    return graph


def ctc_columns(cfg: OWSMConfig) -> int:
    """Columns of a buffer for the encoder's CTC best paths: up to T' + 2 (see build._EncoderGraph)."""
    return cfg.enc_frames + 2


def cache_buckets(capacity: int, smallest: int = 64) -> list[int]:
    """KV-cache lengths a session switches between: 64, 128, ... up to ``capacity``."""
    out, c = [], smallest
    while c < capacity:
        out.append(c)
        c *= 2
    return out + [capacity]


class TimestampRules:
    """ESPnet's ``ScoreFilter``, which every encoder-decoder OWSM search applies: the tokens that may
    come next, given the timestamps emitted so far.

    With <notimestamps> in the primer no timestamp may appear. Otherwise the first token is a
    timestamp, timestamps come in increasing (start, end) pairs, <eos> only after a complete pair, and
    an end time is followed by a start time at or after it, or by <eos>.
    """

    def __init__(self, notime: int, first_time: int, last_time: int, eos: int, vocab: int, device):
        self.notime, self.first_time, self.last_time, self.eos = notime, first_time, last_time, eos
        self.ar = torch.arange(vocab, device=device)
        self.time = self.is_time(self.ar)

    def is_time(self, t: torch.Tensor) -> torch.Tensor:
        return (t >= self.first_time) & (t <= self.last_time)

    def forbidden(
        self,
        notime: torch.Tensor,
        first: torch.Tensor,
        count: torch.Tensor,
        last: torch.Tensor,
        prev: torch.Tensor,
    ) -> torch.Tensor:
        """[R, V] True where the next token is not allowed. Per row: whether the primer has
        <notimestamps>, whether nothing was generated yet, the number of timestamps so far, the last
        timestamp (-1 if none), and the previous token."""
        ar, time = self.ar, self.time
        last = last.unsqueeze(-1)
        open_pair = (ar == self.eos) | (time & (ar <= last))
        after_end = ((ar < last) | (ar > self.last_time)) & (ar != self.eos)
        closed = torch.where(self.is_time(prev).unsqueeze(-1), after_end, torch.ones_like(after_end))
        timed = torch.where((count % 2 == 1).unsqueeze(-1), open_pair, closed)
        timed = torch.where(first.unsqueeze(-1), ~time, timed)
        return torch.where(notime.unsqueeze(-1), time, timed)


class _Session:
    """Static buffers + per-bucket CUDA graphs for decoding ``rows`` sequences in lock step.

    Self-attention reads the whole KV cache every step, so the cache comes in length buckets
    (64, 128, ..., capacity), each with its own buffers and CUDA graph. A sequence starts in the
    smallest bucket that fits and is copied into the next one when it outgrows it.

    Buffers (H = decoder heads, c = bucket length, T = encoder frames):
      caches[c]       2 x [L, rows, H, c, d_k]  KV cache, slot t holds position t
      cross_k/cross_v [L, rows, H, T, d_k]
      logits          [rows, 1, V]              of the last decoder call
    A decoding step (``_body``) feeds ``tok`` at ``pos`` and then selects the next tokens from the
    logits (``_select``); the first selection, from the prefill logits, runs eagerly.
    """

    def __init__(
        self,
        backend,
        cfg: OWSMConfig,
        rows: int,
        capacity: int,
        eos: int,
        dtype: torch.dtype,
        device: torch.device,
        use_cuda_graph: bool,
        chunk: int,
        buckets: list[int] | None,
        rules: TimestampRules | None,
    ):
        self.backend, self.cfg, self.rows, self.capacity = backend, cfg, rows, capacity
        self.rules = rules
        dev = {"device": device}
        # TimestampRules state per row: <notimestamps> given, nothing generated yet, #timestamps, last one
        self.t_notime = torch.zeros(rows, dtype=torch.bool, **dev)
        self.t_first = torch.zeros(rows, dtype=torch.bool, **dev)
        self.t_count = torch.zeros(rows, dtype=torch.long, **dev)
        self.t_last = torch.zeros(rows, dtype=torch.long, **dev)
        self.chunk = chunk  # steps between host checks for the end; larger wastes steps after it
        self.buckets = buckets or cache_buckets(capacity)
        self.device = device
        L, H, dk = cfg.dec_layers, cfg.dec_heads, cfg.d_model // cfg.dec_heads
        self.kw = {"device": device, "dtype": dtype}
        # Zero-init: masked slots still flow through softmax-weighted sums, so they must be finite.
        self.caches = {
            c: (torch.zeros(L, rows, H, c, dk, **self.kw), torch.zeros(L, rows, H, c, dk, **self.kw))
            for c in self.buckets
        }
        self.cross_k = torch.zeros(L, rows, H, cfg.enc_frames, dk, **self.kw)
        self.cross_v = torch.zeros(L, rows, H, cfg.enc_frames, dk, **self.kw)
        self.logits = torch.zeros(rows, 1, cfg.vocab_size, device=device)
        self.new_kv = torch.zeros(L, 2, rows, H, 1, dk, **self.kw)
        self.tok = torch.zeros(rows, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)  # every row's position (lock step)
        self.offs = torch.zeros(rows, dtype=torch.long, device=device)  # the decoder's per-row offsets
        self.eos = eos
        self._steps = {
            c: backend.step_fn(
                self.tok, self.offs, k, v, self.cross_k, self.cross_v, self.logits, self.new_kv
            )
            for c, (k, v) in self.caches.items()
        }
        self.use_cuda_graph = use_cuda_graph
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._feats: dict[int, torch.Tensor] = {}  # per lane: flat, viewed as encoder inputs of any length
        # Encoder graphs are captured on a stream of their own. Graphs captured on one stream share
        # PyTorch's cuBLAS workspace, and decode_stream runs the encoder while decoder graphs run
        # (with the torch backend, sharing it corrupts both).
        self._enc_capture = torch.cuda.Stream(device) if use_cuda_graph and device.type == "cuda" else None
        self._enc_graphs: dict[tuple, tuple] = {}  # (outputs, input shape) -> (closure, input, graph)

    def _check(self) -> None:
        """Let the backend raise errors its kernels flagged (the decoder megakernel's barrier timeout)."""
        check = getattr(self.backend, "check", None)
        if check is not None:
            check()

    def _capture_all(self) -> None:
        if self.use_cuda_graph:
            for c in self.buckets:
                self.graphs[c] = self._capture(c)

    def _select(self, c: int) -> None:
        raise NotImplementedError

    def _rules_start(self, primer: torch.Tensor) -> None:
        if self.rules is not None:
            self.t_notime.copy_((primer == self.rules.notime).any(dim=1))
            self.t_first.fill_(True)
            self.t_count.zero_()
            self.t_last.fill_(-1)

    def _rules_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """``scores`` [rows, V] with the tokens the rules forbid after ``tok`` set to -inf."""
        if self.rules is None:
            return scores
        bad = self.rules.forbidden(self.t_notime, self.t_first, self.t_count, self.t_last, self.tok[:, 0])
        return scores.masked_fill(bad, float("-inf"))

    def _rules_advance(self, token: torch.Tensor, parent: torch.Tensor | None = None) -> None:
        if self.rules is None:
            return
        if parent is not None:
            for buf in (self.t_notime, self.t_count, self.t_last):
                buf.copy_(buf.index_select(0, parent))
        is_time = self.rules.is_time(token)
        self.t_count += is_time
        self.t_last.copy_(torch.where(is_time, token, self.t_last))
        self.t_first.fill_(False)

    def _body(self, c: int) -> None:
        self._feed(c)
        self._select(c)

    def _feed(self, c: int) -> None:
        """Run the decoder on ``tok`` at ``pos`` and append its keys/values to the cache."""
        k, v = self.caches[c]
        self.offs.copy_(self.pos.expand(self.rows))
        self._steps[c]()
        if not getattr(self._steps[c], "writes_cache", False):  # the decoder megakernel writes it itself
            k.index_copy_(3, self.pos, self.new_kv[:, 0])
            v.index_copy_(3, self.pos, self.new_kv[:, 1])
        self.pos += 1

    def _capture(self, c: int) -> torch.cuda.CUDAGraph:
        self.pos.zero_()
        return capture_graph(lambda: self._body(c))

    def _encoder(
        self, feats: torch.Tensor, cross_k, cross_v, ctc_ids, after=None, lane: int = 0
    ) -> torch.Tensor:
        """Run the encoder into the given buffers, replaying a CUDA graph captured on their first use.
        Returns the CTC best paths it wrote: a view of ``ctc_ids`` (see the backends' ``encode``).
        Calls of another ``lane`` may run at the same time as those of lane 0 (on another stream)."""
        if not self.use_cuda_graph:
            paths = self.backend.encode(feats, cross_k, cross_v, ctc_ids)
            if after is not None:
                after()
            return paths
        b, _, f = feats.shape
        if lane not in self._feats:
            self._feats[lane] = torch.zeros(b * self.cfg.num_frames * f, device=feats.device)
        key = (cross_k.data_ptr(), tuple(feats.shape), lane)
        if key not in self._enc_graphs:
            buf = self._feats[lane][: feats.numel()].view(feats.shape)
            if lane:
                run = self.backend.encode_fn(buf, cross_k, cross_v, ctc_ids, lane=lane)
            else:
                run = self.backend.encode_fn(buf, cross_k, cross_v, ctc_ids)

            def body() -> None:
                run()
                if after is not None:
                    after()

            graph = capture_graph(body, self._enc_capture)
            self._enc_graphs[key] = (run, buf, graph)  # `run` keeps the TensorRT context alive
        run, buf, graph = self._enc_graphs[key]
        buf.copy_(feats)
        graph.replay()
        return run.ctc_ids

    def _move(self, c: int, slots: int) -> int:
        """Move the cache from bucket ``c`` to the smallest one with ``slots`` slots (up or down)."""
        new = self._bucket(slots)
        if new != c:
            n = min(c, new)
            for old, buf in zip(self.caches[c], self.caches[new]):
                buf[:, :, :, :n] = old[:, :, :, :n]
        return new

    def _bucket(self, slots: int) -> int:
        """Smallest bucket with at least ``slots`` slots."""
        return next(c for c in self.buckets if c >= slots)

    def _grow(self, c: int, slots: int) -> int:
        """Move the cache from bucket ``c`` to the smallest one with ``slots`` slots."""
        bigger = self._bucket(slots)
        for old, new in zip(self.caches[c], self.caches[bigger]):
            new[:, :, :, :c] = old
        return bigger

    def _prefill(self, primer: torch.Tensor, cross_k: torch.Tensor, cross_v: torch.Tensor):
        """Run the primer [b, P] in one call; returns (last logits [b, V], new K/V, bucket)."""
        b, p = primer.shape
        c = self._bucket(p + 1)
        k, v = self.caches[c]
        self.pos.zero_()
        L, _, _, H, _, dk = self.new_kv.shape
        new_kv = torch.empty(L, 2, b, H, p, dk, **self.kw)
        logits = torch.empty(b, p, self.logits.shape[-1], device=self.device)
        offset = torch.zeros(b, dtype=torch.long, device=self.device)
        self.backend.step_fn(
            primer, offset, k[:, :b], v[:, :b], cross_k, cross_v, logits, new_kv, kind="prefill"
        )()
        return logits[:, -1], new_kv, c

    def _run(self, c: int, pos: int, steps: int, done: torch.Tensor) -> None:
        """Replay decoding steps from cache position ``pos`` until ``done`` or ``steps`` steps."""
        while steps > 0:
            if pos >= c:  # the next step writes slot `pos`: move to a bigger bucket
                c = self._grow(c, pos + 1)
            n = min(self.chunk, steps, c - pos)
            for _ in range(n):
                self.graphs[c].replay() if self.graphs else self._body(c)
            steps -= n
            pos += n
            if self._all_done(done):
                break

    def _all_done(self, done: torch.Tensor) -> bool:
        return bool(done.all())


class GreedySession(_Session):
    """Greedy decoding of ``batch`` sequences.

    Extra buffers: seq [B, C + 1] every token fed or generated so far; ctc_ids [B, T] CTC best path.

    With ``draft_windows``, :meth:`decode_drafted` checks several drafted tokens per row and decoder
    call, using one CUDA graph per (cache bucket, window size).
    """

    def __init__(
        self,
        backend,
        cfg: OWSMConfig,
        batch: int,
        capacity: int,
        eos: int,
        dtype: torch.dtype,
        device: torch.device,
        use_cuda_graph: bool = True,
        chunk: int = 4,
        buckets: list[int] | None = None,
        draft_windows: tuple[int, ...] = (),
        rules: TimestampRules | None = None,
        punctuation: frozenset[int] = frozenset(),
    ):
        super().__init__(
            backend, cfg, batch, capacity, eos, dtype, device, use_cuda_graph, chunk, buckets, rules
        )
        self.punctuation = punctuation  # see _resync
        self.batch = batch
        self.seq = torch.zeros(batch, capacity + 1, dtype=torch.long, device=device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.ctc_ids = torch.zeros(batch, ctc_columns(cfg), dtype=torch.int32, device=device)
        self._eos = torch.full((batch,), eos, dtype=torch.long, device=device)
        self._capture_all()

        self.windows = sorted(set(draft_windows) | {1}) if draft_windows else []
        self._stage: list[tuple[torch.Tensor, ...]] = []  # decode_stream's encoder outputs, two batches
        self._stage_part: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}  # ... of smaller batches
        self._stage_ids: list[torch.Tensor] = []  # ... their CTC best paths, on the host
        self._verify: dict[tuple[int, int], _Bound] = {}
        # single-row calls with nothing alongside (no encoder in decode_stream): a megakernel may take more SMs
        self._verify_solo: dict[tuple[int, int], _Bound] = {}
        solo = batch == 1 and getattr(backend, "solo_differs", False)
        for w in self.windows:
            bufs = _Verify(cfg, batch, w, self.kw, rules, self.t_notime, eos)
            for c, (k, v) in self.caches.items():
                kind = "verify" if w > 1 else "decode"
                # a decoder megakernel can do the argmax itself: plain, or with <notimestamps> (no timestamp
                # tokens); kernels.choose for rows under the other timestamp rules
                extra = {}
                if getattr(backend, "argmax", False):  # (a megakernel backend)
                    extra["inp"] = (bufs.inp, w)
                    extra["pred"] = bufs.pred
                    if rules is not None:
                        extra["notime"] = (self.t_notime, rules.first_time, rules.last_time)
                step = backend.step_fn(
                    bufs.tok,
                    bufs.offs,
                    k,
                    v,
                    self.cross_k,
                    self.cross_v,
                    bufs.logits,
                    bufs.new_kv,
                    kind=kind,
                    **extra,
                )
                self._verify[c, w] = bufs.bind(step, k, v)
                if solo and getattr(backend, "solo_differs", False):
                    step = backend.step_fn(
                        bufs.tok, bufs.offs, k, v, self.cross_k, self.cross_v, bufs.logits, bufs.new_kv, kind=kind,
                        solo=True, **extra,
                    )  # fmt: skip
                    self._verify_solo[c, w] = bufs.bind(step, k, v)
        if use_cuda_graph:
            for ver in (*self._verify.values(), *self._verify_solo.values()):
                ver.capture()

    @property
    def self_k(self) -> torch.Tensor:
        return self.caches[self.buckets[-1]][0]

    @property
    def self_v(self) -> torch.Tensor:
        return self.caches[self.buckets[-1]][1]

    def _select(self, c: int) -> None:
        nxt = torch.where(self.done, self._eos, self._rules_mask(self.logits[:, -1]).argmax(-1))  # [B]
        self.done |= nxt == self._eos
        self.seq.index_copy_(1, self.pos, nxt.unsqueeze(1))
        self._rules_advance(nxt)
        self.tok.copy_(nxt.unsqueeze(1))

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """Run the encoder on ``feats`` [B, T, n_mels]; returns the CTC best paths [B, t] (a view of
        ``ctc_ids``, overwritten by the next call)."""
        return self._encoder(feats, self.cross_k, self.cross_v, self.ctc_ids)

    def decode(self, primer: torch.Tensor, max_new_tokens: int) -> list[list[int]]:
        """Greedy decode after ``primer`` [B, P]; returns generated ids per row (without <eos>)."""
        b, p = primer.shape
        assert b == self.batch
        max_new_tokens = min(max_new_tokens, self.capacity - p)
        if max_new_tokens <= 0:
            raise ValueError(f"prompt of {p} tokens does not fit the cache capacity {self.capacity}")

        logits, new_kv, c = self._prefill(primer, self.cross_k, self.cross_v)
        k, v = self.caches[c]
        k[:, :, :, :p] = new_kv[:, 0]
        v[:, :, :, :p] = new_kv[:, 1]
        self.logits[:, -1] = logits
        self.seq[:, :p] = primer
        self.done.zero_()
        self.pos.fill_(p)
        self._rules_start(primer)
        self._select(c)
        self._run(c, p, max_new_tokens - 1, self.done)

        out = []
        for row in self.seq[:, p : p + max_new_tokens].tolist():
            out.append(row[: row.index(self.eos)] if self.eos in row else row)
        self._check()
        return out

    def decode_drafted(
        self,
        primer: torch.Tensor | list[list[int]],
        drafts: list[list[int]],
        max_new_tokens: int,
        insertions: set[int] = frozenset(),
        give_up: int = 8,
    ) -> tuple[list[list[int]], int]:
        """Greedy decoding that checks drafted tokens (one draft per row) several at a time.

        Each call feeds every row's last emitted token plus its next drafted ones, and keeps the drafted
        tokens that equal the decoder's own argmax, plus the argmax after the last kept one, so the
        output is the greedy output. Rows advance by different amounts, each at its own cache offset.
        The first call feeds the primer [B, P] itself (no separate prefill) unless it is too long for a
        window. (Given as lists, the primer needs no device round trip before that call.)

        ``insertions`` are tokens drafts never contain (timestamps, ...): when the decoder emits one, the
        draft is not advanced. Once ``give_up`` calls with drafted tokens have accepted fewer of them
        than there were calls (e.g. POWSM, whose CTC head writes phones whatever the task), a row goes
        on without its draft. (After 4 calls, a few errors early in a good draft were enough to drop it:
        one LibriSpeech utterance then took 103 calls instead of 27.) Returns (tokens per row, calls).
        """
        if not self.windows:
            raise RuntimeError("the session was created without draft_windows")
        primers = primer.tolist() if isinstance(primer, torch.Tensor) else [list(x) for x in primer]
        b, p = len(primers), len(primers[0])
        assert b == self.batch and len(drafts) == b
        max_new_tokens = min(max_new_tokens, self.capacity - p)
        if max_new_tokens <= 0:
            raise ValueError(f"prompt of {p} tokens does not fit the cache capacity {self.capacity}")
        prefill = p >= self.windows[-1]  # the primer does not fit a window: prefill it on its own
        if prefill or self.rules is not None:
            if not isinstance(primer, torch.Tensor):
                primer = torch.tensor(primers, device=self.device)
            self.tok.fill_(-1)
            self._rules_start(primer)
        rows: list[_Row | None] = [_Row(r, pr, d) for r, (pr, d) in enumerate(zip(primers, drafts))]
        c, calls = None, 0
        if prefill:
            logits, new_kv, c = self._prefill(primer, self.cross_k, self.cross_v)
            k, v = self.caches[c]
            k[:, :, :, :p] = new_kv[:, 0]
            v[:, :, :, :p] = new_kv[:, 1]
            for row, t in zip(rows, self._rules_mask(logits).argmax(-1).tolist()):
                row.advance([], t, p, insertions, self.punctuation, self._time_check())
            calls = 1
        out, n = self._drafted(rows, lambda r: None, c, max_new_tokens, insertions, give_up)
        return [out[r] for r in range(b)], calls + n

    def decode_stream(
        self,
        primer: list[int],
        batches: Iterator[torch.Tensor],
        drafts_of: Callable[[torch.Tensor], list[list[int]]],
        max_new_tokens: int,
        insertions: set[int] = frozenset(),
        give_up: int = 8,
        priority: bool = True,
        finished: Callable[[int, list[int]], None] | None = None,
    ) -> tuple[list[list[int]], int]:
        """:meth:`decode_drafted` of any number of utterances, keeping every row busy.

        ``batches`` yields encoder inputs [E, T, n_mels] (E <= rows) and ``drafts_of`` makes the drafts
        of a batch from its CTC best paths [E, t] (on the host). A row whose utterance is done takes the
        next one for the following call, so that the rows do not wait for the slowest one. The encoder
        runs ahead on a second CUDA stream into two staging buffers in turn; with ``priority``, the
        decoder calls go first on the GPU. Returns (tokens per utterance, in order; decoder calls).

        ``finished(index, tokens)`` is called as soon as an utterance is done, so that ``batches`` can
        depend on the outputs so far (transcribe_long: a window starts where the one before it in the
        recording ended). Such a ``batches`` yields None while it has nothing to encode yet; it is asked
        again after the next decoder call, or when a row has nothing left to decode.
        """
        if not self.windows:
            raise RuntimeError("the session was created without draft_windows")
        p = len(primer)
        if p >= self.windows[-1]:
            raise ValueError(f"a primer of {p} tokens does not fit a draft window")
        max_new_tokens = min(max_new_tokens, self.capacity - p)
        if self.rules is not None:
            self.tok.fill_(-1)
            self._rules_start(torch.tensor([primer] * self.rows, device=self.device))
        cuda = self.device.type == "cuda"
        if not self._stage:
            bufs = (self.cross_k, self.cross_v, self.ctc_ids)
            self._stage = [tuple(torch.zeros_like(t) for t in bufs) for _ in range(2)]
            ids = self.ctc_ids
            self._stage_ids = [torch.zeros(ids.shape, dtype=ids.dtype, pin_memory=cuda) for _ in range(2)]
        main = torch.cuda.current_stream() if cuda else None
        # a stream (and TensorRT contexts) per staging slot when the batch is small (ENCODER_LANES): its
        # encoder calls leave most of the GPU idle, so the next two run at the same time
        lanes = cuda and ENCODER_LANES and self.rows <= ENCODER_LANE_ROWS
        sides = [torch.cuda.Stream() for _ in range(2 if lanes else 1)] if cuda else [None]
        dec = torch.cuda.Stream(priority=-1) if cuda and priority else main
        ready = [torch.cuda.Event() for _ in range(2)] if cuda else []
        copied = [torch.cuda.Event() for _ in range(2)] if cuda else []
        queue: deque[tuple[int, int, int]] = deque()  # (staging slot, row in it, utterance) waiting
        left, drafts, count = [0, 0], [None, None], 0  # per slot: utterances not taken yet, their drafts
        held: list[tuple] = [(), ()]  # per slot: the cross-attention K/V of its batch
        paths: list = [None, None]  # per slot: its CTC best paths, copied to the host after the encoder
        ended = [False]  # batches has no more

        def launch(s: int) -> bool:  # encode the next batch into staging slot s; False: none yet
            nonlocal count
            side = sides[s % len(sides)]
            with torch.cuda.stream(side) if cuda else nullcontext():
                feats = next(batches, _END)
                if feats is None:
                    return False
                if feats is _END:
                    ended[0] = True
                    return True  # no more: the slot stays empty
                e = feats.shape[0]
                # a smaller batch (the last one) gets buffers of its size rather than padding rows
                bufs = self._stage[s] if e == self.rows else self._staging(s, e)
                if cuda:
                    side.wait_event(copied[s])
                ids = self._encoder(feats, *bufs, lane=s if len(sides) > 1 else 0)
                held[s] = bufs[:2]
                paths[s] = self._stage_ids[s].view(-1)[: ids.numel()].view(ids.shape)
                paths[s].copy_(ids, non_blocking=True)  # read on the host once `ready`
                if cuda:
                    ready[s].record(side)
            queue.extend((s, i, count + i) for i in range(e))
            left[s], drafts[s], count = e, None, count + e
            return True

        pending: list[int] = []  # free slots not refilled yet: launched while a decoder call runs

        def flush() -> None:
            pending[:] = [s for s in pending if not launch(s)]

        def refill(r: int) -> _Row | None:
            if not queue:
                flush()  # (the queue holds what pending launches add, in the same order, when it is needed)
                if not queue:
                    return None
            s, i, index = queue.popleft()
            k, v = held[s]
            if drafts[s] is None:
                if cuda:
                    ready[s].synchronize()
                    dec.wait_event(ready[s])
                drafts[s] = drafts_of(paths[s])
            self.cross_k[:, r].copy_(k[:, i])
            self.cross_v[:, r].copy_(v[:, i])
            row = _Row(index, primer, drafts[s][i])
            left[s] -= 1
            if left[s] == 0:  # buffer s is free once these copies are done
                if cuda:
                    copied[s].record(dec)
                pending.append(s)
            return row

        if cuda:
            dec.wait_stream(main)
            for side in sides:
                side.wait_stream(main)

        def concurrent() -> bool:  # whether the encoder may run alongside the next decoder call: it runs, or
            # the call's meanwhile() (flush) launches it
            return (bool(pending) and not ended[0]) or not (ready[0].query() and ready[1].query())

        pending += [0, 1]
        flush()
        with torch.cuda.stream(dec) if cuda else nullcontext():
            out, calls = self._drafted(
                [None] * self.rows, refill, None, max_new_tokens, insertions, give_up, flush, finished,
                concurrent if cuda else None,
            )  # fmt: skip
        if cuda:
            main.wait_stream(dec)
            for side in sides:
                main.wait_stream(side)
        return [out[i] for i in range(count)], calls

    def _staging(self, slot: int, rows: int) -> tuple[torch.Tensor, ...]:
        """decode_stream's encoder outputs for a batch of ``rows`` (< self.rows) in staging ``slot``."""
        key = (slot, rows)
        if key not in self._stage_part:
            self._stage_part[key] = tuple(
                torch.zeros(t.shape[:i] + (rows,) + t.shape[i + 1 :], dtype=t.dtype, device=t.device)
                for t, i in ((self.cross_k, 1), (self.cross_v, 1), (self.ctc_ids, 0))
            )
        return self._stage_part[key]

    def _is_time(self, t: int) -> bool:
        return self.rules is not None and self.rules.first_time <= t <= self.rules.last_time

    def _time_check(self) -> Callable[[int], bool] | None:
        """:meth:`_is_time` for :meth:`_Row.advance`; None without timestamp rules (no timestamps)."""
        return self._is_time if self.rules is not None else None

    def _drafted(
        self,
        rows: list["_Row | None"],
        refill: Callable[[int], "_Row | None"],
        c: int | None,
        max_new_tokens: int,
        insertions: set[int],
        give_up: int,
        meanwhile: Callable[[], None] | None = None,
        finished: Callable[[int, list[int]], None] | None = None,
        concurrent: Callable[[], bool] | None = None,
    ) -> tuple[dict[int, list[int]], int]:
        """The loop of drafted decoding, on cache bucket ``c`` (None: not chosen yet). An empty row, or
        one whose utterance is done, takes ``refill(row)``. ``meanwhile()`` runs while each call runs on
        the GPU (the host would only wait); ``finished(index, tokens)`` when an utterance is done, before
        its row is refilled; ``concurrent()``: whether other GPU work (an encoder) runs alongside the next
        call (None: nothing does). Returns (tokens per utterance, calls).

        The host waits for every call's result, and the GPU for the next call: this loop is kept lean."""
        eos, done, calls = self.eos, {}, 0
        windows, capacity, punctuation = self.windows, self.capacity, self.punctuation
        is_time = self._time_check()
        filler = {w: [eos] * w + [0, 0, -1, -1] for w in windows}  # rows left out: harmless at offset 0
        none = -(1 << 62)
        while True:
            want = top = none
            for r, x in enumerate(rows):
                if x is not None and not x.fresh:
                    if x.last == eos or len(x.out) >= max_new_tokens:
                        done[x.index] = x.out[:max_new_tokens]
                        if finished is not None:
                            finished(x.index, done[x.index])
                        x = None
                    else:
                        x.out.append(x.last)
                        x.head = [x.last]
                if x is None:
                    x = rows[r] = refill(r)
                    if x is None:
                        continue
                # window: each row's head (primer or last emitted token) + as many drafted tokens as any has
                n = len(x.head) + min(len(x.draft) - x.ptr, max_new_tokens - len(x.out))
                if n > want:
                    want = n
                if x.pos > top:
                    top = x.pos
            if want == none:
                self._check()
                return done, calls
            w = windows[min(bisect_left(windows, want), len(windows) - 1)]  # the smallest with `want`
            if top + w > capacity:
                w = min(w, max(x for x in windows if top + x <= capacity))
            # one host copy per call: per row [w tokens | offset | #timestamps, last one | primer end]
            flat, live, last_pos = [], [], none
            for r, x in enumerate(rows):
                h = len(x.head) if x is not None else w + 1
                if h > w:  # no row, or a primer waiting for a wider window
                    flat += filler[w]
                    continue
                drafted = x.draft[x.ptr : x.ptr + w - h]
                flat += x.head
                flat += drafted
                if h + len(drafted) < w:
                    flat += filler[w][: w - h - len(drafted)]
                flat += (x.pos, x.times[0], x.times[1], h - 1 if x.fresh else -1)
                live.append((r, x))
                if x.pos > last_pos:
                    last_pos = x.pos
            if not live:
                raise ValueError("no row fits a draft window")
            c = self._move(c, last_pos + w) if c is not None else self._bucket(last_pos + w)
            solo = self._verify_solo and not (concurrent is not None and concurrent())
            accepted = (self._verify_solo if solo else self._verify)[c, w](flat, meanwhile)
            calls += 1
            for r, x in live:
                n, last = accepted[r]
                h = len(x.head)
                checked = w > h and x.draft
                x.advance(x.draft[x.ptr : x.ptr + n], last, x.pos + h + n, insertions, punctuation, is_time)
                if checked:
                    x.tried, x.accepted = x.tried + 1, x.accepted + n
                    if x.tried >= give_up and x.accepted < x.tried:
                        x.draft = []


class _Row:
    """An utterance in :meth:`GreedySession._drafted`: its tokens so far and its place in the draft."""

    __slots__ = ("index", "head", "fresh", "draft", "out", "last", "pos", "ptr", "times", "tried", "accepted")

    def __init__(self, index: int, primer: list[int], draft: list[int]):
        self.index = index
        self.head = primer  # what the next call feeds before drafted tokens: the primer, later the last token
        self.fresh = True  # the primer has not been fed yet
        self.draft = list(draft)
        self.out: list[int] = []
        self.last = -1  # the last emitted token (not in ``out`` before the next call feeds it)
        self.pos = 0  # cache slot of the next call's first token
        self.ptr = 0  # the next drafted token to check
        self.times = [0, -1]  # timestamps emitted so far, the last one
        self.tried = self.accepted = 0  # calls that checked drafted tokens, drafted tokens accepted

    def advance(
        self,
        got: list[int],
        last: int,
        pos: int,
        insertions: set[int],
        punctuation: frozenset[int],
        is_time: Callable[[int], bool] | None = None,
    ) -> None:
        """The decoder accepted the drafted tokens ``got``, then emitted ``last``; next call at ``pos``.
        ``is_time``: None if no token is a timestamp."""
        out = self.out
        out += got
        if is_time is not None:
            for t in got + [last]:
                if is_time(t):
                    self.times = [self.times[0] + 1, t]
        self.last, self.pos, self.fresh = last, pos, False
        tail = out[-2:]  # all that _resync reads of the output
        tail.append(last)
        self.ptr = _resync(self.draft, self.ptr + len(got), tail, insertions, punctuation)
        if is_time is not None and is_time(last) and last not in insertions and self.times[0] % 2 == 0:
            # Drafts with timestamps (transcribe_long): after an end time, guess the next start time is
            # the same (84% of LibriSpeech long-form utterance boundaries).
            self.draft.insert(self.ptr, last)


def _resync(
    draft: list[int],
    j: int,
    out: list[int],
    insertions: set[int],
    punctuation: frozenset[int] = frozenset(),
    reach: int = 8,
) -> int:
    """Index of the drafted token to continue with, after the decoder emitted ``out[-1]`` where
    ``draft[j]`` was expected.

    The last 3, 2 or 1 emitted tokens are looked up in the draft within ``reach`` of ``j`` (either
    way, so that a wrong guess is undone on the next call), longest first, nearest to ``j`` first; a
    lone punctuation mark recurs too often to anchor on. Without an anchor the token is taken to
    replace ``draft[j]``, unless it is punctuation where the draft has none: CTC drafts often lack
    commas, so that is taken as an insertion and ``draft[j]`` is still expected. (On LibriSpeech
    test-clean these rules make 18% fewer decoder calls than taking every unknown token as an
    insertion and anchoring on anything.)
    """
    last = out[-1]
    if last in insertions:
        return j
    n, lo = len(draft), max(0, j - reach)
    near = draft[lo : j + reach + 3]  # where the anchors, which all end in `last`, can end
    if last in near:
        ends = None
        for k in (3, 2, 1):
            if len(out) < k:
                continue
            tail = out[-k:]
            if k == 1:
                if last in punctuation:
                    continue
            elif tail[0] in insertions or tail[k - 2] in insertions:
                continue
            hi = min(n - k, j + reach)  # anchors draft[i : i + k], lo <= i <= hi
            i = j - k + 1  # the nearest ends at draft[j]: the decoder followed the draft (the usual case)
            if lo <= i <= hi and draft[i : i + k] == tail:
                return j + 1
            if ends is None:
                ends = [lo + e for e, t in enumerate(near) if t == last]
            best = -1  # the nearest other one (where it ends), the first of equally near ones
            for e in ends:
                if lo <= e - k + 1 <= hi and (best < 0 or abs(e - j) < abs(best - j)):
                    if draft[e - k + 1 : e + 1] == tail:
                        best = e
            if best >= 0:
                return best + 1
    if j < n and not (last in punctuation and draft[j] not in punctuation):
        return j + 1
    return j


def _accepted(pred: list[int], window: list[int], s: int, eos: int) -> tuple[int, int]:
    """What :meth:`_Verify.accept` computes for one row, in plain Python (for oracles in tests and
    scripts/draft_sim.py): the number n of drafted tokens ``window[s + 1 :]`` that equal the decoder's
    choice ``pred`` before them, and its choice after the last of those."""
    n = 0  # pred[s + i] is the decoder's choice for window[s + i + 1]
    while s + n + 1 < len(window) and pred[s + n] == window[s + n + 1] and window[s + n + 1] != eos:
        n += 1
    return n, pred[s + n]


class _Verify:
    """Buffers of one window size: every row feeds ``w`` tokens at its own offset, and the decoder's
    argmax after each of them comes back (under the timestamp rules, if any)."""

    def __init__(
        self,
        cfg: OWSMConfig,
        rows: int,
        w: int,
        kw: dict,
        rules: TimestampRules | None,
        notime: torch.Tensor,
        eos: int,
    ):
        L, H, dk = cfg.dec_layers, cfg.dec_heads, cfg.d_model // cfg.dec_heads
        dev = {"device": kw["device"]}
        self.rows, self.w, self.rules, self.notime, self.eos = rows, w, rules, notime, eos
        # per call, one host copy: [w tokens | offset | timestamps so far (count, last) | first-token position]
        self.inp = torch.zeros(rows, w + 4, dtype=torch.long, **dev)
        # Pinned host buffers, copied from and to inside the call's CUDA graph: a call is one graph
        # launch, and its result is read when an event says it is there.
        pinned = kw["device"].type == "cuda"
        self.host = torch.zeros(rows, w + 4, dtype=torch.long, pin_memory=pinned)
        self.host_flat = self.host.numpy().reshape(-1)
        self.res_host = torch.zeros(rows, 2, dtype=torch.long, pin_memory=pinned)
        self.res_np = self.res_host.numpy()
        self.ready = torch.cuda.Event() if pinned else None
        self.tok = torch.zeros(rows, w, dtype=torch.long, **dev)
        self.offs = torch.zeros(rows, dtype=torch.long, **dev)
        self.logits = torch.zeros(rows, w, cfg.vocab_size, **dev)
        self.new_kv = torch.zeros(L, 2, rows, H, w, dk, **kw)
        self.pred = torch.zeros(rows, w, dtype=torch.long, **dev)
        self.res = torch.zeros(rows, 2, dtype=torch.long, **dev)  # what comes back: see accept()
        self.ar = torch.arange(w, **dev)
        self.fast = kernels.available(kw["device"])  # Triton kernels for the work around the decoder call

    def predict(self) -> torch.Tensor:
        """Argmax after each window position [rows, w], with the rules applied position by position."""
        if self.rules is None:
            return self.logits.argmax(-1)
        r, w, tok = self.rows, self.w, self.tok
        times, first = self.inp[:, w + 1 : w + 3], self.inp[:, w + 3]
        later = self.rules.is_time(tok) & (self.ar > 0)  # timestamps the window itself adds
        count = times[:, :1] + torch.cumsum(later.long(), 1)
        last = torch.maximum(times[:, 1:], torch.cummax(torch.where(later, tok, -1), 1).values)
        bad = self.rules.forbidden(
            self.notime.unsqueeze(1).expand(r, w).reshape(-1),
            (self.ar == first.unsqueeze(1)).reshape(-1),
            count.reshape(-1),
            last.reshape(-1),
            tok.reshape(-1),
        )
        return self.logits.masked_fill(bad.view(r, w, -1), float("-inf")).argmax(-1)

    def accept(self) -> torch.Tensor:
        """[rows, 2]: per row, the number n of drafted tokens accepted -- those after the head (the
        primer or the last emitted token, which ends at position s) that equal the decoder's argmax
        before them, up to the first that does not or <eos> -- and the argmax after the last of them."""
        w = self.w
        s = self.inp[:, w + 3].clamp(min=0)
        nxt = self.inp[:, 1:w]
        ok = ((self.pred[:, :-1] == nxt) & (nxt != self.eos)) | (self.ar[:-1] < s.unsqueeze(1))
        end = ok.long().cumprod(1).sum(1)  # s + n
        return torch.stack([end - s, self.pred.gather(1, end.unsqueeze(1)).squeeze(1)], 1)

    def bind(self, step: Callable[[], None], k: torch.Tensor, v: torch.Tensor) -> "_Bound":
        return _Bound(self, step, k, v)


class _Bound:
    """One window size on one cache bucket: the decoder call, the per-row K/V scatter, the argmax and
    the drafted tokens accepted."""

    def __init__(self, bufs: _Verify, step, k, v):
        self.bufs, self.step, self.k, self.v = bufs, step, k, v
        self.graph: torch.cuda.CUDAGraph | None = None

    def body(self) -> None:
        b = self.bufs
        b.inp.copy_(b.host, non_blocking=True)
        if not getattr(self.step, "reads_inp", False):  # (a decoder megakernel reads them in inp)
            b.tok.copy_(b.inp[:, : b.w])
            b.offs.copy_(b.inp[:, b.w])
        self.step()
        written = getattr(self.step, "writes_cache", False)  # the decoder megakernel writes the cache itself
        if b.fast:
            if not written:
                kernels.write_kv(b.new_kv, self.k, self.v, b.offs)
            if not getattr(self.step, "writes_pred", False):
                kernels.choose(b.logits, b.inp, b.notime, b.pred, b.rules)
            elif b.rules is not None:  # (the kernel's argmax holds for the rows with <notimestamps>)
                kernels.choose(b.logits, b.inp, b.notime, b.pred, b.rules, skip_notime=True)
            kernels.accept(b.pred, b.inp, b.res, b.eos)
        else:
            if not written:
                L, _, r, H, w, dk = b.new_kv.shape
                slots = (b.offs.unsqueeze(1) + b.ar)[None, :, None, :, None].expand(L, r, H, w, dk)
                self.k.scatter_(3, slots, b.new_kv[:, 0])
                self.v.scatter_(3, slots, b.new_kv[:, 1])
            b.pred.copy_(b.predict())
            b.res.copy_(b.accept())
        b.res_host.copy_(b.res, non_blocking=True)

    def capture(self) -> None:
        self.graph = capture_graph(self.body)

    def __call__(self, flat: list[int], meanwhile: Callable[[], None] | None = None) -> list[list[int]]:
        """``flat``: per row, the window's tokens + [offset, #timestamps, last timestamp, first position],
        concatenated. Returns per row [drafted tokens accepted, the token emitted after them], after
        running ``meanwhile()`` while the GPU works."""
        b = self.bufs
        b.host_flat[:] = np.frombuffer(array("q", flat), dtype=np.int64)  # read by the call, reused after it
        self.graph.replay() if self.graph is not None else self.body()
        if meanwhile is not None:
            meanwhile()
        if b.ready is not None:
            b.ready.record()
            b.ready.synchronize()
        return b.res_np.tolist()


LOG_ZERO = -1.0e10  # ESPnet's score for slots that must not be expanded
END_DETECT_M, END_DETECT_D = 3, -10.0  # ESPnet's end_detect(M=3, D_end=log(exp(-10)))


class BeamSession(_Session):
    """Beam search over ``batch`` utterances x ``beam`` hypotheses, as ESPnet's ``BatchBeamSearch``.

    Matches ESPnet's s2t default search (decoder score only, no length normalisation or penalty,
    ``maxlenratio=0``): every step keeps the ``beam`` best one-token extensions of the live
    hypotheses of each utterance; an extension ending in <eos> is set aside as finished and its slot
    stops being expanded. An utterance is done when ``end_detect`` fires, when none of its
    hypotheses are live, or at the length limit, where every live hypothesis is finished with an
    appended <eos>. By default it also stops when a finite best finished score strictly exceeds
    every live score. The result is the finished hypothesis with the highest total log-probability.

    Rows are utterance-major (row = utterance * beam + k). Buffers besides :class:`_Session`'s:
      score [N] / active [N]      running log-probability and whether the slot may be expanded
      seq [N, C + 2]               primer + generated tokens of every slot
      best_* [B]                   best finished hypothesis so far
      ended_by_len [B, C + 3]      best finished score per hypothesis length (for end_detect)
    """

    def __init__(
        self,
        backend,
        cfg: OWSMConfig,
        batch: int,
        beam: int,
        capacity: int,
        eos: int,
        dtype: torch.dtype,
        device: torch.device,
        use_cuda_graph: bool = True,
        chunk: int | None = None,
        buckets: list[int] | None = None,
        rules: TimestampRules | None = None,
        upper_bound: bool = True,
        diagnostics: bool = False,
        fused_rank: bool = True,
        fused_finish: bool = True,
        kv_reorder: str = "torch",
    ):
        # H200/1B, beam 5 with bound stopping: every-step checks help batch 1, but hurt batch 3.
        # Explicit intervals stay fixed when set_upper_bound toggles the device flag for ablations.
        if chunk is None:
            chunk = 1 if upper_bound and batch == 1 else 4
        if chunk < 1:
            raise ValueError("beam completion-check interval must be positive")
        if kv_reorder not in ("torch", "full", "prefix"):
            raise ValueError("kv_reorder must be torch, full or prefix")
        n = batch * beam
        super().__init__(backend, cfg, n, capacity, eos, dtype, device, use_cuda_graph, chunk, buckets, rules)
        self.batch, self.beam = batch, beam
        L, H, dk = cfg.dec_layers, cfg.dec_heads, cfg.d_model // cfg.dec_heads
        dev = {"device": device}
        # one row per utterance for the encoder and the primer; copied to every beam slot afterwards
        self.cross_kb = torch.zeros(L, batch, H, cfg.enc_frames, dk, **self.kw)
        self.cross_vb = torch.zeros(L, batch, H, cfg.enc_frames, dk, **self.kw)
        self.ctc_ids = torch.zeros(batch, ctc_columns(cfg), dtype=torch.int32, **dev)
        self.score = torch.zeros(n, **dev)
        self.active = torch.zeros(n, dtype=torch.bool, **dev)
        self.done = torch.zeros(batch, dtype=torch.bool, **dev)
        self.seq = torch.zeros(n, capacity + 2, dtype=torch.long, **dev)
        self.numerical_error = torch.zeros(batch, dtype=torch.bool, **dev)
        self.best_score = torch.zeros(batch, **dev)
        self.best_seq = torch.zeros(batch, capacity + 2, dtype=torch.long, **dev)
        self.best_end = torch.zeros(batch, dtype=torch.long, **dev)  # seq index after its last token
        self.ended_by_len = torch.zeros(batch, capacity + 3, **dev)
        self.step = torch.zeros(1, dtype=torch.long, **dev)  # index i of ESPnet's search loop
        self.last_step = torch.zeros(1, dtype=torch.long, **dev)  # maxlen - 1
        self.row_base = torch.arange(batch, **dev) * beam  # first row of each utterance
        # A device flag lets ON/OFF ablations replay the very same captured graphs.
        self.upper_bound = torch.tensor(upper_bound, dtype=torch.bool, **dev)
        self.fused_rank = fused_rank and kernels.available(device)
        self.fused_finish = fused_finish and kernels.available(device)
        self.kv_reorder = kv_reorder if kernels.available(device) else "torch"
        self._kv_scratch = None
        if self.kv_reorder != "torch":
            self._kv_scratch = torch.empty(2 * L * n * H * max(self.buckets) * dk, **self.kw)
        self._candidates = self._bad_blocks = None
        if self.fused_rank:
            self._candidates = torch.empty(n, cfg.vocab_size, **dev)
            self._bad_blocks = torch.empty(n, (cfg.vocab_size + 1023) // 1024, dtype=torch.bool, **dev)
        self.diagnostics = diagnostics
        self._beam_history = None
        if diagnostics:
            self._beam_history = {
                "scores": torch.zeros(capacity, batch, 2, **dev),
                "state": torch.zeros(capacity, batch, 9, dtype=torch.long, **dev),
            }
        self._host_checks: list[int] = []
        self._capture_all()

    def _body(self, c: int) -> None:
        self._feed(c)
        self.step += 1
        self._select(c)

    def _select(self, c: int) -> None:
        top, parent, token, valid = self._rank()
        self._reorder(c, top, parent, token)
        self._finish(token, valid)

    def _rank(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K = self.batch, self.beam
        V = self.logits.shape[-1]

        # 1. the K best extensions per utterance (scores accumulate fp32 log-probabilities)
        logits = self.logits[:, -1]  # [N, V]
        if self.fused_rank:
            # Full-vocabulary normalization is unchanged. Invalid rows produce only non-finite
            # logp, which the kernel discards while flagging active NaN/+Inf inputs separately.
            cand = beam_kernels.candidates(self, torch.log_softmax(logits, dim=-1))
            top, idx = cand.view(B, K * V).topk(K, dim=-1)
            parent = (idx // V + self.row_base.unsqueeze(1)).view(-1)
            token = (idx % V).view(-1)
            valid = self.active.index_select(0, parent) & torch.isfinite(top.view(-1))
            return top, parent, token, valid
        bad = (torch.isnan(logits) | torch.isposinf(logits)).any(dim=-1)
        live_parent = self.active & ~self.done.repeat_interleave(K)
        self.numerical_error |= (bad & live_parent).view(B, K).any(dim=-1)
        # An all -inf row is exhausted, not a numerical failure. Avoid softmax(-inf, ..., -inf),
        # and never allow dead parents or masked/non-finite extensions into the global top-K.
        usable = live_parent & ~bad & torch.isfinite(logits).any(dim=-1)
        safe = torch.where(usable.unsqueeze(1), logits, 0.0)
        logp = self._rules_mask(torch.log_softmax(safe, dim=-1))  # [N, V]
        cand = (self.score.unsqueeze(1) + logp).masked_fill(~usable.unsqueeze(1), float("-inf"))
        top, idx = cand.view(B, K * V).topk(K, dim=-1)  # [B, K]
        parent = (idx // V + self.row_base.unsqueeze(1)).view(-1)  # [N] row each extension comes from
        token = (idx % V).view(-1)
        valid = live_parent.index_select(0, parent) & torch.isfinite(top.view(-1))
        return top, parent, token, valid

    def _reorder(self, c: int, top: torch.Tensor, parent: torch.Tensor, token: torch.Tensor) -> None:
        self.score.copy_(top.view(-1))
        self.seq.copy_(self.seq.index_select(0, parent))
        self.seq.index_copy_(1, self.pos, token.unsqueeze(1))
        if self.kv_reorder == "torch":
            for buf in self.caches[c]:
                buf.copy_(buf.index_select(1, parent))
        else:
            beam_kernels.reorder_kv(*self.caches[c], self._kv_scratch, parent, self.pos,
                                   prefix=self.kv_reorder == "prefix")
        self._rules_advance(token, parent)

    def _finish(self, token: torch.Tensor, valid: torch.Tensor) -> None:
        if self.fused_finish:
            return beam_kernels.finish(self, token, valid, END_DETECT_M, END_DETECT_D)
        B, K = self.batch, self.beam
        # 2. finished hypotheses. All that finish in one step have the same length (ESPnet's yseq
        # length, which counts the primer): pos + 1, or pos + 2 at the limit where <eos> is appended.
        live = valid & ~self.done.repeat_interleave(K)
        at_limit = self.step == self.last_step  # [1]
        is_eos = token == self.eos
        ended = (is_eos | at_limit) & live
        length = self.pos + 1 + at_limit.long()  # [1]
        ended_score = torch.where(ended, self.score, float("-inf")).view(B, K)
        best_val, best_k = ended_score.max(dim=-1)  # first of equals, as ESPnet's stable sort
        better = best_val > self.best_score
        self.best_score.copy_(torch.where(better, best_val, self.best_score))
        self.best_seq.copy_(
            torch.where(better.unsqueeze(1), self.seq.index_select(0, self.row_base + best_k), self.best_seq)
        )
        self.best_end.copy_(torch.where(better, self.pos + 1, self.best_end))
        col = self.ended_by_len.index_select(1, length)
        self.ended_by_len.index_copy_(1, length, torch.maximum(col, best_val.unsqueeze(1)))

        # 3. end_detect(ended_hyps, i): for each of the lengths i, i-1, i-2 some finished hypothesis
        # has that length, and the best of them is END_DETECT_D below the best finished overall.
        count = torch.zeros(B, dtype=torch.long, device=self.device)
        for m in range(END_DETECT_M):
            n = self.step - m
            s = self.ended_by_len.index_select(1, n.clamp(min=0)).squeeze(1)
            count += ((n >= 0) & (s > float("-inf")) & (s - self.best_score < END_DETECT_D)).long()
        live_after = (~is_eos & live & ~at_limit).view(B, K)
        max_live = self.score.view(B, K).masked_fill(~live_after, float("-inf")).max(dim=-1).values
        bound = torch.isfinite(self.best_score) & (self.best_score > max_live) & live_after.any(dim=-1)
        end_detect = count == END_DETECT_M
        exhausted = ~live_after.any(dim=-1)
        done = self.done | at_limit | exhausted | end_detect | (bound & self.upper_bound)
        if self._beam_history is not None:
            # [B, fields] -> [capacity, B, fields], indexed on-device for CUDA graph replay.
            reason = at_limit.long() | (exhausted.long() * 2) | (end_detect.long() * 4)
            reason |= (bound & self.upper_bound).long() * 8
            state = torch.stack((
                self.pos.expand(B), live_after.sum(dim=-1), (is_eos & live).view(B, K).any(dim=-1),
                better, bound, done, reason, self.numerical_error, self.best_end,
            ), dim=-1)
            self._beam_history["state"].index_copy_(0, self.step, state.unsqueeze(0))
            scores = torch.stack((self.best_score, max_live), dim=-1)
            self._beam_history["scores"].index_copy_(0, self.step, scores.unsqueeze(0))
        self.done.copy_(done)
        self.active.copy_((live_after & ~done.unsqueeze(1)).view(-1))

        self.tok.copy_(token.unsqueeze(1))

    def set_upper_bound(self, enabled: bool) -> None:
        """Change strict-bound stopping without recapture; keep the chosen host-check interval."""
        self.upper_bound.fill_(enabled)

    def _run(self, c: int, pos: int, steps: int, done: torch.Tensor) -> None:
        if not self.diagnostics:
            return super()._run(c, pos, steps, done)
        executed = 0
        while steps > 0:
            if pos >= c:
                c = self._grow(c, pos + 1)
            n = min(self.chunk, steps, c - pos)
            for _ in range(n):
                self.graphs[c].replay() if self.graphs else self._body(c)
            steps -= n
            pos += n
            executed += n
            all_done = self._all_done(done)
            self._host_checks.append(executed)
            if all_done:
                break

    def get_diagnostics(self) -> dict | None:
        """Collect the last decode's fixed device history in bulk; no change to decode's return value.

        Selection 0 follows prefill, without a single-token decoder call. A logical end step is a
        zero-based search index; best_end includes the primer, excludes the appended forced EOS.
        Extra decoder calls include both chunk overshoot and waiting for other batch members.
        """
        if self._beam_history is None:
            return None
        steps = int(self.step) + 1
        state = self._beam_history["state"][:steps].cpu().tolist()
        scores = self._beam_history["scores"][:steps].cpu().tolist()
        fields = ("position", "live_count", "eos", "best_updated", "bound", "done", "stop_mask",
                  "numerical_error", "best_end")
        utterances = []
        for b in range(self.batch):
            rows = [dict(zip(fields, row[b])) for row in state]
            for i, (row, score) in enumerate(zip(rows, scores)):
                row.update(search_step=i, decoder_calls_so_far=i,
                           best_score=score[b][0], max_live_score=score[b][1])
            end = next((i for i, row in enumerate(rows) if row["done"]), None)
            last_update = next((i for i in range(len(rows) - 1, -1, -1) if rows[i]["best_updated"]), None)
            finished_length = (rows[last_update]["best_end"] + bool(rows[last_update]["stop_mask"] & 1)
                               if last_update is not None else None)
            first_check = next((i for i in self._host_checks if end is not None and i >= end), None)
            detected = self._host_checks[-1] if self._host_checks and end is not None else None
            if not self._host_checks and end is not None:
                detected = 0  # max_new_tokens=1: collected at the final output transfer
            utterances.append({
                "history": rows,
                "first_eos_step": next((i for i, row in enumerate(rows) if row["eos"]), None),
                "first_bound_step": next((i for i, row in enumerate(rows) if row["bound"]), None),
                "last_best_update_step": last_update,
                "logical_end_step": end, "host_observed_done_step": detected,
                "extra_decoder_calls": steps - 1 - end if end is not None else None,
                "chunk_overshoot": first_check - end if first_check is not None else 0,
                "stop_mask": rows[end]["stop_mask"] if end is not None else 0,
                "best_end_with_primer": rows[-1]["best_end"],
                "best_finished_length_with_primer": finished_length,
            })
        return {
            "prefill_calls": 1, "prefill_selections": 1, "decoder_calls": steps - 1,
            "selection_calls": steps, "host_check_steps": self._host_checks.copy(),
            "utterances": utterances,
        }

    def _check(self) -> None:
        super()._check()
        # Called after decoding's existing host synchronization, never during graph capture/replay.
        if bool(self.numerical_error.any()):
            rows = self.numerical_error.nonzero().flatten().tolist()
            raise RuntimeError(f"beam search: NaN or positive infinity in live logits for utterances {rows}")
        missing = ~torch.isfinite(self.best_score)
        if bool(missing.any()):
            rows = missing.nonzero().flatten().tolist()
            raise RuntimeError(f"beam search: no valid finished hypothesis for utterances {rows}")

    def _expand_cross(self) -> None:
        self.cross_k.copy_(self.cross_kb.repeat_interleave(self.beam, dim=1))
        self.cross_v.copy_(self.cross_vb.repeat_interleave(self.beam, dim=1))

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        return self._encoder(feats, self.cross_kb, self.cross_vb, self.ctc_ids, after=self._expand_cross)

    def decode(self, primer: torch.Tensor, max_new_tokens: int) -> list[list[int]]:
        """Beam search after ``primer`` [B, P]; returns the best hypothesis per utterance (no <eos>).

        ``max_new_tokens`` is ESPnet's ``maxlen``: the number of search steps.
        """
        b, p = primer.shape
        assert b == self.batch
        K = self.beam
        max_new_tokens = min(max_new_tokens, self.capacity - p)
        if max_new_tokens <= 0:
            raise ValueError(f"prompt of {p} tokens does not fit the cache capacity {self.capacity}")

        logits, new_kv, c = self._prefill(primer, self.cross_kb, self.cross_vb)
        k, v = self.caches[c]
        k[:, :, :, :p] = new_kv[:, 0].repeat_interleave(K, dim=1)
        v[:, :, :, :p] = new_kv[:, 1].repeat_interleave(K, dim=1)
        self.logits[:, -1] = logits.repeat_interleave(K, dim=0)
        self.seq[:, :p] = primer.repeat_interleave(K, dim=0)
        self._rules_start(self.seq[:, :p])
        self.score.zero_()
        self.active.zero_()
        self.active[::K] = True  # the beams of an utterance start identical: expand only the first
        self.done.zero_()
        self.numerical_error.zero_()
        self.best_score.fill_(float("-inf"))
        self.best_end.fill_(p)
        self.ended_by_len.fill_(float("-inf"))
        self.step.zero_()
        self.last_step.fill_(max_new_tokens - 1)
        self.pos.fill_(p)
        if self.diagnostics:
            self._host_checks.clear()
        self._select(c)
        self._run(c, p, max_new_tokens - 1, self.done)

        out = []
        for row, end in zip(self.best_seq.tolist(), self.best_end.tolist()):
            ids = row[p:end]
            out.append(ids[: ids.index(self.eos)] if self.eos in ids else ids)
        self._check()
        return out
