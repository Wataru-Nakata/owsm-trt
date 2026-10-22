"""Execution backends and the CUDA-graph greedy decoding loop.

Both backends expose the same two operations on caller-owned buffers:

* ``encode(feats, cross_k, cross_v)`` -- encoder + per-layer cross-attention K/V projection
* ``step_fn(...)`` -- returns a closure that runs one decoder call writing logits and new K/V into
  fixed buffers, so that the whole decode step can be captured in a CUDA graph.
"""

import importlib.metadata
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

import torch

from .config import OWSMConfig
from .modeling import OWSMModel

log = logging.getLogger(__name__)


def _check_cuda_versions() -> None:
    """Warn when torch and TensorRT ship different CUDA major versions (two runtimes in one process)."""
    trt_major = None
    for dist in importlib.metadata.distributions():
        m = re.fullmatch(r"tensorrt[-_]cu(\d+)[-_]bindings", (dist.metadata["Name"] or "").lower())
        if m:
            trt_major = int(m.group(1))
    torch_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else None
    if trt_major and torch_major and trt_major != torch_major:
        log.warning(
            "torch is built for CUDA %d but TensorRT for CUDA %d; install `tensorrt-cu%d` to match",
            torch_major, trt_major, torch_major,
        )  # fmt: skip


class TorchBackend:
    name = "torch"

    def __init__(self, model: OWSMModel):
        self.model = model
        self.cfg = model.cfg

    def encode(self, feats: torch.Tensor, cross_k: torch.Tensor, cross_v: torch.Tensor) -> None:
        # feats: [B, T, n_mels] fp32; cross_k/v: [L, B, H, T', d_k]
        kv = self.model.decoder.cross_kv(self.model.encoder(feats))  # [L, 2, B, H, T', d_k]
        cross_k.copy_(kv[:, 0])
        cross_v.copy_(kv[:, 1])

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
        prefill: bool = False,
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


class TRTBackend:
    """Runs the encoder / decoder TensorRT engines on torch-owned device buffers."""

    name = "tensorrt"

    def __init__(self, cfg: OWSMConfig, engine_dir: Path):
        import tensorrt as trt

        _check_cuda_versions()
        self.trt = trt
        self.cfg = cfg
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.encoder = self._load(engine_dir / "encoder.engine")
        self.decoder = self._load(engine_dir / "decoder.engine")
        self.enc_ctx = self.encoder.create_execution_context()
        self.profiles = json.loads((engine_dir / "build.json").read_text())["decoder_profiles"]
        prefill = next(i for i, p in enumerate(self.profiles) if p["kind"] == "prefill")
        self.prefill_ctx = self._context(self.decoder, profile=prefill)

    def _decode_profile(self, batch: int) -> int:
        for i, p in enumerate(self.profiles):
            if p["kind"] == "decode" and p["batch"][0] <= batch <= p["batch"][2]:
                return i
        raise ValueError(f"batch size {batch} is not covered by the engine's profiles")

    def _load(self, path: Path):
        engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize {path}; rebuild it with `owsm-trt build`")
        return engine

    def _context(self, engine, profile: int):
        ctx = engine.create_execution_context()
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

    def encode(self, feats: torch.Tensor, cross_k: torch.Tensor, cross_v: torch.Tensor) -> None:
        ctx = self.enc_ctx
        self._bind(ctx, "feats", feats.contiguous(), True)
        self._bind(ctx, "cross_k", cross_k, False)
        self._bind(ctx, "cross_v", cross_v, False)
        if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("encoder engine execution failed")

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
        prefill: bool = False,
    ) -> Callable[[], None]:
        # Decode steps get their own context so a captured CUDA graph never sees shape changes.
        ctx = (
            self.prefill_ctx
            if prefill
            else self._context(self.decoder, self._decode_profile(tokens.shape[0]))
        )
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
            if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError("decoder engine execution failed")

        run.ctx = ctx  # keep the context alive as long as the closure
        return run


def cache_buckets(capacity: int, smallest: int = 64) -> list[int]:
    """KV-cache lengths a session switches between: 64, 128, ... up to ``capacity``."""
    out, c = [], smallest
    while c < capacity:
        out.append(c)
        c *= 2
    return out + [capacity]


class GreedySession:
    """Static buffers + captured CUDA graphs for greedy decoding at a fixed batch size.

    Self-attention reads the whole KV cache every step, so the cache comes in length buckets
    (64, 128, ..., capacity), each with its own buffers and CUDA graph. A sequence starts in the
    smallest bucket that fits and is copied into the next one when it outgrows it.

    Buffers (H = decoder heads, c = bucket length, T = encoder frames):
      caches[c]       2 x [L, B, H, c, d_k]  KV cache, slot t holds position t
      cross_k/cross_v [L, B, H, T, d_k]
      seq             [B, C + 1]             every token fed or generated so far
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
        chunk: int = 4,  # steps between host checks for <eos>; larger wastes steps after the last <eos>
        buckets: list[int] | None = None,
    ):
        self.backend, self.cfg, self.batch, self.capacity = backend, cfg, batch, capacity
        self.chunk = chunk
        self.buckets = buckets or cache_buckets(capacity)
        L, H, dk = cfg.dec_layers, cfg.dec_heads, cfg.d_model // cfg.dec_heads
        kw = {"device": device, "dtype": dtype}
        # Zero-init: masked slots still flow through softmax-weighted sums, so they must be finite.
        self.caches = {
            c: (torch.zeros(L, batch, H, c, dk, **kw), torch.zeros(L, batch, H, c, dk, **kw))
            for c in self.buckets
        }
        self.cross_k = torch.zeros(L, batch, H, cfg.enc_frames, dk, **kw)
        self.cross_v = torch.zeros(L, batch, H, cfg.enc_frames, dk, **kw)
        self.logits = torch.zeros(batch, cfg.vocab_size, device=device)
        self.new_kv = torch.zeros(L, 2, batch, H, 1, dk, **kw)
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.seq = torch.zeros(batch, capacity + 1, dtype=torch.long, device=device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.eos = torch.full((batch,), eos, dtype=torch.long, device=device)

        self._steps = {
            c: backend.step_fn(self.tok, self.pos, k, v, self.cross_k, self.cross_v, self.logits, self.new_kv)
            for c, (k, v) in self.caches.items()
        }
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        if use_cuda_graph:
            for c in self.buckets:
                self.graphs[c] = self._capture(c)

    @property
    def self_k(self) -> torch.Tensor:
        return self.caches[self.buckets[-1]][0]

    @property
    def self_v(self) -> torch.Tensor:
        return self.caches[self.buckets[-1]][1]

    def _body(self, c: int) -> None:
        k, v = self.caches[c]
        self._steps[c]()
        k.index_copy_(3, self.pos, self.new_kv[:, 0])
        v.index_copy_(3, self.pos, self.new_kv[:, 1])
        nxt = torch.where(self.done, self.eos, self.logits.argmax(-1))  # [B]
        self.done |= nxt == self.eos
        self.seq.index_copy_(1, self.pos + 1, nxt.unsqueeze(1))
        self.tok.copy_(nxt.unsqueeze(1))
        self.pos += 1

    def _capture(self, c: int) -> torch.cuda.CUDAGraph:
        self.pos.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # warm up (TRT shape setup, allocator) before capture
                self._body(c)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._body(c)
        return graph

    def _bucket(self, slots: int) -> int:
        """Smallest bucket with at least ``slots`` slots."""
        return next(c for c in self.buckets if c >= slots)

    def encode(self, feats: torch.Tensor) -> None:
        self.backend.encode(feats, self.cross_k, self.cross_v)

    def decode(self, primer: torch.Tensor, max_new_tokens: int) -> list[list[int]]:
        """Greedy decode after ``primer`` [B, P]; returns generated ids per row (without <eos>)."""
        b, p = primer.shape
        assert b == self.batch
        max_new_tokens = min(max_new_tokens, self.capacity - p)
        if max_new_tokens <= 0:
            raise ValueError(f"prompt of {p} tokens does not fit the cache capacity {self.capacity}")

        # Prefill the primer in one call, then append its keys/values at slots 0..P-1.
        c = self._bucket(p + 1)
        k, v = self.caches[c]
        self.pos.zero_()
        new_kv = torch.empty(self.new_kv.shape[:4] + (p, self.new_kv.shape[-1]), **_like(self.new_kv))
        self.backend.step_fn(
            primer, self.pos, k, v, self.cross_k, self.cross_v, self.logits, new_kv, prefill=True
        )()
        k[:, :, :, :p] = new_kv[:, 0]
        v[:, :, :, :p] = new_kv[:, 1]
        nxt = self.logits.argmax(-1)
        self.seq[:, :p] = primer
        self.seq[:, p] = nxt
        self.tok.copy_(nxt.unsqueeze(1))
        self.done.copy_(nxt == self.eos)
        self.pos.fill_(p)

        pos, steps = p, 1  # host mirror of self.pos; tokens generated so far
        while steps < max_new_tokens:
            if pos >= c:  # the next step writes slot `pos`: move to a bigger bucket
                bigger = self._bucket(pos + 1)
                for old, new in zip(self.caches[c], self.caches[bigger]):
                    new[:, :, :, :c] = old
                c = bigger
            n = min(self.chunk, max_new_tokens - steps, c - pos)
            for _ in range(n):
                self.graphs[c].replay() if self.graphs else self._body(c)
            steps += n
            pos += n
            if bool(self.done.all()):
                break

        eos = int(self.eos[0])
        out = []
        for row in self.seq[:, p : p + steps].tolist():
            out.append(row[: row.index(eos)] if eos in row else row)
        return out


def _like(t: torch.Tensor) -> dict:
    return {"dtype": t.dtype, "device": t.device}
