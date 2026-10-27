"""High-level API: ``OWSM.from_pretrained(...).transcribe(...)``."""

import dataclasses
import hashlib
import itertools
import logging
import math
import os
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from . import fp8 as fp8_mod
from . import hub, megakernel
from .audio import AudioLike, duration, load_audio
from .audio_prefetch import prefetch_batches
from .build import DTYPES, BuildOptions, build_engines, buildable, check_config, engine_tag
from .config import load_espnet_config, parse_config
from .modeling import (
    Decoder,
    Frontend,
    OWSMCTCModel,
    OWSMModel,
    convert_espnet_state_dict,
    frames_needed,
    load_checkpoint,
    load_model,
    sinusoidal_positions,
)
from .runtime import (
    BeamSession,
    CTCSession,
    GreedySession,
    TimestampRules,
    TorchBackend,
    TorchCTCBackend,
    TRTBackend,
    TRTCTCBackend,
    ctc_buffers,
)
from .tokenizer import Tokenizer

log = logging.getLogger(__name__)

# Window sizes (drafted tokens + 1) of CTC-drafted greedy decoding; one CUDA graph each per cache bucket.
DRAFT_WINDOWS = (1, 2, 4, 8, 16, 32)
# At most this many tokens (rows x window) per decoder call: wider calls cost more than they save
# (batch 16: 3.7 ms with windows of 16, 5.0 ms with 32, on the 1B model).
DRAFT_TOKENS = 256
# Feature lengths a trimming encoder engine gets (multiples of this, 1 s): one CUDA graph each. The
# subsampling convolutions run on all of them, so a finer step saves work (500 -> 100: batch 16 +2-3%
# RTFx on the 73 LibriSpeech utterances, OWSM v4 1B and OWSM-CTC v4) for more CUDA graphs.
FEATURE_STEP = 100
# OWSM-CTC in batches of at most CTC_LANE_BATCH: this many encoder calls on the device at a time (on
# streams of their own; the same kernels, so the same outputs). A batch-1 call leaves most of the GPU idle:
# two in flight, +20% throughput (73 LibriSpeech utterances, H200).
CTC_LANES = 2
CTC_LANE_BATCH = 1
# transcribe_long: drafts with guessed timestamps (see OWSM._long_window)
TIMESTAMP_GUESSES = True
# transcribe_long with batch_size > 1: a window is cut at the quietest point of its last this many seconds
PAUSE_SEARCH_SECONDS = 15.0
# transcribe_long with batch_size > 1 (ESPnet's windows): the windows of all recordings in one stream,
# a row taking the next window of any recording as soon as it is free (False: round by round)
LONG_STREAM = True


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Result:
    text: str
    tokens: list[int] = field(repr=False)
    segments: list[Segment] = field(default_factory=list)


def ctc_collapse(best_path: list[int], blank: int) -> list[int]:
    """CTC best path -> tokens: merge repeats, drop blanks."""
    out, prev = [], None
    for i in best_path:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def ctc_collapse_rows(best_paths: torch.Tensor, blank: int) -> list[list[int]]:
    """:func:`ctc_collapse` of every row of ``best_paths`` [B, T] (vectorized: drafts are made while the
    decoder waits)."""
    a = best_paths.cpu().numpy()
    keep = a != blank
    keep[:, 1:] &= a[:, 1:] != a[:, :-1]
    return [row[k].tolist() for row, k in zip(a, keep)]


def _window_cut(ids: list[int], tok: Tokenizer, end_id: int) -> tuple[list[int], int | None]:
    """How ESPnet's ``decode_long`` reads one window's output: (the tokens up to the last utterance it
    keeps, the timestamp the next window starts at; None if there are no timestamps).

    An utterance without an end time, or (when there are others) ending after ``end_id``, is cut off
    by the window: it is dropped, and the next window starts where it began."""
    times = [i for i, x in enumerate(ids) if tok.is_time(x)]
    if len(times) == 1:  # only a start time
        ids = ids + [tok.last_time]
        times.append(len(ids) - 1)
    if not times:
        return ids, None
    if len(times) % 2 == 0:
        if len(times) > 2 and ids[times[-1]] > end_id:  # last utterance is incomplete
            next_start = ids[times[-2]]
            times = times[:-2]
        else:
            next_start = ids[times[-1]]
    else:  # last utterance has only a start time
        next_start = ids[times[-1]]
        times = times[:-1]
    return (ids[: times[-1] + 1] if times else []), next_start


def pause_cuts(
    wav: torch.Tensor,
    max_len: int,
    sample_rate: int,
    search_seconds: float = 15.0,
    smooth_seconds: float = 0.2,
) -> list[int]:
    """Sample positions ``[0, ..., len(wav)]`` that cut ``wav`` into pieces of at most ``max_len``
    samples: each cut is at the quietest point (mean energy over ``smooth_seconds``) of the last
    ``search_seconds`` (at most half) of its piece."""
    hop = sample_rate // 100  # 10 ms frames
    n = len(wav) // hop
    if len(wav) <= max_len or n == 0:
        return [0, len(wav)]
    energy = wav[: n * hop].float().reshape(n, hop).pow(2).mean(1).double().cpu()
    k = max(1, round(smooth_seconds * 100))
    total = F.pad(energy.cumsum(0), (1, 0))
    lo = (torch.arange(n) - k // 2).clamp(0, n - 1)
    hi = (lo + k).clamp(max=n)
    level = (total[hi] - total[lo]) / (hi - lo)  # mean energy around each frame
    cuts, start = [0], 0
    while len(wav) - start > max_len:
        a = (start + max_len - min(int(search_seconds * sample_rate), max_len // 2)) // hop + 1
        b = (start + max_len - hop // 2) // hop + 1  # frame i cuts at i * hop + hop // 2 <= start + max_len
        a = min(max(a, start // hop + 1), b - 1)
        seg = level[a:b]
        i = a + int(seg.numel() - 1 - torch.argmin(seg.flip(0)))  # the latest of equally quiet frames
        start = i * hop + hop // 2
        cuts.append(start)
    cuts.append(len(wav))
    return cuts


@dataclass
class _Recording:
    """transcribe_long's progress through one recording."""

    wav: torch.Tensor | None
    offset: int = 0  # first sample of the next window
    prev: str | None = None  # text of the previous window (condition_on_prev_text)
    segments: list[Segment] = field(default_factory=list)


def _chain(first: deque, rest: Iterator):
    """The items of ``first`` (taken from the front as they are reached), then those of ``rest``."""
    while first:
        yield first.popleft()
    yield from rest


def _wait(host: torch.Tensor, done: torch.cuda.Event | None) -> torch.Tensor:
    if done is not None:
        done.synchronize()
    return host


def _joined(segments: list[Segment]) -> Result:
    return Result(
        text=" ".join(s.text.strip() for s in segments if s.text.strip()), tokens=[], segments=segments
    )


class OWSM:
    """OWSM speech recognition / translation with a TensorRT (or PyTorch) backend.

    Works with the encoder-decoder models (OWSM v1-v4, OWLS, POWSM) and the CTC-only OWSM-CTC models.

    >>> model = OWSM.from_pretrained("espnet/owsm_v4_medium_1B")
    >>> model.transcribe("speech.wav", lang="eng").text
    """

    def __init__(
        self,
        cfg,
        tokenizer: Tokenizer,
        frontend: Frontend,
        backend,
        opts: BuildOptions,
        device: torch.device,
        use_cuda_graph: bool = True,
        ctc_draft: bool = True,
        beam_upper_bound: bool = True,
        beam_kernels: bool = True,
        beam_compact_tail: bool = False,
        beam_grouped_cross: bool = False,
        frontend_graph: bool = True,
        lazy_drafts: bool = True,
        draft_token_budget: int = DRAFT_TOKENS,
        greedy_kernels: bool = True,
        encoder_lookahead: bool = False,
        frontend_graph_max_batch: int = 1,
        audio_prefetch_workers: int = 0,
        beam_fused_lse: bool = False,
    ):
        if type(beam_fused_lse) is not bool:
            raise ValueError("beam_fused_lse must be a bool")
        if type(audio_prefetch_workers) is not int or audio_prefetch_workers not in (0, 1, 2):
            raise ValueError("audio_prefetch_workers must be 0, 1 or 2")
        if type(frontend_graph_max_batch) is not int or not 1 <= frontend_graph_max_batch <= 8:
            raise ValueError("frontend_graph_max_batch must be an integer between 1 and 8")
        if type(draft_token_budget) is not int or draft_token_budget < 1:
            raise ValueError("draft_token_budget must be a positive integer")
        self.draft_token_budget = draft_token_budget
        self.cfg, self.tokenizer, self.frontend, self.backend = cfg, tokenizer, frontend, backend
        self.opts, self.device, self.use_cuda_graph = opts, device, use_cuda_graph
        self.ctc_draft = ctc_draft
        self.lazy_drafts = lazy_drafts
        self.beam_upper_bound = beam_upper_bound
        self.beam_kernels = beam_kernels
        self.beam_fused_lse = beam_fused_lse
        self.beam_compact_tail = beam_compact_tail
        self.beam_grouped_cross = beam_grouped_cross
        self._beam_backend = None
        self._encoder_lookahead = encoder_lookahead  # Staging resources are allocated only on eligible calls.
        self.frontend_graph = frontend_graph
        self._frontend_graph_batch_limit = frontend_graph_max_batch
        self._frontend_reuse_warmup_stream = True
        self._audio_prefetch_workers = audio_prefetch_workers
        self.greedy_skip_finished = greedy_kernels
        self.greedy_plain_solo = greedy_kernels
        self.greedy_fused_select = greedy_kernels
        self.greedy_split_choose = greedy_kernels
        self.greedy_fuse_accept = greedy_kernels
        self.greedy_reuse_notime = False
        self.greedy_cache_draft_flags = greedy_kernels
        self.greedy_lazy_verify_graphs = lazy_drafts
        self.greedy_lazy_plain = lazy_drafts
        self._frontend_replay = None
        self.draft_stats = [0, 0]  # tokens emitted, decoder calls made by drafted decoding
        self.stream_priority = True  # decoder calls go first while the encoder works ahead
        self.dtype = DTYPES[opts.dtype]
        t = tokenizer
        self.rules = (
            TimestampRules(t.notime, t.first_time, t.last_time, t.eos, cfg.vocab_size, device)
            if not cfg.ctc_only and t.first_time is not None and t.notime is not None
            else None
        )
        self._sessions: OrderedDict[tuple, GreedySession | BeamSession | CTCSession] = OrderedDict()
        self._lane_streams: list = []  # streams of the CTC lanes (_ctc_stream)
        self._ctc_buffers: dict[int, tuple] = {}  # per lane: buffers its CTC sessions share

    @classmethod
    def from_pretrained(
        cls,
        model: str = "espnet/owsm_v4_medium_1B",
        *,
        backend: str = "tensorrt",
        device: str = "cuda",
        dtype: str = "float16",
        max_batch_size: int = 16,
        capacity: int = 512,
        max_prompt_tokens: int = 256,
        use_cuda_graph: bool = True,
        rebuild: bool = False,
        optimization_level: int | None = None,
        ctc_draft: bool = True,
        fp8_calibration: list[AudioLike] | None = None,
        decoder_kernel: bool | str | None = None,
        beam_upper_bound: bool = True,
        beam_kernels: bool = True,
        beam_compact_tail: bool = False,
        beam_grouped_cross: bool = False,
        frontend_graph: bool = True,
        lazy_drafts: bool = True,
        draft_token_budget: int = DRAFT_TOKENS,
        greedy_kernels: bool = True,
        encoder_lookahead: bool = False,
        frontend_graph_max_batch: int = 1,
        fp8: bool | str = False,
        fp8_margin: float = 1.0,
        fp8_skip: str = "",
        audio_prefetch_workers: int = 0,
        beam_fused_lse: bool = False,
    ) -> "OWSM":
        """Load an OWSM model from a Hugging Face repo id or local ESPnet model directory.

        With ``backend="tensorrt"`` engines are built on first use (a few minutes) and cached under
        ``~/.cache/owsm_trt`` (override with ``OWSM_TRT_CACHE``). ``backend="torch"`` skips TensorRT.

        Cached-engine frontend/decoder loading privately maps supported ZIP checkpoints, reading
        only the required tensors. Legacy/file-like checkpoints or shared-mapping configurations
        use ordinary loading. No tensor precision changes; engine builds and full Torch model
        loading keep their existing path. Decoder megakernel setup skips unused random parameter
        initialization before strict copying; norms/position buffers remain initialized. Measured
        startup improvements cover H200, v4 1B/102M,
        node-local files and warmed OS/JIT caches.

        ``decoder_kernel="ring:wide"``: experimental N route for H200132SM with the102M decoder.
        Only R10/15/16 one-token decode calls use Ring with60/90/96 programs; prefill, other
        rows/windows and unsupported GPU/model combinations retain auto routing. Kernel setup
        failures fall back with a warning. Same-engine measurements show throughput gains but
        punctuation, translation and silence outputs can differ. The default remains ``auto``.

        ``ctc_draft``: for greedy ASR, let the encoder's CTC head propose the transcript and check up
        to 31 proposed tokens per utterance and decoder call. The output is the greedy output; only the
        number of decoder calls changes.

        ``frontend_graph``: reuse CUDA graphs for single-utterance STFT/mel/normalization (default
        True with ``use_cuda_graph``). A bounded 32-shape cache retains private graph buffers; each
        returned feature tensor has its own storage. ``frontend_graph_max_batch`` (integer1..8, default1)
        permits wider captured batches as an opt-in. Each cache owns an exclusive capture/warm-up stream.
        Batches above the selected limit, CPU, autocast and cache saturation run
        eagerly. False restores eager frontend execution. Exact features/output verified on H200
        with v4 medium 1B/base 102M; other GPUs/checkpoints are unmeasured.

        ``audio_prefetch_workers``: 0 (default),1 or2 CPU file readers for greedy ASR streaming
        and sequential fixed batches (beam, translation or long prompts). At most two audio batches
        are submitted; feature/GPU work stays on the caller thread. Only all-file inputs are prefetched; mixed/waveform inputs and CPU
        autocast load synchronously. Worker setup/join is per request. More workers can regress
        small/cheap inputs, so select using file-inclusive measurements. Single batches, encoder
        lookahead, CTC-only and long-form paths are unchanged. Greedy streaming is verified on
        H200/1B/102M; fixed batches also pass public/GPU checks and independent timing replication.
        See docs/runtime-audio-prefetch.md and docs/runtime-fixed-prefetch.md for measured limits.

        ``beam_fused_lse`` (default False): fuse full-vocabulary beam normalization with candidate
        scoring when CUDA beam kernels are available and enabled. This changes fp32 reduction order
        and can change near-tied token/timestamp choices. Greedy and unsupported paths keep their
        existing normalization. See docs/beam-lse.md for same-engine measurements and quality checks.

        ``greedy_kernels`` (default True): split draft-window vocabulary selection across blocks and
        fuse the final argmax with draft acceptance, reuse short-primer flags for fixed-batch
        drafts, and fuse mask/argmax/state updates in plain greedy and translation. Streaming and long-prefill initialization stay unchanged. The same timestamp rules, smallest-index ties
        and decoder scores apply. Serial plain B1 calls also use the backend's existing solo budget
        when encoder lookahead is disabled; drafted verification keeps its own overlap scheduling.
        False restores reference selection, initialization and plain budget; CPU, missing Triton and small
        vocabularies keep the reference path. Exact sampled output and latency checks cover H200,
        v4 medium 1B/base 102M, English/Japanese. Other GPUs are unmeasured. Beam decoding is independent.

        ``lazy_drafts`` (default True): create verification buffers on first draft use, then prepare
        each bucket/window/solo decoder call and graph only when needed. Translation and language
        identification avoid unused verification setup. An unseen shape still incurs cold setup;
        warmed decoder calls are unchanged. Plain decoder calls/graphs are also deferred until a
        request needs more than the first prefill selection (language ID needs none). False eagerly
        prepares both plain and verification calls/graphs. This reduces session-cold setup, not
        steady-state decoder time; validated on H200 with v4 medium 1B/base 102M.

        ``draft_token_budget`` (default 256): maximum rows x verification window for greedy CTC
        drafts (at least one token per row). 128 improved H200/1B timestamped B8 throughput in the
        measured English/Japanese corpora. Shape/kernel changes can change fp16 outputs; this is
        an opt-in numerical tradeoff, not a beam parameter. Smaller is not always faster. A primer
        too long for the resulting window uses the existing fixed-batch/prefill path.

        ``beam_upper_bound``: stop beam search when the best finished score strictly exceeds every
        live score (default True). Existing stopping rules still apply; greedy decoding is unaffected.
        False disables this bound. Bound-enabled batch 1 checks completion every step, other beam
        batches every four steps; see ``BeamSession`` for an explicit interval override.

        ``beam_kernels``: fuse beam candidate masking/scoring, parent/history/timestamp and
        finished-state updates, and valid-prefix KV reordering with Triton; share cross KV across
        beam rows when the decoder supports it
        (default True on CUDA when Triton is available). Full-vocabulary log-softmax and global top-K
        stay in PyTorch. False selects the reference operations; CPU and missing-Triton paths also
        use them. These kernels do not affect greedy decoding.

        ``encoder_lookahead`` (opt-in, default False): for multiple fixed batches, prepare the next
        encoder batch on a separate stream while the current beam/plain batch decodes. Batch shapes,
        sorting and tokens stay unchanged in the measured tests. Existing drafted ASR streaming,
        single-batch calls, CPU and CTC-only paths keep their execution. Extra staging buffers and
        encoder captures cost memory/cold time; warm throughput measurements cover H200 v4 1B/102M.

        ``beam_compact_tail`` (experimental, default False): once only one utterance remains in a
        beam batch, decode its rows only when a shared-cross-KV decoder supports that shape.
        Shape/kernel changes may change tokens. Additional group/bucket graphs increase cold start.

        ``beam_grouped_cross`` (experimental, default False): for K5 with ``beam_kernels=True``,
        group cross-attention queries on an available Gluon decoder. The measured H200/1B preset
        permits10/15 rows with4/2 key slices; five rows and prefill retain8. Other model/GPU shapes
        keep the existing routing limits. This changes reduction order and can change tokens
        (observed on silence and two English segment ends). Greedy keeps the original decoder.
        Measurements currently cover H200/1B only; the option remains off by default.

        ``fp8_calibration`` (experimental, TensorRT): run the encoder's linear layers and subsampling
        convolutions in FP8, with input ranges measured on these recordings.

        ``fp8`` (experimental, TensorRT): the same with the input ranges of a scale file: True takes the one
        the package ships for this checkpoint (``owsm_trt/fp8_scales``, matched by the checkpoint's sha256;
        without one, a warning and an fp16 encoder), a path that file. ``fp8_margin`` multiplies the ranges,
        ``fp8_skip`` (a regular expression on the encoder's layer names) keeps the layers it matches fp16.

        ``decoder_kernel`` (TensorRT, fp16): run small decoder calls as one persistent Gluon kernel per call
        instead of the TensorRT decoder engine (Hopper GPUs; larger calls and prompt prefill stay on TensorRT).
        ``"ring"``: single-row calls of up to 32 tokens on the ring kernel (32 SMs while decode_stream's encoder
        runs alongside, 64 otherwise), calls of up to 16 tokens and 8 rows on the Gluon kernel; ``"auto"``: as
        ``"ring"`` on a Hopper GPU when the kernels build and match the decoder on a test call, TensorRT
        otherwise; True / ``"1"``: the Gluon kernel only; ``"triton"``: every decode/verify call as the
        (slower) Triton megakernel (Ampere or newer); False / ``"0"``: TensorRT only. Kernels that fail leave
        TensorRT (logged). ``"ring:rr"`` uses Ring also for up to8 rows/32 total tokens. On H200/102M,
        its R5..8 schedule uses one CTA per attention head with unchanged split count; sampled
        outputs match the old Ring schedule, but Ring can differ numerically from TensorRT.
        None: the ``OWSM_TRT_MEGAKERNEL`` environment variable (auto / ring / 1 / triton /
        0), ``"auto"`` if unset.
        """
        if type(beam_fused_lse) is not bool:
            raise ValueError("beam_fused_lse must be a bool")
        if type(audio_prefetch_workers) is not int or audio_prefetch_workers not in (0, 1, 2):
            raise ValueError("audio_prefetch_workers must be 0, 1 or 2")
        if type(frontend_graph_max_batch) is not int or not 1 <= frontend_graph_max_batch <= 8:
            raise ValueError("frontend_graph_max_batch must be an integer between 1 and 8")
        if type(draft_token_budget) is not int or draft_token_budget < 1:
            raise ValueError("draft_token_budget must be a positive integer")
        files = hub.resolve(model)
        raw = load_espnet_config(files.config)
        cfg = parse_config(raw)
        tok = Tokenizer(
            raw["token_list"],
            hub.resolve_relative(files, raw["bpemodel"]),
            raw.get("preprocessor_conf", {}).get("notime_symbol", "<notimestamps>"),
        )
        levels = {}  # builder optimization levels: BuildOptions' defaults unless one is given for both
        if optimization_level is not None:
            levels = {"opt_level": optimization_level, "verify_opt_level": optimization_level}
        calibration, scales, digest = None, None, ""
        if fp8_calibration is not None:
            calibration = [load_audio(a, cfg.sample_rate) for a in fp8_calibration]
            digest = hashlib.sha1(b"".join(w.numpy().tobytes() for w in calibration)).hexdigest()[:12]
        elif fp8:
            scales = _fp8_scales(files, fp8)
            if scales is not None:
                digest = "s" + fp8_mod.scales_digest(scales)
        if digest and (fp8_margin != 1.0 or fp8_skip):  # (other ranges, other engines)
            digest += f"-m{fp8_margin:g}" + (
                "-k" + hashlib.sha1(fp8_skip.encode()).hexdigest()[:8] if fp8_skip else ""
            )
        opts = BuildOptions(
            dtype=dtype,
            max_batch=max_batch_size,
            capacity=capacity,
            max_prefill=max_prompt_tokens,
            **levels,
            fp8_encoder=bool(digest),
            fp8_calibration=digest,
        )
        if digest and backend != "tensorrt":
            raise ValueError("FP8 is available for the TensorRT backend only")
        dev = torch.device(device)
        if backend == "tensorrt" and dev.type != "cuda":
            raise ValueError("the TensorRT backend needs a CUDA device")

        if backend == "torch":
            net = load_model(cfg, str(files.checkpoint), dev, DTYPES[dtype])
            graphs = use_cuda_graph and dev.type == "cuda"
            runner = TorchCTCBackend(net) if cfg.ctc_only else TorchBackend(net)
            return cls(cfg, tok, net.frontend, runner, opts, dev, graphs, ctc_draft,
                       beam_upper_bound, beam_kernels, beam_compact_tail, beam_grouped_cross,
                       frontend_graph=frontend_graph, lazy_drafts=lazy_drafts, draft_token_budget=draft_token_budget,
                       greedy_kernels=greedy_kernels, encoder_lookahead=encoder_lookahead,
                       frontend_graph_max_batch=frontend_graph_max_batch, audio_prefetch_workers=audio_prefetch_workers,
                       beam_fused_lse=beam_fused_lse)
        if backend != "tensorrt":
            raise ValueError(f"unknown backend {backend!r} (use 'tensorrt' or 'torch')")

        with torch.cuda.device(dev):
            opts = buildable(opts, cfg)
            engine_dir = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(opts)
            if rebuild or not check_config(cfg, engine_dir):
                log.warning("building TensorRT engines for %s into %s", files.name, engine_dir)
                net = load_model(cfg, str(files.checkpoint), dev, DTYPES[dtype])
                if calibration is not None:
                    _quantize_encoder(net, tok, calibration, opts.max_batch, fp8_margin, fp8_skip)
                elif scales is not None:
                    amax = fp8_mod.load_scales(scales)["amax"]
                    n = fp8_mod.quantize(net.encoder, amax, fp8_margin, fp8_skip)
                    log.warning("FP8 encoder: %d layers, input ranges from %s", n, scales)
                decoders = None
                if opts.fp8_encoder and not cfg.ctc_only:  # an fp16 build's decoder engines are the same
                    plain = (
                        hub.cache_dir()
                        / files.name.replace("/", "--")
                        / engine_tag(dataclasses.replace(opts, fp8_encoder=False, fp8_calibration=""))
                    )
                    if check_config(cfg, plain) and all(
                        (plain / f"{e}.engine").exists() for e in ("decoder", "verify")
                    ):
                        decoders = plain
                build_engines(net, engine_dir, opts, decoders_from=decoders)
                frontend = net.frontend
                del net
                torch.cuda.empty_cache()
            else:
                frontend = _load_frontend(cfg, files.checkpoint, dev)
            runner = TRTCTCBackend(cfg, engine_dir) if cfg.ctc_only else TRTBackend(cfg, engine_dir)
            if decoder_kernel is None:
                decoder_kernel = os.environ.get("OWSM_TRT_MEGAKERNEL", "auto")
            if decoder_kernel and decoder_kernel != "0" and not cfg.ctc_only and dtype == "float16":
                runner = _decoder_kernel(runner, cfg, files.checkpoint, dev, decoder_kernel)
            return cls(cfg, tok, frontend, runner, opts, dev, use_cuda_graph, ctc_draft,
                       beam_upper_bound, beam_kernels, beam_compact_tail, beam_grouped_cross,
                       frontend_graph=frontend_graph, lazy_drafts=lazy_drafts, draft_token_budget=draft_token_budget,
                       greedy_kernels=greedy_kernels, encoder_lookahead=encoder_lookahead,
                       frontend_graph_max_batch=frontend_graph_max_batch, audio_prefetch_workers=audio_prefetch_workers,
                       beam_fused_lse=beam_fused_lse)

    # ------------------------------------------------------------------ helpers

    def _session(self, batch: int, beam: int = 1) -> GreedySession | BeamSession:
        key = (batch, beam)
        if key not in self._sessions:
            if len(self._sessions) >= 2:
                self._sessions.popitem(last=False)
            args = (self.opts.capacity, self.tokenizer.eos, self.dtype, self.device, self.use_cuda_graph)
            if beam == 1:
                windows = self._draft_windows(batch)
                t = self.tokenizer
                # Timestamps count as punctuation for drafts that guess them (transcribe_long): one the
                # decoder writes where the draft has none was inserted, one in place of a guess replaces
                # it. (Drafts without timestamps pass them as insertions, so nothing changes there.)
                times = frozenset(range(t.first_time, t.last_time + 1)) if t.first_time is not None else set()
                self._sessions[key] = GreedySession(
                    self.backend, self.cfg, batch, *args, draft_windows=windows, rules=self.rules,
                    punctuation=t.punctuation | times, lazy_drafts=self.lazy_drafts,
                    split_choose=self.greedy_split_choose, fuse_accept=self.greedy_fuse_accept,
                    reuse_notime=self.greedy_reuse_notime, cache_draft_flags=self.greedy_cache_draft_flags,
                    lazy_verify_graphs=self.greedy_lazy_verify_graphs, fused_select=self.greedy_fused_select,
                    lazy_plain=self.greedy_lazy_plain, skip_finished=self.greedy_skip_finished,
                    skip_finished_groups=self.greedy_skip_finished,
                    plain_solo=self.greedy_plain_solo and not self._encoder_lookahead,
                )  # fmt: skip
            else:
                backend = self.backend
                if self.beam_grouped_cross and self.beam_kernels and beam == 5:
                    if self._beam_backend is None:
                        self._beam_backend = _grouped_beam_backend(self.backend)
                    backend = self._beam_backend
                self._sessions[key] = BeamSession(
                    backend, self.cfg, batch, beam, *args, rules=self.rules,
                    upper_bound=self.beam_upper_bound,
                    fused_rank=self.beam_kernels, fused_finish=self.beam_kernels,
                    kv_reorder="prefix" if self.beam_kernels else "torch",
                    share_cross_kv=self.beam_kernels, fused_state=self.beam_kernels,
                    compact_tail=self.beam_compact_tail,
                    fused_lse=self.beam_fused_lse,
                )
        self._sessions.move_to_end(key)
        return self._sessions[key]

    def _draft_windows(self, batch: int) -> tuple[int, ...]:
        if not self.ctc_draft:
            return ()
        return tuple(
            w for w in DRAFT_WINDOWS if w <= self.opts.max_verify and w * batch <= max(self.draft_token_budget, batch)
        )

    def _check_beam(self, beam: int) -> int:
        if beam < 1 or beam > self.opts.max_batch:
            raise ValueError(f"beam_size must be between 1 and max_batch_size={self.opts.max_batch}")
        return beam

    def _batch_size(self, batch_size: int | None, beam: int) -> int:
        self._check_beam(beam)
        # a decoder call runs batch * beam rows, which the engines allow up to max_batch_size
        return max(1, min(batch_size or self.opts.max_batch, self.opts.max_batch // beam))

    def _features(self, waves: list[torch.Tensor]) -> torch.Tensor:
        n = self.cfg.num_samples
        for w in waves:
            if w.shape[0] > n:
                log.warning(
                    "input of %.1fs is trimmed to %.0fs; use transcribe_long() for long audio",
                    w.shape[0] / self.cfg.sample_rate,
                    self.cfg.speech_seconds,
                )
        frames = self.cfg.num_frames
        if getattr(self.backend, "trimmed", False):  # the encoder fills in the frames of padding only
            t = max(frames_needed(self.cfg, min(w.shape[0], n)) for w in waves)
            frames = min(frames, -(-t // FEATURE_STEP) * FEATURE_STEP)
        # The STFT (center=True) of frames 0 .. frames-1 reads samples up to (frames-1)·hop + n_fft/2 - 1:
        # the padding after them is not transformed (the same frames: the reflection at the end of the
        # shorter input is past them).
        length = min(n, (frames - 1) * self.cfg.hop_length + self.cfg.n_fft // 2 + 1)
        batch = torch.zeros(len(waves), length, device=self.device)  # pad on the device: copy only the audio
        for row, w in zip(batch, waves):
            w = w[:length]
            if batch.is_cuda and w.device.type == "cpu":
                w = w.pin_memory()  # a copy from pageable memory would first wait for the stream's work
            row[: w.shape[0]].copy_(w, non_blocking=True)
        if self.frontend_graph and self.use_cuda_graph and batch.is_cuda and len(waves) <= self._frontend_graph_batch_limit:
            if self._frontend_replay is None:
                from .frontend_graph import FrontendReplay
                self._frontend_replay = FrontendReplay(
                    self.frontend, max_batch=self._frontend_graph_batch_limit,
                    reuse_warmup_stream=self._frontend_reuse_warmup_stream,
                )
            return self._frontend_replay(batch)[:, :frames]
        return self.frontend(batch)[:, :frames]  # [B, frames, n_mels]

    def _primer(self, lang: str, task: str, timestamps: bool, prompt: str | None) -> list[int]:
        t = self.tokenizer
        primer = [t.sos, t.lang_id(lang), t.task_id(task)]
        if not timestamps:
            primer.append(t.notime)
        if prompt:
            primer = [t.sop] + t.encode(prompt) + primer
        return primer

    def _max_new(self) -> int:
        # ESPnet's default (maxlenratio=0) allows as many tokens as encoder frames.
        return self.cfg.enc_frames

    def _wave(self, audio: AudioLike, sr: int | None) -> torch.Tensor:
        return load_audio(audio, self.cfg.sample_rate, sr)

    # ------------------------------------------------------------------ public API

    @torch.inference_mode()
    def transcribe(
        self,
        audio: AudioLike | list[AudioLike],
        *,
        lang: str = "eng",
        task: str = "asr",
        timestamps: bool = False,
        prompt: str | None = None,
        sr: int | None = None,
        batch_size: int | None = None,
        beam_size: int = 1,
    ) -> Result | list[Result]:
        """Transcribe (``task="asr"``) or translate (``task="st_eng"`` etc.) up to 30 s of audio.

        ``audio`` is a path, a waveform (np.ndarray / torch.Tensor, sampled at ``sr`` or 16 kHz),
        or a list of those, which is decoded in batches of ``batch_size``.

        ``beam_size=1`` is greedy decoding; ``beam_size=5`` is ESPnet's default beam search. A batch
        holds at most ``max_batch_size // beam_size`` utterances.
        """
        single = not isinstance(audio, list)
        items = [audio] if single else audio
        if self.cfg.ctc_only:
            if timestamps or beam_size != 1:
                raise ValueError("CTC-only models have no timestamps or beam search; use the defaults")
            bs = min(batch_size or self.opts.max_batch, self.opts.max_batch)
            # batches of similar durations: a batch's features stop after its longest recording
            order = list(range(len(items)))
            if len(order) > 1:
                order.sort(key=lambda i: duration(items[i], sr or self.cfg.sample_rate))
            chunks = [order[i : i + bs] for i in range(0, len(order), bs)]
            batches = ([self._wave(items[j], sr) for j in chunk] for chunk in chunks)
            results: list[Result | None] = [None] * len(items)
            for chunk, paths in zip(chunks, self._ctc_stream(batches, lang, task, prompt)):
                for j, ids in zip(chunk, ctc_collapse_rows(paths, self.tokenizer.blank)):
                    results[j] = self._result(ids, offset=0.0)
            return results[0] if single else results
        bs = self._batch_size(batch_size, beam_size)
        primer = self._primer(lang, task, timestamps, prompt)
        stream = len(items) > bs and beam_size == 1 and task == "asr"
        if stream and len(primer) < max(self._draft_windows(bs), default=0):
            # Rows take the next utterance as soon as theirs is done, and the encoder works ahead. (A
            # single batch goes the direct way below: no staging buffers to copy from.)
            sess = self._session(bs)
            # similar durations in an encoder batch: its features stop after the longest recording
            order = sorted(range(len(items)), key=lambda i: duration(items[i], sr or self.cfg.sample_rate))
            if bs == 1:
                # longest, shortest, 2nd longest, ...: the encoder (a fixed cost per utterance) works ahead
                # while long utterances decode, rather than all the short ones (few calls) coming first
                # (same output, RTFx +1.5% on the 73 test utterances and +2-6% on test-clean)
                order = [order[-1 - i // 2] if i % 2 == 0 else order[i // 2] for i in range(len(order))]
            chunks = (order[i : i + bs] for i in range(0, len(order), bs))
            workers = self._audio_prefetch_workers
            if workers and not all(isinstance(x, (str, os.PathLike)) for x in items):
                workers = 0
            audio_batches = ([items[j] for j in chunk] for chunk in chunks)
            with prefetch_batches(audio_batches, lambda x: self._wave(x, sr), workers) as waves:
                batches = (self._features(batch) for batch in waves)
                outs, calls = sess.decode_stream(
                    primer, batches, self._ctc_drafts, self._max_new(), self.tokenizer.not_drafted,
                    priority=self.stream_priority,
                )  # fmt: skip
            self.draft_stats[0] += sum(len(o) + 1 for o in outs)
            self.draft_stats[1] += calls
            results = [None] * len(items)
            for j, ids in zip(order, outs):
                results[j] = self._result(ids, offset=0.0)
            return results[0] if single else results
        # A batch decodes until its longest member finishes, so batch utterances of similar duration.
        # One item needs no ordering/header read; the loader still validates and decodes it normally.
        order = list(range(len(items)))
        if len(order) > 1:
            order.sort(key=lambda i: duration(items[i], sr or self.cfg.sample_rate))
        results: list[Result | None] = [None] * len(items)
        if (self._encoder_lookahead and self.device.type == "cuda" and len(items) > bs
                and (beam_size > 1 or task != "asr" or not self.ctc_draft)):
            chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
            outs = self._decode_lookahead(items, chunks, primer, beam_size, sr)
            for j, ids in zip(order, outs):
                results[j] = self._result(ids, offset=0.0)
            return results
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
        workers = getattr(self, "_audio_prefetch_workers", 0)
        if workers and (len(chunks) < 2 or not all(isinstance(x, (str, os.PathLike)) for x in items)):
            workers = 0
        audio_batches = ([items[j] for j in idx] for idx in chunks)
        with prefetch_batches(audio_batches, lambda x: self._wave(x, sr), workers) as batches:
            for idx, waves in zip(chunks, batches):
                draft = task == "asr"  # the CTC head is trained on transcripts
                for j, ids in zip(idx, self._decode(waves, [primer] * len(idx), beam_size, draft)):
                    results[j] = self._result(ids, offset=0.0)
        return results[0] if single else results

    def _decode(
        self, waves: list[torch.Tensor], primers: list[list[int]], beam: int = 1, draft: bool = False
    ) -> list[list[int]]:
        sess = self._session(len(waves), beam)
        paths = sess.encode(self._features(waves))
        if draft and isinstance(sess, GreedySession) and sess.windows:
            proposals = self._ctc_drafts(paths)
            outs, calls = sess.decode_drafted(primers, proposals, self._max_new(), self.tokenizer.not_drafted)
            self.draft_stats[0] += sum(len(o) + 1 for o in outs)
            self.draft_stats[1] += calls
            return outs
        return sess.decode(torch.tensor(primers, device=self.device), self._max_new())

    def _decode_lookahead(self, items, chunks, primer, beam: int, sr: int | None) -> list[list[int]]:
        """Fixed batches, one encoder ahead. Batch/primer/decoder shapes match sequential decode."""
        sessions = {b: self._session(b, beam) for b in sorted({len(c) for c in chunks})}
        main = torch.cuda.current_stream(self.device)
        side = torch.cuda.Stream(device=self.device)
        dec = torch.cuda.Stream(device=self.device, priority=-1) if self.stream_priority else main
        side.wait_stream(main)
        dec.wait_stream(main)
        ready = [torch.cuda.Event() for _ in range(2)]
        copied = [torch.cuda.Event() for _ in range(2)]
        staged = [None, None]
        outs = []
        # Prepare lazy decoder graphs before any asynchronous encoder work. Encoder graphs have
        # their own capture stream/cuBLAS workspace; lane1 keeps TRT scratch isolated as well.
        with torch.cuda.stream(dec):
            for sess in sessions.values():
                if isinstance(sess, GreedySession):
                    sess._prepare_plain()

        def targets(sess):
            return ((sess.cross_kb, sess.cross_vb) if isinstance(sess, BeamSession)
                    else (sess.cross_k, sess.cross_v)) + (sess.ctc_ids,)

        def launch(index):
            slot = index % 2
            chunk = chunks[index]
            sess = sessions[len(chunk)]
            with torch.cuda.stream(side):
                side.wait_event(copied[slot])
                waves = [self._wave(items[j], sr) for j in chunk]
                feats = self._features(waves)
                if not hasattr(sess, "_lookahead_stage"):
                    sess._lookahead_stage = {}
                if slot not in sess._lookahead_stage:
                    sess._lookahead_stage[slot] = tuple(torch.empty_like(t) for t in targets(sess))
                bufs = sess._lookahead_stage[slot]
                sess._encoder(feats, *bufs, lane=1)
                staged[slot] = (sess, bufs)
                ready[slot].record(side)

        try:
            launch(0)
            for index in range(len(chunks)):
                slot = index % 2
                sess, bufs = staged[slot]
                with torch.cuda.stream(dec):
                    dec.wait_event(ready[slot])
                    # [L,B,H,T,D] remains utterance-sized; expand only where the decoder requires it.
                    for target, source in zip(targets(sess), bufs):
                        target.copy_(source)
                    if isinstance(sess, BeamSession):
                        sess._expand_cross()
                    copied[slot].record(dec)
                if index + 1 < len(chunks):
                    launch(index + 1)
                with torch.cuda.stream(dec):
                    tokens = torch.tensor([primer] * len(chunks[index]), device=self.device)
                    outs.extend(sess.decode(tokens, self._max_new()))
        finally:
            # Also join a prelaunched encoder on exceptions. Persistent staging owns all addresses
            # referenced by cached graphs; callers may safely reuse ordinary sessions afterwards.
            main.wait_stream(side)
            main.wait_stream(dec)
        return outs

    def _ctc_draft(self, best_path: list[int]) -> list[int]:
        return ctc_collapse(best_path, self.tokenizer.blank)

    def _ctc_drafts(self, best_paths: torch.Tensor) -> list[list[int]]:
        """Drafts from CTC best paths [B, t] (GreedySession.encode)."""
        return ctc_collapse_rows(best_paths, self.tokenizer.blank)

    def _ctc_paths(
        self, waves: list[torch.Tensor], lang: str, task: str, prompt: str | None, lane: int = 0
    ) -> torch.Tensor:
        """OWSM-CTC best paths [B, 2 + T'] (frame-wise argmax, the first two frames are the prefix).

        The tensor is a buffer shared by the CTC sessions of ``lane``, overwritten by their next call."""
        t = self.tokenizer
        prev = t.encode(prompt) if prompt else [t.na]
        if t.na in prev:
            prev = [t.na]
        b = len(waves)
        key = ("ctc", b, len(prev), lane)
        if key not in self._sessions:
            # CTC sessions are light (their buffers are shared): one per batch size and prompt length
            if len(self._sessions) >= 2 * self.opts.max_batch:
                self._sessions.popitem(last=False)
            if lane not in self._ctc_buffers:
                rows = self.opts.max_batch if lane == 0 else CTC_LANE_BATCH
                self._ctc_buffers[lane] = ctc_buffers(self.cfg, rows, self.device)
            self._sessions[key] = CTCSession(
                self.backend, self.cfg, b, len(prev), self.device, self.use_cuda_graph, self._ctc_buffers[lane],
                lane=lane,
            )  # fmt: skip
        self._sessions.move_to_end(key)
        sess = self._sessions[key]
        conditioning = (tuple(prev), lang, task)
        if sess.conditioning == conditioning:  # already in the session's buffers
            return sess(self._features(waves))
        sess.conditioning = conditioning
        return sess(
            self._features(waves),
            torch.tensor([prev] * b, device=self.device),
            torch.tensor([[t.lang_id(lang), t.task_id(task)]] * b, device=self.device),
        )

    def _ctc_stream(
        self, batches: Iterable[list[torch.Tensor]], lang: str, task: str, prompt: str | None
    ) -> Iterator[torch.Tensor]:
        """:meth:`_ctc_paths` of each batch of waveforms, as tensors on the host, in order. Each batch is
        queued on the device before the host waits for the best paths of the one before (and makes their
        transcripts, while the device works)."""
        pending: deque = deque()
        streams: list | None = None
        batches = iter(batches)
        ahead = deque(itertools.islice(batches, 2))
        for i, waves in enumerate(_chain(ahead, batches)):
            if self.device.type != "cuda":
                yield self._ctc_paths(waves, lang, task, prompt).clone()
                continue
            if streams is None:  # batches of one, more than one: CTC_LANES of them on the device at a time
                lanes = CTC_LANES if len(waves) <= CTC_LANE_BATCH and ahead else 1
                main = torch.cuda.current_stream()
                if lanes > 1 and len(self._lane_streams) < lanes:
                    self._lane_streams = [torch.cuda.Stream() for _ in range(lanes)]
                streams = [main] if lanes == 1 else self._lane_streams[:lanes]
            lane = i % len(streams) if len(waves) <= CTC_LANE_BATCH else 0
            stream = streams[lane]
            if stream is not main:
                stream.wait_stream(main)  # (whatever the caller queued before)
            with torch.cuda.stream(stream):  # features, encoder and the copy on the lane's stream
                paths = self._ctc_paths(waves, lang, task, prompt, lane if len(streams) > 1 else 0)
                host = torch.empty(paths.shape, dtype=paths.dtype, pin_memory=True)
                host.copy_(paths, non_blocking=True)
                done = torch.cuda.Event()
                done.record()
            pending.append((host, done))
            if len(pending) > len(streams):
                yield _wait(*pending.popleft())
        while pending:
            yield _wait(*pending.popleft())
        for stream in streams or []:
            torch.cuda.current_stream().wait_stream(stream)

    def _result(self, ids: list[int], offset: float) -> Result:
        t = self.tokenizer
        segments = []
        times = [i for i, x in enumerate(ids) if t.is_time(x)]
        for a, b in zip(times[::2], times[1::2]):
            segments.append(
                Segment(
                    start=round(offset + (ids[a] - t.first_time) * self.cfg.time_resolution, 2),
                    end=round(offset + (ids[b] - t.first_time) * self.cfg.time_resolution, 2),
                    text=t.decode(ids[a + 1 : b]),
                )
            )
        return Result(text=t.decode(ids), tokens=ids, segments=segments)

    @torch.inference_mode()
    def detect_language(
        self, audio: AudioLike, *, sr: int | None = None, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Most likely spoken languages with probabilities, from the first decoder step."""
        if self.cfg.ctc_only:
            raise NotImplementedError("language identification needs a decoder; OWSM-CTC has none")
        sess = self._session(1)
        sess.encode(self._features([self._wave(audio, sr)]))
        t = self.tokenizer
        sess.decode(torch.tensor([[t.sos]], device=self.device), max_new_tokens=1)
        names = list(t.languages)
        ids = torch.tensor([t.languages[n] for n in names], device=self.device)
        probs = sess.logits[0, -1, ids].softmax(-1)
        top = probs.topk(min(top_k, len(names)))
        return [(names[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]

    @torch.inference_mode()
    def transcribe_long(
        self,
        audio: AudioLike | list[AudioLike],
        *,
        lang: str = "eng",
        task: str = "asr",
        sr: int | None = None,
        condition_on_prev_text: bool = False,
        end_time_threshold: float = 29.0,
        skip_last_chunk_seconds: float = 0.2,
        beam_size: int = 1,
        batch_size: int = 1,
        cut_at_pauses: bool = False,
    ) -> Result | list[Result]:
        """Long-form decoding with timestamps (port of ESPnet's ``Speech2Text.decode_long``).

        ``audio`` is one recording or a list of them. A recording is decoded one 30 s window at a
        time, each window starting at the last complete utterance of the one before, as ESPnet does.
        With ``batch_size > 1`` the windows of several recordings are decoded together, a row taking
        the next window of any recording as soon as it is free (the same output, up to fp16 rounding;
        a single recording gains nothing).

        ``cut_at_pauses=True`` (opt-in, faster, not ESPnet's output): each recording is cut at pauses
        into windows of at most 30 s (the quietest point of the last 15 s of a window, see
        :func:`pause_cuts`), and the windows of all recordings are decoded ``batch_size`` at a time, a
        single recording too. Words are lost or doubled only where a cut falls inside a word.

        OWSM-CTC models are decoded in overlapping windows instead (ESPnet's ``ctc_log_probs``) and
        return a single segment.
        """
        single = not isinstance(audio, list)
        waves = [self._wave(a, sr) for a in ([audio] if single else audio)]
        if self.cfg.ctc_only:
            results = self._long_ctc(waves, lang, task)
        elif cut_at_pauses:
            if condition_on_prev_text:
                raise ValueError("condition_on_prev_text needs ESPnet's windows (cut_at_pauses=False)")
            results = self._long_batched(waves, lang, task, batch_size, beam_size, skip_last_chunk_seconds)
        else:
            results = self._long_sequential(
                waves, lang, task, condition_on_prev_text, end_time_threshold, skip_last_chunk_seconds,
                self._check_beam(beam_size), batch_size,
            )  # fmt: skip
        return results[0] if single else results

    def _long_sequential(
        self,
        waves: list[torch.Tensor],
        lang: str,
        task: str,
        condition_on_prev_text: bool,
        end_time_threshold: float,
        skip_last_chunk_seconds: float,
        beam: int,
        batch_size: int,
    ) -> list[Result]:
        """ESPnet's ``decode_long`` on each recording, window by window; with ``batch_size > 1`` the
        windows of all recordings are decoded together (:meth:`_long_stream`, or round by round)."""
        t, cfg = self.tokenizer, self.cfg
        fs, seg_len = cfg.sample_rate, cfg.num_samples
        end_id = t.first_time + round(end_time_threshold / cfg.time_resolution)
        recs = [_Recording(w) for w in waves]

        def live(r: _Recording) -> bool:
            return r.offset < len(r.wav) and (len(r.wav) - r.offset) / fs >= skip_last_chunk_seconds

        together = batch_size > 1 and not condition_on_prev_text
        if together and LONG_STREAM and self._stream_drafts(batch_size, beam, lang, task):
            self._long_stream(recs, live, lang, task, batch_size, end_id)
            return [_joined(r.segments) for r in recs]
        for group in [recs] if together else [[r] for r in recs]:
            if not together and self.device.type == "cuda":  # one recording at a time on the device:
                group[0].wav = group[0].wav.to(self.device)  # its windows need no host copy
            while True:
                now = [r for r in group if live(r)]
                if not now:
                    break
                chunks = [r.wav[r.offset : r.offset + seg_len] for r in now]
                if len(now) > 1:
                    outs = self._decode_windows(chunks, lang, task, batch_size, beam, fixed=True)
                    outs = [res.tokens for res in outs]
                else:
                    primer = self._primer(lang, task, True, now[0].prev if condition_on_prev_text else None)
                    outs = [self._long_window(chunks[0], primer, beam, task == "asr")]
                for r, chunk, ids in zip(now, chunks, outs):
                    self._long_advance(r, len(chunk), ids, end_id)
            for r in group:
                r.wav = None  # done with it (and with its copy on the device)
        return [_joined(r.segments) for r in recs]

    def _long_advance(self, r: "_Recording", samples: int, ids: list[int], end_id: int) -> None:
        """Take in the output of recording ``r``'s window (of ``samples``, at ``r.offset``), and move
        ``r.offset`` to where ESPnet's ``decode_long`` starts the next one."""
        t, cfg = self.tokenizer, self.cfg
        fs = cfg.sample_rate
        kept, next_start = _window_cut(ids, t, end_id)
        if next_start is None:  # no timestamps at all
            r.segments.append(Segment(r.offset / fs, (r.offset + samples) / fs, t.decode(ids)))
            r.offset += cfg.num_samples
            return
        res = self._result(kept, r.offset / fs)
        r.segments += res.segments
        r.prev = "".join(s.text for s in res.segments)
        advance = round((next_start - t.first_time) * cfg.time_resolution * fs)
        r.offset += advance if advance > 0 else cfg.num_samples

    def _stream_drafts(self, batch_size: int, beam: int, lang: str, task: str) -> GreedySession | None:
        """The session that decodes windows as a stream with drafts (greedy ASR), or None."""
        if beam != 1 or task != "asr" or not TIMESTAMP_GUESSES:
            return None
        sess = self._session(self._batch_size(batch_size, beam))
        primer = self._primer(lang, task, True, None)
        return sess if sess.windows and len(primer) < sess.windows[-1] else None

    def _long_stream(
        self,
        recs: list["_Recording"],
        live: Callable[["_Recording"], bool],
        lang: str,
        task: str,
        batch_size: int,
        end_id: int,
    ) -> None:
        """ESPnet's windows of all recordings in one :meth:`GreedySession.decode_stream`: a row whose
        window is done takes the next window of any recording whose previous window is done (it starts
        where that one's output says). Recordings with the most audio left go first, so that the long
        ones do not end up alone; the encoder works ahead on whole batches while there are enough
        windows to start (2 batches queued), else on whatever there is."""
        t, seg_len = self.tokenizer, self.cfg.num_samples
        sess = self._stream_drafts(batch_size, 1, lang, task)
        rows = sess.rows
        ready = [r for r in recs if live(r)]  # recordings whose next window can start
        started: list[tuple[_Recording, int]] = []  # per window, in stream order: recording, length
        ends: deque[list[int]] = deque()  # the guessed last timestamps of the batches not drafted yet
        done = [0]

        def batches() -> Iterator[torch.Tensor | None]:
            while ready or len(started) > done[0]:
                waiting = len(started) - done[0]  # windows encoded or decoding
                if not ready or (len(ready) < rows and waiting >= rows + rows // 2):
                    yield None  # wait for more windows to start (or for the stream to need them)
                    continue
                n = min(rows, 1 << (len(ready).bit_length() - 1))  # (few batch sizes: few CUDA graphs)
                ready.sort(key=lambda r: len(r.wav) - r.offset, reverse=True)
                pick = ready[:n]
                del ready[:n]
                chunks = [r.wav[r.offset : r.offset + seg_len] for r in pick]
                started.extend((r, len(c)) for r, c in zip(pick, chunks))
                ends.append([self._end_guess(len(c)) for c in chunks])
                yield self._features(chunks)

        def drafts_of(best_paths: torch.Tensor) -> list[list[int]]:  # called in batch order
            return [[t.first_time] + d + [g] for d, g in zip(self._ctc_drafts(best_paths), ends.popleft())]

        def finished(i: int, ids: list[int]) -> None:
            r, samples = started[i]
            done[0] += 1
            self._long_advance(r, samples, ids, end_id)
            if live(r):
                ready.append(r)

        primer = self._primer(lang, task, True, None)
        insertions = t.not_drafted - set(range(t.first_time, t.last_time + 1))
        outs, calls = sess.decode_stream(
            primer, batches(), drafts_of, self._max_new(), insertions, give_up=16,
            priority=self.stream_priority, finished=finished,
        )  # fmt: skip
        self.draft_stats[0] += sum(len(o) + 1 for o in outs)
        self.draft_stats[1] += calls
        for r in recs:
            r.wav = None

    def _end_guess(self, samples: int) -> int:
        """The timestamp at the end of a window of ``samples``: a guess of its last one."""
        t, cfg = self.tokenizer, self.cfg
        return t.first_time + min(
            round(samples / cfg.sample_rate / cfg.time_resolution), t.last_time - t.first_time
        )

    def _long_window(self, chunk: torch.Tensor, primer: list[int], beam: int, draft: bool) -> list[int]:
        """One window of :meth:`_long_sequential` (as :meth:`_decode`).

        Its CTC draft gets timestamps guessed: the window opens with the first one (it starts where the
        previous window's last utterance did, or ended: 96% on LibriSpeech), it ends with the one at
        its end (its last utterance runs into the end of the window: 83%), and after an end time the
        session guesses the next start time to be the same (84%, see runtime._Row.advance). Guesses are
        drafted tokens like any other: the output is the greedy output, in 26% fewer decoder calls."""
        t = self.tokenizer
        sess = self._session(1, beam)
        if not (draft and TIMESTAMP_GUESSES and isinstance(sess, GreedySession) and sess.windows):
            return self._decode([chunk], [primer], beam, draft)[0]
        (path,) = self._ctc_drafts(sess.encode(self._features([chunk])))
        proposal = [t.first_time] + path + [self._end_guess(len(chunk))]
        insertions = t.not_drafted - set(range(t.first_time, t.last_time + 1))  # timestamps are drafted
        # (a draft is given up after 16 calls that accepted fewer tokens than calls, not 8: a window
        # has 11 calls on average, and timestamps end many of them early)
        (out,), calls = sess.decode_drafted([primer], [proposal], self._max_new(), insertions, give_up=16)
        self.draft_stats[0] += len(out) + 1
        self.draft_stats[1] += calls
        return out

    def _long_batched(
        self, waves: list[torch.Tensor], lang: str, task: str, batch_size: int, beam: int, skip_seconds: float
    ) -> list[Result]:
        """Windows cut at pauses, decoded in batches (transcribe_long with ``cut_at_pauses=True``)."""
        fs = self.cfg.sample_rate
        windows, owner = [], []  # waveform pieces; (recording, first sample) of each
        for r, wav in enumerate(waves):
            cuts = pause_cuts(wav, self.cfg.num_samples, fs, PAUSE_SEARCH_SECONDS)
            for a, b in zip(cuts, cuts[1:]):
                if (b - a) / fs >= skip_seconds:
                    windows.append(wav[a:b])
                    owner.append((r, a))
        outs = self._decode_windows(windows, lang, task, batch_size, beam) if windows else []
        segments: list[list[Segment]] = [[] for _ in waves]
        for (r, a), res in zip(owner, outs):
            off = a / fs
            segments[r] += [
                Segment(round(s.start + off, 2), round(s.end + off, 2), s.text) for s in res.segments
            ]
        return [_joined(segs) for segs in segments]

    def _decode_windows(
        self,
        windows: list[torch.Tensor],
        lang: str,
        task: str,
        batch_size: int,
        beam: int,
        fixed: bool = False,
    ) -> list[Result]:
        """``transcribe(windows, timestamps=True)``, with drafts that guess timestamps as in
        :meth:`_long_window` (greedy ASR): the first one is <0.00> (92% of pause-cut windows), the last
        one the end of the window, and a start time the end time before it (91%). ``fixed``: the
        session has ``batch_size`` rows however few windows there are (no session per count)."""
        t = self.tokenizer
        bs = self._batch_size(batch_size, beam)
        bs = bs if fixed else min(bs, len(windows))
        primer = self._primer(lang, task, True, None)
        sess = self._session(bs) if beam == 1 and task == "asr" and TIMESTAMP_GUESSES else None
        if sess is None or not sess.windows or len(primer) >= sess.windows[-1]:
            # No drafts: transcribe() decodes fixed batches, and a smaller one would get a session of its
            # own size (rebuilt every round). With ``fixed``, copies of the last window fill it up.
            n = len(windows)
            padded = windows + windows[-1:] * (-n % bs if fixed else 0)
            out = self.transcribe(
                padded, lang=lang, task=task, timestamps=True, batch_size=bs, beam_size=beam
            )
            return out[:n]
        order = sorted(range(len(windows)), key=lambda i: len(windows[i]))  # as transcribe()
        ends = iter([self._end_guess(len(windows[i])) for i in order])  # drafts are made in batch order

        def drafts_of(best_paths: torch.Tensor) -> list[list[int]]:
            return [[t.first_time] + d + [next(ends)] for d in self._ctc_drafts(best_paths)]

        batches = (self._features([windows[j] for j in order[i : i + bs]]) for i in range(0, len(order), bs))
        insertions = t.not_drafted - set(range(t.first_time, t.last_time + 1))
        outs, calls = sess.decode_stream(
            primer, batches, drafts_of, self._max_new(), insertions, give_up=16, priority=self.stream_priority
        )
        self.draft_stats[0] += sum(len(o) + 1 for o in outs)
        self.draft_stats[1] += calls
        results: list[Result] = [None] * len(windows)
        for j, ids in zip(order, outs):
            results[j] = self._result(ids, offset=0.0)
        return results

    def _long_ctc(
        self, waves: list[torch.Tensor], lang: str, task: str, context_seconds: float = 2.0
    ) -> list[Result]:
        """OWSM-CTC over recordings: windows of the training length overlapping by ``context_seconds``
        on each side, keeping each window's middle frames (as ESPnet's ``ctc_log_probs``). The windows of
        all recordings go through the encoder ``max_batch_size`` at a time."""
        cfg, fs = self.cfg, self.cfg.sample_rate
        context = int(fs * context_seconds)
        chunk = cfg.num_samples - 2 * context
        windows, owner = [], []
        for r, wav in enumerate(waves):
            padded = F.pad(wav, (context, context))
            for start in range(0, len(padded), chunk):
                windows.append(padded[start : start + cfg.num_samples])
                owner.append(r)
                if len(windows[-1]) < cfg.num_samples:
                    break
        fps = fs / cfg.hop_length / math.prod(cfg.strides)
        # ESPnet cuts the window's frames before dropping the two prefix frames, so it keeps these:
        keep = slice(int(fps * context_seconds), int(fps * cfg.speech_seconds) - int(fps * context_seconds))
        bs = self.opts.max_batch
        batches = (windows[i : i + bs] for i in range(0, len(windows), bs))
        frames: list[list[torch.Tensor]] = [[] for _ in waves]
        for i, paths in enumerate(self._ctc_stream(batches, lang, task, None)):
            for r, row in zip(owner[i * bs : (i + 1) * bs], paths[:, keep]):
                frames[r].append(row)
        results = []
        for wav, rows in zip(waves, frames):
            wanted = int(round(len(wav) / fs * fps))
            ids = ctc_collapse(torch.cat(rows)[:wanted].tolist(), self.tokenizer.blank)
            text = self.tokenizer.decode(ids)
            results.append(
                Result(text=text, tokens=ids, segments=[Segment(0.0, round(len(wav) / fs, 2), text)])
            )
        return results


def _fp8_scales(files, fp8: bool | str) -> Path | None:
    """The FP8 scale file of ``from_pretrained(fp8=...)``: a given path, or the one the package ships for
    this checkpoint (None, with a warning, if there is none)."""
    sha = fp8_mod.checkpoint_sha256(files.checkpoint, hub.cache_dir() / "checkpoint_sha256.json")
    if not isinstance(fp8, bool):
        path = Path(fp8)
        if fp8_mod.load_scales(path).get("checkpoint_sha256") != sha:
            log.warning("FP8 scales %s were not calibrated on this checkpoint (sha256 %s)", path, sha[:12])
        return path
    path = fp8_mod.packaged_scales(sha)
    if path is None:
        log.warning(
            "no FP8 scales shipped for %s (checkpoint sha256 %s): fp16 encoder (fp8_calibration=[recordings] "
            "calibrates on given audio)", files.name, sha[:12],
        )  # fmt: skip
    return path


@torch.inference_mode()
def _quantize_encoder(
    net: OWSMModel | OWSMCTCModel,
    tok: Tokenizer,
    recordings: list[AudioLike],
    batch: int,
    margin: float = 1.0,
    skip: str = "",
) -> None:
    """Measure the encoder's layer inputs on ``recordings`` (as English ASR) and switch them to FP8."""
    prefix = [tok.lang_id("eng"), tok.task_id("asr")]
    n = net.cfg.num_samples
    waves = [load_audio(a, net.cfg.sample_rate)[:n] for a in recordings]
    dev = next(net.encoder.parameters()).device
    batches = []
    for i in range(0, len(waves), batch):
        x = torch.zeros(len(waves[i : i + batch]), n, device=dev)
        for row, w in zip(x, waves[i : i + batch]):
            row[: len(w)] = w.to(dev)
        batches.append(net.frontend(x))
    if net.cfg.ctc_only:  # the encoder also reads the prompt and the language/task prefix

        def run(feats: torch.Tensor) -> None:
            b = feats.shape[0]
            net(feats, torch.tensor([[tok.na]] * b, device=dev), torch.tensor([prefix] * b, device=dev))

    else:
        run = net.encoder
    amax = fp8_mod.calibrate(net.encoder, run, batches)
    log.warning(
        "FP8 encoder: %d layers calibrated on %d recordings",
        fp8_mod.quantize(net.encoder, amax, margin, skip),
        len(waves),
    )


def _grouped_beam_backend(base):
    """Independent experimental dispatch, sharing weights/engines; greedy retains ``base``."""
    from .megakernel_gluon import BEAM_CROSS_SPLITS, GluonDecoder

    if not isinstance(base, megakernel.MegaBackend):
        log.warning("grouped beam attention needs an available Gluon decoder; using the existing backend")
        return base
    source = base.rows_kernel[0] if base.rows_kernel is not None else base.kernel
    if not isinstance(source, GluonDecoder) or id(source) in base.failed:
        log.warning("grouped beam attention has no available multi-row Gluon kernel; using the existing backend")
        return base
    # Wider routing is measured only for this decoder/GPU. Keep all default and greedy
    # routing limits unchanged; unsupported models retain the existing grouped path.
    m = source.m
    wide = False
    if (tuple(getattr(m, k, None) for k in ("D", "F", "L", "H")) == (1024, 4096, 18, 16)
            and source.P == 132 and source.nw == 4 and source.pfa):
        gpu = torch.cuda.get_device_properties(m.device)
        wide = "H200" in gpu.name and gpu.multi_processor_count == 132
    kernel = GluonDecoder(source.m, programs=source.P, num_warps=source.nw,
                          prefetch_attention=source.pfa, group_cross_queries=True,
                          cross_splits=BEAM_CROSS_SPLITS if wide else 8)
    backend = megakernel.MegaBackend(base.inner, base.kernel, base.max_tokens, base.max_rows,
                                    rows_kernel=base.rows_kernel, trial=base.trial)
    backend.failed = set(base.failed)
    if base.rows_kernel is None:
        backend.kernel = kernel
        if wide:
            backend.max_rows = 16
    else:
        backend.rows_kernel = (kernel, base.rows_kernel[1], 16 if wide else base.rows_kernel[2])
    return backend


def _decoder_kernel(runner, cfg, checkpoint: Path, dev: torch.device, which):
    """``runner`` with its small decoder calls on a megakernel (see from_pretrained's ``decoder_kernel``).
    Anything that fails while the kernels are made (unsupported shapes, out of memory, ...) leaves ``runner``
    (logged); so does any call whose first run fails (MegaBackend's trial)."""
    from . import megakernel_gluon

    if which == "triton":
        if not megakernel.available(dev):
            log.warning("the Triton decoder megakernel needs Triton and an Ampere or newer GPU; not used")
            return runner
        return megakernel.MegaBackend(runner, megakernel.MegaDecoder(_load_decoder(cfg, checkpoint, skip_init=True), dev))
    if not megakernel_gluon.available(dev):
        if which != "auto":
            log.warning("the Gluon decoder megakernels need Triton with Gluon and a Hopper GPU; not used")
        elif dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] == 9:  # (auto: Hopper only)
            log.warning(
                "decoder megakernel not used: this Triton (%s) has no Gluon for Hopper; the TensorRT decoder"
                " is ~20%% slower at batch 1 (the megakernels are tested with Triton %s, which comes with torch %s)",
                megakernel_gluon.triton_version(),
                *megakernel_gluon.TESTED,
            )
        return runner
    wide_ring = which == "ring:wide"
    auto = which == "auto" or wide_ring
    try:
        decoder = _load_decoder(cfg, checkpoint, skip_init=True)
        mega = megakernel.MegaDecoder(decoder, dev)
        try:
            gk = megakernel_gluon.GluonDecoder(mega)
        except ValueError as e:  # (e.g. a model width that is not a power of 2)
            if not (auto or (isinstance(which, str) and which.startswith("ring"))):
                raise
            log.info("Gluon decoder megakernel not used for calls of several rows: %s", e)
            gk = None
        route = megakernel_gluon.ROUTE_TOKENS, megakernel_gluon.ROUTE_ROWS
        rows_kernel = None
        if auto or (isinstance(which, str) and which.startswith("ring")):
            # ring[:<programs>][:r<P>][:s<P>][:c<n>][:a<n>][:w<warps>][:x][:rr]: single-row calls on the ring
            # kernel (few SMs while decode_stream's encoder runs beside it), calls of several rows on the Gluon
            # kernel (all SMs), with rr on the ring kernel too
            from . import megakernel_ring

            kw, opts = {}, which.split(":")[1:]
            for o in opts:
                if o.isdigit():
                    kw["programs"] = int(o)
                elif o == "x":
                    kw["exclusive"] = True
                elif o == "t":  # (profiling: time stamps)
                    kw["ts"] = True
                elif o[:1] in ("r", "s", "c", "a", "w") and o[1:].isdigit():
                    key = {
                        "r": "row_programs",
                        "s": "solo_programs",
                        "c": "chunk",
                        "a": "ahead",
                        "w": "num_warps",
                    }
                    kw[key[o[0]]] = int(o[1:])
            kernel = megakernel_ring.RingDecoder(mega, **kw)
            if "rr" not in opts:
                rows_kernel = (gk, *route)
            route = megakernel_ring.ROUTE_TOKENS, megakernel_ring.ROUTE_ROWS
        elif gk is not None:
            kernel = gk
        if auto:
            _check_kernel(kernel, decoder, dev, 1, 2)
            if gk is not None:
                try:
                    _check_kernel(gk, decoder, dev, 2, 1)
                except Exception as e:  # noqa: BLE001
                    log.warning("Gluon decoder megakernel not used for calls of several rows: %s", e)
                    rows_kernel = (None, *rows_kernel[1:])
    except Exception as e:  # noqa: BLE001 (any failure: TensorRT)
        log.warning(
            "decoder megakernel not used (%s: %s); the TensorRT decoder is ~20%% slower at batch 1 (the megakernels"
            " are tested with Triton %s, which comes with torch %s; this is Triton %s)",
            type(e).__name__,
            e,
            *megakernel_gluon.TESTED,
            megakernel_gluon.triton_version(),
        )
        torch.cuda.empty_cache()
        return runner
    torch.cuda.empty_cache()  # (the fp32 copies the weights were folded in)
    backend = megakernel.MegaBackend(runner, kernel, *route, rows_kernel=rows_kernel, trial=True)
    if wide_ring:
        m = kernel.m
        gpu = torch.cuda.get_device_properties(dev)
        if ((m.D, m.F, m.L, m.H) != (384, 1536, 6, 6) or "H200" not in gpu.name
                or gpu.multi_processor_count != 132):
            log.warning("wide Ring routing requires102M/H200132SM; retaining auto dispatch")
            return backend
        try:
            routes = {r: megakernel.MegaBackend(runner, megakernel_ring.RingDecoder(m, row_programs=r*6),
                      max_tokens=r, max_rows=r, trial=True) for r in (10, 15, 16)}
            return megakernel.WideRingBackend(backend, routes)
        except Exception as e:  # noqa: BLE001 (opt-in setup must retain a usable auto backend)
            log.warning("wide Ring setup failed (%s: %s); retaining auto dispatch", type(e).__name__, e)
    return backend


@torch.no_grad()
def _check_kernel(kernel, decoder: Decoder, dev: torch.device, rows: int, window: int) -> None:
    """Raise unless ``kernel``'s logits for a small random call match ``decoder``'s (fp32, CPU)."""
    m = kernel.m
    L, H, dk, C, T = m.L, m.H, m.D // m.H, 64, 64
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, m.V, (rows, window), generator=g)
    offset = torch.full((rows,), 5, dtype=torch.long)
    k, v, ck, cv = (torch.randn(L, rows, H, n, dk, generator=g).half() for n in (C, C, T, T))
    ref, _ = decoder.step(
        tokens, offset, list(k.float()), list(v.float()), list(ck.float()), list(cv.float())
    )
    logits = torch.zeros(rows, window, m.V, device=dev)
    new_kv = torch.zeros(L, 2, rows, H, window, dk, device=dev, dtype=torch.float16)
    cuda = [t.to(dev) for t in (tokens, offset, k, v, ck, cv)]
    kernel.step_fn(*cuda, logits, new_kv)()
    torch.cuda.synchronize(dev)
    kernel.check()
    out = logits.cpu()
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    if not err < 0.02 or agree < 0.5:
        raise RuntimeError(
            f"{kernel.name}: logits off ({err:.3g} of their range, argmax agreement {agree:.2f})"
        )


def _load_decoder(cfg, checkpoint: Path, *, empty_init: bool = False, skip_init: bool = False) -> Decoder:
    """The decoder's weights (fp32, on the CPU) for the decoder megakernel."""
    if empty_init and skip_init:
        raise ValueError("meta and direct skipped initialization are separate ablations")
    sd = convert_espnet_state_dict(load_checkpoint(checkpoint, memory_map=True))
    if empty_init:
        with torch.device("meta"):
            dec = Decoder(cfg)
        # All parameters are overwritten by strict loading. The nonpersistent position
        # table is the sole buffer and must be regenerated after empty allocation.
        if set(dict(dec.named_buffers())) != {"pe"}:
            raise RuntimeError("empty decoder initialization requires an explicit buffer initialization policy")
        dec.to_empty(device="cpu")
        dec.pe = sinusoidal_positions(dec.pe.shape[0], cfg.d_model)
    else:
        dec = Decoder(cfg, initialize=not skip_init)
    dec.load_state_dict({k[len("decoder.") :]: v for k, v in sd.items() if k.startswith("decoder.")})
    return dec.eval().requires_grad_(False)


def _load_frontend(cfg, checkpoint: Path, device: torch.device) -> Frontend:
    """Only the fp32 frontend buffers are needed at runtime when the engines are cached."""
    sd = load_checkpoint(checkpoint, memory_map=True)
    fe = Frontend(cfg)
    state = {"melmat": sd["frontend.logmel.melmat"]}
    if fe.global_mvn:
        state.update(mean=sd["normalize.mean"], std=sd["normalize.std"])
    fe.load_state_dict(state)
    return fe.to(device).eval()


__all__ = ["OWSM", "OWSMModel", "Result", "Segment"]
