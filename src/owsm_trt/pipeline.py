"""High-level API: ``OWSM.from_pretrained(...).transcribe(...)``."""

import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from . import fp8, hub, megakernel
from .audio import AudioLike, duration, load_audio
from .build import DTYPES, BuildOptions, build_engines, check_config, engine_tag
from .config import load_espnet_config, parse_config
from .modeling import (
    Decoder,
    Frontend,
    OWSMCTCModel,
    OWSMModel,
    convert_espnet_state_dict,
    frames_needed,
    load_model,
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
)
from .tokenizer import Tokenizer

log = logging.getLogger(__name__)

# Window sizes (drafted tokens + 1) of CTC-drafted greedy decoding; one CUDA graph each per cache bucket.
DRAFT_WINDOWS = (1, 2, 4, 8, 16, 32)
# At most this many tokens (rows x window) per decoder call: wider calls cost more than they save
# (batch 16: 3.7 ms with windows of 16, 5.0 ms with 32, on the 1B model).
DRAFT_TOKENS = 256
# Feature lengths a trimming encoder engine gets (multiples of this, 5 s): one CUDA graph each.
FEATURE_STEP = 500
# transcribe_long: drafts with guessed timestamps (see OWSM._long_window)
TIMESTAMP_GUESSES = True
# transcribe_long with batch_size > 1: a window is cut at the quietest point of its last this many seconds
PAUSE_SEARCH_SECONDS = 15.0


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
    ):
        self.cfg, self.tokenizer, self.frontend, self.backend = cfg, tokenizer, frontend, backend
        self.opts, self.device, self.use_cuda_graph = opts, device, use_cuda_graph
        self.ctc_draft = ctc_draft
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
    ) -> "OWSM":
        """Load an OWSM model from a Hugging Face repo id or local ESPnet model directory.

        With ``backend="tensorrt"`` engines are built on first use (a few minutes) and cached under
        ``~/.cache/owsm_trt`` (override with ``OWSM_TRT_CACHE``). ``backend="torch"`` skips TensorRT.

        ``ctc_draft``: for greedy ASR, let the encoder's CTC head propose the transcript and check up
        to 31 proposed tokens per utterance and decoder call. The output is the greedy output; only the
        number of decoder calls changes.

        ``fp8_calibration`` (experimental, TensorRT): run the encoder's linear layers and subsampling
        convolutions in FP8, with input ranges measured on these recordings.

        ``decoder_kernel`` (TensorRT, fp16): run decoder calls of up to 16 tokens (rows x drafted window) as one
        persistent Gluon kernel per call instead of the TensorRT decoder engine (Hopper GPUs; larger calls
        and prompt prefill stay on TensorRT). ``"triton"``: every decode/verify call as the (slower) Triton
        megakernel (Ampere or newer). None: the ``OWSM_TRT_MEGAKERNEL`` environment variable (1 / triton / 0),
        off if unset.
        """
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
        opts = BuildOptions(
            dtype=dtype,
            max_batch=max_batch_size,
            capacity=capacity,
            max_prefill=max_prompt_tokens,
            **levels,
            fp8_encoder=fp8_calibration is not None,
        )
        if fp8_calibration is not None and backend != "tensorrt":
            raise ValueError("FP8 is available for the TensorRT backend only")
        dev = torch.device(device)
        if backend == "tensorrt" and dev.type != "cuda":
            raise ValueError("the TensorRT backend needs a CUDA device")

        if backend == "torch":
            net = load_model(cfg, str(files.checkpoint), dev, DTYPES[dtype])
            graphs = use_cuda_graph and dev.type == "cuda"
            runner = TorchCTCBackend(net) if cfg.ctc_only else TorchBackend(net)
            return cls(cfg, tok, net.frontend, runner, opts, dev, graphs, ctc_draft)
        if backend != "tensorrt":
            raise ValueError(f"unknown backend {backend!r} (use 'tensorrt' or 'torch')")

        with torch.cuda.device(dev):
            engine_dir = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(opts)
            if rebuild or not check_config(cfg, engine_dir):
                log.warning("building TensorRT engines for %s into %s", files.name, engine_dir)
                net = load_model(cfg, str(files.checkpoint), dev, DTYPES[dtype])
                if fp8_calibration is not None:
                    _quantize_encoder(net, tok, fp8_calibration, opts.max_batch)
                build_engines(net, engine_dir, opts)
                frontend = net.frontend
                del net
                torch.cuda.empty_cache()
            else:
                frontend = _load_frontend(cfg, files.checkpoint, dev)
            runner = TRTCTCBackend(cfg, engine_dir) if cfg.ctc_only else TRTBackend(cfg, engine_dir)
            if decoder_kernel is None:
                decoder_kernel = os.environ.get("OWSM_TRT_MEGAKERNEL", "0")
            if decoder_kernel and decoder_kernel != "0" and not cfg.ctc_only and dtype == "float16":
                runner = _decoder_kernel(runner, cfg, files.checkpoint, dev, decoder_kernel)
            return cls(cfg, tok, frontend, runner, opts, dev, use_cuda_graph, ctc_draft)

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
                    punctuation=t.punctuation | times,
                )  # fmt: skip
            else:
                self._sessions[key] = BeamSession(
                    self.backend, self.cfg, batch, beam, *args, rules=self.rules
                )
        self._sessions.move_to_end(key)
        return self._sessions[key]

    def _draft_windows(self, batch: int) -> tuple[int, ...]:
        if not self.ctc_draft:
            return ()
        return tuple(
            w for w in DRAFT_WINDOWS if w <= self.opts.max_verify and w * batch <= max(DRAFT_TOKENS, batch)
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
        batch = torch.zeros(len(waves), n, device=self.device)  # pad on the device: copy only the audio
        for row, w in zip(batch, waves):
            w = w[:n]
            if batch.is_cuda and w.device.type == "cpu":
                w = w.pin_memory()  # a copy from pageable memory would first wait for the stream's work
            row[: w.shape[0]].copy_(w, non_blocking=True)
        feats = self.frontend(batch)  # [B, T, n_mels]
        if getattr(self.backend, "trimmed", False):  # the encoder fills in the frames of padding only
            t = max(frames_needed(self.cfg, w.shape[0]) for w in waves)
            feats = feats[:, : min(self.cfg.num_frames, -(-t // FEATURE_STEP) * FEATURE_STEP)]
        return feats

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
            results = []
            bs = min(batch_size or self.opts.max_batch, self.opts.max_batch)
            for i in range(0, len(items), bs):
                waves = [self._wave(a, sr) for a in items[i : i + bs]]
                for row in self._ctc_paths(waves, lang, task, prompt).tolist():
                    results.append(self._result(ctc_collapse(row, self.tokenizer.blank), offset=0.0))
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
            chunks = (order[i : i + bs] for i in range(0, len(order), bs))
            batches = (self._features([self._wave(items[j], sr) for j in chunk]) for chunk in chunks)
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
        order = sorted(range(len(items)), key=lambda i: duration(items[i], sr or self.cfg.sample_rate))
        results: list[Result | None] = [None] * len(items)
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            waves = [self._wave(items[j], sr) for j in idx]
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

    def _ctc_draft(self, best_path: list[int]) -> list[int]:
        return ctc_collapse(best_path, self.tokenizer.blank)

    def _ctc_drafts(self, best_paths: torch.Tensor) -> list[list[int]]:
        """Drafts from CTC best paths [B, t] (GreedySession.encode)."""
        return ctc_collapse_rows(best_paths, self.tokenizer.blank)

    def _ctc_paths(self, waves: list[torch.Tensor], lang: str, task: str, prompt: str | None) -> torch.Tensor:
        """OWSM-CTC best paths [B, 2 + T'] (frame-wise argmax, the first two frames are the prefix).

        The tensor is a session buffer, overwritten by the next call."""
        t = self.tokenizer
        prev = t.encode(prompt) if prompt else [t.na]
        if t.na in prev:
            prev = [t.na]
        b = len(waves)
        key = ("ctc", b, len(prev))
        if key not in self._sessions:
            if len(self._sessions) >= 2:
                self._sessions.popitem(last=False)
            self._sessions[key] = CTCSession(
                self.backend, self.cfg, b, len(prev), self.device, self.use_cuda_graph
            )
        self._sessions.move_to_end(key)
        return self._sessions[key](
            self._features(waves),
            torch.tensor([prev] * b, device=self.device),
            torch.tensor([[t.lang_id(lang), t.task_id(task)]] * b, device=self.device),
        )

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
        With ``batch_size > 1`` the next windows of up to that many recordings are decoded together
        (the same output, up to fp16 rounding; a single recording gains nothing).

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
            results = [self._transcribe_long_ctc(w, lang, task) for w in waves]
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
        current windows of all recordings are decoded together, round by round."""
        t, cfg = self.tokenizer, self.cfg
        fs, seg_len = cfg.sample_rate, cfg.num_samples
        end_id = t.first_time + round(end_time_threshold / cfg.time_resolution)
        recs = [_Recording(w) for w in waves]
        together = batch_size > 1 and not condition_on_prev_text
        for group in [recs] if together else [[r] for r in recs]:
            if not together and self.device.type == "cuda":  # one recording at a time on the device:
                group[0].wav = group[0].wav.to(self.device)  # its windows need no host copy
            while True:
                live = [
                    r
                    for r in group
                    if r.offset < len(r.wav) and (len(r.wav) - r.offset) / fs >= skip_last_chunk_seconds
                ]
                if not live:
                    break
                chunks = [r.wav[r.offset : r.offset + seg_len] for r in live]
                if len(live) > 1:
                    outs = self._decode_windows(chunks, lang, task, batch_size, beam, fixed=True)
                    outs = [res.tokens for res in outs]
                else:
                    primer = self._primer(lang, task, True, live[0].prev if condition_on_prev_text else None)
                    outs = [self._long_window(chunks[0], primer, beam, task == "asr")]
                for r, chunk, ids in zip(live, chunks, outs):
                    kept, next_start = _window_cut(ids, t, end_id)
                    if next_start is None:  # no timestamps at all
                        r.segments.append(Segment(r.offset / fs, (r.offset + len(chunk)) / fs, t.decode(ids)))
                        r.offset += seg_len
                        continue
                    res = self._result(kept, r.offset / fs)
                    r.segments += res.segments
                    r.prev = "".join(s.text for s in res.segments)
                    advance = round((next_start - t.first_time) * cfg.time_resolution * fs)
                    r.offset += advance if advance > 0 else seg_len
            for r in group:
                r.wav = None  # done with it (and with its copy on the device)
        return [_joined(r.segments) for r in recs]

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

    def _transcribe_long_ctc(
        self, wav: torch.Tensor, lang: str, task: str, context_seconds: float = 2.0
    ) -> Result:
        """OWSM-CTC over a recording: windows of the training length overlapping by ``context_seconds``
        on each side, keeping each window's middle frames (as ESPnet's ``ctc_log_probs``)."""
        cfg, fs = self.cfg, self.cfg.sample_rate
        context = int(fs * context_seconds)
        chunk = cfg.num_samples - 2 * context
        padded = F.pad(wav, (context, context))
        windows = []
        for start in range(0, len(padded), chunk):
            windows.append(padded[start : start + cfg.num_samples])
            if len(windows[-1]) < cfg.num_samples:
                break
        fps = fs / cfg.hop_length / math.prod(cfg.strides)
        # ESPnet cuts the window's frames before dropping the two prefix frames, so it keeps these:
        keep = slice(int(fps * context_seconds), int(fps * cfg.speech_seconds) - int(fps * context_seconds))
        frames = []
        for i in range(0, len(windows), self.opts.max_batch):
            paths = self._ctc_paths(windows[i : i + self.opts.max_batch], lang, task, None)
            frames.append(paths[:, keep].reshape(-1).clone())  # paths is reused by the next call
        wanted = int(round(len(wav) / fs * fps))
        ids = ctc_collapse(torch.cat(frames)[:wanted].tolist(), self.tokenizer.blank)
        text = self.tokenizer.decode(ids)
        return Result(text=text, tokens=ids, segments=[Segment(0.0, round(len(wav) / fs, 2), text)])


@torch.inference_mode()
def _quantize_encoder(
    net: OWSMModel | OWSMCTCModel, tok: Tokenizer, recordings: list[AudioLike], batch: int
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
    amax = fp8.calibrate(net.encoder, run, batches)
    log.warning(
        "FP8 encoder: %d layers calibrated on %d recordings", fp8.quantize(net.encoder, amax), len(waves)
    )


def _decoder_kernel(runner, cfg, checkpoint: Path, dev: torch.device, which):
    """``runner`` with its small decoder calls on a megakernel (see from_pretrained's ``decoder_kernel``)."""
    from . import megakernel_gluon

    if which == "triton":
        if not megakernel.available(dev):
            log.warning("the Triton decoder megakernel needs Triton and an Ampere or newer GPU; not used")
            return runner
        return megakernel.MegaBackend(runner, megakernel.MegaDecoder(_load_decoder(cfg, checkpoint), dev))
    if not megakernel_gluon.available(dev):
        log.warning("the Gluon decoder megakernel needs Triton with Gluon and a Hopper GPU; not used")
        return runner
    mega = megakernel.MegaDecoder(_load_decoder(cfg, checkpoint), dev)
    try:
        kernel = megakernel_gluon.GluonDecoder(mega)
    except ValueError as e:
        log.warning("decoder megakernel not used: %s", e)
        return runner
    return megakernel.MegaBackend(runner, kernel, megakernel_gluon.ROUTE_TOKENS)


def _load_decoder(cfg, checkpoint: Path) -> Decoder:
    """The decoder's weights (fp32, on the CPU) for the decoder megakernel."""
    sd = convert_espnet_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    dec = Decoder(cfg)
    dec.load_state_dict({k[len("decoder.") :]: v for k, v in sd.items() if k.startswith("decoder.")})
    return dec.eval().requires_grad_(False)


def _load_frontend(cfg, checkpoint: Path, device: torch.device) -> Frontend:
    """Only the fp32 frontend buffers are needed at runtime when the engines are cached."""
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fe = Frontend(cfg)
    state = {"melmat": sd["frontend.logmel.melmat"]}
    if fe.global_mvn:
        state.update(mean=sd["normalize.mean"], std=sd["normalize.std"])
    fe.load_state_dict(state)
    return fe.to(device).eval()


__all__ = ["OWSM", "OWSMModel", "Result", "Segment"]
