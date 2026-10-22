"""High-level API: ``OWSM.from_pretrained(...).transcribe(...)``."""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from . import hub
from .audio import AudioLike, duration, load_audio
from .build import DTYPES, BuildOptions, build_engines, check_config, engine_tag
from .config import load_espnet_config, parse_config
from .modeling import Frontend, OWSMModel, load_model
from .runtime import GreedySession, TorchBackend, TRTBackend
from .tokenizer import Tokenizer

log = logging.getLogger(__name__)


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


class OWSM:
    """OWSM speech recognition / translation with a TensorRT (or PyTorch) backend.

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
    ):
        self.cfg, self.tokenizer, self.frontend, self.backend = cfg, tokenizer, frontend, backend
        self.opts, self.device, self.use_cuda_graph = opts, device, use_cuda_graph
        self.dtype = DTYPES[opts.dtype]
        self._sessions: OrderedDict[int, GreedySession] = OrderedDict()

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
        optimization_level: int = 3,
    ) -> "OWSM":
        """Load an OWSM model from a Hugging Face repo id or local ESPnet model directory.

        With ``backend="tensorrt"`` engines are built on first use (a few minutes) and cached under
        ``~/.cache/owsm_trt`` (override with ``OWSM_TRT_CACHE``). ``backend="torch"`` skips TensorRT.
        """
        files = hub.resolve(model)
        raw = load_espnet_config(files.config)
        cfg = parse_config(raw)
        tok = Tokenizer(
            raw["token_list"],
            hub.resolve_relative(files, raw["bpemodel"]),
            raw.get("preprocessor_conf", {}).get("notime_symbol", "<notimestamps>"),
        )
        opts = BuildOptions(
            dtype=dtype,
            max_batch=max_batch_size,
            capacity=capacity,
            max_prefill=max_prompt_tokens,
            opt_level=optimization_level,
        )
        dev = torch.device(device)
        if backend == "tensorrt" and dev.type != "cuda":
            raise ValueError("the TensorRT backend needs a CUDA device")

        if backend == "torch":
            net = load_model(cfg, str(files.checkpoint), dev, DTYPES[dtype])
            return cls(
                cfg, tok, net.frontend, TorchBackend(net), opts, dev, use_cuda_graph and dev.type == "cuda"
            )
        if backend != "tensorrt":
            raise ValueError(f"unknown backend {backend!r} (use 'tensorrt' or 'torch')")

        with torch.cuda.device(dev):
            engine_dir = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(opts)
            if rebuild or not check_config(cfg, engine_dir):
                log.warning("building TensorRT engines for %s into %s", files.name, engine_dir)
                net = load_model(cfg, str(files.checkpoint), dev, DTYPES[dtype])
                build_engines(net, engine_dir, opts)
                frontend = net.frontend
                del net
                torch.cuda.empty_cache()
            else:
                frontend = _load_frontend(cfg, files.checkpoint, dev)
            return cls(cfg, tok, frontend, TRTBackend(cfg, engine_dir), opts, dev, use_cuda_graph)

    # ------------------------------------------------------------------ helpers

    def _session(self, batch: int) -> GreedySession:
        if batch not in self._sessions:
            if len(self._sessions) >= 2:
                self._sessions.popitem(last=False)
            self._sessions[batch] = GreedySession(
                self.backend,
                self.cfg,
                batch,
                self.opts.capacity,
                self.tokenizer.eos,
                self.dtype,
                self.device,
                use_cuda_graph=self.use_cuda_graph,
            )
        self._sessions.move_to_end(batch)
        return self._sessions[batch]

    def _features(self, waves: list[torch.Tensor]) -> torch.Tensor:
        n = self.cfg.num_samples
        for w in waves:
            if w.shape[0] > n:
                log.warning(
                    "input of %.1fs is trimmed to %.0fs; use transcribe_long() for long audio",
                    w.shape[0] / self.cfg.sample_rate,
                    self.cfg.speech_seconds,
                )
        batch = torch.stack([F.pad(w[:n], (0, n - min(n, w.shape[0]))) for w in waves]).to(self.device)
        return self.frontend(batch)  # [B, T, n_mels]

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
    ) -> Result | list[Result]:
        """Transcribe (``task="asr"``) or translate (``task="st_eng"`` etc.) up to 30 s of audio.

        ``audio`` is a path, a waveform (np.ndarray / torch.Tensor, sampled at ``sr`` or 16 kHz),
        or a list of those, which is decoded in batches of ``batch_size``.
        """
        single = not isinstance(audio, list)
        items = [audio] if single else audio
        bs = min(batch_size or self.opts.max_batch, self.opts.max_batch)
        primer = self._primer(lang, task, timestamps, prompt)
        # A batch decodes until its longest member finishes, so batch utterances of similar duration.
        order = sorted(range(len(items)), key=lambda i: duration(items[i], sr or self.cfg.sample_rate))
        results: list[Result | None] = [None] * len(items)
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            waves = [self._wave(items[j], sr) for j in idx]
            for j, ids in zip(idx, self._decode(waves, [primer] * len(idx))):
                results[j] = self._result(ids, offset=0.0)
        return results[0] if single else results

    def _decode(self, waves: list[torch.Tensor], primers: list[list[int]]) -> list[list[int]]:
        sess = self._session(len(waves))
        sess.encode(self._features(waves))
        return sess.decode(torch.tensor(primers, device=self.device), self._max_new())

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
        sess = self._session(1)
        sess.encode(self._features([self._wave(audio, sr)]))
        t = self.tokenizer
        sess.decode(torch.tensor([[t.sos]], device=self.device), max_new_tokens=1)
        names = list(t.languages)
        ids = torch.tensor([t.languages[n] for n in names], device=self.device)
        probs = sess.logits[0, ids].softmax(-1)
        top = probs.topk(min(top_k, len(names)))
        return [(names[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]

    @torch.inference_mode()
    def transcribe_long(
        self,
        audio: AudioLike,
        *,
        lang: str = "eng",
        task: str = "asr",
        sr: int | None = None,
        condition_on_prev_text: bool = False,
        end_time_threshold: float = 29.0,
        skip_last_chunk_seconds: float = 0.2,
    ) -> Result:
        """Long-form decoding with timestamps (port of ESPnet's ``Speech2Text.decode_long``)."""
        t, cfg = self.tokenizer, self.cfg
        wav = self._wave(audio, sr)
        fs, seg_len = cfg.sample_rate, cfg.num_samples
        end_id = t.first_time + round(end_time_threshold / cfg.time_resolution)
        segments: list[Segment] = []
        offset, prev = 0, None
        while offset < len(wav):
            chunk = wav[offset : offset + seg_len]
            if len(chunk) / fs < skip_last_chunk_seconds:
                break
            primer = self._primer(lang, task, True, prev if condition_on_prev_text else None)
            ids = self._decode([chunk], [primer])[0]
            times = [i for i, x in enumerate(ids) if t.is_time(x)]
            if len(times) == 1:  # only a start time
                ids.append(t.last_time)
                times.append(len(ids) - 1)
            if not times:
                segments.append(Segment(offset / fs, (offset + len(chunk)) / fs, t.decode(ids)))
                offset += seg_len
                continue
            if len(times) % 2 == 0:
                if len(times) > 2 and ids[times[-1]] > end_id:  # last utterance is incomplete
                    next_start = ids[times[-2]]
                    times = times[:-2]
                else:
                    next_start = ids[times[-1]]
            else:  # last utterance has only a start time
                next_start = ids[times[-1]]
                times = times[:-1]
            res = self._result(ids[: times[-1] + 1] if times else [], offset / fs)
            segments += res.segments
            prev = "".join(s.text for s in res.segments)
            advance = round((next_start - t.first_time) * cfg.time_resolution * fs)
            offset += advance if advance > 0 else seg_len
        text = " ".join(s.text.strip() for s in segments if s.text.strip())
        return Result(text=text, tokens=[], segments=segments)


def _load_frontend(cfg, checkpoint: Path, device: torch.device) -> Frontend:
    """Only the fp32 frontend buffers are needed at runtime when the engines are cached."""
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fe = Frontend(cfg)
    fe.load_state_dict(
        {"melmat": sd["frontend.logmel.melmat"], "mean": sd["normalize.mean"], "std": sd["normalize.std"]}
    )
    return fe.to(device).eval()


__all__ = ["OWSM", "OWSMModel", "Result", "Segment"]
