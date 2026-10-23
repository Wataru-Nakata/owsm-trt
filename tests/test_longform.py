"""transcribe_long on a tiny random model (CPU): window bookkeeping, drafts, the batched mode."""

import dataclasses

import pytest
import torch

from owsm_trt.build import BuildOptions
from owsm_trt.config import OWSMConfig
from owsm_trt.modeling import OWSMModel
from owsm_trt.pipeline import OWSM, _window_cut, pause_cuts
from owsm_trt.runtime import TorchBackend

FIRST_T, LAST_T, EOS = 20, 49, 1
# 30 timestamps span the 2 s window
TINY = OWSMConfig(
    vocab_size=50, d_model=32, n_heads=4, enc_layers=2, enc_ffn=64, cgmlp_units=64, cgmlp_kernel=5,
    merge_kernel=3, subsample="conv2d8", dec_layers=2, dec_ffn=64, dec_heads=4, n_mels=16, n_fft=64,
    win_length=48, hop_length=32, sample_rate=1600, speech_seconds=2.0, time_resolution=2.0 / 29,
)  # fmt: skip


class _Tokenizer:
    sos, eos, notime, blank, na, sop = 2, EOS, 5, 0, 6, 7
    first_time, last_time = FIRST_T, LAST_T
    punctuation = frozenset({8})
    not_drafted = frozenset(range(FIRST_T, LAST_T + 1)) | {5}

    def lang_id(self, lang: str) -> int:
        return 3

    def task_id(self, task: str) -> int:
        return 4

    def is_time(self, i: int) -> bool:
        return FIRST_T <= i <= LAST_T

    def encode(self, text: str) -> list[int]:
        return [9 + len(w) % 10 for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"w{i}" for i in ids if i > 8 and not self.is_time(i))


@pytest.fixture(scope="module")
def model() -> OWSMModel:
    torch.manual_seed(0)
    m = OWSMModel(TINY).eval().requires_grad_(False)
    m.frontend.melmat.uniform_(0, 1)
    with torch.no_grad():  # make <eos> and timestamps likely enough that windows end in different ways
        m.decoder.output_layer.bias[EOS] += 1.5
        m.decoder.output_layer.bias[FIRST_T : LAST_T + 1] += 1.0
    return m


def _owsm(model: OWSMModel, draft: bool = True) -> OWSM:
    opts = BuildOptions(dtype="float32", max_batch=4, capacity=64)
    return OWSM(
        TINY, _Tokenizer(), model.frontend, TorchBackend(model), opts, torch.device("cpu"), use_cuda_graph=False,
        ctc_draft=draft,
    )  # fmt: skip


def test_window_cut():
    tok = _Tokenizer()
    end = FIRST_T + 26
    t = FIRST_T
    # complete utterances: keep all, continue at the last end time
    assert _window_cut([t, 10, t + 5, t + 6, 11, t + 9], tok, end) == (
        [t, 10, t + 5, t + 6, 11, t + 9],
        t + 9,
    )
    # the last one ends after the threshold: dropped, continue at its start
    assert _window_cut([t, 10, t + 5, t + 6, 11, t + 28], tok, end) == ([t, 10, t + 5], t + 6)
    # ... unless it is the only one
    assert _window_cut([t + 1, 10, t + 28], tok, end) == ([t + 1, 10, t + 28], t + 28)
    # the last one has no end time: dropped, continue at its start
    assert _window_cut([t, 10, t + 5, t + 6, 11], tok, end) == ([t, 10, t + 5], t + 6)
    # a single start time: closed at the end of the window
    assert _window_cut([t + 3, 10, 11], tok, end) == ([t + 3, 10, 11, LAST_T], LAST_T)
    assert _window_cut([10, 11], tok, end) == ([10, 11], None)


def test_pause_cuts():
    torch.manual_seed(0)
    sr, max_len = 1600, 3200
    wav = torch.randn(20000)
    quiet = [2500, 5000, 7600]  # 0.25 s pauses (centres) within reach of every cut
    for q in quiet:
        wav[q - 200 : q + 200] *= 1e-3
    cuts = pause_cuts(wav, max_len, sr, search_seconds=1.0)  # the last second of each piece
    assert cuts[0] == 0 and cuts[-1] == len(wav)
    assert all(0 < b - a <= max_len for a, b in zip(cuts, cuts[1:]))
    for q, c in zip(quiet, cuts[1:]):
        assert abs(c - q) < 120, (cuts, quiet)  # the cut lands in the pause
    assert pause_cuts(wav[:3000], max_len, sr) == [0, 3000]


def test_sequential_long_form_drafts_are_exact(model):
    """With CTC drafts each window is the greedy output: the same segments as without drafts."""
    torch.manual_seed(1)
    wav = torch.randn(int(TINY.sample_rate * 9.3))
    kw = {"end_time_threshold": 1.8}
    fast, plain = (
        _owsm(model).transcribe_long(wav, **kw),
        _owsm(model, draft=False).transcribe_long(wav, **kw),
    )
    assert fast.segments == plain.segments
    assert len(fast.segments) > 3
    starts = [s.start for s in fast.segments]
    assert starts == sorted(starts) and fast.segments[-1].end <= 9.3 + 2.0
    # a list of recordings: one result each, as one at a time
    both = _owsm(model).transcribe_long([wav, wav[:5000]], **kw)
    assert both[0].segments == fast.segments
    assert both[1].segments == _owsm(model).transcribe_long(wav[:5000], **kw).segments


def test_batched_long_form(model):
    """batch_size > 1: windows cut at pauses, decoded in batches, timestamps shifted to the recording."""
    torch.manual_seed(2)
    sr = TINY.sample_rate
    waves = [torch.randn(int(sr * s)) for s in (7.1, 3.3, 1.2)]
    m = _owsm(model)
    out = m.transcribe_long(waves, batch_size=3)
    assert len(out) == 3
    for wav, res in zip(waves, out):
        cuts = pause_cuts(wav, TINY.num_samples, sr)
        want = []
        assert all(b - a >= TINY.num_samples // 2 for a, b in zip(cuts, cuts[1:-1]))
        for a, b in zip(cuts, cuts[1:]):
            if b - a < 0.2 * sr:  # skip_last_chunk_seconds
                continue
            one = m.transcribe(wav[a:b], timestamps=True)
            want += [(round(s.start + a / sr, 2), round(s.end + a / sr, 2), s.text) for s in one.segments]
        assert [dataclasses.astuple(s) for s in res.segments] == want
        assert res.text == " ".join(s.text for s in res.segments if s.text)
    with pytest.raises(ValueError):
        m.transcribe_long(waves[0], batch_size=2, condition_on_prev_text=True)
