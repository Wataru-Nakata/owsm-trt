"""Prefill sharing preserves the full candidate layout, ties, masks and decode state."""

import os

import pytest
import torch
from test_beam import ready, session

from owsm_trt import beam_kernels
from owsm_trt.runtime import TimestampRules


@pytest.fixture
def device():
    if beam_kernels.triton is None:
        pytest.skip("Triton required")
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("CUDA or Triton interpreter required")


@pytest.mark.parametrize("beam,vocab", [(2, 8), (5, 2053), (5, 50002), (8, 2053)])
@pytest.mark.parametrize("prompt", [1, 37])
@pytest.mark.parametrize("kind", ["finite", "tie", "near_tie", "single", "exhausted", "nan", "inf"])
def test_shared_prefill_candidates_exact(device, beam, vocab, prompt, kind):
    s = session(batch=3, beam=beam, device=device, fused_rank=True, fused_finish=False)
    ready(s)
    s.fused_rank = True  # Also exercises the Triton interpreter on CPU.
    s.logits = torch.empty(s.rows, 1, vocab, device=device)
    s._candidates = torch.empty(s.rows, vocab, device=device)
    s._bad_blocks = torch.empty(s.rows, (vocab + 1023) // 1024, dtype=torch.bool, device=device)
    s.rules = TimestampRules(vocab - 1, vocab - 3, vocab - 2, 0, vocab, device)
    s.t_first.fill_(True)
    s.t_notime[s.beam:2*s.beam] = True
    s.done[2] = True
    storage = torch.empty(3, prompt, vocab, device=device)
    logits = storage[:, -1]  # The real prefill returns this noncontiguous view when B,P>1.
    logits.copy_(torch.arange(vocab, device=device).float()[None, :] * -.001)
    if kind == "tie":
        logits.zero_()
    elif kind == "near_tie":
        logits.zero_()
        logits[:, -3] = 1.
        logits[:, -2] = torch.nextafter(torch.tensor(1., device=device), torch.tensor(torch.inf, device=device))
    elif kind == "single":
        logits[0].fill_(-torch.inf)
        logits[0, -3] = 0.  # Exactly one permitted timestamp, fewer than K candidates.
    elif kind == "exhausted":
        logits[0].fill_(-torch.inf)
    elif kind in ("nan", "inf"):
        logits[0, 1] = float(kind)
        logits[2, 1] = float(kind)  # Ended utterance must not acquire a numerical-error flag.
    s.logits[:, -1].copy_(logits.repeat_interleave(beam, 0))
    reference = beam_kernels.candidates(s, torch.log_softmax(s.logits[:, -1], -1)).clone()
    error = s.numerical_error.clone()
    s.numerical_error.zero_()
    s.logits.fill_(torch.nan)  # Shared selection must not read stale per-beam logits.
    actual = beam_kernels.candidates(s, torch.log_softmax(logits, -1), logits)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(s.numerical_error, error, rtol=0, atol=0)
    for left, right in zip(actual.view(3, -1).topk(beam), reference.view(3, -1).topk(beam)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("fused_state", [False, True])
@pytest.mark.parametrize("timestamps", [False, True])
def test_prefill_shared_decode_graph_and_reuse(device, batch, fused_state, timestamps):
    if device.type != "cuda":
        pytest.skip("CUDA graph required")
    rules = TimestampRules(7, 5, 6, 0, 8, device)
    options = dict(batch=batch, beam=5, device=device, rules=rules, diagnostics=True,
                   use_cuda_graph=True, upper_bound=False, fused_state=fused_state)
    reference = session(**options)
    shared = session(**options, prefill_shared=True)
    assert shared.prefill_shared
    for length in (3, 7):
        primer = torch.tensor([[2] * (length - 1) + [3 if timestamps else 7]] * batch, device=device)
        for limit in (16-length, 5, 1):
            assert shared.decode(primer, limit) == reference.decode(primer, limit)
            assert shared.get_diagnostics() == reference.get_diagnostics()
            for key in ("active", "done", "score", "best_score", "best_end", "best_seq", "seq"):
                torch.testing.assert_close(getattr(shared, key), getattr(reference, key), rtol=0, atol=0)


def test_prefill_shared_cpu_fallback():
    s = session(prefill_shared=True)
    assert not s.prefill_shared

