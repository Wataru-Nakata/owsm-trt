"""Exact state/candidate checks on CUDA, or small CPU states with TRITON_INTERPRET=1."""

import copy
import os

import pytest
import torch
from test_beam import ready, session

from owsm_trt import beam_kernels
from owsm_trt.runtime import END_DETECT_D, END_DETECT_M, TimestampRules


@pytest.fixture
def device():
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("CUDA or Triton interpreter required")


@pytest.mark.parametrize("step, force, bound", [(0, False, True), (7, False, True),
                                               (7, False, False), (12, True, True)])
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("ending", ["mixed", "no_eos", "all_eos", "invalid"])
def test_finish_exact_state(device, step, force, bound, diagnostics, ending):
    s = session(batch=3, beam=5, device=device, diagnostics=diagnostics, upper_bound=bound,
                fused_rank=False, fused_finish=False)
    ready(s)
    s.step.fill_(step)
    s.pos.fill_(step + 3)
    s.last_step.fill_(step if force else 14)
    s.best_score.copy_(torch.tensor([-4., -torch.inf, -6.], device=device))
    s.done[2] = True
    s.seq.copy_(torch.arange(s.seq.numel(), device=device).view_as(s.seq))
    s.score.copy_(torch.tensor([-4., -4., -5., -torch.inf, -10.] * 3, device=device))
    # Explicitly exercise all three end_detect lengths and its score subtraction.
    if step >= 2:
        s.ended_by_len[:, step-2:step+1] = -30
    tokens = torch.tensor([0, 0, 1, 2, 3] * 3, device=device)
    valid = torch.isfinite(s.score)
    if ending == "no_eos":
        tokens.fill_(1)
    elif ending == "all_eos":
        tokens.zero_()
    elif ending == "invalid":
        valid.zero_()
    expected = copy.deepcopy(s)
    expected._finish(tokens, valid)
    beam_kernels.finish(s, tokens, valid, END_DETECT_M, END_DETECT_D)
    for attr in ("active", "done", "best_score", "best_seq", "best_end", "ended_by_len", "tok"):
        torch.testing.assert_close(getattr(s, attr), getattr(expected, attr), rtol=0, atol=0)
    if diagnostics:
        for field in ("scores", "state"):
            torch.testing.assert_close(s._beam_history[field], expected._beam_history[field], rtol=0, atol=0)


@pytest.mark.parametrize("rules_enabled", [False, True])
@pytest.mark.parametrize("vocab", [8, 2053])
def test_candidates_exact_masks_errors_and_ties(device, rules_enabled, vocab):
    s = session(batch=3, beam=5, device=device, fused_rank=False, fused_finish=False)
    ready(s)
    s.logits = torch.zeros(s.rows, 1, vocab, device=device)
    if rules_enabled:
        s.rules = TimestampRules(vocab - 1, 5, vocab - 2, 0, vocab, device)
    # Different rule states, inactive and done parents, partial/all -inf, NaN and +Inf rows.
    s.active.fill_(True)
    s.done[2] = True
    s.active[9] = False
    s.logits[:, 0].copy_(torch.arange(vocab, device=device).float().expand(15, -1) * -.1)
    s.logits[0, 0].zero_()  # exact ties
    s.logits[1, 0, 2] = -torch.inf
    s.logits[2, 0].fill_(-torch.inf)
    s.logits[3, 0, -1] = torch.nan
    s.logits[4, 0, -1] = torch.inf
    s.logits[9:, 0, 1] = torch.nan  # must not add errors for inactive/done rows
    s.t_notime[0] = True
    s.t_first[1] = True
    s.t_count.copy_(torch.tensor([0, 0, 1, 2, 3] * 3, device=device))
    s.t_last.fill_(5)
    s.tok[:, 0] = torch.tensor([1, 1, 5, 6, 2] * 3, device=device)
    s.score[5] = -2e10
    expected = copy.deepcopy(s)
    rank = expected._rank()
    logits = s.logits[:, -1]
    bad = (torch.isnan(logits) | torch.isposinf(logits)).any(-1)
    live = s.active & ~s.done.repeat_interleave(s.beam)
    usable = live & ~bad & torch.isfinite(logits).any(-1)
    ref = s._rules_mask(torch.log_softmax(torch.where(usable[:, None], logits, 0.), -1))
    ref = (s.score[:, None] + ref).masked_fill(~usable[:, None], -torch.inf)
    s._candidates = torch.empty_like(logits)
    s._bad_blocks = torch.empty(s.rows, (vocab + 1023) // 1024, dtype=torch.bool, device=device)
    result = beam_kernels.candidates(s, torch.log_softmax(logits, -1))
    torch.testing.assert_close(result, ref, rtol=0, atol=0)
    torch.testing.assert_close(s.numerical_error, expected.numerical_error)
    s.fused_rank = True  # Also exercises the interpreter on CPU.
    for got, want in zip(s._rank(), rank):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("fused_rank, fused_finish", [(True, False), (False, True), (True, True)])
def test_fused_graph_bucket_crossing(device, fused_rank, fused_finish):
    if device.type != "cuda":
        pytest.skip("CUDA graph required")
    rules = TimestampRules(7, 5, 6, 0, 8, device)
    args = dict(batch=3, beam=3, device=device, rules=rules, diagnostics=True,
                use_cuda_graph=True, upper_bound=False)
    reference = session(**args, fused_rank=False, fused_finish=False)
    fused = session(**args, fused_rank=fused_rank, fused_finish=fused_finish)
    primer = torch.tensor([[2, 3, 7]] * 3, device=device)
    for limit in (13, 7, 1):
        assert fused.decode(primer, limit) == reference.decode(primer, limit)
        for key in ("scores", "state"):
            n = int(reference.step) + 1
            torch.testing.assert_close(fused._beam_history[key][:n], reference._beam_history[key][:n],
                                       rtol=0, atol=0)


def test_fused_cpu_falls_back():
    s = session(fused_rank=True, fused_finish=True)
    assert not s.fused_rank and not s.fused_finish
    assert s._candidates is None and s._bad_blocks is None
