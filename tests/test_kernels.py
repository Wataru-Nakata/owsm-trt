"""Triton kernels against the PyTorch code they replace (CUDA only)."""

import dataclasses

import pytest
import torch
from test_model import TINY

from owsm_trt import kernels
from owsm_trt.runtime import TimestampRules, _Verify

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or kernels.triton is None, reason="needs CUDA + Triton"
)


@pytest.mark.parametrize("rules_on", [False, True], ids=["free", "rules"])
def test_choose_matches_torch(rules_on):
    torch.manual_seed(0)
    V, R, W = 3000, 64, 8
    first_time, last_time, notime_tok, eos = 100, 200, 50, 2
    cfg = dataclasses.replace(TINY, vocab_size=V)
    notime = torch.rand(R, device="cuda") < 0.2
    rules = (
        TimestampRules(notime_tok, first_time, last_time, eos, V, torch.device("cuda")) if rules_on else None
    )
    bufs = _Verify(cfg, R, W, {"device": torch.device("cuda"), "dtype": torch.float16}, rules, notime, eos)
    # windows of text and timestamps, and every rule state: before the first token, inside / after a pair
    tok = torch.randint(300, V, (R, W), device="cuda")
    tok = torch.where(
        torch.rand(R, W, device="cuda") < 0.3,
        torch.randint(first_time, last_time + 1, (R, W), device="cuda"),
        tok,
    )
    count = torch.randint(0, 4, (R, 1), device="cuda")
    last = torch.where(count > 0, torch.randint(first_time, last_time + 1, (R, 1), device="cuda"), -1)
    first = torch.randint(-1, W, (R, 1), device="cuda")
    bufs.inp.copy_(torch.cat([tok, torch.zeros_like(count), count, last, first], 1))
    bufs.tok.copy_(tok)
    bufs.logits.normal_()
    bufs.logits[:, :, eos] += 2.0  # make the rules matter for <eos> too
    ref = bufs.predict()
    out = torch.full_like(ref, -7)
    kernels.choose(bufs.logits, bufs.inp, bufs.notime, out, rules)
    assert torch.equal(out, ref)


def test_write_kv_matches_scatter():
    torch.manual_seed(0)
    L, R, H, S, D, C = 3, 5, 4, 7, 64, 32
    new_kv = torch.randn(L, 2, R, H, S, D, device="cuda", dtype=torch.float16)
    offs = torch.randint(0, C - S, (R,), device="cuda")
    k, v = (torch.randn(L, R, H, C, D, device="cuda", dtype=torch.float16) for _ in range(2))
    rk, rv = k.clone(), v.clone()
    slots = (offs.unsqueeze(1) + torch.arange(S, device="cuda"))[None, :, None, :, None].expand(L, R, H, S, D)
    rk.scatter_(3, slots, new_kv[:, 0])
    rv.scatter_(3, slots, new_kv[:, 1])
    kernels.write_kv(new_kv, k, v, offs)
    assert torch.equal(k, rk) and torch.equal(v, rv)


@pytest.mark.parametrize("W", [1, 2, 5, 16, 32])
def test_accept_matches_torch(W):
    torch.manual_seed(W)
    R, eos = 33, 3
    bufs = _Verify(TINY, R, W, {"device": torch.device("cuda"), "dtype": torch.float16}, None, None, eos)
    tok = torch.randint(0, 6, (R, W), device="cuda")  # few tokens: long accepted runs, <eos> among them
    pred = torch.where(torch.rand(R, W, device="cuda") < 0.8, tok.roll(-1, 1), torch.randint_like(tok, 0, 6))
    first = torch.randint(-1, W, (R, 1), device="cuda")  # a primer ending anywhere, or none
    bufs.inp.copy_(
        torch.cat([tok, torch.zeros_like(first), torch.zeros_like(first), -torch.ones_like(first), first], 1)
    )
    bufs.pred.copy_(pred)
    ref = bufs.accept()
    kernels.accept(bufs.pred, bufs.inp, bufs.res, eos)
    assert torch.equal(bufs.res, ref)
    assert W < 5 or ref[:, 0].max() > 1
