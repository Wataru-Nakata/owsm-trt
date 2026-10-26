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


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("rules_on", [False, True], ids=["free", "rules"])
def test_choose_matches_torch(rules_on, split):
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
    kernels.choose(
        bufs.logits,
        bufs.inp,
        bufs.notime,
        out,
        rules,
        workspace=kernels.choose_workspace(R, W, V, "cuda") if split else None,
    )
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


@pytest.mark.parametrize(
    "shape", [(1, 1, 64), (1, 8, 64), (5, 8, 64), (16, 16, 128), (3, 32, 512), (2, 17, 64), (2, 256, 512)]
)
def test_attention_kernels_match_reference(shape):
    """attn_plugin's Triton kernels (every tactic) == their PyTorch implementations; rows at their own
    offsets, garbage in the cache slots past them."""
    from owsm_trt import attn_plugin as ap

    torch.manual_seed(0)
    B, S, C = shape
    H, D, T = 4, 64, 37
    f16 = {"device": "cuda", "dtype": torch.float16}
    qkv = torch.randn(B, S, 3 * H * D, **f16)
    k_cache, v_cache = torch.randn(B, H, C, D, **f16), torch.randn(B, H, C, D, **f16)
    offset = torch.randint(0, C - S + 1, (B,), device="cuda")
    offset[0] = C - S
    ref_self = ap.self_attn(qkv, k_cache, v_cache, offset)
    q, k, v = torch.randn(B, S, H * D, **f16), torch.randn(B, H, T, D, **f16), torch.randn(B, H, T, D, **f16)
    ref_cross = ap.cross_attn(q, k, v)
    for config in ap.TACTICS["self_attn"]:
        y = ap.self_attn_triton(qkv, k_cache, v_cache, offset, config=config)
        torch.testing.assert_close(y, ref_self, rtol=1e-2, atol=2e-3)
    # offsets out of range (TensorRT times the plugin on arbitrary data) are clamped to the cache
    assert ap.self_attn_triton(qkv, k_cache, v_cache, torch.full_like(offset, 1 << 40)).isfinite().all()
    for config in ap.TACTICS["cross_attn"]:
        torch.testing.assert_close(
            ap.cross_attn_triton(q, k, v, config=config), ref_cross, rtol=1e-2, atol=2e-3
        )


@pytest.mark.parametrize("shape", [(1, 1, 2051), (1, 16, 51865), (3, 8, 50003), (8, 4, 3000)])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@torch.inference_mode()
def test_split_choose_edges_graph_and_nonfinite(shape, skip, fused):
    R, W, V = shape
    rules = TimestampRules(50, 100, 200, 2, V, torch.device("cuda"))
    cfg = dataclasses.replace(TINY, vocab_size=V)
    notime = torch.arange(R, device="cuda") % 2 == 1
    bufs = _Verify(
        cfg,
        R,
        W,
        {"device": torch.device("cuda"), "dtype": torch.float16},
        rules,
        notime,
        2,
        split_choose=True,
    )
    bufs.inp[:, :W] = 300
    bufs.inp[:, W + 1] = 1  # inside timestamp pair: text and later timestamp, no EOS
    bufs.inp[:, W + 2] = 100
    bufs.inp[:, W + 3] = -1
    expected = torch.empty_like(bufs.pred)

    def call():
        if fused:
            kernels.choose_accept(
                bufs.logits,
                bufs.inp,
                notime,
                bufs.pred,
                rules,
                bufs.res,
                2,
                bufs.choose_workspace,
                skip_notime=skip,
            )
        else:
            kernels.choose(
                bufs.logits, bufs.inp, notime, bufs.pred, rules, skip, workspace=bufs.choose_workspace
            )

    call()  # JIT outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for mode in ("random", "tie", "near_tie", "invalid", "inf", "nan", "all_nan", "first", "after_end"):
        bufs.logits.normal_()
        bufs.inp[:, W + 1] = 1
        bufs.inp[:, W + 2] = 100
        bufs.inp[:, W + 3] = -1
        bufs.inp[:, :W] = 300
        if mode in ("tie", "near_tie", "invalid", "inf"):
            bufs.logits.fill_(float("-inf"))
        if mode in ("tie", "near_tie", "inf"):
            bufs.logits[..., 2047] = 1 if mode != "inf" else float("inf")
            bufs.logits[..., 2049] = 1 if mode != "inf" else float("inf")
            if mode == "near_tie":
                bufs.logits[..., 2049] = torch.nextafter(
                    torch.tensor(1.0, device="cuda"), torch.tensor(2.0, device="cuda")
                )
        if mode == "nan":
            bufs.logits[..., 2050:] = float("nan")
            bufs.logits[..., 300] = float("nan")
        if mode == "all_nan":
            bufs.logits.fill_(float("nan"))
        if mode == "first":
            bufs.inp[:, W + 3] = 0
            bufs.logits[..., 100] = bufs.logits[..., 200] = 50
        if mode == "after_end":
            bufs.inp[:, :W] = 200
            bufs.inp[:, W + 1] = 2
            bufs.inp[:, W + 2] = 200
            bufs.logits[..., 2] = bufs.logits[..., 200] = 50
        expected.fill_(-7)
        bufs.pred.fill_(-7)
        kernels.choose(bufs.logits, bufs.inp, notime, expected, rules, skip)
        graph.replay()
        assert torch.equal(bufs.pred, expected), mode
        if fused:
            assert torch.equal(bufs.res, bufs.accept()), mode


@pytest.mark.parametrize("W", [1, 5, 32])
@torch.inference_mode()
def test_fused_choose_accepts_runs_and_eos(W):
    R, eos, V = 7, 3, 2053
    cfg = dataclasses.replace(TINY, vocab_size=V)
    notime = torch.zeros(R, dtype=torch.bool, device="cuda")
    b = _Verify(
        cfg,
        R,
        W,
        {"device": torch.device("cuda"), "dtype": torch.float16},
        None,
        notime,
        eos,
        split_choose=True,
    )
    tok = torch.full((R, W), 5, device="cuda", dtype=torch.long)
    tok[1, W // 2] = eos
    first = torch.arange(R, device="cuda").remainder(W).unsqueeze(1)
    b.inp.copy_(
        torch.cat([tok, torch.zeros_like(first), torch.zeros_like(first), -torch.ones_like(first), first], 1)
    )
    b.pred.copy_(tok.roll(-1, 1))
    b.pred[2, W // 2] = 8
    b.logits.fill_(-10)
    b.logits.scatter_(-1, b.pred.unsqueeze(-1), 10)
    expected = b.accept().clone()
    kernels.choose_accept(b.logits, b.inp, notime, b.pred, None, b.res, eos, b.choose_workspace)
    assert torch.equal(b.res, expected)
    assert W < 5 or expected[:, 0].max() > 1
