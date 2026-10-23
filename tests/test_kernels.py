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
    bufs = _Verify(cfg, R, W, {"device": torch.device("cuda"), "dtype": torch.float16}, rules, notime)
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
