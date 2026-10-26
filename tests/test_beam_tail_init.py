"""Prefix-only stores require zero unused caches across request, bucket and graph reuse."""

import pytest
import torch
from test_beam import session
from test_beam_kernels import device as device

from owsm_trt import beam_kernels


@pytest.mark.parametrize("skip_identity", [False, True])
@pytest.mark.parametrize("pos", [0, 1, 31, 33, 65])
@pytest.mark.parametrize("mapping", [[0, 1, 2, 3, 4], [2, 2, 2, 2, 2], [4, 1, 1, 3, 0]])
def test_prefix_only_copy_leaves_initialized_tail(device, skip_identity, pos, mapping):
    shape = (1, 5, 2, 65, 64)
    k = (torch.arange(41600, device=device) % 97).reshape(shape).float()
    v = -k - 1
    parent = torch.tensor(mapping, device=device)
    # Nonzero finite sentinels distinguish "untouched" from "zeroed". Runtime clears to zero.
    for buf in (k, v):
        buf[:, :, :, pos:] = 7
    expected = [buf.index_select(1, parent) for buf in (k, v)]
    scratch = torch.full((2*k.numel(),), torch.nan, device=device)
    beam_kernels.reorder_kv(k, v, scratch, parent, torch.tensor([pos], device=device),
                           prefix=True, skip_identity=skip_identity, clear_tail=False)
    for actual, ref in zip((k, v), expected):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)


class TailCheckingBackend:
    def step_fn(self, tokens, offset, k, v, ck, cv, logits, new_kv, kind="decode"):
        def run():
            pos = int(offset[0])
            for cache in (k, v):
                assert torch.isfinite(cache).all()
                assert (cache[:, :, :, pos:] == 0).all(), (kind, pos, cache.shape)
            logits.fill_(-10.)
            logits[..., 0] = -100.
            logits[..., 1:4] = torch.tensor([1., 2., 3.])
            new_kv.fill_(3.)
        return run


def test_once_initialization_before_prefill_growth_and_repeated_requests():
    s = session(batch=3, backend=TailCheckingBackend(), kv_clear_once=True)
    assert s.kv_reorder == "torch" and not s.kv_clear_once
    # Exercise host initialization ordering on tiny CPU state; the GPU tests cover Triton stores.
    s.kv_clear_once = True
    for length, limit in ((3, 13), (4, 12), (8, 8), (15, 1), (3, 10)):
        for pair in s.caches.values():
            for cache in pair:
                cache.fill_(torch.nan)
        out = s.decode(torch.full((3, length), 2), limit)
        assert all(len(ids) == limit for ids in out)


@pytest.mark.parametrize("mode", ["torch", "full"])
def test_once_initialization_requires_prefix_mode(mode):
    with pytest.raises(ValueError, match="kv_clear_once"):
        session(kv_reorder=mode, kv_clear_once=True)


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_real_decoder_nan_tail_bucket_boundary_and_reuse(device, identity, graphs):
    if device.type != "cuda":
        pytest.skip("real decoder requires compute GPU")
    from test_model import TINY

    from owsm_trt.modeling import OWSMModel
    from owsm_trt.runtime import BeamSession, TorchBackend

    torch.manual_seed(19)
    model = OWSMModel(TINY).eval().to(device)
    args = (TorchBackend(model), TINY, 3, 3, 32, -1, torch.float32, device)
    kwargs = dict(use_cuda_graph=graphs, buckets=[4, 8, 16, 32], diagnostics=True,
                  kv_reorder="identity" if identity else "prefix")
    ref = BeamSession(*args, **kwargs)
    fast = BeamSession(*args, **kwargs, kv_clear_once=True)
    for primer_len, limit in ((3, 29), (4, 28), (8, 24), (16, 16), (31, 1), (3, 20)):
        for pair in fast.caches.values():
            for cache in pair:
                cache.fill_(torch.nan)
        primer = torch.full((3, primer_len), 2, device=device)
        assert fast.decode(primer, limit) == ref.decode(primer, limit)
        torch.testing.assert_close(fast.logits, ref.logits, rtol=0, atol=0)
        n = int(ref.step) + 1
        for key in ("scores", "state"):
            torch.testing.assert_close(fast._beam_history[key][:n], ref._beam_history[key][:n], rtol=0, atol=0)
