"""Parent gathers on small states; real decoder/cache checks require CUDA allocation."""

import pytest
import torch
from test_beam import session
from test_beam_kernels import device as device

from owsm_trt import beam_kernels


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("pos", [0, 1, 7, 8])
@pytest.mark.parametrize("mapping", [[0, 1, 2, 3, 4], [2, 2, 2, 2, 2],
                                     [4, 1, 1, 3, 0], [4, 3, 2, 1, 0]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_kv_gather_exact_prefix_and_duplicate_parents(device, prefix, pos, mapping, dtype):
    shape = (2, 5, 2, 8, 3)
    k = (torch.arange(480, device=device) % 97).reshape(shape).to(dtype)
    v = -k - 1
    parent = torch.tensor(mapping, device=device)
    expected = [buf.index_select(1, parent) for buf in (k, v)]
    if prefix:
        # Unused input/scratch NaNs must never survive into the output's masked V tail.
        for src, target in zip((k, v), expected):
            src[:, :, :, pos:] = torch.nan
            target[:, :, :, pos:] = 0
    scratch = torch.full((2 * k.numel(),), torch.nan, dtype=dtype, device=device)
    beam_kernels.reorder_kv(k, v, scratch, parent, torch.tensor([pos], device=device), prefix)
    for actual, ref in zip((k, v), expected):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("pos", [31, 33, 65])
def test_kv_gather_tile_boundary_and_workspace_reuse(device, prefix, pos):
    shape = (1, 5, 2, 65, 64)
    k = (torch.arange(41600, device=device) % 97).reshape(shape).float()
    v = -k - 1
    scratch = torch.empty(2 * k.numel(), device=device)
    for mapping in ([4, 1, 1, 3, 0], [2, 2, 2, 2, 2]):
        parent = torch.tensor(mapping, device=device)
        expected = [buf.index_select(1, parent) for buf in (k, v)]
        if prefix:
            for buf in expected:
                buf[:, :, :, pos:] = 0
        beam_kernels.reorder_kv(k, v, scratch, parent, torch.tensor([pos], device=device), prefix)
        for actual, ref in zip((k, v), expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["full", "prefix"])
@torch.inference_mode()
def test_real_decoder_reorder_graphs_and_repeated_decode(device, mode):
    if device.type != "cuda":
        pytest.skip("Decoder inference and CUDA graphs require a compute allocation")
    from test_model import TINY

    from owsm_trt.modeling import OWSMModel
    from owsm_trt.runtime import BeamSession, TorchBackend

    torch.manual_seed(19)
    model = OWSMModel(TINY).eval().to(device)
    backend = TorchBackend(model)
    args = (backend, TINY, 3, 3, 32, -1, torch.float32, device)
    ref = BeamSession(*args, buckets=[4, 8, 16, 32], kv_reorder="torch", diagnostics=True)
    fast = BeamSession(*args, buckets=[4, 8, 16, 32], kv_reorder=mode, diagnostics=True)
    for primer_len, limit in ((3, 29), (9, 7), (3, 20)):
        primer = torch.full((3, primer_len), 2, device=device)
        assert fast.decode(primer, limit) == ref.decode(primer, limit)
        torch.testing.assert_close(fast.logits, ref.logits, rtol=0, atol=0)
        n = int(ref.step) + 1
        for key in ("scores", "state"):
            torch.testing.assert_close(fast._beam_history[key][:n], ref._beam_history[key][:n], rtol=0, atol=0)


def test_kv_reorder_cpu_fallback_and_invalid_option():
    s = session(kv_reorder="prefix")
    assert s.kv_reorder == "torch" and s._kv_scratch is None
    with pytest.raises(ValueError, match="kv_reorder"):
        session(kv_reorder="unknown")
