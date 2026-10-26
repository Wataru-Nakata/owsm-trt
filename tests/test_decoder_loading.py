"""Decoder loading exactness; run these model tests in an allocated compute job."""

import hashlib

import pytest
import torch
from test_model import TINY

from owsm_trt.modeling import Decoder
from owsm_trt.pipeline import _load_decoder


@pytest.mark.parametrize("mode", ["empty_init", "skip_init"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("legacy", [False, True])
@torch.inference_mode()
def test_empty_decoder_initialization_preserves_parameters_buffers_and_logits(tmp_path, dtype, legacy, mode):
    torch.manual_seed(79)
    source = Decoder(TINY)
    state = {"decoder." + k: v.to(dtype) for k, v in source.state_dict().items()}
    path = tmp_path / "decoder.pth"
    torch.save(state, path, _use_new_zipfile_serialization=not legacy)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    reference = _load_decoder(TINY, path)
    candidate = _load_decoder(TINY, path, **{mode: True})
    assert not candidate.training
    for a, b in zip(reference.parameters(), candidate.parameters()):
        assert not b.requires_grad and b.dtype == a.dtype == torch.float32
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(reference.pe, candidate.pe, rtol=0, atol=0)
    assert not any(t.is_meta for t in (*candidate.parameters(), *candidate.buffers()))
    rows, window, capacity, length = 3, 1, 16, 13
    shape = (TINY.dec_layers, rows, TINY.dec_heads, capacity, TINY.d_model // TINY.dec_heads)
    k, v = [torch.randn(shape) for _ in range(2)]
    cshape = (*shape[:3], length, shape[-1])
    ck, cv = [torch.randn(cshape) for _ in range(2)]
    tokens = torch.tensor([[3], [5], [7]])
    offsets = torch.tensor([0, 4, 15])
    assert tokens.shape == (rows, window)
    args = (tokens, offsets, list(k), list(v), list(ck), list(cv))
    for a, b in zip(reference.step(*args), candidate.step(*args)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Ordinary strict copy semantics keep writable module parameters independent of the file.
    candidate.embed[0].weight.zero_()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
    assert reference.embed[0].weight.abs().sum() > 0


@pytest.mark.parametrize("mode", ["empty_init", "skip_init"])
@pytest.mark.parametrize("invalid", ["missing", "unexpected", "shape"])
def test_empty_decoder_keeps_strict_checkpoint_errors(tmp_path, invalid, mode):
    state = {"decoder." + k: v for k, v in Decoder(TINY).state_dict().items()}
    if invalid == "missing":
        del state["decoder.output_layer.bias"]
    elif invalid == "unexpected":
        state["decoder.unknown"] = torch.zeros(1)
    else:
        state["decoder.output_layer.bias"] = torch.zeros(1)
    path = tmp_path / "bad.pth"
    torch.save(state, path)
    messages = []
    for empty in (False, True):
        with pytest.raises(RuntimeError) as error:
            _load_decoder(TINY, path, **{mode: empty})
        messages.append(str(error.value))
    assert messages[0] == messages[1]


def test_direct_skip_preserves_rng_and_initializes_nonpersistent_buffers(tmp_path):
    source = Decoder(TINY)
    path = tmp_path / "decoder.pth"
    torch.save({"decoder." + k: v for k, v in source.state_dict().items()}, path)
    before = torch.random.get_rng_state()
    candidate = _load_decoder(TINY, path, skip_init=True)
    assert torch.equal(torch.random.get_rng_state(), before)
    torch.testing.assert_close(candidate.pe, source.pe, rtol=0, atol=0)
    assert type(source.output_layer) is torch.nn.Linear  # Default/full-model construction is unchanged.
    with pytest.raises(ValueError, match="separate ablations"):
        _load_decoder(TINY, path, skip_init=True, empty_init=True)
