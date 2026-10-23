"""The decoder megakernel against Decoder.step and in the decoding sessions (CUDA + Triton only)."""

import copy
import dataclasses

import pytest
import torch
from test_model import TINY

from owsm_trt import megakernel
from owsm_trt.modeling import OWSMModel
from owsm_trt.runtime import GreedySession, TorchBackend

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not megakernel.available(torch.device("cuda")),
    reason="needs CUDA + Triton",
)

# head size 64 as in every OWSM model (tl.dot needs >= 16)
CFG = dataclasses.replace(
    TINY, d_model=256, n_heads=4, dec_heads=4, dec_ffn=512, dec_layers=3, vocab_size=1000
)


@pytest.fixture(scope="module")
def model() -> OWSMModel:
    torch.manual_seed(0)
    m = OWSMModel(CFG).eval().requires_grad_(False)
    m.frontend.melmat.uniform_(0, 1)
    for name, p in m.decoder.named_parameters():  # non-trivial LayerNorms and biases
        if "norm" in name:
            p.add_(0.2 * torch.randn_like(p))
        elif name.endswith("bias"):
            p.copy_(0.2 * torch.randn_like(p))
    m.decoder.output_layer.weight.mul_(8.0)  # clear argmax margins
    m.fuse_projections()
    return m.cuda()


@pytest.mark.parametrize("rows,window", [(1, 1), (2, 5), (1, 32), (4, 16), (16, 16), (16, 1)])
def test_step_matches_decoder(model, rows, window):
    dec = model.decoder
    L, H, dk, C, T = CFG.dec_layers, CFG.dec_heads, CFG.d_model // CFG.dec_heads, 64, CFG.enc_frames
    torch.manual_seed(rows * 100 + window)
    dev = torch.device("cuda")
    tokens = torch.randint(0, CFG.vocab_size, (rows, window), device=dev)
    offset = torch.randint(0, C - window, (rows,), device=dev)
    offset[0] = 0  # an empty cache
    f16 = {"device": dev, "dtype": torch.float16}
    k, v = torch.randn(L, rows, H, C, dk, **f16), torch.randn(L, rows, H, C, dk, **f16)
    ck, cv = torch.randn(L, rows, H, T, dk, **f16), torch.randn(L, rows, H, T, dk, **f16)
    ref, ref_kv = dec.step(
        tokens, offset, list(k.float()), list(v.float()), list(ck.float()), list(cv.float())
    )
    logits = torch.zeros(rows, window, CFG.vocab_size, device=dev)
    new_kv = torch.zeros(L, 2, rows, H, window, dk, **f16)
    k2, v2 = k.clone(), v.clone()
    mk = megakernel.MegaDecoder(dec, dev)
    run = mk.step_fn(tokens, offset, k2, v2, ck, cv, logits, new_kv)
    assert run.writes_cache
    run()
    mk.check()
    scale = ref.abs().max().item()
    assert (logits - ref).abs().max().item() < 0.02 * scale
    assert (logits.argmax(-1) == ref.argmax(-1)).float().mean().item() >= 0.95
    torch.testing.assert_close(new_kv.float(), ref_kv, atol=0.05, rtol=0.02)
    slots = (offset[:, None] + torch.arange(window, device=dev))[None, :, None, :, None].expand(
        L, rows, H, window, dk
    )
    k.scatter_(3, slots, new_kv[:, 0])
    v.scatter_(3, slots, new_kv[:, 1])
    assert torch.equal(k2, k) and torch.equal(v2, v)  # the new keys/values at each row's offset, nothing else


def _session(backend, batch, windows=()):
    return GreedySession(
        backend, CFG, batch, 64, eos=1, dtype=torch.float16, device=torch.device("cuda"), buckets=[16, 32, 64],
        draft_windows=windows,
    )  # fmt: skip


@pytest.mark.parametrize("drafted", [False, True], ids=["steps", "drafted"])
def test_greedy_session_matches_torch(model, drafted):
    """Greedy decoding with the megakernel for decoder calls gives the PyTorch backend's output."""
    m16 = copy.deepcopy(model).half()
    m16.frontend.float()
    torch_be = TorchBackend(m16)
    mega_be = megakernel.MegaBackend(
        TorchBackend(m16), megakernel.MegaDecoder(model.decoder, torch.device("cuda"))
    )
    batch = 3
    feats = m16.frontend(torch.randn(batch, CFG.num_samples, device="cuda"))
    primer = torch.tensor([[2, 3, 4]] * batch, device="cuda")
    outs = []
    for be in (torch_be, mega_be):
        sess = _session(be, batch, (2, 4, 8) if drafted else ())
        sess.encode(feats)
        if drafted:
            drafts = [[5, 6, 7, 8, 9, 10], [11, 12], []]
            out, _ = sess.decode_drafted(primer, drafts, 20)
        else:
            out = sess.decode(primer, 20)
        outs.append(out)
    assert outs[0] == outs[1]
