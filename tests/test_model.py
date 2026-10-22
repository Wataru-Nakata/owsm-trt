import math

import pytest
import torch

from owsm_trt.audio import resample
from owsm_trt.config import OWSMConfig
from owsm_trt.modeling import OWSMModel
from owsm_trt.runtime import GreedySession, TorchBackend

TINY = OWSMConfig(
    vocab_size=50, d_model=32, n_heads=4, enc_layers=2, enc_ffn=64, cgmlp_units=64, cgmlp_kernel=5,
    merge_kernel=3, subsample_convs=3, dec_layers=2, dec_ffn=64, dec_heads=4, n_mels=16, n_fft=64,
    win_length=48, hop_length=32, sample_rate=1600, speech_seconds=2.0, time_resolution=0.02,
)  # fmt: skip


@pytest.fixture(scope="module")
def model() -> OWSMModel:
    torch.manual_seed(0)
    m = OWSMModel(TINY).eval().requires_grad_(False)
    m.frontend.melmat.uniform_(0, 1)
    return m


def _session(model: OWSMModel, batch: int, capacity: int = 32, eos: int = 1, buckets=None) -> GreedySession:
    return GreedySession(
        TorchBackend(model), TINY, batch, capacity, eos=eos, dtype=torch.float32,
        device=torch.device("cpu"), use_cuda_graph=False, buckets=buckets,
    )  # fmt: skip


def test_shapes(model):
    assert TINY.num_frames == 101 and TINY.enc_frames == 11
    kv = model.encode(torch.randn(3, TINY.num_samples))
    assert kv.shape == (2, 2, 3, 4, 11, 8)


def test_prefill_matches_incremental(model):
    sess = _session(model, batch=2)
    sess.encode(model.frontend(torch.randn(2, TINY.num_samples)))
    tokens = torch.randint(2, 50, (2, 6))
    sk, sv = list(sess.self_k), list(sess.self_v)
    ck, cv = list(sess.cross_k), list(sess.cross_v)
    full, _ = model.decoder.step(tokens, torch.zeros(1, dtype=torch.long), sk, sv, ck, cv)
    for t in range(6):
        logits, new = model.decoder.step(tokens[:, t : t + 1], torch.tensor([t]), sk, sv, ck, cv)
        sess.self_k[:, :, :, t] = new[:, 0, :, :, 0]
        sess.self_v[:, :, :, t] = new[:, 1, :, :, 0]
    torch.testing.assert_close(logits, full, rtol=1e-4, atol=1e-4)


def test_greedy_session_matches_naive(model):
    sess = _session(model, batch=2)
    feats = model.frontend(torch.randn(2, TINY.num_samples))
    sess.encode(feats)
    primer = torch.tensor([[2, 3, 4], [2, 5, 4]])
    out = sess.decode(primer, max_new_tokens=10)

    # naive: re-run the whole prefix every step with an empty cache
    empty = torch.zeros_like(sess.self_k)
    for b in range(2):
        seq = primer[b : b + 1]
        naive = []
        for _ in range(10):
            logits, _ = model.decoder.step(
                seq, torch.zeros(1, dtype=torch.long), list(empty[:, b : b + 1]), list(empty[:, b : b + 1]),
                list(sess.cross_k[:, b : b + 1]), list(sess.cross_v[:, b : b + 1]),
            )  # fmt: skip
            nxt = int(logits.argmax(-1))
            if nxt == 1:
                break
            naive.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)
        assert out[b] == naive


def test_resample_preserves_tone():
    sr_in, sr_out, f = 44100, 16000, 440.0
    t = torch.arange(sr_in) / sr_in
    y = resample(torch.sin(2 * math.pi * f * t), sr_in, sr_out)
    assert y.shape[0] == sr_out
    ref = torch.sin(2 * math.pi * f * torch.arange(sr_out) / sr_out)
    mid = slice(1000, -1000)  # ignore edge effects
    assert (y[mid] - ref[mid]).abs().max() < 1e-2


def test_fused_projections_match(model):
    import copy

    fused = copy.deepcopy(model)
    fused.fuse_projections()
    speech = torch.randn(2, TINY.num_samples)
    torch.testing.assert_close(fused.encode(speech), model.encode(speech), rtol=1e-4, atol=1e-5)
    s1, s2 = _session(model, batch=2), _session(fused, batch=2)
    feats = model.frontend(speech)
    s1.encode(feats)
    s2.encode(feats)
    primer = torch.tensor([[2, 3, 4], [2, 5, 4]])
    assert s1.decode(primer, 12) == s2.decode(primer, 12)
    torch.testing.assert_close(s1.logits, s2.logits, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("split", [True, False])
def test_bucket_switching_matches_single_cache(model, split):
    """Growing through tiny cache buckets gives the same tokens and logits as one big cache."""
    model.decoder.split_attention = split
    try:
        feats = model.frontend(torch.randn(2, TINY.num_samples))
        primer = torch.tensor([[2, 3, 4], [2, 5, 4]])
        outs, logits = [], []
        for buckets in ([4, 8, 16, 32], None):
            sess = _session(model, batch=2, eos=-1, buckets=buckets)  # eos never fires: always 20 steps
            sess.encode(feats)
            outs.append(sess.decode(primer, 20))
            logits.append(sess.logits.clone())
        assert outs[0] == outs[1] and len(outs[0][0]) == 20
        torch.testing.assert_close(logits[0], logits[1], rtol=1e-4, atol=1e-4)
    finally:
        model.decoder.split_attention = True


def test_decoder_profile_shapes_are_consistent():
    from owsm_trt.build import BuildOptions, decoder_input_names, decoder_profile_shapes, decoder_profiles

    opts = BuildOptions(max_batch=4, capacity=256, max_prefill=32)
    profiles = decoder_profiles(opts.max_batch)
    assert [p["kind"] for p in profiles] == ["decode", "decode", "prefill"]
    for p in profiles:
        shapes = decoder_profile_shapes(TINY, opts, p)
        assert set(shapes) == set(decoder_input_names(TINY.dec_layers))
        for name, (lo, opt, hi) in shapes.items():
            assert len(lo) == len(opt) == len(hi), name
            assert all(a <= b <= c for a, b, c in zip(lo, opt, hi)), (name, lo, opt, hi)
    # every batch size 1..max_batch is covered by exactly one decode profile
    for b in range(1, opts.max_batch + 1):
        assert sum(p["kind"] == "decode" and p["batch"][0] <= b <= p["batch"][2] for p in profiles) == 1
