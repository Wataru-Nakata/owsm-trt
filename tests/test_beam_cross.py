"""Utterance-major cross-KV sharing and fallback; inference tests require a GPU allocation."""

import pytest
import torch
from test_beam import ScriptedBackend, session

from owsm_trt import megakernel, megakernel_gluon


@pytest.mark.parametrize("batch,beam", [(1, 5), (2, 3)])
def test_shared_buffer_layout_and_unsupported_fallback(batch, beam):
    class Backend(ScriptedBackend):
        def supports_shared_cross(self, rows):
            return rows <= 8

    shared = session(batch=batch, beam=beam, backend=Backend(), share_cross_kv=True)
    assert shared.share_cross_kv and shared.cross_k is shared.cross_kb
    assert shared.cross_v is shared.cross_vb and shared.cross_k.shape[1] == batch
    shared._expand_cross()
    fallback = session(batch=batch, beam=beam, share_cross_kv=True)
    assert not fallback.share_cross_kv and fallback.cross_kv_fallback
    fallback.cross_kb.fill_(3)
    fallback._expand_cross()
    assert fallback.cross_k.shape[1] == batch * beam and (fallback.cross_k == 3).all()


def test_trial_failure_refreshes_expanded_cross_after_each_encode():
    class BrokenKernel:
        shared_cross_kv = True
        name = "broken-shared-kernel"

        def step_fn(self, *args, **kwargs):
            raise ValueError("deliberate trial failure")

    class Backend(ScriptedBackend):
        name = "scripted"

        def step_fn(self, tokens, offset, k, v, ck, cv, logits, new_kv, **kwargs):
            assert ck.shape[1] == tokens.shape[0]

            def run():
                logits[:, 0, 0].copy_(ck[0, :, 0, 0, 0] + cv[0, :, 0, 0, 0])
            return run

    backend = megakernel.MegaBackend(Backend(), BrokenKernel(), 16, 8, trial=True)
    s = session(batch=2, beam=3, backend=backend, share_cross_kv=True)
    assert s.share_cross_kv and backend.failed
    for delta in (0, 100):
        for b in range(2):
            s.cross_kb[:, b].fill_(delta + b)
            s.cross_vb[:, b].fill_(10 + b)
        s._expand_cross()
        for run in s._steps.values():
            assert run.dispatch == "scripted" and "expanded after encode" in run.fallback_reason
            run()
            assert s.logits[:, 0, 0].tolist() == [delta + 10] * 3 + [delta + 12] * 3
    assert not backend.supports_shared_cross(6)


gluon_only = pytest.mark.skipif(
    not torch.cuda.is_available() or not megakernel_gluon.available(torch.device("cuda")),
    reason="Hopper + Gluon in a compute allocation required",
)


@gluon_only
@pytest.mark.parametrize("batch,beam", [(1, 5), (2, 3)])
@pytest.mark.parametrize("prefetch", [False, True])
@torch.inference_mode()
def test_shared_gluon_logits_kv_and_graph_replay(batch, beam, prefetch):
    from test_megakernel import CFG, _model

    cfg, dev = CFG, torch.device("cuda")
    model = _model(cfg)
    kernel = megakernel_gluon.GluonDecoder(megakernel.MegaDecoder(model.decoder, dev),
                                         prefetch_attention=prefetch)
    rows, c = batch * beam, 64
    shape = (cfg.dec_layers, rows, cfg.dec_heads, c, cfg.d_model // cfg.dec_heads)
    k, v = [torch.randn(shape, device=dev, dtype=torch.float16) for _ in range(2)]
    cross_shape = (cfg.dec_layers, batch, cfg.dec_heads, cfg.enc_frames, cfg.d_model // cfg.dec_heads)
    ck, cv = [torch.randn(cross_shape, device=dev, dtype=torch.float16) for _ in range(2)]
    tokens = torch.randint(0, cfg.vocab_size, (rows, 1), device=dev)
    offsets = torch.arange(rows, device=dev) * 5
    outputs, graphs, inputs = [], [], []
    for shared in (False, True):
        cross = (ck, cv) if shared else tuple(t.repeat_interleave(beam, dim=1) for t in (ck, cv))
        kc, vc = k.clone(), v.clone()
        logits = torch.empty(rows, 1, cfg.vocab_size, device=dev)
        new = torch.empty(cfg.dec_layers, 2, rows, cfg.dec_heads, 1, cfg.d_model // cfg.dec_heads,
                          dtype=torch.float16, device=dev)
        run = kernel.step_fn(tokens, offsets, kc, vc, *cross, logits, new)
        run()
        torch.cuda.synchronize()
        kernel.check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graphs.append(graph)
        outputs.append((logits, new, kc, vc))
        inputs.append(cross)
    for _ in range(2):
        ck.add_(.01)
        cv.mul_(.95)
        for source, expanded in zip((ck, cv), inputs[0]):
            expanded.copy_(source.repeat_interleave(beam, dim=1))
        for output, graph in zip(outputs, graphs):
            output[2].copy_(k)
            output[3].copy_(v)
            graph.replay()
        torch.cuda.synchronize()
        kernel.check()
        for a, b in zip(*outputs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@gluon_only
@torch.inference_mode()
def test_shared_beam_repeated_prompt_and_bucket_growth():
    from test_megakernel import CFG, _model

    from owsm_trt.runtime import BeamSession, TorchBackend

    model, dev = _model(CFG), torch.device("cuda")
    decoder = megakernel_gluon.GluonDecoder(megakernel.MegaDecoder(model.decoder, dev))
    backend = megakernel.MegaBackend(TorchBackend(model.half()), decoder, 16, 8)
    args = (backend, CFG, 2, 3, 32, -1, torch.float16, dev)
    ref = BeamSession(*args, buckets=[4, 8, 16, 32], diagnostics=True, share_cross_kv=False)
    fast = BeamSession(*args, buckets=[4, 8, 16, 32], diagnostics=True, share_cross_kv=True)
    assert fast.share_cross_kv
    for primer_len, limit in ((3, 29), (9, 7), (3, 20)):
        for name in ("cross_kb", "cross_vb"):
            value = torch.randn_like(getattr(ref, name))
            getattr(ref, name).copy_(value)
            getattr(fast, name).copy_(value)
        ref._expand_cross()
        fast._expand_cross()
        primer = torch.full((2, primer_len), 2, device=dev)
        assert fast.decode(primer, limit) == ref.decode(primer, limit)
        torch.testing.assert_close(fast.logits, ref.logits, rtol=0, atol=0)
        n = int(ref.step) + 1
        for name in ("scores", "state"):
            torch.testing.assert_close(fast._beam_history[name][:n], ref._beam_history[name][:n], rtol=0, atol=0)
