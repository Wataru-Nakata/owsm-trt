"""Smaller producer rings must preserve graph/cache state through wraparound and EOS reuse."""

import pytest
import torch
from test_megakernel import CFG, _model

from owsm_trt import megakernel, megakernel_ring
from owsm_trt.runtime import capture_graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute GPU required")
@pytest.mark.parametrize("window", [1, 8, 32])
@pytest.mark.parametrize("solo", [False, True])
@torch.inference_mode()
def test_ring_capacity_graph_wraparound_logits_kv_and_restart(window, solo):
    if not megakernel_ring.available(torch.device("cuda")):
        pytest.skip("Hopper Ring required")
    model = _model(CFG)
    mega = megakernel.MegaDecoder(model.decoder, torch.device("cuda"))
    torch.manual_seed(71 + window)
    f16 = dict(device="cuda", dtype=torch.float16)
    shape = (CFG.dec_layers, 1, CFG.dec_heads, 128, 64)
    initial = [torch.randn(shape, **f16) for _ in range(2)]
    cross = [torch.randn(CFG.dec_layers, 1, CFG.dec_heads, CFG.enc_frames, 64, **f16) for _ in range(2)]
    tokens = torch.randint(CFG.vocab_size, (1, window), device="cuda")
    offset = torch.tensor([7], device="cuda")
    stop = torch.zeros(1, device="cuda", dtype=torch.bool) if window == 1 else None
    expected = None
    for kb in (128, 64, 32):
        ring = megakernel_ring.RingDecoder(mega, ring_kb=kb)
        k, v = (x.clone() for x in initial)
        logits = torch.zeros(1, window, CFG.vocab_size, device="cuda")
        kv = torch.zeros(CFG.dec_layers, 2, 1, CFG.dec_heads, window, 64, **f16)
        pred = torch.full((1, window), -1, device="cuda", dtype=torch.long)
        run = ring.step_fn(tokens, offset, k, v, *cross, logits, kv, solo=solo, pred=pred, stop=stop)
        graph = capture_graph(run)
        results = []
        for pos in (7, 63, 7):
            offset.fill_(pos)
            k.copy_(initial[0])
            v.copy_(initial[1])
            if stop is not None:
                before = [x.clone() for x in (logits, kv, k, v, pred)]
                stop.fill_(True)
                graph.replay()
                for actual, old in zip((logits, kv, k, v, pred), before):
                    torch.testing.assert_close(actual, old, rtol=0, atol=0)
                stop.fill_(False)
            graph.replay()
            results.append([x.clone() for x in (logits, kv, k, v, pred)])
        ring.check()
        if expected is None:
            expected = results
        else:
            for actuals, refs in zip(results, expected):
                for actual, ref in zip(actuals, refs):
                    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
