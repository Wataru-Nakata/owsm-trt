"""Compute-only diagnosis of R10/15/16 S1; production Ring routing remains capped at8."""

from dataclasses import replace

import pytest
import torch

from owsm_trt import megakernel, megakernel_ring
from owsm_trt.runtime import capture_graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason='compute GPU required')
@pytest.mark.parametrize('rows', [10, 15, 16])
@pytest.mark.parametrize('time', [64, 374])
@torch.inference_mode()
def test_wide_one_token_schedule_cache_graph_and_fp32_reference(rows, time):
    from test_megakernel import CFG384, _model

    if (not megakernel_ring.available(torch.device('cuda')) or 'H200' not in torch.cuda.get_device_name()
            or torch.cuda.get_device_properties('cuda').multi_processor_count != 132):
        pytest.skip('diagnosed configuration requires H200132SM')
    cfg = replace(CFG384, dec_ffn=1536, dec_layers=6)
    model = _model(cfg)
    mega = megakernel.MegaDecoder(model.decoder, torch.device('cuda'))
    f16 = dict(device='cuda', dtype=torch.float16)
    shape = (cfg.dec_layers, rows, cfg.dec_heads, 64, 64)
    initial = [torch.randn(shape, **f16) for _ in range(2)]
    cross = [torch.randn(cfg.dec_layers, rows, cfg.dec_heads, time, 64, **f16) for _ in range(2)]
    tokens = torch.randint(cfg.vocab_size, (rows, 1), device='cuda')
    offsets = torch.arange(rows, device='cuda')
    graphs, owners, buffers = [], [], []
    for programs in (24, rows*cfg.dec_heads):
        kernel = megakernel_ring.RingDecoder(mega, row_programs=programs)
        k, v = (x.clone() for x in initial)
        logits = torch.zeros(rows, 1, cfg.vocab_size, device='cuda')
        kv = torch.zeros(cfg.dec_layers, 2, rows, cfg.dec_heads, 1, 64, **f16)
        pred = torch.zeros(rows, 1, dtype=torch.long, device='cuda')
        run = kernel.step_fn(tokens, offsets, k, v, *cross, logits, kv, pred=pred)
        assert run.const['NS'] == 1 and run.programs == programs
        owners.append(run)
        graphs.append(capture_graph(run))
        buffers.append((logits, kv, k, v, pred))
    for base in (0, 32, 0):
        offsets.copy_(torch.arange(rows, device='cuda') + base)
        ref, ref_kv = model.decoder.step(tokens, offsets, *[list(x.float()) for x in (*initial, *cross)])
        for graph, (logits, kv, k, v, pred) in zip(graphs, buffers):
            k.copy_(initial[0])
            v.copy_(initial[1])
            graph.replay()
            scale = ref.abs().max().item()
            assert (logits-ref).abs().max().item() < .02*scale
            assert (logits.argmax(-1)==ref.argmax(-1)).float().mean().item() >= .95
            torch.testing.assert_close(kv.float(), ref_kv, atol=.05, rtol=.02)
            assert torch.equal(pred, logits.argmax(-1))
            slots = offsets.view(1, rows, 1, 1, 1).expand(cfg.dec_layers, rows, cfg.dec_heads, 1, 64)
            for actual, source, new in zip((k, v), initial, (kv[:, 0], kv[:, 1])):
                expected = source.clone()
                expected.scatter_(3, slots, new)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for a, b in zip(*buffers):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    mega.check()
