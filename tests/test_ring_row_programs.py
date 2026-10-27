"""Program routing must preserve attention splits and exact active Ring computations."""

from dataclasses import replace

import pytest
import torch
from profile_ring_row_programs import RowPrograms, row_budget

from owsm_trt import megakernel, megakernel_ring
from owsm_trt.runtime import capture_graph


@pytest.mark.parametrize('rows', range(2, 9))
@pytest.mark.parametrize('policy', ['heads', 'tiles'])
def test_budgets_cover_heads_without_changing_attention_splits(rows, policy):
    programs = row_budget(rows, policy)
    assert rows*6 <= programs <= 48
    assert max(1, programs // (rows*6)) == max(1, 24 // (rows*6))


@pytest.mark.parametrize('rows', [1, 9])
def test_budget_refuses_unmeasured_rows(rows):
    with pytest.raises(ValueError, match='2..8'):
        row_budget(rows, 'heads')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='compute GPU required')
@pytest.mark.parametrize('rows', range(2, 9))
@pytest.mark.parametrize('window', [1, 4])
@torch.inference_mode()
def test_row_program_graph_logits_cache_and_argmax_exact(rows, window):
    from test_megakernel import CFG384, _model

    if not megakernel_ring.available(torch.device('cuda')):
        pytest.skip('Hopper Ring required')
    cfg = replace(CFG384, dec_ffn=1536, dec_layers=6)
    model = _model(cfg)
    mega = megakernel.MegaDecoder(model.decoder, torch.device('cuda'))
    original = megakernel_ring.RingDecoder(mega, row_programs=24)
    public = megakernel_ring.RingDecoder(mega)
    instrumented = megakernel_ring.RingDecoder(mega, ts=True)
    assert not original.row_head_parallel
    expected_policy = 'H200' in torch.cuda.get_device_name() and torch.cuda.get_device_properties('cuda').multi_processor_count == 132
    assert public.row_head_parallel is expected_policy
    assert instrumented.ts.numel() >= (8*cfg.dec_layers+3)*(48 if expected_policy else 24)*8
    kernels = [original, RowPrograms(original, 'heads'), RowPrograms(original, 'tiles'), public, instrumented]
    shape = (cfg.dec_layers, rows, cfg.dec_heads, 64, cfg.d_model // cfg.dec_heads)
    k, v = torch.randn(shape, device='cuda', dtype=torch.float16), torch.randn(shape, device='cuda', dtype=torch.float16)
    cross = [torch.randn(cfg.dec_layers, rows, cfg.dec_heads, cfg.enc_frames, cfg.d_model//cfg.dec_heads,
                        device='cuda', dtype=torch.float16) for _ in range(2)]
    tokens = torch.full((rows, window), 4, device='cuda', dtype=torch.long)
    offset = torch.full((rows,), 3, device='cuda', dtype=torch.long)
    owners, graphs, buffers = [], [], []
    for kernel in kernels:
        kc, vc = k.clone(), v.clone()
        logits = torch.zeros(rows, window, cfg.vocab_size, device='cuda')
        kv = torch.zeros(cfg.dec_layers, 2, rows, cfg.dec_heads, window, cfg.d_model//cfg.dec_heads,
                         dtype=torch.float16, device='cuda')
        pred = torch.zeros(rows, window, device='cuda', dtype=torch.long)
        run = kernel.step_fn(tokens, offset, kc, vc, *cross, logits, kv, pred=pred)
        owners.append(run)
        graphs.append(capture_graph(run))
        buffers.append((logits, kc, vc, kv, pred))
    assert all(run.const['NS'] == owners[0].const['NS'] for run in owners)
    assert owners[3].programs == (rows*6 if expected_policy and rows>=5 else 24)
    for position in (3, 31, 3):
        offset.fill_(position)
        for graph, values in zip(graphs, buffers):
            values[1].copy_(k)
            values[2].copy_(v)
            graph.replay()
        for actual in buffers[1:]:
            for a, b in zip(actual, buffers[0]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    original.check()
