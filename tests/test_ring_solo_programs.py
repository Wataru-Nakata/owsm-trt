"""Exact serial-budget changes must preserve graph state, cache writes, EOS skip and argmax."""

from dataclasses import replace

import pytest
import torch

from owsm_trt import megakernel, megakernel_ring
from owsm_trt.runtime import capture_graph


@pytest.mark.skipif(not torch.cuda.is_available(),reason='compute GPU required')
@pytest.mark.parametrize('window',[1,8,32])
@pytest.mark.parametrize('time',[64,374])
@torch.inference_mode()
def test_solo_budget_graph_logits_cache_stop_and_reuse(window,time):
    from test_megakernel import CFG384, _model

    if (not megakernel_ring.available(torch.device('cuda')) or 'H200' not in torch.cuda.get_device_name()
            or torch.cuda.get_device_properties('cuda').multi_processor_count!=132):
        pytest.skip('H200132SM required')
    cfg = replace(CFG384,dec_ffn=1536,dec_layers=6)
    model = _model(cfg)
    mega = megakernel.MegaDecoder(model.decoder,torch.device('cuda'))
    f16 = dict(device='cuda',dtype=torch.float16)
    initial = [torch.randn(cfg.dec_layers,1,cfg.dec_heads,128,64,**f16) for _ in range(2)]
    cross = [torch.randn(cfg.dec_layers,1,cfg.dec_heads,time,64,**f16) for _ in range(2)]
    tok = torch.randint(cfg.vocab_size,(1,window),device='cuda')
    offset = torch.tensor([7],device='cuda')
    stop = torch.zeros(1,dtype=torch.bool,device='cuda') if window==1 else None
    graphs,owners,buffers = [],[],[]
    for programs in (24,32,48):
        ring = megakernel_ring.RingDecoder(mega,solo_programs=programs)
        k,v = (x.clone() for x in initial)
        logits = torch.zeros(1,window,cfg.vocab_size,device='cuda')
        kv = torch.zeros(cfg.dec_layers,2,1,cfg.dec_heads,window,64,**f16)
        pred = torch.zeros(1,window,dtype=torch.long,device='cuda')
        run = ring.step_fn(tok,offset,k,v,*cross,logits,kv,solo=True,pred=pred,stop=stop)
        assert run.const['NS']==2 and run.programs==programs
        owners.append(run)
        graphs.append(capture_graph(run))
        buffers.append((logits,kv,k,v,pred))
    for position in (0,63,0):
        offset.fill_(position)
        for graph,values in zip(graphs,buffers):
            values[2].copy_(initial[0])
            values[3].copy_(initial[1])
            if stop is not None:
                stop.fill_(True)
                before = [x.clone() for x in values]
                graph.replay()
                for a,b in zip(values,before):
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
                stop.fill_(False)
            graph.replay()
        for values in buffers[1:]:
            for a,b in zip(values,buffers[0]):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
    mega.check()
