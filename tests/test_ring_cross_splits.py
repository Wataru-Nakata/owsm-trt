"""Explicit N split changes retain deterministic solo/concurrent math and graph/cache state."""

from dataclasses import replace

import pytest
import torch

from owsm_trt import megakernel, megakernel_ring
from owsm_trt.runtime import capture_graph


@pytest.mark.parametrize('splits',[0,3,9,-1])
def test_invalid_split_count_fails_before_gpu_setup(splits):
    with pytest.raises(ValueError,match='single_cross_splits'):
        megakernel_ring.RingDecoder(None,single_cross_splits=splits)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='compute GPU required')
@pytest.mark.parametrize('width',[384,1024])
@pytest.mark.parametrize('window',[1,8,32])
@pytest.mark.parametrize('time',[64,374])
@torch.inference_mode()
def test_cross_splits_graph_fp32_cache_and_solo_repeat(width,window,time):
    from test_megakernel import CFG, _model

    if (not megakernel_ring.available(torch.device('cuda'))
            or torch.cuda.get_device_properties('cuda').multi_processor_count<64):
        pytest.skip('Hopper Ring required')
    cfg = replace(CFG,d_model=width,n_heads=width//64,dec_heads=width//64,dec_ffn=width*4,dec_layers=2)
    model = _model(cfg)
    mega = megakernel.MegaDecoder(model.decoder,torch.device('cuda'))
    f16 = dict(device='cuda',dtype=torch.float16)
    shape = (cfg.dec_layers,1,cfg.dec_heads,128,64)
    initial = [torch.randn(shape,**f16)*.1 for _ in range(2)]
    cross = [torch.randn(cfg.dec_layers,1,cfg.dec_heads,time,64,**f16)*.1 for _ in range(2)]
    tok = torch.randint(cfg.vocab_size,(1,window),device='cuda')
    offset = torch.tensor([0],device='cuda')
    stop = torch.zeros(1,dtype=torch.bool,device='cuda') if window==1 else None
    graphs,owners,buffers = {},[],{}
    for splits in (None,2,4,8):
        kernel = megakernel_ring.RingDecoder(mega,programs=cfg.dec_heads*2,solo_programs=64,single_cross_splits=splits)
        for solo in (False,True):
            k,v = (x.clone() for x in initial)
            logits = torch.zeros(1,window,cfg.vocab_size,device='cuda')
            kv = torch.zeros(cfg.dec_layers,2,1,cfg.dec_heads,window,64,**f16)
            pred = torch.zeros(1,window,dtype=torch.long,device='cuda')
            run = kernel.step_fn(tok,offset,k,v,*cross,logits,kv,pred=pred,solo=solo,stop=stop)
            assert run.const['NS']==(splits or 2)
            assert run.programs==(64 if solo else cfg.dec_heads*2)
            owners.append(run)
            graphs[splits,solo] = capture_graph(run)
            buffers[splits,solo] = logits,kv,k,v,pred
    first = {}
    for iteration,position in enumerate((0,63,0)):
        offset.fill_(position)
        ref,refkv = model.decoder.step(tok,offset,[x.float() for x in initial[0]],
                                       [x.float() for x in initial[1]],*[list(x.float()) for x in cross])
        for key,graph in graphs.items():
            values = buffers[key]
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
            assert all(bool(torch.isfinite(x).all()) for x in values)
            assert float((values[0]-ref).abs().max()) < .02*float(ref.abs().max())
            assert float((values[0].argmax(-1)==ref.argmax(-1)).float().mean())>=.95
            torch.testing.assert_close(values[1].float(),refkv,atol=.05,rtol=.02)
            assert torch.equal(values[4],values[0].argmax(-1))
            for cache,original in zip(values[2:4],initial):
                assert torch.equal(cache[:,:,:,:position],original[:,:,:,:position])
                assert torch.equal(cache[:,:,:,position+window:],original[:,:,:,position+window:])
            assert torch.equal(values[2][:,:,:,position:position+window],values[1][:,0])
            assert torch.equal(values[3][:,:,:,position:position+window],values[1][:,1])
        if iteration==0:
            first = {key:[x.clone() for x in values] for key,values in buffers.items()}
        elif iteration==2:
            for key,values in buffers.items():
                for a,b in zip(values,first[key]):
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
        for splits in (None,2,4,8):
            for a,b in zip(buffers[splits,False],buffers[splits,True]):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
        for a,b in zip(buffers[None,True],buffers[2,True]):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    mega.check()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='compute GPU required')
@torch.inference_mode()
def test_single_row_override_leaves_multi_row_math_unchanged():
    from test_megakernel import CFG384, _model

    if (not megakernel_ring.available(torch.device('cuda'))
            or torch.cuda.get_device_properties('cuda').multi_processor_count<64):
        pytest.skip('Hopper Ring required')
    cfg = replace(CFG384,dec_ffn=1536)
    model = _model(cfg)
    mega = megakernel.MegaDecoder(model.decoder,torch.device('cuda'))
    f16 = dict(device='cuda',dtype=torch.float16)
    initial = [torch.randn(cfg.dec_layers,2,6,64,64,**f16)*.1 for _ in range(2)]
    cross = [torch.randn(cfg.dec_layers,2,6,374,64,**f16)*.1 for _ in range(2)]
    tok = torch.randint(cfg.vocab_size,(2,1),device='cuda')
    pos = torch.tensor([3,7],device='cuda')
    expected = None
    for splits in (None,4,8):
        ring = megakernel_ring.RingDecoder(mega,single_cross_splits=splits)
        k,v = (x.clone() for x in initial)
        logits = torch.zeros(2,1,cfg.vocab_size,device='cuda')
        kv = torch.zeros(cfg.dec_layers,2,2,6,1,64,**f16)
        run = ring.step_fn(tok,pos,k,v,*cross,logits,kv)
        assert run.const['NS']==2 and run.programs==24
        graph = capture_graph(run)
        graph.replay()
        values = logits,kv,k,v
        if expected is None:
            expected = values
        else:
            for a,b in zip(values,expected):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
    mega.check()
