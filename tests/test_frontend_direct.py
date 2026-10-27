"""Direct graph input filling preserves padding, output lifetime and exception stream ordering."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_model import TINY

from owsm_trt.frontend_graph import FrontendReplay
from owsm_trt.modeling import Frontend


def test_direct_input_only_uses_a_warm_supported_entry():
    replay = FrontendReplay(None)
    assert replay.try_waves([],1601,torch.device('cuda:0')) is None
    replay.entries['sentinel'] = None
    assert replay.try_waves([torch.ones(7)],1601,torch.device('cpu')) is None
    with torch.enable_grad():
        assert replay.try_waves([torch.ones(7)],1601,torch.device('cuda:0')) is None
    with torch.inference_mode():
        assert replay.try_waves([torch.ones(7)]*2,1601,torch.device('cuda:0')) is None
    assert replay.direct_replays==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='compute allocation required')
@pytest.mark.parametrize('fused',[False,True])
@pytest.mark.parametrize('normalize',['global_mvn','utterance_mvn'])
@pytest.mark.parametrize('batch',[1,3,8])
@torch.inference_mode()
def test_direct_padding_crop_dtype_and_retained_outputs(fused,normalize,batch):
    frontend = Frontend(replace(TINY,normalize=normalize)).eval().cuda()
    frontend.melmat.uniform_(0,1)
    if frontend.global_mvn:
        frontend.mean.normal_()
        frontend.std.uniform_(.1,2)
    frontend.fused_power = frontend.fused_logmel = fused
    replay = FrontendReplay(frontend,max_entries=1,max_batch=batch)
    device = torch.device('cuda',torch.cuda.current_device())
    lengths = (0,1,800,1601,2400)
    groups = [[torch.randn(lengths[(i+j)%len(lengths)],dtype=torch.float64 if j%2 else torch.float32,
                           device=device if i%2 else 'cpu') for j in range(batch)] for i in range(5)]
    def padded(waves):
        inp = torch.zeros(batch,1601,device=device)
        for row,wave in zip(inp,waves):
            wave = wave[:1601]
            row[:len(wave)].copy_(wave)
        return inp
    expected = [frontend(padded(waves)) for waves in groups]
    assert replay.try_waves(groups[0],1601,device) is None
    replay(padded(groups[0]))
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(),torch.cuda.Stream()]
    actual = []
    for i in range(15):
        with torch.cuda.stream(streams[i%2]):
            actual.append(replay.try_waves(groups[i%5],1601,torch.device('cuda') if i%2 else device))
    torch.cuda.synchronize()
    for i,value in enumerate(actual):
        torch.testing.assert_close(value,expected[i%5],atol=0,rtol=0)
    assert len({v.data_ptr() for v in actual})==len(actual)
    assert replay.direct_replays==15 and len(replay.entries)==1
    assert replay.try_waves(groups[0],3200,device) is None
    frontend.fused_power = not fused
    assert replay.try_waves(groups[0],1601,device) is None
    frontend.fused_power = fused
    with torch.autocast('cuda',dtype=torch.float16):
        assert replay.try_waves(groups[0],1601,device) is None


@pytest.mark.skipif(not torch.cuda.is_available(),reason='compute allocation required')
@torch.inference_mode()
def test_direct_pin_failure_records_prior_writes_before_another_stream():
    frontend = Frontend(TINY).eval().cuda()
    frontend.melmat.uniform_(0,1)
    replay = FrontendReplay(frontend)
    speech = torch.randn(1,1601,device='cuda')
    expected = replay(speech)
    torch.cuda.synchronize()
    class FailedWave:
        device = SimpleNamespace(type='cpu')
        def __getitem__(self,key):
            return self
        def pin_memory(self):
            raise RuntimeError('synthetic pin failure')
    streams = [torch.cuda.Stream(),torch.cuda.Stream()]
    with torch.cuda.stream(streams[0]),pytest.raises(RuntimeError,match='synthetic pin failure'):
        replay.try_waves([FailedWave()],1601,speech.device)
    with torch.cuda.stream(streams[1]):
        actual = replay.try_waves([speech[0]],1601,speech.device)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert replay.direct_replays==1
