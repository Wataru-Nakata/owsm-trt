"""Only the measured hardware/model schedule may change; explicit budgets and other models retain control."""

from types import SimpleNamespace

import pytest
import torch

from owsm_trt import megakernel_ring


@pytest.mark.parametrize('shape,name,sms,options,expected', [
    ((384,1536,6,6),'NVIDIA H200',132,{},(12,64,24,True)),
    ((384,1536,6,6),'NVIDIA H200',132,{'solo_programs':24},(12,24,24,True)),
    ((384,1536,6,6),'NVIDIA H200',132,{'solo_programs':32},(12,32,24,True)),
    ((384,1536,6,6),'NVIDIA H200',132,{'row_programs':60},(12,64,60,False)),
    ((384,1536,6,6),'NVIDIA H100',132,{},(12,24,24,False)),
    ((384,1536,6,6),'NVIDIA H200',120,{},(12,24,24,False)),
    ((384,1536,6,6),'NVIDIA H200',16,{},(12,16,16,False)),
    ((384,1536,6,6),'NVIDIA H200',132,{'num_warps':8},(6,12,12,False)),
    ((1024,4096,18,16),'NVIDIA H200',132,{},(32,64,64,False)),
])
def test_serial_schedule_scope_and_explicit_overrides(monkeypatch,shape,name,sms,options,expected):
    mega = SimpleNamespace(**dict(zip(('D','F','L','H'),shape)),device=torch.device('cpu'),err=None)
    for attr in ('wqkv','wo','wcq','wco','w1','w2','wout'):
        setattr(mega,attr,None)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(multi_processor_count=sms))
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _:name)
    monkeypatch.setattr(megakernel_ring,'weight_desc',lambda w,bm:None,raising=False)
    ring = megakernel_ring.RingDecoder(mega,ts=True,**options)
    assert (ring.P,ring.P_solo,ring.P_rows,ring.row_head_parallel)==expected
    assert ring.ts.numel()>=(8*mega.L+3)*max(ring.P,ring.P_solo,ring.P_rows)*8
