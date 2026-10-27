"""Public N split grouping is explicit, shape guarded and keeps the automatic numerical gate."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from owsm_trt import megakernel, megakernel_gluon, megakernel_ring, pipeline


@pytest.mark.parametrize('which,shape,name,sms,available,failure,expected', [
    ('auto',(1024,4096,18,16),'NVIDIA H200',132,True,None,None),
    ('auto',(384,1536,6,6),'NVIDIA H200',132,True,None,None),
    ('ring:split',(1024,4096,18,16),'NVIDIA H200',132,True,None,4),
    ('ring:split',(384,1536,6,6),'NVIDIA H200',132,True,None,8),
    ('ring:split',(384,1536,6,6),'NVIDIA H100',132,True,None,None),
    ('ring:split',(384,1536,6,6),'NVIDIA H200',120,True,None,None),
    ('ring:split',(384,1536,6,6),'NVIDIA B200',132,True,None,None),
    ('ring:split',(512,2048,6,8),'NVIDIA H200',132,True,None,None),
    ('ring:split',(384,1536,6,6),'NVIDIA H200',132,False,None,'inner'),
    ('ring:split',(384,1536,6,6),'NVIDIA H200',132,True,'constructor','inner'),
    ('ring:split',(384,1536,6,6),'NVIDIA H200',132,True,'gate','inner'),
])
def test_split_factory(monkeypatch,which,shape,name,sms,available,failure,expected):
    model = SimpleNamespace(**dict(zip(('D','F','L','H'),shape)))
    checks = []
    made = []
    def ring(m,**kw):
        made.append(kw)
        if failure=='constructor':
            raise RuntimeError('synthetic setup failure')
        return SimpleNamespace(m=m,single_cross_splits=kw.get('single_cross_splits'))
    def gluon(_):
        raise ValueError('synthetic unsupported Gluon model')
    def check(kernel,decoder,dev,rows,window):
        checks.append((rows,window))
        if failure=='gate':
            raise RuntimeError('synthetic numerical failure')
    monkeypatch.setattr(pipeline,'_load_decoder',lambda *a,**k:object())
    monkeypatch.setattr(megakernel,'MegaDecoder',lambda *a:model)
    monkeypatch.setattr(megakernel_gluon,'available',lambda _:available)
    monkeypatch.setattr(megakernel_gluon,'GluonDecoder',gluon)
    monkeypatch.setattr(megakernel_ring,'RingDecoder',ring)
    monkeypatch.setattr(pipeline,'_check_kernel',check)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(name=name,multi_processor_count=sms))
    inner = SimpleNamespace(name='tensorrt')
    actual = pipeline._decoder_kernel(inner,None,Path('/not-loaded'),torch.device('cuda'),which)
    assert checks==([] if not available or failure=='constructor' else [(1,2)])
    if expected=='inner':
        assert actual is inner
    else:
        assert isinstance(actual,megakernel.MegaBackend) and actual.inner is inner
        assert actual.kernel.single_cross_splits==expected
        assert actual.rows_kernel[0] is None and actual.trial
        assert made==([{}] if expected is None else [{'single_cross_splits':expected}])
