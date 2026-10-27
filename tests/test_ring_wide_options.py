"""Public opt-in retains auto numerical setup and refuses unmeasured GPU/model combinations."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from owsm_trt import megakernel, megakernel_gluon, megakernel_ring, pipeline


@pytest.mark.parametrize('which,shape,name,sms,fail,expected', [
    ('auto', (384,1536,6,6), 'NVIDIA H200', 132, False, False),
    ('ring:wide', (384,1536,6,6), 'NVIDIA H200', 132, False, True),
    ('ring:wide', (1024,4096,18,16), 'NVIDIA H200', 132, False, False),
    ('ring:wide', (384,1536,6,6), 'NVIDIA H100', 132, False, False),
    ('ring:wide', (384,1536,6,6), 'NVIDIA H200', 120, False, False),
    ('ring:wide', (384,1536,6,6), 'NVIDIA H200', 132, True, False),
])
def test_wide_factory_is_opt_in_and_preserves_auto_gate(monkeypatch, which, shape, name, sms, fail, expected):
    model = SimpleNamespace(**dict(zip(('D','F','L','H'),shape)))
    checks = []
    made = []

    def ring(m, **kw):
        made.append(kw)
        if fail and 'row_programs' in kw:
            raise RuntimeError('synthetic setup failure')
        return SimpleNamespace(m=m, row_programs=kw.get('row_programs'))

    def gluon(_):
        raise ValueError('synthetic unsupported Gluon model')

    monkeypatch.setattr(pipeline, '_load_decoder', lambda *a,**k: object())
    monkeypatch.setattr(megakernel, 'MegaDecoder', lambda *a: model)
    monkeypatch.setattr(megakernel_gluon, 'available', lambda _: True)
    monkeypatch.setattr(megakernel_gluon, 'GluonDecoder', gluon)
    monkeypatch.setattr(megakernel_ring, 'RingDecoder', ring)
    monkeypatch.setattr(pipeline, '_check_kernel', lambda kernel,decoder,dev,rows,window: checks.append((rows,window)))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(name=name,multi_processor_count=sms))
    inner = SimpleNamespace(name='tensorrt')
    actual = pipeline._decoder_kernel(inner, None, Path('/not-loaded'), torch.device('cuda'), which)
    assert checks == [(1,2)]
    assert isinstance(actual, megakernel.WideRingBackend) == expected
    if expected:
        assert set(actual.routes) == {10,15,16}
        assert {r:b.kernel.row_programs for r,b in actual.routes.items()} == {10:60,15:90,16:96}
        assert all(b.trial and b.max_rows==r and b.max_tokens==r for r,b in actual.routes.items())
        assert actual.base.inner is inner and actual.base.rows_kernel[0] is None
    else:
        assert isinstance(actual, megakernel.MegaBackend) and actual.inner is inner
    assert len(made) == (4 if expected else 2 if fail else 1)
