"""Solo scheduling must not replace prefill, drafted verification or multi-row dispatch."""

from types import SimpleNamespace

import pytest
import torch
from profile_plain_solo import PlainSolo


@pytest.mark.parametrize('shape,kind', [((1,1),'decode'),((1,1),'verify'),((1,3),'prefill'),((3,1),'decode')])
@pytest.mark.parametrize('supported', [False,True])
def test_plain_solo_forwards_only_supported_plain_calls(shape,kind,supported):
    calls = []
    def step_fn(tokens,*args,**kwargs):
        calls.append(kwargs)
        return 'bound'
    base = SimpleNamespace(solo_differs=supported,step_fn=step_fn,failed=set())
    wrapper = PlainSolo(base)
    assert wrapper.failed is base.failed
    assert wrapper.step_fn(torch.zeros(shape,dtype=torch.long),kind=kind,stop='same flag') == 'bound'
    assert calls[0]['kind']==kind and calls[0]['stop']=='same flag'
    assert calls[0].get('solo',False) is (supported and shape==(1,1) and kind=='decode')
