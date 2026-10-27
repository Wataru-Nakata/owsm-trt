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


@pytest.mark.parametrize('enabled', [False,True])
@pytest.mark.parametrize('lookahead', [False,True])
@pytest.mark.parametrize('supported,batch', [(True,1),(False,1),(True,3)])
@pytest.mark.parametrize('lazy', [False,True])
def test_public_plain_solo_excludes_overlap_and_unsupported_calls(enabled,lookahead,supported,batch,lazy):
    from test_beam import CFG
    from test_finished_skip import ScriptedStopBackend

    from owsm_trt.pipeline import OWSM

    class Backend(ScriptedStopBackend):
        solo_differs = supported
        def step_fn(self,*args,solo=False,**kwargs):
            bound = super().step_fn(*args,**kwargs)
            bound.solo = solo
            return bound
    tok = SimpleNamespace(first_time=None,notime=None,eos=0,punctuation=frozenset())
    model = OWSM(CFG,tok,None,Backend(2),SimpleNamespace(dtype='float32',capacity=16),torch.device('cpu'),
                 use_cuda_graph=False,ctc_draft=False,greedy_kernels=enabled,encoder_lookahead=lookahead,lazy_drafts=lazy)
    session = model._session(batch)
    expected = enabled and not lookahead and supported and batch==1
    assert session.plain_solo is expected
    assert session.decode(torch.tensor([[2,3,4]]*batch),10)==[[1,1]]*batch
    assert session._steps and all(fn.solo is expected for fn in session._steps.values())
    assert model._session(batch) is session
