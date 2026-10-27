"""The public N-class normalizer is opt-in and forwarded only to beam sessions."""

from types import SimpleNamespace

import pytest
import torch
from test_model import TINY

from owsm_trt import pipeline


@pytest.mark.parametrize("value", [0, 1, None, "true", 0.5])
@pytest.mark.parametrize("pretrained", [False, True])
def test_invalid_lse_option_fails_before_loading(value, pretrained):
    with pytest.raises(ValueError, match="beam_fused_lse"):
        if pretrained:
            pipeline.OWSM.from_pretrained("never-load-this", beam_fused_lse=value)
        else:
            pipeline.OWSM(None, None, None, None, None, torch.device("cpu"), beam_fused_lse=value)


@pytest.mark.parametrize("options, expected", [({}, False), ({"beam_fused_lse": False}, False),
                                               ({"beam_fused_lse": True}, True)])
def test_public_lse_value_reaches_beam_session(options, expected, monkeypatch):
    model = pipeline.OWSM(
        TINY, SimpleNamespace(first_time=None, eos=1), None, None,
        SimpleNamespace(dtype="float32", capacity=16), torch.device("cpu"),
        use_cuda_graph=False, **options,
    )
    received = []

    def session(*args, **kwargs):
        received.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(pipeline, "BeamSession", session)
    assert model.beam_fused_lse == expected
    assert model._session(1, 5).fused_lse == expected
    assert model._session(1, 5).fused_lse == expected
    assert len(received) == 1
