"""Public worker counts validate before model loading or thread creation."""

from types import SimpleNamespace

import pytest
import torch
from test_model import TINY

from owsm_trt.pipeline import OWSM


@pytest.mark.parametrize('value', [-1, 3, True, 1.5, None])
@pytest.mark.parametrize('pretrained', [False, True])
def test_prefetch_validation_precedes_loading(value, pretrained):
    with pytest.raises(ValueError, match='audio_prefetch_workers'):
        if pretrained:
            OWSM.from_pretrained('never-load-this', audio_prefetch_workers=value)
        else:
            OWSM(None, None, None, None, None, torch.device('cpu'), audio_prefetch_workers=value)


@pytest.mark.parametrize('workers', [0, 1, 2])
def test_public_prefetch_constructor_value(workers):
    tokenizer = SimpleNamespace(first_time=None)
    model = OWSM(TINY, tokenizer, None, None, SimpleNamespace(dtype='float32'), torch.device('cpu'),
                 use_cuda_graph=False, audio_prefetch_workers=workers)
    assert model._audio_prefetch_workers == workers
