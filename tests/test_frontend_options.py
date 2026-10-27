"""Public graph limits reject invalid settings before checkpoint/model work."""

from types import SimpleNamespace

import pytest
import torch
from test_model import TINY

from owsm_trt.pipeline import OWSM


@pytest.mark.parametrize("value", [0, 9, -1, True, 1.5, None])
@pytest.mark.parametrize("pretrained", [False, True])
def test_frontend_batch_limit_validation_precedes_loading(value, pretrained):
    with pytest.raises(ValueError, match="frontend_graph_max_batch"):
        if pretrained:
            OWSM.from_pretrained("never-load-this", frontend_graph_max_batch=value)
        else:
            OWSM(None, None, None, None, None, torch.device("cpu"), frontend_graph_max_batch=value)


@pytest.mark.parametrize("limit", [1, 3, 8])
def test_public_batch_limit_and_serialized_warmup_defaults(limit):
    tokenizer = SimpleNamespace(first_time=None)
    model = OWSM(TINY, tokenizer, None, None, SimpleNamespace(dtype="float32"), torch.device("cpu"),
                 use_cuda_graph=False, frontend_graph_max_batch=limit)
    assert model._frontend_graph_batch_limit == limit
    assert model._frontend_reuse_warmup_stream
    assert model._frontend_replay is None  # Construction retains no graph/stream/workspace.
