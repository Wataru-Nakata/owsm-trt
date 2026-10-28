"""Session retention policy, without model operations or CUDA allocation."""

from types import SimpleNamespace

import pytest
import torch

from owsm_trt import pipeline


def model_with_fake_sessions(monkeypatch):
    model = pipeline.OWSM(
        SimpleNamespace(ctc_only=False), SimpleNamespace(first_time=None, eos=1), None, None,
        SimpleNamespace(dtype="float32", capacity=16), torch.device("cpu"), use_cuda_graph=False,
    )
    created = []

    def session(*args, **kwargs):
        result = SimpleNamespace(batch=args[2], beam=args[3], options=kwargs)
        created.append(result)
        return result

    monkeypatch.setattr(pipeline, "BeamSession", session)
    return model, created


@pytest.mark.parametrize("limit, expected_creations", [(2, 9), (3, 3)])
def test_three_shape_cycle_reuses_only_retained_sessions(monkeypatch, limit, expected_creations):
    model, created = model_with_fake_sessions(monkeypatch)
    assert model._session_cache_limit == 2
    model._session_cache_limit = limit
    first = {}
    for cycle in range(3):
        for batch in (1, 3, 2):
            session = model._session(batch, 5)
            assert (session.batch, session.beam) == (batch, 5)
            if cycle:
                assert (session is first[batch]) == (limit == 3)
            else:
                first[batch] = session
            assert len(model._sessions) <= limit
    assert len(created) == expected_creations


def test_cache_hit_refreshes_lru_before_another_shape_is_created(monkeypatch):
    model, created = model_with_fake_sessions(monkeypatch)
    a = model._session(1, 5)
    model._session(3, 5)
    assert model._session(1, 5) is a
    model._session(2, 5)
    assert list(model._sessions) == [(1, 5), (2, 5)]
    assert len(created) == 3
