"""Session retention policy, without model operations or CUDA allocation."""

from types import SimpleNamespace

import pytest
import torch

from owsm_trt import pipeline


def model_with_fake_sessions(monkeypatch, limit=2):
    model = pipeline.OWSM(
        SimpleNamespace(ctc_only=False), SimpleNamespace(first_time=None, eos=1), None, None,
        SimpleNamespace(dtype="float32", capacity=16), torch.device("cpu"), use_cuda_graph=False,
        session_cache_size=limit,
    )
    created = []

    def session(*args, **kwargs):
        result = SimpleNamespace(batch=args[2], beam=args[3], options=kwargs)
        created.append(result)
        return result

    monkeypatch.setattr(pipeline, "BeamSession", session)
    return model, created


@pytest.mark.parametrize("limit, expected_creations", [(1, 9), (2, 9), (3, 3)])
def test_three_shape_cycle_reuses_only_retained_sessions(monkeypatch, limit, expected_creations):
    model, created = model_with_fake_sessions(monkeypatch, limit)
    assert model.session_cache_size == limit
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


@pytest.mark.parametrize("limit", [0, -1, True, False, 2.5, "3", None])
@pytest.mark.parametrize("entry", ["init", "from_pretrained"])
def test_invalid_cache_size_fails_before_loading(monkeypatch, limit, entry):
    def no_load(*args, **kwargs):
        pytest.fail("invalid cache size reached model resolution")

    monkeypatch.setattr(pipeline.hub, "resolve", no_load)
    with pytest.raises(ValueError, match="session_cache_size must be a positive integer"):
        if entry == "from_pretrained":
            pipeline.OWSM.from_pretrained(session_cache_size=limit)
        else:
            pipeline.OWSM(None, None, None, None, None, None, session_cache_size=limit)
