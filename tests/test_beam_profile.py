"""Decoder ablations must never change the backend used by the reference sessions."""

from types import SimpleNamespace

import pytest


def test_gluon_options_keep_reference_dispatch_and_weights(monkeypatch):
    import profile_beam

    from owsm_trt import megakernel, megakernel_gluon

    class Kernel:
        def __init__(self, mega, programs=132, num_warps=4, prefetch_attention=True, group_cross_queries=False,
                     cross_splits=None):
            self.m, self.P, self.nw, self.pfa = mega, programs, num_warps, prefetch_attention

    monkeypatch.setattr(megakernel_gluon, "GluonDecoder", Kernel)
    monkeypatch.setattr(profile_beam.torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(multi_processor_count=132))
    weights = object()
    reference = Kernel(weights)
    for multi in (False, True):
        base = megakernel.MegaBackend(object(), object() if multi else reference, 16, 8,
                                     rows_kernel=(reference, 16, 8) if multi else None, trial=True)
        model = SimpleNamespace(backend=base, device="cuda")
        profile_beam.configure_gluon(model, programs=128, prefetch=False)
        actual = model._profile_backend
        changed = actual.rows_kernel[0] if multi else actual.kernel
        assert changed is not reference and changed.m is weights
        assert (changed.P, changed.pfa) == (128, False)
        assert (reference.P, reference.pfa) == (132, True)
        assert actual.inner is base.inner and actual.trial is base.trial
        assert actual.max_tokens == base.max_tokens and actual.max_rows == base.max_rows
        actual.failed.add(id(changed))
        assert not base.failed
        with pytest.raises(ValueError, match="SMs"):
            profile_beam.configure_gluon(model, programs=133)


def test_public_beam_options_are_scoped_and_preserve_reference(monkeypatch):
    import torch
    from test_beam import CFG

    from owsm_trt import megakernel, megakernel_gluon, pipeline
    from owsm_trt.build import BuildOptions

    class Kernel:
        def __init__(self, mega, programs=132, num_warps=4, prefetch_attention=True,
                     group_cross_queries=False, cross_splits=None):
            self.m, self.P, self.nw, self.pfa = mega, programs, num_warps, prefetch_attention
            self.group_cross_queries, self.cross_splits = group_cross_queries, cross_splits

    monkeypatch.setattr(megakernel_gluon, 'GluonDecoder', Kernel)
    monkeypatch.setattr(pipeline, 'BeamSession', lambda backend, *args, **kw: SimpleNamespace(backend=backend, options=kw))
    monkeypatch.setattr(pipeline, 'GreedySession', lambda backend, *args, **kw: SimpleNamespace(backend=backend, options=kw))
    tok = SimpleNamespace(notime=None, first_time=None, eos=0, punctuation=frozenset())
    weights = object()
    original = Kernel(weights)
    base = megakernel.MegaBackend(object(), object(), 32, 1, rows_kernel=(original, 16, 8), trial=True)
    model = pipeline.OWSM(CFG, tok, None, base, BuildOptions(), torch.device('cpu'),
                          use_cuda_graph=False, ctc_draft=False, beam_compact_tail=True, beam_grouped_cross=True)
    s = model._session(1, 5)
    grouped = s.backend.rows_kernel[0]
    assert grouped is not original and grouped.m is weights
    assert grouped.group_cross_queries and grouped.cross_splits == 8
    assert s.options['compact_tail'] and s.options['fused_state']
    assert original.cross_splits is None and not original.group_cross_queries
    assert model._session(3, 5).backend.rows_kernel[0] is grouped
    assert model._session(1, 3).backend is base
    assert model._session(1, 1).backend is base
    assert model._session(1, 5).backend.rows_kernel[0] is grouped  # survives session eviction
    s.backend.failed.add(id(grouped))
    assert not base.failed
    assert pipeline._grouped_beam_backend(object()).__class__ is object
