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
        profile_beam.configure_gluon(model, max_rows=16)
        wider = model._profile_backend
        assert (wider.rows_kernel[1:] if multi else (wider.max_tokens, wider.max_rows)) == (16, 16)
        assert (base.rows_kernel[1:] if multi else (base.max_tokens, base.max_rows)) == (16, 8)
        with pytest.raises(ValueError, match="row limit"):
            profile_beam.configure_gluon(model, max_rows=32)


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


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("case", ["measured", "other_model", "other_gpu", "other_sms", "other_programs"])
def test_wide_grouped_beam_routing_is_shape_scoped(monkeypatch, multi, case):
    from owsm_trt import megakernel, megakernel_gluon, pipeline

    class Kernel:
        def __init__(self, mega, programs=132, num_warps=4, prefetch_attention=True,
                     group_cross_queries=False, cross_splits=None):
            self.m, self.P, self.nw, self.pfa = mega, programs, num_warps, prefetch_attention
            self.group_cross_queries, self.cross_splits = group_cross_queries, cross_splits

    monkeypatch.setattr(megakernel_gluon, "GluonDecoder", Kernel)
    weights = SimpleNamespace(D=1024, F=4096, L=18, H=16, device="cuda")
    gpu = SimpleNamespace(name="NVIDIA H200", multi_processor_count=132)
    if case == "other_model":
        weights.L = 6
    elif case == "other_gpu":
        gpu.name = "NVIDIA H100"
    elif case == "other_sms":
        gpu.multi_processor_count = 120
    original = Kernel(weights, programs=128 if case == "other_programs" else 132)
    monkeypatch.setattr(pipeline.torch.cuda, "get_device_properties", lambda _: gpu)
    base = megakernel.MegaBackend(object(), object() if multi else original, 16, 8,
                                 rows_kernel=(original, 16, 8) if multi else None, trial=True)
    backend = pipeline._grouped_beam_backend(base)
    grouped = backend.rows_kernel[0] if multi else backend.kernel
    assert grouped.m is weights and grouped is not original
    assert grouped.cross_splits == (megakernel_gluon.BEAM_CROSS_SPLITS if case == "measured" else 8)
    assert (backend.rows_kernel[1:] if multi else (backend.max_tokens, backend.max_rows)) == (
        16, 16 if case == "measured" else 8)
    assert (base.rows_kernel[1:] if multi else (base.max_tokens, base.max_rows)) == (16, 8)
    assert original.cross_splits is None
    backend.failed.add(id(grouped))
    assert not base.failed


@pytest.mark.parametrize("splits", [{}, {0: 2}, {"5": 8}, {5: 3}])
def test_row_slice_mapping_rejects_invalid_settings(splits):
    from owsm_trt.megakernel_gluon import GluonDecoder

    with pytest.raises(ValueError, match="maps positive row counts"):
        GluonDecoder(None, cross_splits=splits)



def test_ring_rows_ablation_preserves_reference_routing_and_weights():
    from profile_ring_programs import ring_rows_backend

    from owsm_trt.megakernel import MegaBackend

    kernel = object()
    base = MegaBackend(object(), kernel, 32, 8, rows_kernel=(None, 16, 8), trial=True)
    candidate = ring_rows_backend(base)
    assert candidate.inner is base.inner and candidate.kernel is kernel
    assert base.rows_kernel == (None, 16, 8)
    assert candidate.rows_kernel == (kernel, 32, 8)
    assert candidate.max_tokens == base.max_tokens and candidate.max_rows == base.max_rows
    candidate.failed.add(id(kernel))
    assert not base.failed



def test_row_route_diagnostics_count_calls_after_output_and_restore_methods(monkeypatch):
    import torch
    from profile_ring_programs import count_decoder_dispatch

    from owsm_trt.runtime import BeamSession, GreedySession

    greedy = SimpleNamespace(pos=torch.tensor([8]), rows=3)
    beam = SimpleNamespace(step=torch.tensor([6]), rows=5)
    primer = torch.zeros(1, 4, dtype=torch.long)

    def decode(session, primer):
        return [[1, 2]]

    monkeypatch.setattr(GreedySession, 'decode', decode)
    monkeypatch.setattr(BeamSession, 'decode', decode)

    def call():
        GreedySession.decode(greedy, primer)
        BeamSession.decode(beam, primer)

    stats = count_decoder_dispatch(SimpleNamespace(_sessions={}), call)
    assert [r['decoder_calls'] for r in stats['plain_batches']] == [4, 6]
    assert [r['beam'] for r in stats['plain_batches']] == [False, True]
    assert not stats['drafted_calls']
    assert GreedySession.decode is decode and BeamSession.decode is decode



@pytest.mark.parametrize("message,passed,expected", [
    ("megakernel-ring: logits off (0.0744 of their range, argmax agreement 0.88)", False, True),
    ("megakernel-ring: logits off (0.1 of their range, argmax agreement 0.88)", False, False),
    ("megakernel-ring: logits off (0.0744 of their range, argmax agreement 0.75)", False, False),
    ("megakernel-ring: logits off (0.0744 of their range, argmax agreement 0.88)", True, False),
    ("grid barrier timeout", False, False),
])
def test_experimental_row_exception_requires_the_diagnosed_failure(message, passed, expected):
    from profile_ring_programs import diagnosed_row_failure

    evidence = dict(existing_check_pass=passed, max_relative_range=.074428, argmax_agreement=.875)
    assert diagnosed_row_failure(RuntimeError(message), evidence) is expected
    assert not diagnosed_row_failure(RuntimeError(message), None)
