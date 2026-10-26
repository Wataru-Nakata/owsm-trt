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
