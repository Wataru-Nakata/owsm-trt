"""Keep unreviewed shapes and prefill on the existing backend in the isolated N experiment."""

from types import SimpleNamespace

import pytest
from profile_ring_wide import WideBackend

from owsm_trt.megakernel import WideRingBackend


@pytest.mark.parametrize('implementation', [WideBackend, WideRingBackend])
@pytest.mark.parametrize('rows,window,kind,expected', [
    (10, 1, 'decode', 'wide10'), (15, 1, 'decode', 'wide15'), (16, 1, 'decode', 'wide16'),
    (8, 1, 'decode', 'base'), (9, 1, 'decode', 'base'), (12, 1, 'decode', 'base'),
    (17, 1, 'decode', 'base'), (10, 2, 'decode', 'base'), (15, 1, 'prefill', 'base'),
    (16, 1, 'verify', 'base'),
])
def test_only_reviewed_decode_shapes_are_routed(implementation, rows, window, kind, expected):
    calls = []

    def backend(label):
        def step(tokens, *args, **kwargs):
            calls.append((label, tokens.shape, args, kwargs))
            return label
        return SimpleNamespace(step_fn=step, failed=set(), marker='delegated')

    base = backend('base')
    routes = {r:backend(f'wide{r}') for r in (10, 15, 16)}
    wrapper = implementation(base, routes)
    assert wrapper.step_fn(SimpleNamespace(shape=(rows,window)), 'sentinel', kind=kind, pred='prediction') == expected
    assert calls == [(expected, (rows,window), ('sentinel',), dict(kind=kind, pred='prediction'))]
    assert wrapper.marker == 'delegated'
    assert wrapper.failed == set()
    routes[15].failed.add(41)
    base.failed.add(52)
    assert wrapper.failed == {41,52}
