"""Host diagnostics must not change GC policy or leave a callback behind after errors."""

import gc

import pytest
from host_latency import HostLatency


@pytest.mark.parametrize('fail', [False, True])
def test_host_observer_preserves_gc_policy_and_callback_lifetime(monkeypatch, fail):
    monkeypatch.setattr('host_latency.torch.cuda.synchronize', lambda: None)
    original = list(gc.callbacks)
    state = gc.isenabled(), gc.get_threshold()
    with HostLatency() as host:
        def call():
            gc.collect(0)
            if fail:
                raise ValueError('expected')
            return 7
        if fail:
            with pytest.raises(ValueError, match='expected'):
                host.measure(call)
        else:
            result, ms = host.measure(call)
            assert result == 7 and ms >= 0
    assert gc.callbacks == original
    assert (gc.isenabled(), gc.get_threshold()) == state
    report = host.report([{'variant': 'reference'}])
    assert report['samples'][0]['gc'][0]['generation'] == 0
    assert report['samples'][0]['gc_overlap_ms'] >= 0
    assert not report['unmatched_gc_starts']


def test_report_clips_gc_overlap_without_subtracting_from_wall():
    host = HostLatency()
    host.events = [('start', 5, 2, 0, 0), ('stop', 25, 2, 3, 0)]
    host.samples = [dict(start_ns=10, end_ns=20), dict(start_ns=30, end_ns=40)]
    report = host.report([dict(wall_ms=10), dict(wall_ms=10)])
    assert report['samples'][0]['gc_overlap_ms'] == 10/1e6
    assert report['samples'][1]['gc_overlap_ms'] == 0
    assert all(row['wall_ms'] == 10 for row in report['samples'])
    with pytest.raises(ValueError, match='counts differ'):
        host.report([])
