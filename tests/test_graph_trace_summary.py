"""Do not attribute uncorrelated or other-thread GPU work to a replay label."""

import json
from types import SimpleNamespace

import pytest
from profile_graph_launches import graph_labels, trace_summary


def test_trace_requires_same_thread_enclosed_launch_and_matching_correlation(tmp_path):
    events = [
        dict(cat='user_annotation', ph='X', name='graph/decoder/a', ts=10, dur=5, tid=1),
        dict(cat='user_annotation', ph='X', name='graph/encoder/b', ts=10, dur=20, tid=2),
        dict(cat='cuda_runtime', name='cudaGraphLaunch', ts=11, dur=1, tid=1, args={'correlation': 7}),
        dict(cat='cuda_runtime', name='cudaGraphLaunch', ts=11, dur=1, tid=2, args={'correlation': 8}),
        dict(cat='cuda_runtime', name='cudaGraphLaunch', ts=14, dur=3, tid=1, args={'correlation': 9}),
        dict(cat='kernel', name='a', ts=50, dur=2, args={'correlation': 7}),
        dict(cat='kernel', name='b', ts=52, dur=4, args={'correlation': 8}),
        dict(cat='kernel', name='outside', ts=56, dur=6, args={'correlation': 9}),
        dict(cat='kernel', name='missing', ts=62, dur=8, args={}),
    ]
    path = tmp_path/'trace.json'
    path.write_text(json.dumps({'traceEvents': events}))
    result = trace_summary(path)
    assert result['gpu_kernel_sum_us'] == 20
    assert result['graph_correlated_kernel_sums_us'] == {
        'graph/decoder/a': 2, 'graph/encoder/b': 4, 'unattributed': 14,
    }
    assert result['cuda_api']['cudaGraphLaunch']['total_us'] == 5


def test_missing_gpu_activity_is_unknown_not_zero_cost(tmp_path):
    path = tmp_path/'trace.json'
    path.write_text(json.dumps({'traceEvents': []}))
    result = trace_summary(path)
    assert result['gpu_kernel_sum_us'] is None
    assert result['graph_correlated_kernel_sums_us'] is None


@pytest.mark.parametrize('error', [False, True])
def test_replay_labels_restore_all_captured_objects_even_after_error(error):
    class Graph:
        calls = 0

        def replay(self):
            self.calls += 1

    graph = Graph()
    bound = SimpleNamespace(graph=graph, notime_graph=graph)
    session = SimpleNamespace(graphs={64: graph}, _tail_graphs={(64, 1): graph},
                              _enc_graphs={('buffer', (1, 100, 80)): ('run', 'buf', graph)},
                              _verify={(64, 2): bound})
    frontend = SimpleNamespace(entries={'shape': ('input', 'output', graph)})
    model = SimpleNamespace(_sessions={(1, 1): session}, _frontend_replay=frontend)
    failure = ValueError('diagnostic failure')
    try:
        with graph_labels(model):
            session.graphs[64].replay()
            bound.notime_graph.replay()
            assert graph.calls == 2
            if error:
                raise failure
    except ValueError as caught:
        assert caught is failure
    assert session.graphs[64] is session._tail_graphs[64, 1] is graph
    assert session._enc_graphs['buffer', (1, 100, 80)][2] is graph
    assert bound.graph is bound.notime_graph is frontend.entries['shape'][2] is graph
