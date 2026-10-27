"""Tiny metadata checks for focused batch reproduction; no model or audio operations."""

import copy
import itertools

import pytest
from reproduce_batch_corner import select_group, summarize_runs


def test_corner_group_uses_exact_key_and_original_neighbours():
    rows=[dict(key=k,lang='zho',audio_sha256=k+'-hash') for k in ('one','two','three')]
    case=dict(key='three',input_group=['one','two','three'],source=dict(audio_sha256='three-hash'))
    assert select_group(rows[::-1],case)==rows
    with pytest.raises(AssertionError):
        select_group(rows,case|dict(source=dict(audio_sha256='different-audio')))
    with pytest.raises(AssertionError):
        select_group(rows,case|dict(input_group=['one','one','three']))


def diagnostic_runs():
    runs=[]
    for route,batch,kernels,bound,repeat in itertools.product(('auto','tensorrt'),(1,3),
                                                            (False,True),(False,True),range(2)):
        # N differences across decoder/batch configurations must not count as E failures.
        token=1 if route=='auto' and batch==1 else 2
        value=dict(tokens=[token],timestamp_tokens=[],text=str(token),segments=[])
        runs.append(dict(route=route,batch=batch,kernels=kernels,bound=bound,repeat=repeat,
                         outputs={'three':value}))
    return runs


def test_diagnostic_summary_separates_numerical_routes_from_exact_controls():
    runs=diagnostic_runs()
    summary=summarize_runs(runs,2)
    assert summary['complete'] and summary['runs']==32
    assert not summary['unstable'] and not summary['exact_failures']
    changed=copy.deepcopy(runs)
    changed[-1]['outputs']['three']['tokens']=[99]
    summary=summarize_runs(changed,2)
    assert len(summary['unstable'])==1
    assert {r['knob'] for r in summary['exact_failures']}=={'kernels','bound'}
    for invalid in (runs[:-1],runs+[runs[-1]]):
        with pytest.raises(AssertionError):
            summarize_runs(invalid,2)
