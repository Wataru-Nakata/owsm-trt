"""Synthetic timing records only; no inference or timing on the login node."""

import pytest
from summarize_expanded_greedy import aggregate


def samples(streaming=False):
    return [dict(variant=v,repeat=r,ids=ids,wall_ms=ms,audio_seconds=len(ids),
                 peak_allocated_bytes=100,draft_decoder_calls=len(ids)*2)
            for v,ms in [('reference',10),('candidate',5)] for r in range(3)
            for ids in ([['a','b']] if streaming else [['a'],['b']])]


@pytest.mark.parametrize('streaming',[False,True])
def test_summary_pairs_repeat_totals_and_keeps_request_denominators(streaming):
    result=aggregate(samples(streaming),['a','b'],3,streaming)
    assert result['pooled_rtfx_change_percent']==100
    assert result['paired_rtfx_change_percent']['samples']==[100,100,100]
    assert result['variants']['candidate']['request_wall_ms']['n']==(3 if streaming else 6)
    assert result['repeat_totals_ms']['reference']==([10]*3 if streaming else [20]*3)
    assert result['draft_call_differences']==[]


@pytest.mark.parametrize('corruption',['missing','duplicate','nan','duration'])
def test_summary_rejects_incomplete_or_unpaired_timing(corruption):
    rows=samples()
    if corruption=='missing':
        rows.pop()
    elif corruption=='duplicate':
        rows.append(rows[0])
    elif corruption=='nan':
        rows[0]['wall_ms']=float('nan')
    else:
        rows[0]['audio_seconds']=9
    with pytest.raises(AssertionError):
        aggregate(rows,['a','b'],3,False)
