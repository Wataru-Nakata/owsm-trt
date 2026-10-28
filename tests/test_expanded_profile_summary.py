"""Synthetic timing records only; no inference or timing on the login node."""

import json

import pytest
from summarize_expanded_greedy import aggregate, beam_diagnostics, output_agreement, summarize


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


def test_output_differences_are_retained_only_for_explicit_n_comparison():
    value=dict(tokens=[1],timestamp_tokens=[],text='one',segments=[])
    records=dict(reference={'a':value},candidate={'a':value|dict(tokens=[2],text='two')})
    with pytest.raises(AssertionError,match='E output difference'):
        output_agreement(records,['a'],exact=True)
    result=output_agreement(records,['a'],exact=False)
    assert result['tokens']==dict(equal=0,total=1,different_ids=['a'])
    assert result['timestamp_tokens']==dict(equal=1,total=1,different_ids=[])
    with pytest.raises(AssertionError):
        output_agreement(records,['a','missing'],exact=False)


def test_instrumented_job_cannot_be_reported_as_warm_latency(tmp_path):
    (tmp_path/'profile').mkdir()
    (tmp_path/'exit_status.txt').write_text('0\n')
    for name,value in [('pbs-final.json',dict(state='F',exit_status=0)),('source.json',{}),
                       ('profile/environment.json',dict(host_diagnostics=True))]:
        (tmp_path/name).write_text(json.dumps(value))
    with pytest.raises(AssertionError,match='instrumented host diagnostics'):
        summarize(tmp_path)


def beam_records(single, candidate):
    sizes=[1]*32 if single else [3]*10+[2]
    sessions={}
    for size in set(sizes):
        tail=candidate and not single
        gluon=single or candidate
        keys=[str(bucket) for bucket in (64,128,256,512)]
        if tail:
            keys.extend(f'tail/group={group}/cache={bucket}'
                        for group in range(size) for bucket in (64,128,256,512))
        sessions[f'({size}, 5)']=dict(
            dispatch={key:dict(implementation='megakernel-gluon' if gluon else 'tensorrt',
                               fallback_reason=None if gluon else 'shape limit') for key in keys},
            fused_lse=candidate,compact_tail=tail,chunk=1 if single else 4,last_request_tail_start=None)
    return dict(plain_batches=[dict(beam=True,rows=5*size,decoder_calls=4,output_lengths=[2]*size)
                               for size in sizes],sessions=sessions,scope='separate diagnostic pass')


@pytest.mark.parametrize('single',[True,False])
@pytest.mark.parametrize('candidate',[True,False])
def test_beam_replays_exclude_prefill_and_validate_partial_batch(single,candidate):
    result=beam_diagnostics(beam_records(single,candidate),
        'beam_single_ts' if single else 'beam_stream_ts_b3','candidate' if candidate else 'reference')
    assert result['prefill_selections']==(32 if single else 11)
    assert result['decoder_calls']==4*result['prefill_selections']


@pytest.mark.parametrize('corruption',['route','missing_tail','missing_b2','tail_on_b1','fractional_calls'])
def test_beam_summary_rejects_wrong_effective_routes_or_counts(corruption):
    single=corruption=='tail_on_b1'
    observed=beam_records(single,True)
    session=observed['sessions']['(1, 5)' if single else '(3, 5)']
    if corruption=='route':
        session['dispatch']['64']['implementation']='tensorrt'
    elif corruption=='missing_tail':
        del session['dispatch']['tail/group=2/cache=512']
    elif corruption=='missing_b2':
        del observed['sessions']['(2, 5)']
    elif corruption=='tail_on_b1':
        session['compact_tail']=True
    else:
        observed['plain_batches'][0]['decoder_calls']=4.5
    with pytest.raises(AssertionError):
        beam_diagnostics(observed,'beam_single_ts' if single else 'beam_stream_ts_b3','candidate')
