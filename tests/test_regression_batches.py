"""Saved-output fixtures exercise comparison denominators and provenance, without model/audio work."""

import gzip
import hashlib
import json

import compare_regression_batches
import pytest
from compare_regression_batches import boundary_displacements, compare_batches


def write_fixture(root,batch):
    root.mkdir()
    config=dict(mode='beam_bound',case=f'beam_b{batch}',commit='fixture',checkpoint_sha256='checkpoint',
                index_sha256='corpus',selected_samples=3,
                selected_keys_sha256=hashlib.sha256(b'speech\nlong\nstress').hexdigest())
    (root/'config.json').write_text(json.dumps(config))
    digest=hashlib.sha256((root/'config.json').read_bytes()).hexdigest()
    (root/'environment.json').write_text(json.dumps(dict(engine_sha256={'decoder':'same'},
                                                        torch='fixture',cuda='fixture',gpu='fixture')))
    rows=[]
    for key in ('speech','long','stress'):
        text='word' if key!='stress' or batch==3 else ''
        tokens=[1] if text else []
        value=dict(text=text,tokens=tokens if key!='long' else [],timestamp_tokens=[],
                   token_ids_available=key!='long',segments=[dict(start=0,end=1,text=text)],
                   quality=dict(metric='WER',reference_units=0 if key=='stress' else 1,
                                errors=int(key=='stress' and bool(text))))
        if key=='speech' and batch==3:
            value['segments'][0]['end']=1.02
        rows.append(dict(key=key,dataset='stress' if key=='stress' else 'speech',lang='eng',source={},
                         config_sha256=digest,reference=value,candidate=value,differences=[],
                         input_group=['long'] if key=='long' else ['speech','stress'],
                         api='transcribe_long' if key=='long' else 'transcribe'))
    save_rows(root,rows)


def save_rows(root,rows):
    path=root/'chunk-00000.jsonl.gz'
    with gzip.open(path,'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    (root/'progress.json').write_text(json.dumps(dict(chunks={path.name:dict(
        samples=3,sha256=hashlib.sha256(path.read_bytes()).hexdigest())})))


def test_batch_comparison_separates_missing_ids_nonspeech_and_boundary_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(compare_regression_batches,'runtime_tree',lambda _: 'same')
    first,third=tmp_path/'b1',tmp_path/'b3'
    write_fixture(first,1)
    write_fixture(third,3)
    result=compare_batches(first,third)
    assert result['complete'] and result['samples']==3 and result['different_inputs']==2
    assert result['agreement']['tokens']==dict(equal=1,total=2)
    assert result['agreement']['segments']==dict(equal=1,total=3)
    assert result['quality']['eng']['samples']==2
    assert result['quality']['eng']['b3_errors']==0
    assert result['nonspeech']['eng']['b3_insertions']==1
    assert result['actual_groups']['transcribe_long:group1']['different_inputs']==0
    boundaries=result['boundary_displacements']
    assert boundaries['comparable_inputs']==2 and boundaries['excluded_inputs']==1
    assert boundaries['all_boundaries']['count']==4
    assert boundaries['changed_boundaries']['count']==1
    assert boundaries['changed_boundaries']['p95']==pytest.approx(.02)
    assert result['changed_inputs'][0]['max_boundary_displacement']==pytest.approx(.02)
    assert result['changed_inputs'][1]['max_boundary_displacement'] is None


@pytest.mark.parametrize('corruption',['hash','config','within_batch','order'])
def test_comparison_rejects_corrupt_or_incompatible_saved_records(tmp_path,monkeypatch,corruption):
    monkeypatch.setattr(compare_regression_batches,'runtime_tree',lambda _: 'same')
    first,third=tmp_path/'b1',tmp_path/'b3'
    write_fixture(first,1)
    write_fixture(third,3)
    path=third/'chunk-00000.jsonl.gz'
    with gzip.open(path,'rt') as stream:
        rows=[json.loads(line) for line in stream]
    if corruption=='hash':
        path.write_bytes(b'corrupt')
    else:
        if corruption=='config':
            rows[0]['config_sha256']='unrelated source/config'
        elif corruption=='within_batch':
            rows[0]['candidate']['tokens']=[99]
            rows[0]['differences']=['tokens']
        else:
            rows.reverse()
        save_rows(third,rows)
    with pytest.raises(AssertionError):
        compare_batches(first,third)


def test_boundary_displacement_does_not_align_different_segment_texts():
    value=dict(text='one two',segments=[dict(text='one',start=0,end=1),dict(text='two',start=1,end=2)])
    assert boundary_displacements(value,value)==[0,0,0,0]
    assert boundary_displacements(value,value|dict(segments=[dict(text='one two',start=0,end=2)])) is None
    assert boundary_displacements(value,value|dict(segments=[dict(text='one',start=0,end=float('nan')),
                                                            dict(text='two',start=1,end=2)])) is None
