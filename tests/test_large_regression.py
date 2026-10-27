"""Small byte/metadata fixtures cover shard integrity and regression accounting, without audio/model work."""

import gzip
import hashlib
import json
import tarfile

import pytest
from audit_large_regression import (
    capture_failure,
    load_completed,
    record_differences,
    repetition,
    select_rows,
)
from prepare_large_regression import safe_key, write_shard


@pytest.mark.parametrize('key',['','../outside','folder/item','a.b','bad key'])
def test_shard_keys_cannot_escape_or_confuse_extensions(key):
    with pytest.raises(ValueError):
        safe_key(key)


def test_shard_preserves_bytes_and_metadata(tmp_path):
    source = tmp_path/'source.flac'
    source.write_bytes(b'tiny encoded-audio placeholder')
    row = dict(key='dataset--one',source_path=str(source),text='reference',lang='eng')
    dest = tmp_path/'shard.tar'
    result = write_shard(dest,[row])
    assert result['samples']==1 and result['sha256']==hashlib.sha256(dest.read_bytes()).hexdigest()
    with tarfile.open(dest) as data:
        assert data.extractfile('dataset--one.flac').read()==source.read_bytes()
        saved = json.load(data.extractfile('dataset--one.json'))
        assert saved==row and saved['audio_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        write_shard(tmp_path/'duplicate.tar',[row,row])
    with pytest.raises(ValueError):
        write_shard(tmp_path/'empty.tar',[])


def test_differences_include_timestamps_and_segments_not_just_text():
    ref = dict(tokens=[1,2],timestamp_tokens=[1],segments=[dict(start=0,end=1,text='same')],text='same')
    actual = ref|dict(tokens=[3,2],timestamp_tokens=[3],segments=[dict(start=.02,end=1,text='same')])
    assert record_differences(ref,actual)==['tokens','timestamp_tokens','segments']
    assert record_differences(ref,dict(ref))==[]
    assert repetition([1]*8)['max_identical_token_run']==8
    assert repetition([1,2,3,4]*3)['max_fourgram_count']==3
    assert repetition([])['max_fourgram_count']==0


def test_deterministic_selection_and_separate_long_form():
    rows=[dict(key=str(i),lang=lang,dataset=dataset,long_form=long) for i,(lang,dataset,long) in enumerate([
        ('eng','clean',False),('jpn','jvs',False),('eng','long',True),('deu','fleurs',False)])]
    selected=select_rows(rows,'beam_b1',None)
    assert [r['lang'] for r in selected]==['jpn','eng','deu']
    assert selected==select_rows(rows[::-1],'beam_b1',None)
    assert select_rows(rows,'long',None)==[rows[2]]
    assert select_rows(rows,'beam_b1',['clean'])==[rows[0]]


def test_failure_keeps_reproduction_context_and_reraises(tmp_path,monkeypatch):
    monkeypatch.setenv('PBS_JOBID','synthetic')
    out,published = tmp_path/'local',tmp_path/'published'
    out.mkdir()
    context = dict(variant='candidate',inputs=[dict(key='sample',shard='audio-0001.tar')])
    def fail():
        raise RuntimeError('exhausted')
    with pytest.raises(RuntimeError,match='exhausted'):
        capture_failure(fail,context,out,published)
    data = json.loads((published/'failure-synthetic.json').read_text())
    assert data['inputs']==context['inputs'] and data['error_type']=='RuntimeError'
    assert (out/'failure-synthetic.json').read_bytes()==(published/'failure-synthetic.json').read_bytes()


def test_resume_rejects_partial_wrong_config_and_changed_chunks(tmp_path):
    rows = [dict(key=str(i)) for i in range(3)]
    value = dict(tokens=[],timestamp_tokens=[],segments=[],text='')
    records = [dict(key=str(i),config_sha256='digest',differences=[],reference=value,candidate=value) for i in range(2)]
    path = tmp_path/'chunk-00000.jsonl.gz'
    def write(values):
        with gzip.open(path,'wt') as f:
            for row in values:
                f.write(json.dumps(row)+'\n')
    write(records)
    completed = load_completed(tmp_path,rows,2,'digest')
    assert completed[path.name]['samples']==2
    with pytest.raises(AssertionError):
        load_completed(tmp_path,rows,2,'different config')
    write(records[:1])
    with pytest.raises(AssertionError):
        load_completed(tmp_path,rows,2,'digest')
    write(records)
    (tmp_path/'progress.json').write_text(json.dumps(dict(chunks=completed)))
    records[0]['candidate'] = value|dict(text='changed')
    records[0]['differences'] = ['text']
    write(records)
    with pytest.raises(AssertionError):
        load_completed(tmp_path,rows,2,'digest')
