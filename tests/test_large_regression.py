"""Small byte/metadata fixtures cover shard integrity and regression accounting, without audio/model work."""

import gzip
import hashlib
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import audit_large_regression
import prepare_large_regression
import pytest
from audit_large_regression import (
    beam_variants,
    capture_failure,
    draft_dispatch,
    input_groups,
    load_completed,
    record_differences,
    repetition,
    select_rows,
)
from prepare_large_regression import (
    jvs_nonpara_inventory,
    missing_inputs,
    recover_missing_inputs,
    restore_dev_other,
    safe_key,
    write_shard,
)
from segment_metrics import invalid_segments
from summarize_large_regression import summarize


@pytest.mark.parametrize('mode,case,message',[
    ('draft_budget','draft_ts','draft_budget requires draft_bulk'),
    ('frontend_batches','draft_ts','frontend_batches requires draft_bulk'),
    ('beam_preset','beam_prompt','beam_preset requires beam_b1 or beam_b3'),
    ('ring_split','draft_bulk','single-row scheduling knobs are not bulk ablations'),
])
def test_bulk_ablation_rejects_inactive_or_unrelated_scheduling_options(monkeypatch,capsys,mode,case,message):
    monkeypatch.setenv('PBS_JOBID','metadata-only')
    monkeypatch.setattr('sys.argv',['audit','--model','unused','--data','unused','--out','unused',
                                  '--publish','unused','--mode',mode,'--case',case])
    with pytest.raises(SystemExit) as error:
        audit_large_regression.main()
    assert error.value.code==2 and message in capsys.readouterr().err


def test_numerical_preset_keeps_existing_search_optimizations_on_in_both_variants():
    modes=beam_variants('beam_preset')
    for name,settings in modes.items():
        assert settings['beam_upper_bound'] and settings['beam_kernels']
        assert all(settings[key] is (name=='candidate') for key in (
            'beam_grouped_cross','beam_compact_tail','beam_fused_lse'))
    assert beam_variants('beam_bound')['reference']==dict(beam_upper_bound=False,beam_kernels=True)
    assert beam_variants('beam_kernels')['reference']==dict(beam_upper_bound=True,beam_kernels=False)
    assert beam_variants('beam_combined')['reference']==dict(beam_upper_bound=False,beam_kernels=False)


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


@pytest.mark.parametrize('corrupt_copy',[False,True])
def test_reuse_verifies_staged_bytes_before_marking_local_corpus_complete(tmp_path,monkeypatch,corrupt_copy):
    published,local = tmp_path/'published',tmp_path/'local'
    published.mkdir()
    shard = published/'audio-0000.tar'
    shard.write_bytes(b'small encoded-shard fixture')
    manifest = dict(samples=1,files=[dict(name=shard.name,sha256=hashlib.sha256(shard.read_bytes()).hexdigest())])
    (published/'complete.json').write_text(json.dumps(manifest))
    original_copy = shutil.copyfile
    def copy(source,target):
        result = original_copy(source,target)
        if corrupt_copy and Path(source).suffix=='.tar':
            Path(target).write_bytes(b'corrupt staged copy')
        return result
    monkeypatch.setattr(prepare_large_regression.shutil,'copyfile',copy)
    monkeypatch.setenv('PBS_JOBID','synthetic')
    monkeypatch.setattr('sys.argv',['prepare_large_regression','--repo',str(tmp_path),'--jvs',str(tmp_path/'jvs'),
                                  '--out',str(local),'--publish',str(published)])
    if corrupt_copy:
        with pytest.raises(AssertionError):
            prepare_large_regression.main()
        assert not (local/'complete.json').exists()
    else:
        prepare_large_regression.main()
        assert (local/shard.name).read_bytes()==shard.read_bytes()
        assert json.loads((local/'complete.json').read_text())==manifest


def test_differences_include_timestamps_and_segments_not_just_text():
    ref = dict(tokens=[1,2],timestamp_tokens=[1],segments=[dict(start=0,end=1,text='same')],text='same')
    actual = ref|dict(tokens=[3,2],timestamp_tokens=[3],segments=[dict(start=.02,end=1,text='same')])
    assert record_differences(ref,actual)==['tokens','timestamp_tokens','segments']
    assert record_differences(ref,dict(ref))==[]
    assert repetition([1]*8)['max_identical_token_run']==8
    assert repetition([1,2,3,4]*3)['max_fourgram_count']==3
    assert repetition([])['max_fourgram_count']==0


def test_segment_range_allows_rounding_at_existing_tolerance_but_rejects_real_errors():
    assert 8.30 > 8.28 + .02  # The previous diagnostic incorrectly flagged this exact boundary.
    assert invalid_segments([dict(start=0,end=8.30)],8.28)==0
    assert invalid_segments([dict(start=0,end=8.300001)],8.28)==1
    assert invalid_segments([dict(start=0,end=8.32)],8.28)==1
    assert invalid_segments([dict(start=-.02,end=0),dict(start=1,end=.98),
                             dict(start=float('nan'),end=1),dict(start=0,end=float('inf'))],8.28)==4


def test_deterministic_selection_and_separate_long_form():
    rows=[dict(key=str(i),lang=lang,dataset=dataset,long_form=long) for i,(lang,dataset,long) in enumerate([
        ('eng','clean',False),('jpn','jvs',False),('eng','long',True),('deu','fleurs',False)])]
    selected=select_rows(rows,'beam_b1',None)
    assert [r['lang'] for r in selected]==['jpn','eng','deu']
    assert selected==select_rows(rows[::-1],'beam_b1',None)
    assert select_rows(rows,'long',None)==[rows[2]]
    assert select_rows(rows,'beam_long',None)==[rows[2]]
    assert select_rows(rows,'beam_b1',['clean'])==[rows[0]]


def test_bulk_groups_keep_language_and_long_api_boundaries():
    rows=[dict(key=str(i),lang='eng' if i<75 else 'jpn',seconds=31 if i==69 else i%30+1)
          for i in range(100)]
    groups=list(input_groups(rows,64,30,False))
    assert [r for batch,_ in groups for r in batch]==rows
    assert any(len(batch)>8 and not joined for batch,joined in groups)
    for batch,joined in groups:
        assert len(batch)<=64 and len({r['lang'] for r in batch})==1
        assert all((r['seconds']>30)==joined for r in batch)


def test_draft_metadata_never_constructs_or_runs_a_lazy_bound():
    def fail():
        raise AssertionError('metadata must not execute GPU work')
    fail.dispatch='test-decoder'
    ready=SimpleNamespace(step=fail,graph=object(),notime_graph=None)
    deferred=SimpleNamespace(step=None,factory=fail)
    session=SimpleNamespace(_verify={(64,8):ready,(128,32):deferred},_verify_solo={(64,4):ready})
    result=draft_dispatch(session)
    assert set(result)=={'draft/solo=False/cache=64/window=8','draft/solo=True/cache=64/window=4'}
    assert all(v['implementation']=='test-decoder' and v['graph_captured'] for v in result.values())
    assert deferred.step is None and deferred.factory is fail


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


def test_prompt_keeps_full_reference_only_for_supported_input_windows():
    rows = [dict(key=str(i),lang='eng',dataset='speech',long_form=False,seconds=seconds)
            for i,seconds in enumerate((1,30,30.01,45))]
    with pytest.raises(ValueError,match='input-window limit'):
        select_rows(rows,'beam_prompt',None)
    assert {r['seconds'] for r in select_rows(rows,'beam_prompt',None,30)}=={1,30}
    assert len(select_rows(rows,'beam_b1',None))==4


def test_mixed_batch_does_not_change_short_input_api_or_output_order():
    rows = [dict(key=str(i),lang=lang,seconds=seconds) for i,(lang,seconds) in enumerate([
        ('eng',2),('eng',31),('eng',4),('eng',7),('eng',9),('jpn',10)])]
    groups = list(input_groups(rows,3,30,False))
    assert [r for batch,_ in groups for r in batch]==rows
    assert [(len(batch),joined) for batch,joined in groups]==[(1,False),(1,True),(1,False),(2,False),(1,False)]
    assert all(len({r['lang'] for r in batch})==1 for batch,_ in groups)
    assert all(joined for _,joined in input_groups(rows,3,30,True))


def test_recover_local_archive_verifies_transcript_and_rejects_links(tmp_path,monkeypatch):
    path = tmp_path/'tiny.tar.gz'
    prefix = 'LibriSpeech/dev-other/116/288045/'
    with tarfile.open(path,'w:gz') as archive:
        for name,data in [('116-288045-0000.flac',b'original encoded bytes'),
                          ('116-288045.trans.txt',b'116-288045-0000 HELLO\n')]:
            info = tarfile.TarInfo(prefix+name)
            info.size = len(data)
            archive.addfile(info,io.BytesIO(data))
        info = tarfile.TarInfo(prefix+'116-288045-0001.flac')
        info.type = tarfile.SYMTYPE
        info.linkname = '/outside'
        archive.addfile(info)
    row = dict(id='116-288045-0000',text='HELLO',source_path='/missing',key='sample',dataset='librispeech_dev_other')
    restored = dict(row)
    restore_dev_other([restored],path,tmp_path/'local')
    assert restored['original_source_path']=='/missing'
    assert Path(restored['source_path']).read_bytes()==b'original encoded bytes'
    assert missing_inputs([restored])==[]
    cached = tmp_path/'.cache/regression-inputs/dev-other.tar.gz'
    cached.parent.mkdir(parents=True)
    cached.write_bytes(path.read_bytes())
    local = tmp_path/'recovery'
    local.mkdir()
    monkeypatch.setattr('prepare_large_regression.DEV_OTHER_MD5',hashlib.md5(path.read_bytes()).hexdigest())
    recovered = dict(row)
    report = recover_missing_inputs([recovered],tmp_path,local,tmp_path/'published')
    assert report['restored_archive']['samples']==1 and not report['unresolved']
    assert Path(recovered['source_path']).read_bytes()==b'original encoded bytes'
    monkeypatch.setattr('prepare_large_regression.DEV_OTHER_MD5','0'*32)
    with pytest.raises(ValueError,match='checksum mismatch'):
        recover_missing_inputs([dict(row)],tmp_path,local,tmp_path/'published')
    with pytest.raises(ValueError,match='transcript differs'):
        restore_dev_other([row|dict(text='different')],path,tmp_path/'bad')
    with pytest.raises(ValueError,match='invalid LibriSpeech id'):
        restore_dev_other([row|dict(id='../../outside')],path,tmp_path/'bad')
    # Matching transcript does not make a symlink an admissible audio member.
    with tarfile.open(path,'w:gz') as archive:
        data = b'116-288045-0001 HELLO\n'
        info = tarfile.TarInfo(prefix+'116-288045.trans.txt')
        info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
        info = tarfile.TarInfo(prefix+'116-288045-0001.flac')
        info.type=tarfile.SYMTYPE
        info.linkname='/outside'
        archive.addfile(info)
    with pytest.raises(ValueError,match='regular archive member'):
        restore_dev_other([row|dict(id='116-288045-0001')],path,tmp_path/'bad')


def test_missing_preflight_reports_all_missing_inputs_before_any_download(tmp_path,monkeypatch):
    local,published = tmp_path/'local',tmp_path/'published'
    local.mkdir()
    rows = [dict(key=str(i),dataset=dataset,source_path=str(tmp_path/str(i)))
            for i,dataset in enumerate(('librispeech_dev_other','fleurs_ja_validation'))]
    def no_download(*args,**kwargs):
        raise AssertionError('must report unknown missing inputs before downloading')
    monkeypatch.setattr('prepare_large_regression.subprocess.run',no_download)
    with pytest.raises(FileNotFoundError,match='all paths'):
        recover_missing_inputs(rows,tmp_path,local,published)
    report = json.loads((published/'preflight.json').read_text())
    assert len(report['before_restore'])==2
    assert report['missing_by_dataset']=={'librispeech_dev_other':1,'fleurs_ja_validation':1}


def test_jvs_archive_distinguishes_unused_transcripts_from_missing_recordings(tmp_path):
    jvs = tmp_path/'jvs_ver1'
    root = jvs/'jvs001/nonpara30'
    wavs = root/'wav24kHz16bit'
    wavs.mkdir(parents=True)
    (root/'transcripts_utf8.txt').write_text('RECORDED:hello\nDELETED:missing file\nUNRECORDED:extra text\n')
    (wavs/'RECORDED.wav').write_bytes(b'fixture')
    with zipfile.ZipFile(jvs.with_suffix('.zip'),'w') as archive:
        for name in ('RECORDED','DELETED'):
            archive.writestr('jvs_ver1/jvs001/nonpara30/wav24kHz16bit/'+name+'.wav',b'fixture')
    rows,provenance = jvs_nonpara_inventory(jvs,range(1,2))
    assert {r['id'] for r in rows}=={'jvs001-RECORDED','jvs001-DELETED'}
    assert provenance['transcript_entries_without_audio']=={'jvs001':['UNRECORDED']}
    assert [r['key'] for r in missing_inputs(rows)]==['jvs_nonpara90--jvs001-DELETED']


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


def test_summary_separates_nonspeech_and_excludes_unavailable_token_ids(tmp_path):
    config = dict(selected_samples=3,selected_keys_sha256=hashlib.sha256(b'a\nb\nc').hexdigest())
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    value = dict(tokens=[1],timestamp_tokens=[],segments=[dict(start=0,end=8.30)],
        input_seconds=8.28,text='word',token_ids_available=True,
        empty_segments=0,invalid_segments=1,unpaired_timestamp=False,nonmonotonic_timestamps=False,
        repetition=repetition([1]),quality=dict(metric='WER',errors=1,reference_units=10))
    probe = value|dict(quality=dict(metric='CER',errors=9,reference_units=0))
    joined = value|dict(tokens=[],token_ids_available=False,
                       segments=[dict(start=0,end=3),dict(start=2,end=4)])
    records = [dict(key=key,lang=lang,dataset=dataset,duration_bucket='<=10s',config_sha256=digest,
                    reference=record,candidate=record,differences=[])
               for key,lang,dataset,record in [('a','eng','speech',value),('b','jpn','stress',probe),
                                                ('c','eng','long',joined)]]
    path = tmp_path/'chunk-00000.jsonl.gz'
    with gzip.open(path,'wt') as f:
        for record in records:
            f.write(json.dumps(record)+'\n')
    (tmp_path/'progress.json').write_text(json.dumps(dict(chunks={
        path.name:dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    })))
    result = summarize(tmp_path)
    assert result['complete'] and result['completed']==3
    total = result['strata']['all']
    assert total['agreement']['tokens']==dict(equal=2,total=2)
    assert total['agreement']['text']==dict(equal=3,total=3)
    assert total['quality']['candidate']['rate'] is None
    assert total['quality']['candidate']['errors']==2
    assert total['quality']['candidate']['excluded_nonspeech_errors']==9
    assert total['flags']['candidate']['timestamp_token_records']==2
    assert total['flags']['candidate']['nonmonotonic_segment_boundaries']==1
    assert total['flags']['candidate']['recorded_invalid_segments']==3
    assert total['flags']['candidate']['invalid_segments']==0
    assert result['strata']['language:eng']['quality']['candidate']['rate']==.1
    assert result['strata']['dataset:stress']['quality']['candidate']['rate'] is None
    path.write_bytes(b'changed')
    with pytest.raises(AssertionError):
        summarize(tmp_path)
