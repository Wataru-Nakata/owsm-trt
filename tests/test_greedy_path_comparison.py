"""Small saved-result fixtures; no audio/model work."""

import copy
import gzip
import hashlib
import json

import compare_greedy_paths
import pytest


def saved_pair(tmp_path,monkeypatch,failed_control=False):
    monkeypatch.setattr(compare_greedy_paths,'runtime_tree',lambda _: 'same-runtime')
    roots=[]
    for index,(mode,case) in enumerate((('serial64','plain_ts'),('greedy_kernels','draft_ts'))):
        root=tmp_path/case
        root.mkdir()
        roots.append(root)
        config=dict(mode=mode,case=case,commit=str(index),checkpoint_sha256='weights',index_sha256='inputs',
                    selected_samples=2,selected_keys_sha256=hashlib.sha256(b'short\nlong').hexdigest())
        config_path=root/'config.json'
        config_path.write_text(json.dumps(config))
        digest=hashlib.sha256(config_path.read_bytes()).hexdigest()
        environment=dict(engine_sha256={'decoder':'engine'},torch='torch',cuda='cuda',gpu='gpu',
                         decoder_settings=dict(candidate=dict(solo_programs=64)))
        (root/'environment.json').write_text(json.dumps(environment))
        rows=[]
        for key in ('short','long'):
            available=key=='short'
            text='one' if available else ('two' if not index else 'three')
            output=dict(text=text,tokens=[101+index,1] if available else [],
                timestamp_tokens=[101+index] if available else [],token_ids_available=available,
                segments=[dict(start=.02*index if available else 0,end=1,text=text)],input_seconds=1,
                quality=dict(metric='WER',errors=0 if available else 1+index,reference_units=3))
            reference=copy.deepcopy(output)
            if failed_control and index and available:
                reference['tokens']=[999]
            rows.append(dict(key=key,dataset='speech',lang='eng',source={'reference_text':key},
                input_group=[key],api='transcribe' if available else 'transcribe_long',
                config_sha256=digest,reference=reference,candidate=output,differences=[]))
        path=root/'chunk-00000.jsonl.gz'
        with gzip.open(path,'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
        (root/'status.json').write_text(json.dumps(dict(complete=True,completed=2,differences=0,
            chunks={path.name:dict(samples=2,sha256=hashlib.sha256(path.read_bytes()).hexdigest())})))
    return roots


def test_numerical_comparison_retains_differences_and_excludes_missing_ids(tmp_path,monkeypatch):
    result=compare_greedy_paths.compare_paths(*saved_pair(tmp_path,monkeypatch))
    assert result['complete'] and result['different_inputs']==2
    assert result['agreement']['tokens']==dict(equal=0,total=1)
    assert result['agreement']['timestamp_tokens']==dict(equal=0,total=1)
    assert result['agreement']['text']==dict(equal=1,total=2)
    assert result['quality']['eng']['error_delta']==1
    assert result['boundary_displacements']['comparable_inputs']==1
    assert result['boundary_displacements']['excluded_inputs']==1
    assert result['boundary_displacements']['changed_boundaries']['maximum']==.02


def test_comparison_rejects_a_failed_within_run_control(tmp_path,monkeypatch):
    roots=saved_pair(tmp_path,monkeypatch,failed_control=True)
    with pytest.raises(AssertionError):
        compare_greedy_paths.compare_paths(*roots)


def test_comparison_rejects_changed_engine(tmp_path,monkeypatch):
    roots=saved_pair(tmp_path,monkeypatch)
    path=roots[1]/'environment.json'
    env=json.loads(path.read_text())
    env['engine_sha256']['decoder']='different-engine'
    path.write_text(json.dumps(env))
    with pytest.raises(AssertionError,match='engine_sha256'):
        compare_greedy_paths.compare_paths(*roots)
