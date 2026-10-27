"""Reproduce a retained batch-sensitive input on fixed engines; diagnostic output checks only."""

import argparse
import gc
import hashlib
import io
import itertools
import json
import os
import tarfile
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path

from audit_large_regression import capture_failure, record_differences, repetition
from prepare_large_regression import publish_file, sha256


def select_group(index: list[dict],case: dict) -> list[dict]:
    by_key={r['key']:r for r in index}
    assert len(by_key)==len(index), 'duplicate corpus key'
    keys=case['input_group']
    assert len(keys)==3 and len(set(keys))==3 and case['key'] in keys
    rows=[by_key[k] for k in keys]
    assert len({r['lang'] for r in rows})==1
    assert by_key[case['key']]['audio_sha256']==case['source']['audio_sha256']
    return rows


def load_group_waves(rows: list[dict],data_dir: Path):
    import soundfile as sf
    import torch

    from owsm_trt.audio import load_audio

    waves=[]
    with ExitStack() as stack:
        archives={shard:stack.enter_context(tarfile.open(data_dir/shard))
                  for shard in dict.fromkeys(r['shard'] for r in rows)}
        for row in rows:
            raw=archives[row['shard']].extractfile(row['audio_member']).read()
            assert hashlib.sha256(raw).hexdigest()==row['audio_sha256']
            data,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
            mono=data[:,0] if data.shape[1]==1 else data.mean(axis=1)
            waves.append(load_audio(torch.from_numpy(mono),sr=sr))
    return waves


def summarize_runs(runs: list[dict],repeats: int) -> dict:
    by_mode={}
    for run in runs:
        key=tuple(run[k] for k in ('route','batch','kernels','bound','repeat'))
        assert key not in by_mode, 'duplicate diagnostic run'
        by_mode[key]=run['outputs']
    expected=set(itertools.product(('auto','tensorrt'),(1,3),(False,True),(False,True),range(repeats)))
    assert set(by_mode)==expected, 'incomplete diagnostic matrix'
    def differences(a,b):
        assert a.keys()==b.keys(), 'different diagnostic input sets'
        return {key:fields for key in a if (fields:=record_differences(a[key],b[key]))}
    unstable=[]
    exact_failures=[]
    for (route,batch,kernels,bound,repeat),outputs in by_mode.items():
        if repeat:
            changed=differences(by_mode[route,batch,kernels,bound,0],outputs)
            if changed:
                unstable.append(dict(route=route,batch=batch,kernels=kernels,bound=bound,
                                     repeat=repeat,differences=changed))
        for knob in ('kernels','bound'):
            if (kernels if knob=='kernels' else bound):
                baseline=(route,batch,False if knob=='kernels' else kernels,
                          False if knob=='bound' else bound,repeat)
                changed=differences(by_mode[baseline],outputs)
                if changed:
                    exact_failures.append(dict(route=route,batch=batch,kernels=kernels,bound=bound,
                                               repeat=repeat,knob=knob,differences=changed))
    return dict(complete=True,runs=len(runs),unstable=unstable,exact_failures=exact_failures,
        scope='Only fixed-route/fixed-batch repeat stability and search-kernel/bound toggles are E checks. '
              'Changing batch or decoder route is N; encoder batch shape also changes. No latency claim.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--case-artifact',type=Path,required=True)
    ap.add_argument('--audit-artifact',type=Path,required=True)
    ap.add_argument('--key',required=True)
    ap.add_argument('--repeat',type=int,default=3)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--publish',type=Path,required=True)
    args=ap.parse_args()
    if not os.environ.get('PBS_JOBID') or not os.environ.get('PBS_LOCALDIR'):
        ap.error('model/audio work requires a compute allocation')
    if args.repeat<2:
        ap.error('at least two repeats are needed to assess stability')
    cases=json.loads(args.case_artifact.read_text())['cases']
    matching=[c for c in cases if c['key']==args.key]
    assert len(matching)==1, 'corner key is missing or ambiguous'
    case=matching[0]
    audit=json.loads(args.audit_artifact.read_text())
    assert audit['summary']['complete'], 'original audit is incomplete'
    rows=select_group(json.loads((args.data/'index.json').read_text()),case)
    args.out.mkdir(parents=True,exist_ok=True)
    args.publish.mkdir(parents=True,exist_ok=True)

    import torch
    from eval_wer import CER_LANGS, characters, edit_distance, normalize
    from profile_beam import command, dispatch, result_record

    from owsm_trt import OWSM, hub, megakernel
    from owsm_trt.build import engine_tag

    waves=load_group_waves(rows,args.data)
    files=hub.resolve(args.model)
    runs=[]
    with torch.inference_mode():
        model=OWSM.from_pretrained(args.model,decoder_kernel='auto')
        assert isinstance(model.backend,megakernel.MegaBackend), 'auto backend unavailable'
        assert all(r['seconds']<=model.cfg.speech_seconds for r in rows), 'short-form case required'
        backends={'auto':model.backend,'tensorrt':model.backend.inner}
        engines=hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
        environment=dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),job=os.environ['PBS_JOBID'],
            checkpoint_sha256=sha256(files.checkpoint),index_sha256=sha256(args.data/'index.json'),
            case_sha256=sha256(args.case_artifact),engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            nvidia_smi=command(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv']))
        for field in ('checkpoint_sha256','index_sha256'):
            assert environment[field]==audit['summary']['config'][field],field
        for field in ('engine_sha256','torch','cuda','gpu'):
            assert environment[field]==audit['environment'][field],field
        # Separate sessions avoid reusing graphs captured under another route or search setting.
        for route,batch,kernels,bound in itertools.product(('auto','tensorrt'),(1,3),(False,True),(False,True)):
            model.backend=backends[route]
            model.beam_kernels=kernels
            model.beam_upper_bound=bound
            model.ctc_draft=False
            model._sessions=OrderedDict()
            for repeat in range(args.repeat):
                values=capture_failure(lambda:model.transcribe(waves,lang=rows[0]['lang'],task='asr',
                    timestamps=True,beam_size=5,batch_size=batch),
                    dict(environment=environment,inputs=rows,route=route,batch=batch,kernels=kernels,
                         bound=bound,repeat=repeat),args.out,args.publish)
                assert len(values)==len(rows)
                assert not backends['auto'].failed, 'unexpected megakernel failure'
                outputs={}
                for row,value in zip(rows,values):
                    record=result_record(model,value,row,row['seconds'])
                    units=characters if row['lang'] in CER_LANGS else normalize
                    reference,hypothesis=units(row['text']),units(record['text'])
                    record.update(token_ids_available=True,repetition=repetition(record['tokens']),
                        quality=dict(metric='CER' if row['lang'] in CER_LANGS else 'WER',
                                     errors=edit_distance(reference,hypothesis),reference_units=len(reference)))
                    outputs[row['key']]=record
                runs.append(dict(route=route,batch=batch,kernels=kernels,bound=bound,repeat=repeat,
                    outputs=outputs,dispatch={str(k):dispatch(s) for k,s in model._sessions.items()},
                    original_audit_differences=record_differences(case[f'b{batch}'],outputs[case['key']])))
            torch.cuda.synchronize()
            progress=dict(job=os.environ['PBS_JOBID'],completed_runs=len(runs),expected_runs=16*args.repeat,
                          route=route,batch=batch,kernels=kernels,bound=bound,complete=False)
            path=args.out/'progress.json'
            path.write_text(json.dumps(progress,indent=2)+'\n')
            publish_file(path,args.publish/path.name)
            # Keep completed diagnostic runs even if a subsequent condition fails.
            (args.out/'runs.json').write_text(json.dumps(dict(environment=environment,inputs=rows,runs=runs),
                                                         indent=2,ensure_ascii=False)+'\n')
            print(json.dumps(progress),flush=True)
            model._sessions.clear()
            gc.collect()
            torch.cuda.empty_cache()
    summary=summarize_runs(runs,args.repeat)
    result=dict(environment=environment,inputs=rows,target=case['key'],summary=summary,runs=runs)
    path=args.out/'result.json'
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    publish_file(path,args.publish/path.name)
    print(json.dumps(summary),flush=True)
    if summary['unstable'] or summary['exact_failures']:
        raise SystemExit(3)


if __name__=='__main__':
    main()
