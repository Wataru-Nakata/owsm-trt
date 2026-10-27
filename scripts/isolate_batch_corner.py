"""Cross fixed encoder conditioning with decoder batch shape for a retained corner case."""

import argparse
import itertools
import json
import os
from pathlib import Path

from audit_large_regression import capture_failure, record_differences, repetition
from prepare_large_regression import publish_file, sha256
from reproduce_batch_corner import load_group_waves, select_group


def summarize_stages(runs: list[dict],expected: dict,repeats: int) -> dict:
    by_mode={}
    for run in runs:
        key=tuple(run[k] for k in ('conditioning_batch','decoder_batch','repeat'))
        assert key not in by_mode
        by_mode[key]=run['outputs']
    assert set(by_mode)==set(itertools.product((1,3),(1,3),range(repeats)))
    unstable=[]
    native_failures=[]
    for (conditioning,decoder,repeat),outputs in by_mode.items():
        assert outputs.keys()==expected[decoder].keys(), 'missing diagnostic input'
        if repeat:
            first=by_mode[conditioning,decoder,0]
            assert outputs.keys()==first.keys()
            changed={k:d for k in outputs if (d:=record_differences(first[k],outputs[k]))}
            if changed:
                unstable.append(dict(conditioning_batch=conditioning,decoder_batch=decoder,
                                     repeat=repeat,differences=changed))
        if conditioning==decoder:
            assert outputs.keys()==expected[decoder].keys()
            changed={k:d for k in outputs if (d:=record_differences(expected[decoder][k],outputs[k]))}
            if changed:
                native_failures.append(dict(batch=decoder,repeat=repeat,differences=changed))
    return dict(complete=True,runs=len(runs),unstable=unstable,native_failures=native_failures,
        scope='TensorRT only; bound/search kernels ON. Conditioning includes frontend, encoder and '
              'cross projections. Decoder batch includes prefill and beam decode. Mixed conditioning/'
              'decode shapes are diagnostic N interventions, not a production optimization. '
              'Native1/1 and3/3 must match the previously recorded public API outputs.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--case-artifact',type=Path,required=True)
    ap.add_argument('--reproduction-artifact',type=Path,required=True)
    ap.add_argument('--key',required=True)
    ap.add_argument('--repeat',type=int,default=3)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--publish',type=Path,required=True)
    args=ap.parse_args()
    if not os.environ.get('PBS_JOBID') or not os.environ.get('PBS_LOCALDIR'):
        ap.error('model/audio work requires a compute allocation')
    if args.repeat<2:
        ap.error('at least two repeats required')
    cases=[c for c in json.loads(args.case_artifact.read_text())['cases'] if c['key']==args.key]
    assert len(cases)==1
    rows=select_group(json.loads((args.data/'index.json').read_text()),cases[0])
    previous=json.loads(args.reproduction_artifact.read_text())
    assert previous['target']==args.key and previous['summary']['complete']
    expected={r['batch']:r['outputs'] for r in previous['canonical_runs'] if r['route']=='tensorrt'}
    assert set(expected)=={1,3}
    args.out.mkdir(parents=True,exist_ok=True)
    args.publish.mkdir(parents=True,exist_ok=True)

    import torch
    from eval_wer import CER_LANGS, characters, edit_distance, normalize
    from profile_beam import command, dispatch, result_record

    from owsm_trt import OWSM, hub
    from owsm_trt.build import engine_tag
    from owsm_trt.runtime import BeamSession, TRTBackend

    waves=load_group_waves(rows,args.data)
    # Preserve the public transcribe ordering; otherwise row permutations become another variable.
    order=sorted(range(len(waves)),key=lambda i:len(waves[i]))
    rows,waves=[rows[i] for i in order],[waves[i] for i in order]
    files=hub.resolve(args.model)
    with torch.inference_mode():
        model=OWSM.from_pretrained(args.model,decoder_kernel='0')
        assert isinstance(model.backend,TRTBackend)
        assert all(r['seconds']<=model.cfg.speech_seconds for r in rows)
        engines=hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
        environment=dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),job=os.environ['PBS_JOBID'],
            checkpoint_sha256=sha256(files.checkpoint),index_sha256=sha256(args.data/'index.json'),
            case_sha256=sha256(args.case_artifact),reproduction_sha256=sha256(args.reproduction_artifact),
            engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            nvidia_smi=command(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv']))
        for f in ('checkpoint_sha256','index_sha256','engine_sha256','torch','cuda','gpu'):
            assert environment[f]==previous['environment'][f],f
        sessions={b:BeamSession(model.backend,model.cfg,b,5,model.opts.capacity,model.tokenizer.eos,
                               model.dtype,model.device,model.use_cuda_graph,rules=model.rules,
                               upper_bound=True,diagnostics=True) for b in (1,3)}
        primer=model._primer(rows[0]['lang'],'asr',True,None)
        conditioning={}
        one_k,one_v=[],[]
        for wave in waves:
            sessions[1].encode(model._features([wave]))
            one_k.append(sessions[1].cross_kb.clone())
            one_v.append(sessions[1].cross_vb.clone())
        # [L, 1, H, S, D] -> [L, 3, H, S, D]; rows use the same duration order as public B3.
        conditioning[1]=(torch.cat(one_k,dim=1),torch.cat(one_v,dim=1))
        sessions[3].encode(model._features(waves))
        conditioning[3]=(sessions[3].cross_kb.clone(),sessions[3].cross_vb.clone())
        metrics={}
        for label,a,b in zip(('key','value'),conditioning[1],conditioning[3]):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            delta=(a.float()-b.float()).abs()
            metrics[label]=dict(shape=list(a.shape),equal=bool(torch.equal(a,b)),
                max_absolute=float(delta.max()),mean_absolute=float(delta.mean()),
                per_input_max=[float(delta[:,i].max()) for i in range(len(rows))])
        runs=[]
        for enc_batch,dec_batch,repeat in itertools.product((1,3),(1,3),range(args.repeat)):
            outputs={}
            diagnostics={}
            session=sessions[dec_batch]
            for begin in range(0,len(rows),dec_batch):
                subset=rows[begin:begin+dec_batch]
                # Copy conditioning only; never recompute the encoder inside this decode control.
                for target,source in zip((session.cross_kb,session.cross_vb),conditioning[enc_batch]):
                    target.copy_(source[:,begin:begin+dec_batch])
                session._expand_cross()
                ids=capture_failure(lambda:session.decode(torch.tensor([primer]*dec_batch,device=model.device),
                                                          model._max_new()),
                    dict(environment=environment,inputs=subset,conditioning_batch=enc_batch,
                         decoder_batch=dec_batch,repeat=repeat),args.out,args.publish)
                diagnostic=session.get_diagnostics()
                assert len(ids)==len(diagnostic['utterances'])==len(subset)
                for row,tokens,detail in zip(subset,ids,diagnostic['utterances']):
                    record=result_record(model,model._result(tokens,0),row,row['seconds'])
                    units=characters if row['lang'] in CER_LANGS else normalize
                    ref,hyp=units(row['text']),units(record['text'])
                    record.update(token_ids_available=True,repetition=repetition(record['tokens']),
                        quality=dict(metric='CER' if row['lang'] in CER_LANGS else 'WER',
                                     errors=edit_distance(ref,hyp),reference_units=len(ref)))
                    outputs[row['key']]=record
                    diagnostics[row['key']]={k:v for k,v in detail.items() if k!='history'}
            runs.append(dict(conditioning_batch=enc_batch,decoder_batch=dec_batch,repeat=repeat,
                             outputs=outputs,diagnostics=diagnostics,dispatch=dispatch(session)))
            result=dict(environment=environment,inputs=rows,target=args.key,conditioning_metrics=metrics,
                        runs=runs,summary=dict(complete=False,runs=len(runs)))
            path=args.out/'result.json'
            path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
            publish_file(path,args.publish/path.name)
            print(f'conditioning={enc_batch},decoder={dec_batch},repeat={repeat}: {len(runs)}/{4*args.repeat}',flush=True)
        result['summary']=summarize_stages(runs,expected,args.repeat)
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        publish_file(path,args.publish/path.name)
        print(json.dumps(result['summary']),flush=True)
        if result['summary']['unstable'] or result['summary']['native_failures']:
            raise SystemExit(3)


if __name__=='__main__':
    main()
