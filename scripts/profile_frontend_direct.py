"""E experiment: fill warm frontend graph inputs directly, retaining all arithmetic and shapes."""

import argparse
import gc
import importlib.metadata
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import command, save, sha256, stats, wall
from profile_frontend_batches import BATCH_CASES
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import run_case

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.frontend_graph import FrontendReplay


@torch.inference_mode()
def features(model,waves,out):
    records = {}
    for batch in (1,3,8):
        model._frontend_graph_batch_limit = batch
        model._frontend_replay = FrontendReplay(model.frontend,max_batch=batch)
        model._frontend_direct_input = False
        expected,cold = wall(lambda:model._features(waves[:batch]))
        samples = {'reference':[],'direct':[]}
        for repeat in range(31):
            for mode in (list(samples) if repeat%2==0 else list(samples)[::-1]):
                model._frontend_direct_input = mode=='direct'
                actual,ms = wall(lambda:model._features(waves[:batch]))
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                samples[mode].append(ms)
        pool = model._frontend_replay
        assert pool.direct_replays==31
        records[str(batch)] = dict(features_exact=True,shape=list(expected.shape),cold_ms=cold,
            samples_ms=samples,warm_ms={v:stats(x) for v,x in samples.items()},cache=pool.info(),
            scope='Feature padding/pinning/H2D/frontend wall only. Same warmed graph and shape. B3/8 graph limit is explicit here; default request paths use limit1.')
        save(out/'feature_checks.json',records)
    model._frontend_graph_batch_limit = 1
    model._frontend_replay = None
    model._frontend_direct_input = False
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def api_checks(model,waves,args):
    rng = torch.Generator().manual_seed(19)
    synthetic = [torch.zeros(model.cfg.sample_rate*2),torch.randn(model.cfg.sample_rate*2,generator=rng)*.01]
    full = waves[0].repeat(-(-model.cfg.num_samples//len(waves[0])))[:model.cfg.num_samples]
    cases = [
        ('plain',waves[:3],dict(batch_size=1,timestamps=True),False,False,1),
        ('drafted_b8',waves[:8],dict(batch_size=8,timestamps=True),True,False,1),
        ('beam_b1',waves[:3],dict(batch_size=1,beam_size=5,timestamps=True),False,False,1),
        ('beam_b3',waves[:3],dict(batch_size=3,beam_size=5,timestamps=True),False,False,8),
        ('translation_b8',waves[:8],dict(batch_size=8,task='st_eng'),False,False,8),
        ('silence_noise',synthetic+waves[:1],dict(batch_size=1,timestamps=True),True,False,1),
        ('full_window',[full],dict(batch_size=1,timestamps=True),True,False,1),
        ('long_prompt',waves[:3],dict(batch_size=1,prompt='Previous speech context. '*20,timestamps=True),False,False,1),
        ('lookahead',waves[:8],dict(batch_size=1,task='st_eng'),False,True,1),
        ('lookahead_b3',waves[:8],dict(batch_size=3,beam_size=5,timestamps=True),False,True,8),
        ('gpu_inputs',[w.to(model.device) for w in waves[:3]],dict(batch_size=1,timestamps=True),True,False,1),
        ('plain_again',waves[:3],dict(batch_size=1,timestamps=True),False,False,1),
    ]
    records = {}
    for name,audio,options,drafted,ahead,limit in cases:
        records[name] = {}
        for mode in ('reference','direct'):
            model._sessions.clear()
            model._frontend_replay = None
            gc.collect()
            model.ctc_draft,model._encoder_lookahead = drafted,ahead
            model._frontend_direct_input = mode=='direct'
            model._frontend_graph_batch_limit = limit
            warm = [asdict(x) for x in model.transcribe(audio,lang=args.lang,**options)]
            outputs = [asdict(x) for x in model.transcribe(audio,lang=args.lang,**options)]
            assert warm==outputs
            pool = model._frontend_replay
            if name!='drafted_b8':
                assert pool is not None and (pool.direct_replays>0)==(mode=='direct')
            records[name][mode] = dict(outputs=outputs,cache=pool.info() if pool else None)
            assert not model.backend.failed
        assert records[name]['reference']['outputs']==records[name]['direct']['outputs'],name
        save(args.out/'api_checks.json',records)
        if progress:=os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(progress).open('a') as f:
                f.write(f'{args.out.name} frontend direct {name} exact\n')
    return len(cases)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--lang',default='eng')
    ap.add_argument('--long-audio',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--repeat',type=int,default=11)
    ap.add_argument('--rotated-order',action='store_true')
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('compute allocation required')
    if args.repeat<1:
        ap.error('positive repeat required')
    args.trace,args.variants = False,['direct','reference'] if args.rotated_order else ['reference','direct']
    model,load_ms = wall(lambda:OWSM.from_pretrained(args.model))
    assert not model._frontend_direct_input and model._frontend_graph_batch_limit==1
    files = hub.resolve(args.model)
    engines = hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
    save(args.out/'environment.json',dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),
        checkpoint_sha256=sha256(files.checkpoint),engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},
        gpu=torch.cuda.get_device_name(),sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,
        cuda=torch.version.cuda,versions={n:importlib.metadata.version(n) for n in ('torch','triton','tensorrt')},
        load_ms=load_ms,variant_order=args.variants,
        nvidia_smi=command(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv']),
        scope='E direct fill of an existing graph input. Cold/cache-miss/eager retain existing path. Same checkpoint/engines/precision/batch shape/decoder. Both session pools resident; no standalone memory/cold claim. Warm loaded-input wall, every output exact required.'))
    items = json.loads(args.manifest.read_text())
    waves = [load_audio(x['path']) for x in items]
    features(model,waves,args.out)
    pools = {v:FrontendReplay(model.frontend) for v in args.variants}
    def configure(m,mode):
        m._frontend_direct_input = mode=='direct'
        m._frontend_replay = pools[mode]
    cases = ['plain_greedy','plain_ts','translation','single_greedy','single_ts','stream_greedy_b1','beam_ts',*BATCH_CASES]
    if args.long_audio:
        cases.append('long_greedy')
    summary,caches = {},{}
    for name in cases:
        summary[name] = run_case(model,args,name,items,waves,configure=configure,observe=count_decoder_dispatch)
        save(args.out/'summary.json',summary)
        assert all(v['equal']==v['total'] for m in summary[name].values() for v in m['agreement'].values()),name
        assert not model.backend.failed
        caches[name] = {v:pool.info() for v,pool in pools.items()}
        save(args.out/'frontend_caches.json',caches)
    assert pools['reference'].direct_replays==0 and pools['direct'].direct_replays>0
    count = api_checks(model,waves,args)
    save(args.out/'complete.json',dict(cases=len(cases),api_cases=count,all_equal=True))


if __name__=='__main__':
    main()
