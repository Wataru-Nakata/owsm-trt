"""Opt-in B3/B8 frontend graphs with exact features and separate cold/memory observations."""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from profile_beam import command, save, sha256, stats, wall
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import CASES, run_case

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.frontend_graph import FrontendReplay

BATCH_CASES = {
    'plain_ts_b3': (3,1,True,False,True,'asr',None,None),
    'translation_b8': (8,1,False,False,True,'st_eng',None,None),
    'beam_ts_b3': (3,5,True,False,True,'asr',None,None),
    'drafted_ts_b8': (8,1,True,True,True,'asr',None,None),
}
CASES.update(BATCH_CASES)


@torch.inference_mode()
def feature_check(model,waves,out):
    results = {}
    for batch in (3,8):
        model._frontend_graph_batch_limit = 1
        model._frontend_replay = None
        expected,_ = wall(lambda:model._features(waves[:batch]))
        baseline_bytes = torch.cuda.memory_allocated()
        model._frontend_graph_batch_limit = 8
        actual,cold_ms = wall(lambda:model._features(waves[:batch]))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        capture_bytes = torch.cuda.memory_allocated()-baseline_bytes
        replay = model._frontend_replay
        samples = {'reference':[],'graph':[]}
        for repeat in range(15):
            for mode in (list(samples) if repeat%2==0 else list(samples)[::-1]):
                model._frontend_graph_batch_limit = 1 if mode=='reference' else 8
                features,ms = wall(lambda:model._features(waves[:batch]))
                torch.testing.assert_close(features,expected,rtol=0,atol=0)
                samples[mode].append(ms)
        results[batch] = dict(features_exact=True,shape=list(expected.shape),cold_ms=cold_ms,
            capture_allocated_delta_bytes=capture_bytes,cache=replay.info(),samples_ms=samples,
            warm_ms={mode:stats(values) for mode,values in samples.items()},
            scope='Feature/H2D only, not decoder/request latency. Delta also includes one retained output and warmed library workspaces.')
        save(out/'feature_checks.json',results)
        model._frontend_replay = None
        del replay,actual,expected,features
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--lang',default='eng')
    ap.add_argument('--repeat',type=int,default=11)
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('audio/model work requires a compute allocation')
    if args.repeat < 1:
        ap.error('repeat must be positive')
    args.trace,args.long_audio,args.variants = False,None,['reference','graph']
    model,load_ms = wall(lambda:OWSM.from_pretrained(args.model))
    files = hub.resolve(args.model)
    engines = hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
    save(args.out/'environment.json',dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),
        checkpoint_sha256=sha256(files.checkpoint),engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},
        manifest_sha256=sha256(args.manifest),gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,cuda=torch.version.cuda,
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','tensorrt')},load_ms=load_ms,
        nvidia_smi=command(['nvidia-smi','--query-gpu=name,driver_version','--format=csv']),
        classification='E intent; same frontend operations/batch shape, graph replay versus eager for B2..8. '
                       'B1 graph remains on both modes.32-entry cap per variant. No runtime default expansion. '
                       'Both variants resident: peak memory is not standalone deployment footprint.'))
    items = json.loads(args.manifest.read_text())
    if len(items)<8:
        ap.error('at least8 inputs required')
    waves,disk_ms = wall(lambda:[load_audio(it['path']) for it in items])
    save(args.out/'disk_load.json',dict(aggregate_ms=disk_ms,included_in_wall=False))
    feature_check(model,waves,args.out)
    pools = {mode:FrontendReplay(model.frontend,max_batch=1 if mode=='reference' else 8) for mode in args.variants}
    summary,caches = {},{}
    def configure(m,mode):
        m.frontend_graph = True
        m._frontend_graph_batch_limit = 1 if mode=='reference' else 8
        m._frontend_replay = pools[mode]
    for name in BATCH_CASES:
        summary[name] = run_case(model,args,name,items,waves,configure=configure,observe=count_decoder_dispatch)
        save(args.out/'summary.json',summary)
        assert not getattr(model.backend,'failed',set())
        assert all(v['equal']==v['total'] for mode in summary[name].values() for v in mode['agreement'].values()),name
        caches[name] = {mode:pool.info() for mode,pool in pools.items()}
        save(args.out/'frontend_caches.json',caches)
        if path := os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(path).open('a') as stream:
                stream.write(f'{args.out.name}: {name} frontend batch comparison complete\n')


if __name__ == '__main__':
    main()
