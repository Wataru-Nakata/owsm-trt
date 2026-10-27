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
def cache_memory(model,waves,out,batch=8,reuse_comparison=False):
    """One variant's frontend cache at a time; model and expected features are fixed residents."""
    order = sorted(waves,key=len)
    groups = [order[i:i+batch] for i in range(0,len(order),batch)]
    model._frontend_replay = None
    model.frontend_graph = False
    expected = [model._features(group) for group in groups]
    torch.cuda.synchronize()
    results = {}
    for mode,limit in (('reference',batch if reuse_comparison else 1),('graph',batch if reuse_comparison else 8)):
        model._frontend_replay = None
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        model.frontend_graph = True
        model._frontend_graph_batch_limit = limit
        model._frontend_reuse_warmup_stream = reuse_comparison and mode == 'graph'
        cold_samples,peak = [],baseline
        for group,reference in zip(groups,expected):
            torch.cuda.reset_peak_memory_stats()
            features,ms = wall(lambda group=group:model._features(group))
            cold_samples.append(ms)
            peak = max(peak,torch.cuda.max_memory_allocated())
            torch.testing.assert_close(features,reference,rtol=0,atol=0)
            del features
        results[mode] = dict(batch=batch,batches=len(groups),features_exact=True,
            cold_pass_ms=sum(cold_samples),cold_batch_ms=cold_samples,
            retained_allocated_delta_bytes=torch.cuda.memory_allocated()-baseline,
            reserved_delta_bytes=torch.cuda.memory_reserved()-baseline_reserved,
            peak_allocated_delta_bytes=peak-baseline,
            cache=model._frontend_replay.info() if model._frontend_replay else None)
        model._frontend_replay = None
        gc.collect()
        torch.cuda.empty_cache()
        results[mode]['post_release_allocated_delta_bytes'] = torch.cuda.memory_allocated()-baseline
    model.frontend_graph = True
    model._frontend_graph_batch_limit = 1
    model._frontend_reuse_warmup_stream = False
    save(out/'isolated_cache_memory.json',dict(variants=results,reuse_comparison=reuse_comparison,
        scope='Frontend cache only, one variant resident at a time; shared model and exact eager features '
              'remain fixed. PyTorch allocated/reserved memory, not total device process memory. '
              'Cold sums synchronized feature/H2D calls, excludes equality checks; not request latency. '
              f'Batches match sorted B{batch} encoder supply; decoder/session memory excluded.'))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--lang',default='eng')
    ap.add_argument('--repeat',type=int,default=11)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cases',nargs='+',choices=[*BATCH_CASES,'single_ts'],default=list(BATCH_CASES))
    ap.add_argument('--cache-memory',action='store_true',help='measure isolated B8 frontend cache allocation before request timings')
    ap.add_argument('--warmup-reuse',action='store_true',help='same graph batch limit8 on both modes; only warmup stream reuse differs')
    ap.add_argument('--memory-only',type=int,choices=[1,8],help='fresh-process memory comparison at one batch size; no request timing')
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
        cases=args.cases,isolated_cache_memory=args.cache_memory,
        warmup_reuse=args.warmup_reuse,memory_only=args.memory_only,
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','tensorrt')},load_ms=load_ms,
        nvidia_smi=command(['nvidia-smi','--query-gpu=name,driver_version','--format=csv']),
        classification=('E intent; only frontend warmup stream reuse, same graph arithmetic/batch shape and serialized replay. '
                       if args.warmup_reuse else 'E intent; same frontend operations/batch shape, graph replay versus eager for B2..8. ') +
                       'B1 graph remains on both modes.32-entry cap per variant. No runtime default expansion. '
                       'Both variants resident: peak memory is not standalone deployment footprint.'))
    items = json.loads(args.manifest.read_text())
    if len(items)<8:
        ap.error('at least8 inputs required')
    waves,disk_ms = wall(lambda:[load_audio(it['path']) for it in items])
    save(args.out/'disk_load.json',dict(aggregate_ms=disk_ms,included_in_wall=False))
    if args.memory_only:
        cache_memory(model,waves,args.out,batch=args.memory_only,reuse_comparison=args.warmup_reuse)
        return
    feature_check(model,waves,args.out)
    if args.cache_memory:
        cache_memory(model,waves,args.out,reuse_comparison=args.warmup_reuse)
    pools = {mode:FrontendReplay(model.frontend,max_batch=8 if args.warmup_reuse or mode=='graph' else 1,
                                reuse_warmup_stream=args.warmup_reuse and mode=='graph')
             for mode in args.variants}
    summary,caches = {},{}
    def configure(m,mode):
        m.frontend_graph = True
        m._frontend_graph_batch_limit = 8 if args.warmup_reuse or mode=='graph' else 1
        m._frontend_reuse_warmup_stream = args.warmup_reuse and mode=='graph'
        m._frontend_replay = pools[mode]
    for name in args.cases:
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
