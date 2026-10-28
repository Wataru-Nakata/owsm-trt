"""Repeated same-engine greedy switch timing on duration/dataset-stratified speech from the large audit."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import tarfile
from collections import defaultdict, deque
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace

COMPARISON_CASES = {
    'greedy_kernels': ('single_ts','stream_ts_b8'),
    'draft_budget': ('stream_ts_b8',),
    'frontend_batches': ('stream_ts_b8',),
    'beam_preset': ('beam_single_ts','beam_stream_ts_b3'),
}


def select_inputs(rows: list[dict], per_language: int, window_seconds: float) -> dict[str, list[dict]]:
    if per_language<1 or window_seconds<=0:
        raise ValueError('positive sample count and window required')
    if len({r['key'] for r in rows})!=len(rows):
        raise ValueError('duplicate corpus input keys')
    languages=sorted({r['lang'] for r in rows if r['dataset']!='stress' and not r['long_form']})
    selected={}
    for lang in languages:
        strata=defaultdict(list)
        for row in rows:
            if (row['lang']==lang and row['dataset']!='stress' and not row['long_form']
                    and not row['screening_overlap'] and 0<row['seconds']<=window_seconds):
                strata[row['duration_bucket'],row['dataset']].append(row)
        queues=[deque(sorted(values,key=lambda r:hashlib.sha256(r['key'].encode()).hexdigest()))
                for _,values in sorted(strata.items())]
        group=[]
        while queues and len(group)<per_language:
            for queue in queues:
                if queue and len(group)<per_language:
                    group.append(queue.popleft())
            queues=[q for q in queues if q]
        if len(group)!=per_language:
            raise ValueError(f'insufficient eligible speech for {lang}: {len(group)}/{per_language}')
        selected[lang]=group
    if not selected:
        raise ValueError('no eligible languages')
    return selected


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--per-language',type=int,default=32)
    ap.add_argument('--repeat',type=int,default=7)
    ap.add_argument('--comparison',choices=list(COMPARISON_CASES),default='greedy_kernels')
    ap.add_argument('--host-diagnostics',action='store_true',help='separate instrumented GC/CPU pass; not warm latency')
    ap.add_argument('--candidate-first',action='store_true',help='reverse the initial warmup/alternating variant order')
    args=ap.parse_args()
    if not os.environ.get('PBS_JOBID') or not os.environ.get('PBS_LOCALDIR'):
        ap.error('model/audio profiling requires a compute allocation')
    if args.repeat<3:
        ap.error('at least three alternating repeats required')
    import soundfile as sf
    import torch
    from profile_beam import command, save, sha256, wall
    from profile_ring_programs import count_decoder_dispatch
    from profile_runtime import run_case

    from owsm_trt import OWSM, hub, megakernel
    from owsm_trt.audio import load_audio
    from owsm_trt.build import engine_tag

    with torch.inference_mode():
        variants=['candidate','reference'] if args.candidate_first else ['reference','candidate']
        budget=args.comparison=='draft_budget'
        beam=args.comparison=='beam_preset'
        numerical=budget or beam
        frontend=args.comparison=='frontend_batches'
        cases=COMPARISON_CASES[args.comparison]
        beam_settings=None
        if beam:
            from audit_large_regression import beam_variants
            from profile_beam_presets import PRESET_CASES
            from profile_runtime import CASES
            CASES.update(PRESET_CASES)
            beam_settings=beam_variants('beam_preset')
        model,load_ms=wall(lambda:OWSM.from_pretrained(args.model,greedy_kernels=True,
            draft_token_budget=128 if budget else 256,frontend_graph_max_batch=8 if frontend else 1,
            **(beam_settings['candidate'] if beam else {})))
        if not isinstance(model.backend,megakernel.MegaBackend):
            raise RuntimeError('expected auto decoder backend unavailable')
        reference=OWSM(model.cfg,model.tokenizer,model.frontend,model.backend,model.opts,
                       model.device,use_cuda_graph=model.use_cuda_graph,greedy_kernels=numerical or frontend,
                       draft_token_budget=256,**(beam_settings['reference'] if beam else {}))
        settings={name:{k:v for k,v in vars(m).items() if k.startswith('greedy_')}
                  for name,m in [('reference',reference),('candidate',model)]}
        budgets=dict(reference=reference.draft_token_budget,candidate=model.draft_token_budget)
        if budget:
            assert settings['reference']==settings['candidate'] and budgets==dict(reference=256,candidate=128)
        if beam:
            gpu=torch.cuda.get_device_properties(model.device)
            assert (model.cfg.dec_layers,model.cfg.d_model,gpu.name,gpu.multi_processor_count)==(
                18,1024,'NVIDIA H200',132),'beam_preset comparison requires H200/1B'
            resolved={v:{key:getattr(m,key) for key in beam_settings[v]}
                      for v,m in [('reference',reference),('candidate',model)]}
            assert resolved==beam_settings and settings['reference']==settings['candidate']
        frontend_limits=dict(reference=reference._frontend_graph_batch_limit,candidate=model._frontend_graph_batch_limit)
        if frontend:
            assert frontend_limits==dict(reference=1,candidate=8)
            assert settings['reference']==settings['candidate'] and budgets==dict(reference=256,candidate=256)
        selected=select_inputs(json.loads((args.data/'index.json').read_text()),args.per_language,
                               model.cfg.speech_seconds)
        files=hub.resolve(args.model)
        engines=hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
        save(args.out/'selection.json',selected)
        save(args.out/'environment.json',dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),
            job=os.environ['PBS_JOBID'],checkpoint_sha256=sha256(files.checkpoint),
            index_sha256=sha256(args.data/'index.json'),selection_sha256=sha256(args.out/'selection.json'),
            engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},
            gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,torch=torch.__version__,
            versions={n:importlib.metadata.version(n) for n in ('torch','triton','tensorrt')},
            nvidia_smi=command(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv']),
            load_ms=load_ms,greedy_settings=settings,draft_token_budgets=budgets,frontend_batch_limits=frontend_limits,
            beam_settings=beam_settings,
            comparison=args.comparison,classification='N' if numerical else 'E',cases=cases,
            host_diagnostics=args.host_diagnostics,variant_order=variants,
            repeat=args.repeat,per_language=args.per_language,
            scope=('INSTRUMENTED host diagnostic; not warm latency. ' if args.host_diagnostics else
                   'Uninstrumented warm request timing. ')+('N: public draft budget256/128, greedy kernels ON both. ' if budget
                   else 'N: public grouped cross/tail/LSE beam preset, optimized default control. ' if beam
                   else 'Public frontend graph batch limit1/8. ' if frontend
                   else 'Combined public greedy switch. ')+'Same engine/checkpoint. Balanced language/duration/dataset '
                  'sample excluding prior screening overlap, nonspeech and long-form. Not corpus-weighted '
                  'performance or an independent quality holdout. Loaded audio outside measured calls; '
                  'cold/capture and diagnostic dispatch counts separate. Both session pools resident.'))

        def configure(m,variant):
            for key,value in settings[variant].items():
                setattr(m,key,value)
            m.draft_token_budget=budgets[variant]
            if beam:
                for key,value in beam_settings[variant].items():
                    setattr(m,key,value)
            if frontend:
                m._frontend_graph_batch_limit=frontend_limits[variant]
                m._frontend_replay=frontend_pools[variant]

        def observe(m,call):
            result=count_decoder_dispatch(m,call)
            if beam:
                from profile_beam import dispatch
                result['sessions']={str(key):dict(dispatch=dispatch(s),fused_lse=s.fused_lse,
                    compact_tail=s.compact_tail,chunk=s.chunk,last_request_tail_start=s._tail_started)
                    for key,s in m._sessions.items()}
                result['scope']+=' Beam decoder_calls counts post-prefill decoder replays; selection0 after prefill is separate.'
                assert all(s.fused_lse==m.beam_fused_lse for s in m._sessions.values())
                assert not getattr(m._beam_backend,'failed',set()),'grouped decoder failed'
            return result

        summaries={}
        with ExitStack() as stack:
            archives={}
            for lang,rows in selected.items():
                if frontend:
                    from owsm_trt.frontend_graph import FrontendReplay
                    # A fresh bounded cache per language prevents previous languages from using up
                    # the32 entries before this shape comparison starts; the large audit retains saturation.
                    frontend_pools={v:FrontendReplay(model.frontend,max_batch=limit)
                                    for v,limit in frontend_limits.items()}
                def read_audio():
                    waves=[]
                    for row in rows:
                        if row['shard'] not in archives:
                            archives[row['shard']]=stack.enter_context(tarfile.open(args.data/row['shard']))
                        raw=archives[row['shard']].extractfile(row['audio_member']).read()
                        assert hashlib.sha256(raw).hexdigest()==row['audio_sha256']
                        values,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
                        mono=values[:,0] if values.shape[1]==1 else values.mean(axis=1)
                        waves.append(load_audio(torch.from_numpy(mono),sr=sr))
                    return waves
                waves,audio_ms=wall(read_audio)
                items=[dict(id=r['key'],text=r['text']) for r in rows]
                save(args.out/lang/'audio_load.json',dict(milliseconds=audio_ms,included_in_timing=False))
                options=SimpleNamespace(out=args.out/lang,lang=lang,repeat=args.repeat,
                                        variants=variants,trace=False,long_audio=None)
                for case in cases:
                    if args.host_diagnostics:
                        from host_latency import HostLatency
                    with HostLatency() if args.host_diagnostics else nullcontext() as host:
                        result=run_case(model,options,case,items,waves,configure=configure,
                                        observe=observe,measure=host.measure if host else None)
                    if host:
                        samples=json.loads((args.out/lang/case/'wall.json').read_text())
                        save(args.out/lang/case/'host.json',host.report(samples))
                    summaries[lang+'/'+case]=result
                    save(args.out/'summary.json',summaries)
                    assert not model.backend.failed,'decoder failed; refuse fallback timing'
                    assert not getattr(model._beam_backend,'failed',set()),'grouped decoder failed'
                    if not numerical:
                        assert all(a['equal']==a['total'] for v in result.values()
                                   for a in v['agreement'].values()),f'output differences: {lang}/{case}'
                    if frontend:
                        save(args.out/lang/case/'frontend_cache.json',{
                            v:pool.info() | dict(shapes=[list(key[0]) for key in pool.entries])
                            for v,pool in frontend_pools.items()})
        save(args.out/'complete.json',dict(job=os.environ['PBS_JOBID'],complete=True,
             conditions=len(summaries),samples=sum(map(len,selected.values())),repeat=args.repeat))


if __name__=='__main__':
    main()
