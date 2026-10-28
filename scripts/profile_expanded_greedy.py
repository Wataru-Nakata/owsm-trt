"""Repeated same-engine greedy switch timing on duration/dataset-stratified speech from the large audit."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import tarfile
from collections import defaultdict, deque
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace


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
    ap.add_argument('--comparison',choices=['greedy_kernels','draft_budget'],default='greedy_kernels')
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
        numerical=args.comparison=='draft_budget'
        cases=('stream_ts_b8',) if numerical else ('single_ts','stream_ts_b8')
        model,load_ms=wall(lambda:OWSM.from_pretrained(args.model,greedy_kernels=True,
            draft_token_budget=128 if numerical else 256))
        if not isinstance(model.backend,megakernel.MegaBackend):
            raise RuntimeError('expected auto decoder backend unavailable')
        reference=OWSM(model.cfg,model.tokenizer,model.frontend,model.backend,model.opts,
                       model.device,use_cuda_graph=model.use_cuda_graph,greedy_kernels=numerical,
                       draft_token_budget=256)
        settings={name:{k:v for k,v in vars(m).items() if k.startswith('greedy_')}
                  for name,m in [('reference',reference),('candidate',model)]}
        budgets=dict(reference=reference.draft_token_budget,candidate=model.draft_token_budget)
        if numerical:
            assert settings['reference']==settings['candidate'] and budgets==dict(reference=256,candidate=128)
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
            load_ms=load_ms,greedy_settings=settings,draft_token_budgets=budgets,
            comparison=args.comparison,classification='N' if numerical else 'E',cases=cases,
            repeat=args.repeat,per_language=args.per_language,
            scope=('N: public draft budget256/128, greedy kernels ON both. ' if numerical
                   else 'Combined public greedy switch. ')+'Same engine/checkpoint. Balanced language/duration/dataset '
                  'sample excluding prior screening overlap, nonspeech and long-form. Not corpus-weighted '
                  'performance or an independent quality holdout. Loaded audio outside measured calls; '
                  'cold/capture and diagnostic dispatch counts separate. Both session pools resident.'))

        def configure(m,variant):
            for key,value in settings[variant].items():
                setattr(m,key,value)
            m.draft_token_budget=budgets[variant]

        summaries={}
        with ExitStack() as stack:
            archives={}
            for lang,rows in selected.items():
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
                                        variants=['reference','candidate'],trace=False,long_audio=None)
                for case in cases:
                    result=run_case(model,options,case,items,waves,configure=configure,
                                    observe=count_decoder_dispatch)
                    summaries[lang+'/'+case]=result
                    save(args.out/'summary.json',summaries)
                    assert not model.backend.failed,'decoder failed; refuse fallback timing'
                    if not numerical:
                        assert all(a['equal']==a['total'] for v in result.values()
                                   for a in v['agreement'].values()),f'output differences: {lang}/{case}'
        save(args.out/'complete.json',dict(job=os.environ['PBS_JOBID'],complete=True,
             conditions=len(summaries),samples=sum(map(len,selected.values())),repeat=args.repeat))


if __name__=='__main__':
    main()
