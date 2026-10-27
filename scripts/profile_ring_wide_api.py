"""Public wide route must reproduce the frozen experiment and retain its numerical differences."""

import argparse
import gc
import importlib.metadata
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import command, save, sha256, wall
from profile_ring_programs import count_decoder_dispatch
from profile_ring_wide import WIDE_CASES, WideBackend
from profile_runtime import run_case

from owsm_trt import OWSM, hub, megakernel, megakernel_ring
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--baseline-results', type=Path, required=True)
    ap.add_argument('--long-audio', type=Path, required=True)
    ap.add_argument('--lang', default='eng')
    ap.add_argument('--repeat', type=int, default=11)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('compute allocation required')
    args.trace, args.variants = False, ['reference', 'wide']
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, decoder_kernel='ring:wide'))
    public = model.backend
    assert isinstance(public, megakernel.WideRingBackend), 'public route did not activate'
    base = public.base
    files = hub.resolve(args.model)
    engines = hub.cache_dir()/files.name.replace('/', '--')/engine_tag(model.opts)
    fingerprint = dict(checkpoint_sha256=sha256(files.checkpoint),
                       engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')})
    previous = json.loads((args.baseline_results/'environment.json').read_text())
    assert all(previous[k]==v for k,v in fingerprint.items())
    routes = {r:megakernel.MegaBackend(base.inner, megakernel_ring.RingDecoder(base.kernel.m,row_programs=r*6),
              max_tokens=r,max_rows=r,trial=True) for r in (10,15,16)}
    experimental = WideBackend(base,routes)
    items = json.loads(args.manifest.read_text())
    waves = [load_audio(item['path']) for item in items]
    save(args.out/'environment.json', dict(**fingerprint,commit=os.environ.get('OWSM_PROFILE_COMMIT'),
         gpu=torch.cuda.get_device_name(),sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,
         versions={n:importlib.metadata.version(n) for n in ('torch','triton','tensorrt')},cuda=torch.version.cuda,
         nvidia_smi=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,compute_cap,memory.total','--format=csv']),
         load_ms=load_ms,manifest_sha256=sha256(args.manifest),
         scope='Publicring:wide vs auto, exact match required to frozen experimental N outputs for eachmode. '
               'Same engines/weights;11alternatingwarm repetitions. Both pools resident, not isolated deploymentmemory. '
               'No reference translation/time annotations; previous Ndifferences must remain visible.'))

    def configure(m, mode):
        m.backend = base if mode=='reference' else public

    summary = {}
    for case in ('beam_ts',*WIDE_CASES):
        summary[case] = run_case(model,args,case,items,waves,configure=configure,observe=count_decoder_dispatch)
        current = json.loads((args.out/case/'tokens.json').read_text())
        frozen = json.loads((args.baseline_results/case/'tokens.json').read_text())
        assert current==frozen, f'public route differs from frozen experiment: {case}'
        save(args.out/'summary.json',summary)
        assert not public.failed

    rng = torch.Generator().manual_seed(0)
    silence = torch.zeros(model.cfg.sample_rate*2)
    noise = torch.randn(model.cfg.sample_rate*2,generator=rng)*.01
    full = waves[0].repeat(-(-model.cfg.num_samples//len(waves[0])))[:model.cfg.num_samples]
    cases = [
        ('beam_b2',waves[:5],dict(beam_size=5,batch_size=2,timestamps=True),False,False),
        ('beam_b3',waves[:7],dict(beam_size=5,batch_size=3,timestamps=True),False,False),
        ('plain_b10',waves[:10],dict(batch_size=10,timestamps=True),False,False),
        ('translation_b10',waves,dict(batch_size=10,task='st_eng'),False,False),
        ('translation_b16',waves,dict(batch_size=16,task='st_eng'),False,False),
        ('draft_b10',waves,dict(batch_size=10,timestamps=True),True,False),
        ('no_timestamps',waves[:3],dict(beam_size=5,batch_size=3),False,False),
        ('long_prompt',waves[:3],dict(beam_size=5,batch_size=3,prompt='Previous speech context. '*20,timestamps=True),False,False),
        ('silence_noise',[silence,noise,waves[0]],dict(beam_size=5,batch_size=3,timestamps=True),False,False),
        ('full_window',[full,waves[1],waves[2]],dict(beam_size=5,batch_size=3,timestamps=True),False,False),
        ('lookahead',waves[:7],dict(beam_size=5,batch_size=3,timestamps=True),False,True),
        ('long_audio',[load_audio(args.long_audio)],dict(beam_size=5,batch_size=1),False,False),
        ('beam_again',waves[:7],dict(beam_size=5,batch_size=3,timestamps=True),False,False),
    ]
    checks = {}
    for name,audio,options,drafted,ahead in cases:
        checks[name] = {}
        for mode,backend in (('experimental',experimental),('public',public)):
            model._sessions.clear()
            gc.collect()
            model.backend,model.ctc_draft,model._encoder_lookahead = backend,drafted,ahead
            fn = model.transcribe_long if name=='long_audio' else model.transcribe
            checks[name][mode] = [asdict(x) for x in fn(audio,lang=args.lang,**options)]
            assert not backend.failed
        save(args.out/'api_checks.json',checks)
        assert checks[name]['experimental']==checks[name]['public'], name
        if path:=os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(path).open('a') as f:
                f.write(f'{args.lang} public wide {name} exact\n')
    save(args.out/'complete.json',dict(frozen_corpus_modes_exact=True,api_cases=len(checks),all_api_equal=True,
         previous_environment_sha256=sha256(args.baseline_results/'environment.json')))


if __name__=='__main__':
    main()
