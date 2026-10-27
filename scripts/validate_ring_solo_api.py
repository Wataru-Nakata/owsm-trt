"""Public serial Ring budget versus explicit24 rollback, including unchanged concurrent/row routes."""

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import dispatch, save, sha256

from owsm_trt import OWSM, hub, megakernel, megakernel_ring
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--long-audio',type=Path,required=True)
    ap.add_argument('--lang',default='eng')
    ap.add_argument('--expected-programs',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('compute allocation required')
    model = OWSM.from_pretrained(args.model)
    public = model.backend
    assert isinstance(public,megakernel.MegaBackend)
    assert (public.kernel.P,public.kernel.P_solo)==(12,args.expected_programs)
    rollback = megakernel.MegaBackend(public.inner,megakernel_ring.RingDecoder(public.kernel.m,solo_programs=24),
        public.max_tokens,public.max_rows,rows_kernel=public.rows_kernel,trial=True)
    items = json.loads(args.manifest.read_text())[:8]
    waves = [load_audio(it['path']) for it in items]
    rng = torch.Generator().manual_seed(18)
    synthetic = [torch.zeros(model.cfg.sample_rate*2),torch.randn(model.cfg.sample_rate*2,generator=rng)*.01]
    full = waves[0].repeat(-(-model.cfg.num_samples//len(waves[0])))[:model.cfg.num_samples]
    cases = [
        ('plain',waves[:3],dict(batch_size=1),False,False),
        ('plain_ts',waves[:3],dict(batch_size=1,timestamps=True),False,False),
        ('drafted',waves[:3],dict(batch_size=1,timestamps=True),True,False),
        ('drafted_notime',waves[:3],dict(batch_size=1),True,False),
        ('drafted_b8',waves,dict(batch_size=8,timestamps=True),True,False),
        ('translation',waves[:3],dict(batch_size=1,task='st_eng'),False,False),
        ('beam_b1',waves[:3],dict(batch_size=1,beam_size=5,timestamps=True),False,False),
        ('beam_b3',waves[:3],dict(batch_size=3,beam_size=5,timestamps=True),False,False),
        ('silence_noise_plain',synthetic+waves[:1],dict(batch_size=1,timestamps=True),False,False),
        ('silence_noise_drafted',synthetic+waves[:1],dict(batch_size=1,timestamps=True),True,False),
        ('full_window',[full],dict(batch_size=1,timestamps=True),True,False),
        ('long_prompt',waves[:3],dict(batch_size=1,prompt='Previous speech context. '*20,timestamps=True),False,False),
        ('lookahead',waves,dict(batch_size=1,task='st_eng'),False,True),
        ('long_audio',[load_audio(args.long_audio)],dict(batch_size=1),True,False),
        ('plain_again',waves[:3],dict(batch_size=1,timestamps=True),False,False),
    ]
    records = {}
    for name,audio,options,drafted,ahead in cases:
        records[name] = {}
        for mode,backend in (('reference',rollback),('public',public)):
            model._sessions.clear()
            gc.collect()
            model.backend,model.ctc_draft,model._encoder_lookahead = backend,drafted,ahead
            fn = model.transcribe_long if name=='long_audio' else model.transcribe
            results = fn(audio,lang=args.lang,**options)
            configurations = {}
            for key,session in model._sessions.items():
                calls = list(session._steps.values())
                for label in ('_verify','_verify_solo'):
                    calls += [b.step for b in getattr(session,label,{}).values() if b.step is not None]
                configurations[str(key)] = [dict(programs=f.programs,ns=f.const['NS']) for f in calls if hasattr(f,'programs')]
                if session.rows==1:
                    solo = args.expected_programs if mode=='public' else 24
                    for f in calls:
                        if hasattr(f,'programs'):
                            assert f.programs in (12,solo) and f.const['NS']==2
                    if ahead:
                        assert not session.plain_solo
                        assert all(getattr(f,'programs',12)==12 for f in session._steps.values())
            records[name][mode] = dict(outputs=[asdict(x) for x in results],configurations=configurations,
                                      dispatch={str(k):dispatch(s) for k,s in model._sessions.items()})
            assert not backend.failed
        save(args.out/'checks.json',records)
        assert records[name]['reference']['outputs']==records[name]['public']['outputs'],name
        if progress:=os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(progress).open('a') as f:
                f.write(f'{args.lang} public Ring solo {name} exact\n')
    files = hub.resolve(args.model)
    engine_dir = hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
    save(args.out/'complete.json',dict(all_equal=True,cases=len(cases),commit=os.environ.get('OWSM_PROFILE_COMMIT'),
        checkpoint_sha256=sha256(files.checkpoint),engine_sha256={p.name:sha256(p) for p in engine_dir.glob('*.engine')},
        note='Same Ring arithmetic: public default vs explicit solo_programs24 rollback; auto multi-row route unchanged. '
             'Long-form joined text/segments only. No boundary/translation references. No timed performance claim.'))


if __name__=='__main__':
    main()
