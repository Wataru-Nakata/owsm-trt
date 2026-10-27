"""Public Ring policy versus explicit24 rollback, including edge inputs and session reuse."""

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
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--long-audio', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('audio/model work requires a compute allocation')
    model = OWSM.from_pretrained(args.model, decoder_kernel='ring:rr')
    public = model.backend
    assert public.kernel.row_head_parallel
    rollback = megakernel.MegaBackend(public.inner, megakernel_ring.RingDecoder(public.kernel.m, row_programs=24),
                                     public.max_tokens, public.max_rows, trial=True)
    items = json.loads(args.manifest.read_text())[:8]
    waves = [load_audio(it['path']) for it in items]
    generator = torch.Generator().manual_seed(18)
    synthetic = [torch.zeros(model.cfg.sample_rate*2), torch.randn(model.cfg.sample_rate*2,generator=generator)*.01]
    full = waves[0].repeat((model.cfg.num_samples+len(waves[0])-1)//len(waves[0]))[:model.cfg.num_samples]
    cases = [
        ('plain_b3',waves,dict(batch_size=3),False,False),
        ('plain_b8',waves,dict(batch_size=8,timestamps=True),False,False),
        ('beam_ts',waves[:3],dict(batch_size=1,beam_size=5,timestamps=True),False,False),
        ('draft_b8',waves,dict(batch_size=8,timestamps=True),True,False),
        ('translation',waves,dict(batch_size=8,task='st_eng'),False,False),
        ('silence_noise',synthetic+waves[:3],dict(batch_size=5,timestamps=True),False,False),
        ('full_window',[full]+waves[:4],dict(batch_size=5,timestamps=True),False,False),
        ('long_prompt',waves[:5],dict(batch_size=5,prompt='Previous speech context. '*20,timestamps=True),False,False),
        ('lookahead',waves,dict(batch_size=5,task='st_eng'),False,True),
        ('long_audio',[load_audio(args.long_audio)],dict(batch_size=1,beam_size=5),False,False),
        ('plain_again',waves,dict(batch_size=8,timestamps=True),False,False),
    ]
    records = {}
    for name,audio,options,drafted,ahead in cases:
        records[name] = {}
        for mode,backend in (('reference',rollback),('public',public)):
            model._sessions.clear()
            gc.collect()
            model.backend, model.ctc_draft, model._encoder_lookahead = backend,drafted,ahead
            fn = model.transcribe_long if name=='long_audio' else model.transcribe
            results = fn(audio,lang='eng',**options)
            programs = {}
            for key, session in model._sessions.items():
                calls = list(session._steps.values())
                calls += [b.step for b in getattr(session,'_verify',{}).values()]
                calls += [b.step for b in getattr(session,'_verify_solo',{}).values()]
                actual = [fn.programs for fn in calls if hasattr(fn,'programs')]
                if session.rows > 1:
                    expected = session.rows*6 if mode=='public' and 5<=session.rows<=8 else 24
                    assert all(n==expected for n in actual), (name,mode,key,actual)
                programs[str(key)] = actual
            records[name][mode] = dict(outputs=[asdict(x) for x in results], programs=programs,
                                      dispatch={str(k):dispatch(s) for k,s in model._sessions.items()})
            assert not backend.failed
        save(args.out/'checks.json',records)
        assert records[name]['reference']['outputs']==records[name]['public']['outputs'],name
        if path := os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(path).open('a') as stream:
                stream.write(f'public Ring policy: {name} exact\n')
    files = hub.resolve(args.model)
    engine_dir = hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
    save(args.out/'complete.json',dict(all_equal=True,cases=len(cases),commit=os.environ.get('OWSM_PROFILE_COMMIT'),
        checkpoint_sha256=sha256(files.checkpoint), engine_sha256={p.name:sha256(p) for p in engine_dir.glob('*.engine')},
        note='Same explicit Ring route, public policy versus constructor row_programs=24 rollback. '
             'Long-form checks joined text/segments; raw window token IDs unavailable. No reference boundary/translation quality.'))


if __name__ == '__main__':
    main()
