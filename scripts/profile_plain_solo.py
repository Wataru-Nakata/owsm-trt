"""Use the existing Ring solo budget for plain calls without concurrent encoding (profiler only)."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from profile_beam import command, save, sha256, wall
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import CASES, run_case

from owsm_trt import OWSM, hub, megakernel, megakernel_ring
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag

PLAIN_CASES = {
    'plain_b1': (1,1,False,False,False,'asr',None,None),
    'plain_ts_b1': (1,1,True,False,False,'asr',None,None),
    'translation_b1': (1,1,False,False,False,'st_eng',None,None),
}
CASES.update(PLAIN_CASES)


class PlainSolo:
    """Only plain R1/S1 construction changes; drafted verification retains its existing policy."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def step_fn(self, tokens, *args, kind='decode', **kwargs):
        if kind == 'decode' and tuple(tokens.shape) == (1,1) and self.base.solo_differs:
            kwargs['solo'] = True
        return self.base.step_fn(tokens, *args, kind=kind, **kwargs)


@torch.inference_mode()
def check_solo(kernel, frames, out):
    m = kernel.m
    torch.manual_seed(23)
    records = []
    for capacity, offset in ((64,3),(64,63),(128,127),(256,255)):
        shape = (m.L,1,m.H,capacity,m.D//m.H)
        prefix = [torch.randn(shape,device=m.device,dtype=torch.float16)*.1 for _ in range(2)]
        cross = [torch.randn(m.L,1,m.H,frames,m.D//m.H,device=m.device,dtype=torch.float16)*.1 for _ in range(2)]
        token = torch.randint(m.V,(1,1),device=m.device)
        pos = torch.tensor([offset],device=m.device)
        expected, ns = None, None
        for solo in (False,True):
            k,v = (x.clone() for x in prefix)
            logits = torch.zeros(1,1,m.V,device=m.device)
            kv = torch.zeros(m.L,2,1,m.H,1,m.D//m.H,device=m.device,dtype=torch.float16)
            run = kernel.step_fn(token,pos,k,v,*cross,logits,kv,solo=solo)
            run()
            torch.cuda.synchronize()
            kernel.check()
            values = (logits,k,v,kv)
            assert all(bool(torch.isfinite(x).all()) for x in values)
            if expected is None:
                expected,ns = values,run.const['NS']
            else:
                assert run.const['NS']==ns
                for a,b in zip(values,expected):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
            records.append(dict(capacity=capacity,offset=offset,solo=solo,programs=run.programs,cross_splits=ns,exact=True))
            save(out/'shape_checks.json',records)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--lang',default='eng')
    ap.add_argument('--repeat',type=int,default=11)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--public-policy',action='store_true',help='compare actual public plain_solo to its rollback')
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('model/audio work requires a compute allocation')
    if args.repeat < 1:
        ap.error('repeat must be positive')
    args.trace,args.long_audio,args.variants = False,None,['reference','solo']
    model,load_ms = wall(lambda: OWSM.from_pretrained(args.model,ctc_draft=False,encoder_lookahead=False))
    base = model.backend
    if not isinstance(base,megakernel.MegaBackend) or not isinstance(base.kernel,megakernel_ring.RingDecoder) or not base.solo_differs:
        raise RuntimeError('requires the actual Ring single-row route with distinct solo programs')
    backends = {'reference':base,'solo':base if args.public_policy else PlainSolo(base)}
    files = hub.resolve(args.model)
    engines = hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
    save(args.out/'environment.json',dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),
        checkpoint_sha256=sha256(files.checkpoint),engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},
        gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','tensorrt')},
        nvidia_smi=command(['nvidia-smi','--query-gpu=name,driver_version','--format=csv']),
        public_policy=args.public_policy,load_ms=load_ms,manifest_sha256=sha256(args.manifest),programs=dict(reference=base.kernel.P,solo=base.kernel.P_solo),
        scope='E intent, same existing Ring NS/arithmetic, fixed serial requests, no concurrent encoder. '
              'Separate real-checkpoint synthetic-state bitwise checks; existing scalar EOS guard on both modes. '
              'No runtime default change. Mixed-duration per-request samples; cold/cache order and shared resident pools are separate.'))
    check_solo(base.kernel,model.cfg.enc_frames,args.out)
    items = json.loads(args.manifest.read_text())
    waves,disk_ms = wall(lambda:[load_audio(it['path']) for it in items])
    save(args.out/'disk_load.json',dict(aggregate_ms=disk_ms,included_in_wall=False))
    summary = {}
    def configure(m,mode):
        m.backend = backends[mode]
        m.greedy_plain_solo = args.public_policy and mode == 'solo'
        assert not m._encoder_lookahead
    def observe(m,call):
        result = count_decoder_dispatch(m,call)
        enabled = isinstance(m.backend,PlainSolo) or m.greedy_plain_solo
        expected = base.kernel.P_solo if enabled else base.kernel.P
        budgets = {}
        for key,session in m._sessions.items():
            actual = {c:dict(programs=fn.programs,cross_splits=fn.const['NS']) for c,fn in session._steps.items()}
            assert actual and all(x['programs']==expected for x in actual.values())
            if args.public_policy:
                assert session.plain_solo is enabled
            budgets[str(key)] = actual
        return result | {'actual_plain_budgets':budgets}
    for name in PLAIN_CASES:
        summary[name] = run_case(model,args,name,items,waves,configure=configure,observe=observe)
        save(args.out/'summary.json',summary)
        assert not base.failed
        assert all(v['equal']==v['total'] for mode in summary[name].values() for v in mode['agreement'].values()), name
        if path := os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(path).open('a') as stream:
                stream.write(f'{args.out.name}: {name} plain solo comparison complete\n')


if __name__ == '__main__':
    main()
