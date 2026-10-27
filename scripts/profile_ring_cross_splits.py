"""N experiment: change single-row attention splits with CTA budgets and row routing fixed."""

import argparse
import gc
import importlib.metadata
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from diagnose_ring_rows import compare
from profile_beam import command, save, sha256, wall
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import run_case

from owsm_trt import OWSM, hub, megakernel, megakernel_ring
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.pipeline import _load_decoder


@torch.inference_mode()
def numerics(model,kernels,waves,args):
    files = hub.resolve(args.model)
    decoder = _load_decoder(model.cfg,files.checkpoint,skip_init=True)
    m,dev = kernels['reference'].m,model.device
    records = {}

    def check(name,tokens,offset,initial,cross):
        cpu = [x.cpu() for x in (tokens,offset,*initial,*cross)]
        ref,refkv = decoder.step(cpu[0],cpu[1],*[list(x.float()) for x in cpu[2:]])
        window = tokens.shape[1]
        expected = None
        records[name] = {}
        for mode,kernel in kernels.items():
            paired = []
            records[name][mode] = {}
            for solo in (False,True):
                k,v = (x.clone() for x in initial)
                logits = torch.zeros(1,window,m.V,device=dev)
                kv = torch.zeros(m.L,2,1,m.H,window,64,device=dev,dtype=torch.float16)
                pred = torch.zeros(1,window,device=dev,dtype=torch.long)
                run = kernel.step_fn(tokens,offset,k,v,*cross,logits,kv,pred=pred,solo=solo)
                run()
                torch.cuda.synchronize()
                kernel.check()
                saved = [x.clone() for x in (logits,kv,k,v,pred)]
                k.copy_(initial[0])
                v.copy_(initial[1])
                run()
                assert all(torch.equal(a,b) for a,b in zip(saved,(logits,kv,k,v,pred)))
                paired.append(saved)
                row = compare(logits,ref)
                row.update(programs=run.programs,cross_splits=run.const['NS'],repeat_exact=True,
                    all_state_finite=all(bool(torch.isfinite(x).all()) for x in saved),
                    kv_layer_max_abs=(kv.float().cpu()-refkv).abs().flatten(1).amax(1).tolist(),
                    versus_existing=compare(logits,expected) if expected is not None else None)
                records[name][mode]['solo' if solo else 'concurrent'] = row
                save(args.out/'numerics.json',records)
                assert row['finite'] and row['all_state_finite'] and row['existing_check_pass'], (name,mode,solo,row)
            for a,b in zip(*paired):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            if expected is None:
                expected = paired[0][0].cpu()
        if progress:=os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(progress).open('a') as f:
                f.write(f'{args.out.name} {name} split numerics complete\n')

    for time in (64,model.cfg.enc_frames):
        for window in (1,8,32):
            g = torch.Generator().manual_seed(0)
            tokens = torch.randint(m.V,(1,window),generator=g).to(dev)
            offset = torch.full((1,),5,device=dev,dtype=torch.long)
            values = [torch.randn(m.L,1,m.H,n,64,generator=g).half().to(dev) for n in (64,64,time,time)]
            check(f'random_t{time}_w{window}',tokens,offset,values[:2],values[2:])
    session = model._session(1)
    session.encode(model._features(waves[:1]))
    primer = torch.tensor([model._primer(args.lang,'asr',False,None)],device=dev)
    logits,kv,c = session._prefill(primer,session.cross_k,session.cross_v)
    p = primer.shape[1]
    k,v = session.caches[c]
    k[:,:,:,:p].copy_(kv[:,0])
    v[:,:,:,:p].copy_(kv[:,1])
    session._rules_start(primer)
    tokens = session._rules_mask(logits).argmax(-1).unsqueeze(1)
    check('real_prefill',tokens,torch.full((1,),p,device=dev,dtype=torch.long),[k,v],[session.cross_k,session.cross_v])
    model._sessions.clear()
    return records


@torch.inference_mode()
def validate_public(model,public,base,waves,args):
    """Same N prototype/public outputs, including beam controls and encoder concurrency."""
    prototype = megakernel.MegaBackend(base.inner,
        megakernel_ring.RingDecoder(base.kernel.m,single_cross_splits=public.kernel.single_cross_splits),
        base.max_tokens,base.max_rows,rows_kernel=base.rows_kernel,trial=True)
    rng = torch.Generator().manual_seed(18)
    synthetic = [torch.zeros(model.cfg.sample_rate*2),torch.randn(model.cfg.sample_rate*2,generator=rng)*.01]
    full = waves[0].repeat(-(-model.cfg.num_samples//len(waves[0])))[:model.cfg.num_samples]
    cases = [
        ('plain',waves[:3],dict(batch_size=1),False,False),
        ('plain_ts',waves[:3],dict(batch_size=1,timestamps=True),False,False),
        ('drafted',waves[:3],dict(batch_size=1,timestamps=True),True,False),
        ('drafted_notime',waves[:3],dict(batch_size=1),True,False),
        ('drafted_b8',waves[:8],dict(batch_size=8,timestamps=True),True,False),
        ('translation',waves[:3],dict(batch_size=1,task='st_eng'),False,False),
        ('beam_b1',waves[:3],dict(batch_size=1,beam_size=5,timestamps=True),False,False),
        ('beam_b3',waves[:3],dict(batch_size=3,beam_size=5,timestamps=True),False,False),
        ('silence_noise_plain',synthetic+waves[:1],dict(batch_size=1,timestamps=True),False,False),
        ('silence_noise_drafted',synthetic+waves[:1],dict(batch_size=1,timestamps=True),True,False),
        ('full_window',[full],dict(batch_size=1,timestamps=True),True,False),
        ('long_prompt',waves[:3],dict(batch_size=1,prompt='Previous speech context. '*20,timestamps=True),False,False),
        ('lookahead',waves[:8],dict(batch_size=1,task='st_eng'),False,True),
        ('plain_again',waves[:3],dict(batch_size=1,timestamps=True),False,False),
    ]
    if args.long_audio:
        cases.insert(-1,('long_audio',[load_audio(args.long_audio)],dict(batch_size=1),True,False))
    records = {}
    for name,audio,options,drafted,ahead in cases:
        records[name] = {}
        for mode,backend in (('reference',base),('prototype',prototype),('public',public)):
            model._sessions.clear()
            gc.collect()
            model.backend,model.ctc_draft,model._encoder_lookahead = backend,drafted,ahead
            fn = model.transcribe_long if name=='long_audio' else model.transcribe
            warm = [asdict(x) for x in fn(audio,lang=args.lang,**options)]
            outputs = []
            def call():
                decoded = fn(audio,lang=args.lang,**options)
                outputs.extend(asdict(x) for x in decoded)
                return decoded
            result = count_decoder_dispatch(model,call)
            assert outputs==warm, f'nondeterministic API output: {name}/{mode}'
            configurations = {}
            for key,session in model._sessions.items():
                calls = list(session._steps.values())
                for label in ('_verify','_verify_solo'):
                    calls += [b.step for b in getattr(session,label,{}).values() if b.step is not None]
                configurations[str(key)] = [dict(programs=f.programs,ns=f.const['NS']) for f in calls if hasattr(f,'programs')]
                if session.rows==1:
                    expected = 2 if mode=='reference' else public.kernel.single_cross_splits
                    assert all(f.const['NS']==expected for f in calls if hasattr(f,'const'))
                    if ahead:
                        assert not session.plain_solo
                        assert all(getattr(f,'programs',base.kernel.P)==base.kernel.P for f in session._steps.values())
            records[name][mode] = dict(outputs=outputs,observation=result,configurations=configurations)
            assert not backend.failed
        save(args.out/'api_checks.json',records)
        assert records[name]['prototype']==records[name]['public'], f'public/prototype API drift: {name}'
        if progress:=os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(progress).open('a') as f:
                f.write(f'{args.out.name} public split {name} exact to prototype\n')
    return len(cases)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--long-audio',type=Path)
    ap.add_argument('--lang',default='eng')
    ap.add_argument('--repeat',type=int,default=11)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--public',action='store_true')
    ap.add_argument('--baseline-results',type=Path)
    ap.add_argument('--rotated-order',action='store_true')
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('compute allocation required')
    if args.repeat<1:
        ap.error('positive repeat required')
    args.trace,args.variants = False,['reference','ns4','ns8']
    model,load_ms = wall(lambda:OWSM.from_pretrained(args.model,decoder_kernel='ring:split' if args.public else 'auto'))
    base = model.backend
    assert isinstance(base,megakernel.MegaBackend) and isinstance(base.kernel,megakernel_ring.RingDecoder)
    m = base.kernel.m
    if ((m.D,m.F,m.L,m.H) not in ((1024,4096,18,16),(384,1536,6,6))
            or 'H200' not in torch.cuda.get_device_name() or torch.cuda.get_device_properties(model.device).multi_processor_count!=132):
        raise RuntimeError('only H200132SM/1B/102M in scope')
    public = base if args.public else None
    if public:
        selected = 4 if m.D==1024 else 8
        assert public.kernel.single_cross_splits==selected, 'public split option did not activate'
        base = megakernel.MegaBackend(public.inner,megakernel_ring.RingDecoder(m),public.max_tokens,
                                     public.max_rows,rows_kernel=public.rows_kernel,trial=True)
        kernels = {'reference':base.kernel,f'ns{selected}':public.kernel}
        args.variants = list(kernels)
    else:
        kernels = {'reference':base.kernel}|{f'ns{n}':megakernel_ring.RingDecoder(m,single_cross_splits=n) for n in (4,8)}
    if args.rotated_order:
        args.variants = args.variants[-1:]+args.variants[:-1]
    assert all((k.P,k.P_solo,k.P_rows)==(base.kernel.P,64,base.kernel.P_rows) for k in kernels.values())
    backends = {'reference':base}|{name:megakernel.MegaBackend(base.inner,k,base.max_tokens,base.max_rows,
                                  rows_kernel=base.rows_kernel,trial=True) for name,k in kernels.items() if name!='reference'}
    if public:
        backends[f'ns{selected}'] = public
    files = hub.resolve(args.model)
    engines = hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
    if args.baseline_results:
        frozen = json.loads((args.baseline_results/'environment.json').read_text())
        assert frozen['checkpoint_sha256']==sha256(files.checkpoint)
        assert frozen['engine_sha256']=={p.name:sha256(p) for p in engines.glob('*.engine')}
    save(args.out/'environment.json',dict(public_api=args.public,variant_order=args.variants,commit=os.environ.get('OWSM_PROFILE_COMMIT'),
        checkpoint_sha256=sha256(files.checkpoint),engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},
        gpu=torch.cuda.get_device_name(),sm_count=132,cuda=torch.version.cuda,
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','tensorrt')},load_ms=load_ms,
        nvidia_smi=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,compute_cap,memory.total','--format=csv']),
        budgets={name:dict(concurrent=k.P,solo=k.P_solo,rows=k.P_rows,single_cross_splits=k.single_cross_splits) for name,k in kernels.items()},
        scope='N experiment NS2/4/8 for everyR1 call, same within a mode for concurrent/solo: no timing-dependent arithmetic. '
              f'PhysicalCTAcounts/rowroute/precision unchanged. {len(kernels)}poolsresident during timing, sharedcoldJIT; warm loaded-input wall. '
              'Retain all samples/outputdifferences and WER/CER; no human timestamp/translation references.'))
    items = json.loads(args.manifest.read_text())
    waves = [load_audio(it['path']) for it in items]
    numerics(model,kernels,waves,args)
    def configure(m,mode):
        m.backend = backends[mode]
    def observe(m,call):
        result = count_decoder_dispatch(m,call)
        result['prepared_plain_steps'] = {
            str(key):{str(c):dict(programs=getattr(fn,'programs',None),
                cross_splits=getattr(fn,'const',{}).get('NS'),dispatch=getattr(fn,'dispatch',None),
                fallback=getattr(fn,'fallback_reason',None)) for c,fn in session._steps.items()}
            for key,session in m._sessions.items()}
        return result
    summary = {}
    cases = ['plain_greedy','plain_ts','translation','single_greedy','single_ts','stream_greedy_b1','stream_ts_b8']
    if args.long_audio:
        cases.append('long_greedy')
    for case in cases:
        summary[case] = run_case(model,args,case,items,waves,configure=configure,observe=observe)
        if args.baseline_results:
            frozen = json.loads((args.baseline_results/case/'tokens.json').read_text())
            current = json.loads((args.out/case/'tokens.json').read_text())
            assert all(current[v]==frozen[v] for v in args.variants), f'frozen N output drift: {case}'
        save(args.out/'summary.json',summary)
        assert all(not b.failed for b in backends.values())
    rng = torch.Generator().manual_seed(19)
    synthetic = [torch.zeros(model.cfg.sample_rate*2),torch.randn(model.cfg.sample_rate*2,generator=rng)*.01]
    full = waves[0].repeat(-(-model.cfg.num_samples//len(waves[0])))[:model.cfg.num_samples]
    model.ctc_draft = True
    extras = {}
    for case,inputs,opts in [('silence_noise',synthetic+waves[:1],{}),('full_window',[full],{}),
                             ('long_prompt',waves[:2],dict(prompt='Previous speech context. '*20))]:
        extras[case] = {}
        for mode in args.variants:
            model._sessions.clear()
            gc.collect()
            configure(model,mode)
            extras[case][mode] = [asdict(x) for x in model.transcribe(inputs,lang=args.lang,timestamps=True,**opts)]
            assert not backends[mode].failed
        if args.baseline_results:
            frozen = json.loads((args.baseline_results/'extras.json').read_text())
            assert all(extras[case][v]==frozen[case][v] for v in args.variants), f'frozen extra drift: {case}'
        save(args.out/'extras.json',extras)
    api_checks = validate_public(model,public,base,waves,args) if public else None
    save(args.out/'complete.json',dict(cases=len(cases),extras=len(extras),numerical_gate_passed=True,
                                      public_api_cases=api_checks))


if __name__=='__main__':
    main()
