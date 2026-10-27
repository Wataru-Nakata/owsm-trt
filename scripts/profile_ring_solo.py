"""102M serial Ring SM-budget ablation; existing attention splits and exact outputs are required."""

import argparse
import importlib.metadata
import json
import os
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
from host_latency import HostLatency
from profile_beam import command, save, sha256, wall
from profile_graph_launches import LabelledGraph, graph_labels
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import run_case

from owsm_trt import OWSM, hub, megakernel, megakernel_ring
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag


@torch.inference_mode()
def check_solo(kernels, out):
    m = kernels['reference'].m
    torch.manual_seed(91)
    checked = []
    for window in (1,8,32):
        for position in (0,63):
            shape = (m.L,1,m.H,128,64)
            initial = [torch.randn(shape,dtype=torch.float16,device=m.device)*.1 for _ in range(2)]
            cross = [torch.randn(m.L,1,m.H,374,64,dtype=torch.float16,device=m.device)*.1 for _ in range(2)]
            tok = torch.randint(m.V,(1,window),device=m.device)
            offset = torch.full((1,),position,device=m.device,dtype=torch.long)
            expected, splits = None,None
            for mode,kernel in kernels.items():
                k,v = (x.clone() for x in initial)
                logits = torch.zeros(1,window,m.V,device=m.device)
                kv = torch.zeros(m.L,2,1,m.H,window,64,dtype=torch.float16,device=m.device)
                pred = torch.zeros(1,window,dtype=torch.long,device=m.device)
                run = kernel.step_fn(tok,offset,k,v,*cross,logits,kv,pred=pred,solo=True)
                run()
                torch.cuda.synchronize()
                kernel.check()
                values = (logits,kv,k,v,pred)
                assert all(bool(torch.isfinite(x).all()) for x in values)
                if expected is None:
                    expected,splits = values,run.const['NS']
                else:
                    assert run.const['NS']==splits
                    for a,b in zip(values,expected):
                        torch.testing.assert_close(a,b,atol=0,rtol=0)
                checked.append(dict(window=window,position=position,mode=mode,programs=run.programs,
                                    cross_splits=run.const['NS'],all_finite=True,exact_to_existing_ring=True))
                save(out/'shape_checks.json',checked)
    return checked


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--long-audio',type=Path)
    ap.add_argument('--lang',default='eng')
    ap.add_argument('--repeat',type=int,default=11)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--solo-programs',nargs='+',type=int,default=[24,32,48],
                    help='serial CTA budgets, beginning with the existing24 reference')
    ap.add_argument('--host-diagnostics',action='store_true',help='separate GC/CPU diagnostic; instrumented timings')
    ap.add_argument('--rotated-order',action='store_true',help='start with the last budget, then alternate forward/reverse order')
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('compute allocation required')
    if args.repeat < 1:
        ap.error('repeat must be positive')
    if (len(args.solo_programs)<2 or args.solo_programs[0]!=24
            or len(set(args.solo_programs))!=len(args.solo_programs)
            or any(not 1<=n<=132 for n in args.solo_programs)):
        ap.error('unique budgets in1..132 must start with24 and include a comparison')
    args.trace,args.variants = False,['reference']+[f'solo{n}' for n in args.solo_programs[1:]]
    if args.rotated_order:
        args.variants = args.variants[-1:]+args.variants[:-1]
    model,load_ms = wall(lambda: OWSM.from_pretrained(args.model))
    base = model.backend
    if not isinstance(base,megakernel.MegaBackend) or not isinstance(base.kernel,megakernel_ring.RingDecoder):
        raise RuntimeError('existing auto Ring backend required')
    original,m = base.kernel,base.kernel.m
    if ((m.D,m.F,m.L,m.H,original.P,original.P_solo)!=(384,1536,6,6,12,24)
            or 'H200' not in torch.cuda.get_device_name() or torch.cuda.get_device_properties(model.device).multi_processor_count!=132):
        raise RuntimeError('only the existing H200132SM102M12/24 schedule is in scope')
    kernels = {'reference':original} | {f'solo{n}':megakernel_ring.RingDecoder(m,solo_programs=n) for n in args.solo_programs[1:]}
    backends = {'reference':base} | {name:megakernel.MegaBackend(base.inner,kernel,base.max_tokens,base.max_rows,
                 rows_kernel=base.rows_kernel,trial=True) for name,kernel in kernels.items() if name!='reference'}
    check_solo(kernels,args.out)
    files = hub.resolve(args.model)
    engines = hub.cache_dir()/files.name.replace('/', '--')/engine_tag(model.opts)
    save(args.out/'environment.json',dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),
         checkpoint_sha256=sha256(files.checkpoint),engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},
         gpu=torch.cuda.get_device_name(),sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,
         cuda=torch.version.cuda,versions={n:importlib.metadata.version(n) for n in ('torch','triton','tensorrt')},
         nvidia_smi=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,compute_cap,memory.total','--format=csv']),
         manifest_sha256=sha256(args.manifest),load_ms=load_ms,
         host_diagnostics=args.host_diagnostics,variant_order=args.variants,
         budgets={k:dict(concurrent=v.P,solo=v.P_solo,rows=v.P_rows) for k,v in kernels.items()},
         scope='E intent versus existing same Ring arithmetic: only serialR1 CTAcount changes. '
               'Concurrent12 and cross-attention splits unchanged; no productiondefaultchange. '
               'Three residentpools, cold/JITorder shared. Loadedinputs. Longformjoinedtext/segments only.'))
    items = json.loads(args.manifest.read_text())
    waves = [load_audio(item['path']) for item in items]

    def configure(m,mode):
        m.backend = backends[mode]

    def observe(m, call):
        counts = Counter()

        class CountedGraph(LabelledGraph):
            def replay(self):
                counts[self.label] += 1
                self.graph.replay()

        with graph_labels(m, wrapper=CountedGraph):
            result = count_decoder_dispatch(m, call)
        result['actual_graph_replays'] = dict(counts)
        result['plain_step_configuration'] = {
            str(key): {str(c): dict(programs=getattr(fn,'programs',None),
                                   cross_splits=getattr(fn,'const',{}).get('NS'),
                                   dispatch=getattr(fn,'dispatch',None),
                                   fallback=getattr(fn,'fallback_reason',None)) for c,fn in sess._steps.items()}
            for key,sess in m._sessions.items()}
        result['scope'] += ' Separate graph replay counts with prepared bucket configuration; no per-step GPU reads or CUPTI.'
        return result

    summary = {}
    cases = ['plain_greedy','plain_ts','translation','single_ts','stream_ts_b8']
    if args.long_audio:
        cases.append('long_greedy')
    for case in cases:
        with HostLatency() if args.host_diagnostics else nullcontext() as host:
            summary[case] = run_case(model,args,case,items,waves,configure=configure,observe=observe,
                                     measure=host.measure if host else None)
        if host:
            samples = json.loads((args.out/case/'wall.json').read_text())
            save(args.out/case/'host.json',host.report(samples))
        save(args.out/'summary.json',summary)
        assert all(v['equal']==v['total'] for mode in summary[case].values() for v in mode['agreement'].values())
        assert all(not b.failed for b in backends.values())


if __name__=='__main__':
    main()
