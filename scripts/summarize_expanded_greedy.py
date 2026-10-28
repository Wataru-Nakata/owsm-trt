"""Validate expanded greedy timing artifacts and aggregate paired repeats without model operations."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from audit_large_regression import FIELDS


def stats(values: list[float]) -> dict:
    ordered=sorted(values)
    position=(len(ordered)-1)*.95
    lower=int(position)
    p95=ordered[lower]+(ordered[min(lower+1,len(ordered)-1)]-ordered[lower])*(position-lower)
    return dict(n=len(values),p50=statistics.median(values),p95=p95,
                minimum=min(values),maximum=max(values),stdev=statistics.pstdev(values),
                percentile_method='linear interpolation at (n-1)*q')


def aggregate(samples: list[dict], ids: list[str], repeats: int, streaming: bool) -> dict:
    groups=[tuple(ids)] if streaming else [(key,) for key in ids]
    expected={(v,r,g) for v in ('reference','candidate') for r in range(repeats) for g in groups}
    observed={}
    for sample in samples:
        key=(sample['variant'],sample['repeat'],tuple(sample['ids']))
        assert key in expected and key not in observed,'unexpected or duplicate timed request'
        for name in ('wall_ms','audio_seconds'):
            assert math.isfinite(sample[name]) and sample[name]>0,name
        observed[key]=sample
    assert observed.keys()==expected,'incomplete timed requests'
    for group in groups:
        durations=[observed[v,r,group]['audio_seconds'] for v in ('reference','candidate') for r in range(repeats)]
        assert max(durations)-min(durations)<1e-9,'audio duration changed across paired calls'
    totals={v:[sum(observed[v,r,g]['wall_ms'] for g in groups) for r in range(repeats)]
            for v in ('reference','candidate')}
    variants={}
    for variant in ('reference','candidate'):
        rows=[s for (v,_,_),s in observed.items() if v==variant]
        variants[variant]=dict(request_wall_ms=stats([s['wall_ms'] for s in rows]),
            repeat_total_ms=stats(totals[variant]),
            pooled_rtfx=1000*sum(s['audio_seconds'] for s in rows)/sum(totals[variant]),
            peak_allocated_bytes=max(s['peak_allocated_bytes'] for s in rows),
            draft_decoder_calls=sum(s['draft_decoder_calls'] for s in rows))
    paired=[100*(a/b-1) for a,b in zip(totals['reference'],totals['candidate'])]
    per_group=[]
    for group in groups:
        medians={v:statistics.median(observed[v,r,group]['wall_ms'] for r in range(repeats))
                 for v in ('reference','candidate')}
        per_group.append(dict(ids=list(group),median_wall_ms=medians,
                              rtfx_change_percent=100*(medians['reference']/medians['candidate']-1)))
    call_differences=[dict(repeat=r,ids=list(g),reference=observed['reference',r,g]['draft_decoder_calls'],
                          candidate=observed['candidate',r,g]['draft_decoder_calls'])
                      for r in range(repeats) for g in groups
                      if observed['reference',r,g]['draft_decoder_calls']!=observed['candidate',r,g]['draft_decoder_calls']]
    return dict(variants=variants,repeat_totals_ms=totals,
        paired_rtfx_change_percent=dict(samples=paired,stats=stats(paired),positive_repeats=sum(x>0 for x in paired)),
        pooled_rtfx_change_percent=100*(variants['candidate']['pooled_rtfx']/variants['reference']['pooled_rtfx']-1),
        request_median_comparisons=per_group,requests_with_faster_candidate_median=sum(
            x['rtfx_change_percent']>0 for x in per_group),draft_call_differences=call_differences,
        scope='Single: per-input transcribe latency. Streaming: one complete multi-batch request. '
              'Pooled RTFx uses all warm wall samples; paired statistics use repeat totals. '
              'Draft-call counts are not all decoder steps. Memory includes both resident variant pools.')


def output_agreement(tokens: dict, ids: list[str], exact: bool) -> dict:
    assert set(tokens)=={'reference','candidate'}
    assert all(set(records)==set(ids) for records in tokens.values())
    result={field:dict(equal=sum(tokens['reference'][key][field]==tokens['candidate'][key][field] for key in ids),
        total=len(ids),different_ids=[key for key in ids if tokens['reference'][key][field]!=tokens['candidate'][key][field]])
        for field in FIELDS}
    if exact:
        assert all(v['equal']==v['total'] for v in result.values()),'E output difference'
    return result


def summarize(raw: Path) -> dict:
    profile=raw/'profile'
    assert (raw/'exit_status.txt').read_text().strip()=='0'
    scheduler=json.loads((raw/'pbs-final.json').read_text())
    assert scheduler['state']=='F' and scheduler['exit_status']==0
    source=json.loads((raw/'source.json').read_text())
    env=json.loads((profile/'environment.json').read_text())
    assert not env.get('host_diagnostics',False),'instrumented host diagnostics are not warm latency results'
    complete=json.loads((profile/'complete.json').read_text())
    assert complete['complete'] and complete['job']==env['job']==scheduler['job']
    assert env['commit']==source['commit'] and env['repeat']==complete['repeat']==7
    selection_path=profile/'selection.json'
    assert hashlib.sha256(selection_path.read_bytes()).hexdigest()==env['selection_sha256']
    selected=json.loads(selection_path.read_text())
    assert len(selected)==7 and all(len(rows)==32 for rows in selected.values())
    assert complete['samples']==sum(map(len,selected.values()))==224
    summaries=json.loads((profile/'summary.json').read_text())
    comparison=env.get('comparison','greedy_kernels')
    assert comparison in ('greedy_kernels','draft_budget','frontend_batches')
    exact=comparison!='draft_budget'
    cases=('single_ts','stream_ts_b8') if comparison=='greedy_kernels' else ('stream_ts_b8',)
    assert env.get('classification','E')==('E' if exact else 'N')
    assert tuple(env.get('cases',cases))==cases
    if not exact:
        assert env['classification']=='N' and env['draft_token_budgets']==dict(reference=256,candidate=128)
        assert env['greedy_settings']['reference']==env['greedy_settings']['candidate']
    if comparison=='frontend_batches':
        assert env['frontend_batch_limits']==dict(reference=1,candidate=8)
        assert env['draft_token_budgets']==dict(reference=256,candidate=256)
        assert env['greedy_settings']['reference']==env['greedy_settings']['candidate']
    expected={lang+'/'+case for lang in selected for case in cases}
    assert summaries.keys()==expected and complete['conditions']==len(expected)
    result={}
    for name in sorted(expected):
        lang,case=name.split('/')
        folder=profile/name
        ids=[r['key'] for r in selected[lang]]
        assert len(set(ids))==len(ids)
        tokens=json.loads((folder/'tokens.json').read_text())
        agreement=output_agreement(tokens,ids,exact)
        assert agreement==summaries[name]['candidate']['agreement']
        assert all(a==dict(equal=32,total=32,different_ids=[])
                   for a in summaries[name]['reference']['agreement'].values())
        for variant in ('reference','candidate'):
            assert all(v['total']==32 for v in summaries[name][variant]['agreement'].values())
            assert (folder/f'actual_dispatch_{variant}.json').is_file()
        cold=json.loads((folder/'cold.json').read_text())
        samples=json.loads((folder/'wall.json').read_text())
        value=aggregate(samples,ids,env['repeat'],case=='stream_ts_b8')
        for variant in ('reference','candidate'):
            assert math.isclose(value['variants'][variant]['pooled_rtfx'],summaries[name][variant]['rtfx'],rel_tol=1e-12)
        value.update(quality={v:summaries[name][v]['quality'] for v in ('reference','candidate')},
            cold=cold,agreement=summaries[name]['candidate']['agreement'],
            artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.json')})
        if comparison=='frontend_batches':
            caches=json.loads((folder/'frontend_cache.json').read_text())
            for variant,limit in env['frontend_batch_limits'].items():
                cache=caches[variant]
                assert cache['max_batch']==limit and cache['max_entries']==32 and cache['reuse_warmup_stream']
                assert cache['captures']==len(cache['shapes'])<=32
                assert all(1<=shape[0]<=limit for shape in cache['shapes'])
            assert any(shape[0]>1 for shape in caches['candidate']['shapes']),'batched frontend never captured'
            value['frontend_cache']=caches
        result[name]=value
    with (raw/'gpu_utilization.csv').open() as stream:
        telemetry=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(stream)]
    utilization=[float(r['utilization.gpu [%]'].split()[0]) for r in telemetry]
    memory=[float(r['memory.used [MiB]'].split()[0]) for r in telemetry]
    return dict(source=source,environment=env,scheduler=scheduler,complete=complete,conditions=result,
        comparison=comparison,classification='E' if exact else 'N',
        gpu_telemetry=dict(utilization_percent=stats(utilization),device_memory_mib=stats(memory),
            scope='Five-second samples across loading/capture/warm/diagnostic calls and both variants; '
                  'not per-condition utilization or kernel occupancy.'),
        scope=('Same-engine combined public greedy switch. ' if comparison=='greedy_kernels' else
               'Public frontend graph batch limit1/8, fresh bounded cache per language. ' if comparison=='frontend_batches' else
               'N: same-engine public draft budget256/128, greedy kernels ON both; differences retained. ')+
              'Balanced 224-input speech sample, '
              'not an independent quality holdout or corpus-weighted performance. Seven repeats '
              'are repeated timings, not additional evaluation inputs. Raw files retain per-call '
              'wall time/output/dispatch/cold metadata. No timing from the full-corpus one-pass audit.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('raw',type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    result=summarize(args.raw)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({name:dict(pooled_rtfx_change_percent=v['pooled_rtfx_change_percent'],
        positive_repeats=v['paired_rtfx_change_percent']['positive_repeats'],
        call_differences=len(v['draft_call_differences'])) for name,v in result['conditions'].items()},indent=2))


if __name__=='__main__':
    main()
