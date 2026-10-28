"""Compare saved beam B1/B3 or drafted greedy B1/bulk outputs; no model or audio operations."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from audit_large_regression import FIELDS, record_differences


def runtime_tree(commit: str) -> str:
    repo=Path(__file__).resolve().parents[1]
    return subprocess.check_output(['git','rev-parse',commit+':src/owsm_trt'],text=True,cwd=repo).strip()


def boundary_displacements(first: dict,third: dict) -> list[float] | None:
    a,b=first['segments'],third['segments']
    # Ordinal boundaries are comparable only when both segmentations have the same text labels.
    if first['text']!=third['text'] or [s['text'] for s in a]!=[s['text'] for s in b]:
        return None
    values=[abs(x[k]-y[k]) for x,y in zip(a,b) for k in ('start','end')]
    return values if all(math.isfinite(x) for x in values) else None


def distribution(values: list[float]) -> dict:
    ordered=sorted(values)
    return dict(count=len(values),median=statistics.median(values) if values else None,
                p95=ordered[math.ceil(.95*len(values))-1] if values else None,
                maximum=max(values) if values else None,percentile_method='nearest rank')


def compare_batches(first_dir: Path,third_dir: Path,*,greedy_bulk: bool=False) -> dict:
    roots=[first_dir,third_dir]
    second='bulk' if greedy_bulk else 'b3'
    variant='candidate' if greedy_bulk else 'reference'
    configs=[json.loads((r/'config.json').read_text()) for r in roots]
    for field in ('checkpoint_sha256','index_sha256','selected_keys_sha256','selected_samples'):
        assert configs[0][field]==configs[1][field],field
    assert all(c['mode']==('greedy_kernels' if greedy_bulk else 'beam_bound') for c in configs)
    assert [c['case'] for c in configs]==(['draft_ts','draft_bulk'] if greedy_bulk else ['beam_b1','beam_b3'])
    if greedy_bulk:
        assert configs[1]['request_batch_size']==8 and configs[1]['input_group_limit']==64
        assert all(c['greedy_variants']==dict(reference=False,candidate=True) for c in configs)
    environments=[json.loads((r/'environment.json').read_text()) for r in roots]
    for field in ('engine_sha256','torch','cuda','gpu'):
        assert environments[0][field]==environments[1][field],field
    if greedy_bulk:
        for field in ('greedy_settings','decoder_settings'):
            assert environments[0][field]['candidate']==environments[1][field]['candidate'],field
    runtime_trees=[runtime_tree(c['commit']) for c in configs]
    assert runtime_trees[0]==runtime_trees[1], 'runtime source changed as well as batch configuration'
    progress=[json.loads((r/'progress.json').read_text()) for r in roots]
    if greedy_bulk:
        for root,p in zip(roots,progress):
            status=json.loads((root/'status.json').read_text())
            assert status['complete'] and status['completed']==configs[0]['selected_samples']
            assert not status['interrupted']
            assert status['differences']==0 and status['chunks']==p['chunks']
    names=sorted(set(progress[0]['chunks']) & set(progress[1]['chunks']))
    if greedy_bulk:
        assert progress[0]['chunks'].keys()==progress[1]['chunks'].keys()
    counts={f:dict(equal=0,total=0) for f in FIELDS}
    groups=defaultdict(lambda:dict(samples=0,different_inputs=0,
        agreement={f:dict(equal=0,total=0) for f in FIELDS}))
    config_digests=[hashlib.sha256((r/'config.json').read_bytes()).hexdigest() for r in roots]
    ordered=[]
    quality=defaultdict(lambda:dict(samples=0,reference_units=0,b1_errors=0,**{second+'_errors':0}))
    bulk_groups={}
    changed=[]
    samples=0
    boundaries=[]
    boundary_inputs=0
    nonspeech=defaultdict(lambda:dict(samples=0,b1_nonempty=0,b1_insertions=0,
        **{second+'_nonempty':0,second+'_insertions':0}))
    for name in names:
        batches=[]
        for root,p in zip(roots,progress):
            path=root/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==p['chunks'][name]['sha256'],path
            with gzip.open(path,'rt') as stream:
                batches.append([json.loads(line) for line in stream])
        assert [r['key'] for r in batches[0]]==[r['key'] for r in batches[1]],name
        for first,third in zip(*batches):
            assert first['dataset']==third['dataset'] and first['lang']==third['lang']
            assert first['source']==third['source']
            for row,digest in zip((first,third),config_digests):
                assert row['config_sha256']==digest
                assert row['differences']==record_differences(row['reference'],row['candidate'])==[]
            if greedy_bulk:
                assert first['input_group']==[first['key']] and first['requested_batch_size']==1
                assert first['api']==third['api'] and third['requested_batch_size']==8
                members=tuple(third['input_group'])
                assert 0<len(members)<=64 and len(set(members))==len(members) and third['key'] in members
                group_state=bulk_groups.setdefault(members,dict(api=third['api'],language=third['lang'],seen=set()))
                assert group_state['api']==third['api'] and group_state['language']==third['lang']
                assert third['key'] not in group_state['seen']
                group_state['seen'].add(third['key'])
            a,b=first[variant],third[variant]
            assert a['token_ids_available']==b['token_ids_available']
            ordered.append(first['key'])
            group=groups[f"{third['api']}:group{len(third['input_group'])}"]
            group['samples']+=1
            fields=[]
            for field in FIELDS:
                if field in ('tokens','timestamp_tokens') and not (a['token_ids_available'] and b['token_ids_available']):
                    continue
                counts[field]['total']+=1
                counts[field]['equal']+=a[field]==b[field]
                group['agreement'][field]['total']+=1
                group['agreement'][field]['equal']+=a[field]==b[field]
                if a[field]!=b[field]:
                    fields.append(field)
            if fields:
                group['different_inputs']+=1
                changed.append(dict(key=first['key'],language=first['lang'],chunk=name,fields=fields,
                    **{second+'_input_group':third.get('input_group'),second+'_api':third.get('api')}))
            if first['dataset']!='stress':
                qa,qb=a['quality'],b['quality']
                assert qa['metric']==qb['metric'] and qa['reference_units']==qb['reference_units']
                q=quality[first['lang']]
                q['metric']=qa['metric']
                q['samples']+=1
                q['reference_units']+=qa['reference_units']
                q['b1_errors']+=qa['errors']
                q[second+'_errors']+=qb['errors']
                if fields:
                    changed[-1]['quality']=dict(metric=qa['metric'],reference_units=qa['reference_units'],
                        b1_errors=qa['errors'],**{second+'_errors':qb['errors']},error_delta=qb['errors']-qa['errors'])
            else:
                q=nonspeech[first['lang']]
                assert a['quality']['metric']==b['quality']['metric']
                q['metric']=a['quality']['metric']
                q['samples']+=1
                for label,record in [('b1',a),(second,b)]:
                    q[label+'_nonempty']+=bool(record['text'].strip())
                    q[label+'_insertions']+=record['quality']['errors']
            deltas=boundary_displacements(a,b)
            if fields:
                changed[-1].update(segment_counts=[len(a['segments']),len(b['segments'])],
                    comparable_boundary_count=len(deltas) if deltas is not None else None,
                    max_boundary_displacement=max(deltas,default=0) if deltas is not None else None)
            if deltas is not None:
                boundary_inputs+=1
                boundaries.extend(deltas)
            samples+=1
    if samples==configs[0]['selected_samples']:
        assert len(ordered)==len(set(ordered))
        assert hashlib.sha256('\n'.join(ordered).encode()).hexdigest()==configs[0]['selected_keys_sha256']
    if greedy_bulk:
        assert samples==configs[0]['selected_samples']
        for members,state in bulk_groups.items():
            assert set(members)==state['seen'], 'incomplete bulk request membership'
    for q in quality.values():
        for batch in ('b1',second):
            q[batch+'_rate']=q[batch+'_errors']/q['reference_units'] if q['reference_units'] else None
        q['error_delta']=q[second+'_errors']-q['b1_errors']
    result=dict(samples=samples,expected=configs[0]['selected_samples'],complete=samples==configs[0]['selected_samples'],
        classification=('N: request grouping, streaming/refill, batch and verification shapes, default decoder dispatch together.'
                        if greedy_bulk else 'N: batch shape and default decoder dispatch together; not an isolated kernel or upper-bound ablation.'),
        compared_variant=('Greedy kernels ON in both. Within each request configuration OFF/ON outputs must match first.'
                          if greedy_bulk else 'Upperbound OFF in both batches. Within each batch OFF/ON outputs must match before this comparison.'),
        configs=configs,environments=environments,runtime_tree=runtime_trees[0],
        agreement=counts,actual_groups=dict(groups),nonspeech=dict(nonspeech),
        boundary_displacements=dict(comparable_inputs=boundary_inputs,excluded_inputs=samples-boundary_inputs,
            all_boundaries=distribution(boundaries),changed_boundaries=distribution([x for x in boundaries if x>0]),
            scope=f'Absolute B1/{second} boundary displacement in seconds, only identical text and ordered segment texts/count. No ground-truth timing accuracy; excluded text/segmentation changes remain in output differences.'),
        different_inputs=len(changed),quality=dict(quality),changed_inputs=changed,
        timestamps='No human timing annotations; boundary differences are not correctness errors by themselves.',
        timing='No timing claim; these are saved-output regression artifacts.')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('b1',type=Path)
    ap.add_argument('other',type=Path)
    ap.add_argument('--greedy-bulk',action='store_true',help='Compare greedy_kernels draft_ts/draft_bulk candidates instead of beam_bound B1/B3 references')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result=compare_batches(args.b1,args.other,greedy_bulk=args.greedy_bulk)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temp=args.out.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    temp.replace(args.out)
    print(json.dumps({k:v for k,v in result.items() if k not in
                     ('configs','environments','changed_inputs')},indent=2))


if __name__=='__main__':
    main()
